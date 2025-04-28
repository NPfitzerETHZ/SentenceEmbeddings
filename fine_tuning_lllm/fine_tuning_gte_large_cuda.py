import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from transformers import AutoModel, AutoTokenizer
import tqdm
import wandb
import numpy as np

from process_gemini_data import collect_train_test_data_grid_texts

# -------------------------------
# Configuration
# -------------------------------
GTE_MODEL_NAME = 'thenlper/gte-large'
EMBED_SIZE = 1024
HIDDEN_SIZE = 256
BATCH_SIZE = 128
EPOCHS = 500
NUM_EVALS_PER_EPOCH = 2
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 100

EXPERIMENT_NAME = "minimal_finetuning_gte_large_multi_target"
DATA_PATH = "data/language_data_complete_multi_target.json"
CHECKPOINT_DIR = os.path.join("checkpoints", EXPERIMENT_NAME)
FINAL_DECODER_PATH = f"decoder_{EXPERIMENT_NAME}.pth"
FINAL_MODEL_DIR = f"finetuned_{EXPERIMENT_NAME}"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Models
# -------------------------------

class GTEModel(nn.Module):
    def __init__(self, model_name=GTE_MODEL_NAME):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling = lambda x: x.last_hidden_state[:, 0]  # CLS token

    def forward_from_tokens(self, tokens):
        output = self.model(**tokens)
        return self.pooling(output)

def forward_from_tokens(model, tokens):
    if isinstance(model, nn.DataParallel):
        return model.module.forward_from_tokens(tokens)
    return model.forward_from_tokens(tokens)

def freeze_all_but_last_layers(model, n_last_layers=1):
    for param in model.parameters():
        param.requires_grad = False
    for param in list(model.parameters())[-n_last_layers * 16:]:
        param.requires_grad = True

class Decoder(nn.Module):
    def __init__(self, emb_size, out_size, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.norm_input = nn.LayerNorm(emb_size)
        self.l0 = nn.Linear(emb_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, out_size)
        self.norm_hidden = nn.LayerNorm(hidden_size)
        self.act = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.norm_input(x)
        x = self.dropout(self.act(self.l0(x)))
        x = self.norm_hidden(x)
        x = self.dropout(self.act(self.l1(x)))
        return torch.tanh(self.l2(x))

# -------------------------------
# Utility Functions
# -------------------------------

def tokenize_data(tokenizer, texts, device):
    with torch.no_grad():
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    return {k: v.to(device) for k, v in tokens.items()}

def evaluate(model, decoder, tokenized_test, test_goals, batch_size, device):
    with torch.no_grad():
        val_idx = torch.randint(0, test_goals.shape[0] - batch_size, (1,)).item()
        batch_tokens = {k: v[val_idx:val_idx + batch_size] for k, v in tokenized_test.items()}
        val_goal = test_goals[val_idx:val_idx + batch_size].clone().detach().to(device)
        emb = forward_from_tokens(model, batch_tokens)
        pred = decoder(emb)
        return torch.mean(torch.norm(pred - val_goal, dim=-1)).item()

def make_checkpoint_name(name="best_checkpoint"):
    return os.path.join(CHECKPOINT_DIR, f"{name}.pt")

# -------------------------------
# Main Training Loop
# -------------------------------

def train():
    wandb.init(project='grid_decoder_llm_fine_tuned', name=EXPERIMENT_NAME)
    print("Loading data...")
    train, test = collect_train_test_data_grid_texts(DATA_PATH, 0.8, 0.2, DEVICE)

    gte_model = GTEModel().to(DEVICE)
    freeze_all_but_last_layers(gte_model.model, n_last_layers=1)
    decoder = Decoder(EMBED_SIZE, train["goal"].shape[1]).to(DEVICE)

    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        gte_model = nn.DataParallel(gte_model)
        decoder = nn.DataParallel(decoder)

    scaler = GradScaler()

    print("Tokenizing...")
    tokenizer = gte_model.module.tokenizer if isinstance(gte_model, nn.DataParallel) else gte_model.tokenizer
    tokenized_train = tokenize_data(tokenizer, train["task_text"], DEVICE)
    tokenized_test = tokenize_data(tokenizer, test["task_text"], DEVICE)

    params = list(filter(lambda p: p.requires_grad, gte_model.parameters())) + list(decoder.parameters())
    optimizer = optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float('inf')

    num_batches = train["goal"].shape[0] // BATCH_SIZE
    eval_steps = set([
        (num_batches * (i + 1)) // (NUM_EVALS_PER_EPOCH + 1)
        for i in range(NUM_EVALS_PER_EPOCH)
    ])
    pbar = tqdm.tqdm(total=EPOCHS * num_batches)
    patience_counter = 0
    val_loss = 0.0
    for epoch in range(EPOCHS):

        perm = torch.randperm(train["goal"].shape[0])
        tokenized_train = {k: v[perm] for k, v in tokenized_train.items()}
        train["goal"] = train["goal"][perm]

        for i in range(num_batches):
            start, end = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
            tokens = {k: v[start:end] for k, v in tokenized_train.items()}
            goal = torch.tensor(train["goal"][start:end], dtype=torch.float32).to(DEVICE)

            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                emb = forward_from_tokens(gte_model, tokens)
                pred = decoder(emb)
                loss = torch.mean(torch.norm(pred - goal, dim=-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
            if i in eval_steps:
                val_loss = evaluate(gte_model, decoder, tokenized_test, test["goal"], BATCH_SIZE, DEVICE)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save({
                        'epoch': epoch,
                        'decoder_state_dict': decoder.module.state_dict() if isinstance(decoder, nn.DataParallel) else decoder.state_dict(),
                        'gte_model_state_dict': gte_model.module.model.state_dict() if isinstance(gte_model, nn.DataParallel) else gte_model.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss
                    }, make_checkpoint_name())
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f"Early stopping triggered at epoch {epoch}, batch {i}")
                        wandb.finish()
                        return
            
            pbar.update()
            pbar.set_description(f"Epoch: {epoch}, loss: {loss.item():0.4f}, val_loss: {val_loss:0.4f}, best val_loss: {best_val_loss:0.4f}")
            wandb.log({
                "epoch": epoch,
                "loss": loss.item(),
                "eval_loss": val_loss,
                "best_eval_loss": best_val_loss,
            })

    wandb.finish()

    # Final save
    torch.save(decoder.module.state_dict() if isinstance(decoder, nn.DataParallel) else decoder.state_dict(), FINAL_DECODER_PATH)
    model_to_save = gte_model.module if isinstance(gte_model, nn.DataParallel) else gte_model
    model_to_save.model.save_pretrained(FINAL_MODEL_DIR)
    model_to_save.tokenizer.save_pretrained(FINAL_MODEL_DIR)

# -------------------------------
# Entry Point
# -------------------------------

if __name__ == "__main__":
    train()
