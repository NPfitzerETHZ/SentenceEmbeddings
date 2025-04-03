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
CHECKPOINT_DIR = "checkpoints"
DEVICE = "cuda"

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
        val_goal = torch.tensor(test_goals[val_idx:val_idx + batch_size], dtype=torch.float32).to(device)
        emb = model.forward_from_tokens(batch_tokens)
        pred = decoder(emb)
        return torch.mean(torch.norm(pred - val_goal, dim=-1)).item()

# -------------------------------
# Main Training Loop
# -------------------------------

def train():
    wandb.init(project='grid_decoder_llm_fine_tuned', name="minimal_finetuning_gte_large_single_target")

    gte_model = GTEModel().to(DEVICE)
    freeze_all_but_last_layers(gte_model.model, n_last_layers=1)
    scaler = GradScaler()

    print("Loading data...")
    train, test = collect_train_test_data_grid_texts("data/language_data_complete_single_target.json", 0.8, 0.2, DEVICE)

    print("Tokenizing...")
    tokenized_train = tokenize_data(gte_model.tokenizer, train["task_text"], DEVICE)
    tokenized_test = tokenize_data(gte_model.tokenizer, test["task_text"], DEVICE)

    decoder = Decoder(EMBED_SIZE, train["goal"].shape[1]).to(DEVICE)
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
    for epoch in range(EPOCHS):
        
        # Shuffle Data
        perm = torch.randperm(train["goal"].shape[0])
        tokenized_train = {k: v[perm] for k, v in tokenized_train.items()}
        train["goal"] = train["goal"][perm]
        
        for i in range(num_batches):
            start, end = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
            tokens = {k: v[start:end] for k, v in tokenized_train.items()}
            goal = torch.tensor(train["goal"][start:end], dtype=torch.float32).to(DEVICE)

            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                emb = gte_model.forward_from_tokens(tokens)
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
                        'decoder_state_dict': decoder.state_dict(),
                        'gte_model_state_dict': gte_model.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss
                    }, os.path.join(CHECKPOINT_DIR, "best_checkpoint.pt"))
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f"Early stopping triggered at epoch {epoch}, batch {i}")
                        wandb.finish()
                        return  # Exit training early

    wandb.finish()
    torch.save(decoder.state_dict(), "llm_decoder_model_grid_single_target.pth")
    gte_model.model.save_pretrained("finetuned_gte_large_single_target")
    gte_model.tokenizer.save_pretrained("finetuned_gte_large_single_target")

# -------------------------------
# Entry Point
# -------------------------------

if __name__ == "__main__":
    train()
