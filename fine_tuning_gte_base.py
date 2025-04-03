from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.optim as optim
import tqdm
import torch
import wandb
from process_gemini_data import collect_train_test_data_grid_texts

GTE_LARGE_EMBED_SIZE = 1024

class GTEModel(nn.Module):
    def __init__(self, model_name='thenlper/gte-large'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling = lambda x: x.last_hidden_state[:, 0]  # CLS token pooling

    def forward(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
        model_output = self.model(**encoded_input)
        return self.pooling(model_output)
    
def freeze_all_but_last_layers(model, n_last_layers=2):
    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in list(model.named_parameters())[-n_last_layers * 16:]:  # 16 params per layer
        param.requires_grad = True
    
    # for i, (name, _) in enumerate(model.named_parameters()):
    #     print(i, name)

class Decoder(nn.Module):
    def __init__(self, emb_size, out_size):
        super(Decoder, self).__init__()
        hidden_size = 256
        self.l0 = nn.Linear(emb_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, embed):
        x = self.relu(self.l0(embed))
        x = self.relu(self.l1(x))
        return torch.tanh(self.l2(x))

# Define loss function
def loss_fn(model, emb, goal):
    pred = model(emb)
    return torch.mean(torch.norm(pred - goal, dim=-1))


results = {}
wandb.init(project='grid_decoder_llm_fine_tuned', name="finetuning_gte_large")
json_data_file = "data/language_data_complete_danger&target.json"

batch_size = 128
epochs = 0
device = "mps"

train, test =  collect_train_test_data_grid_texts(json_path=json_data_file, train_ratio=0.8,test_ratio=0.2, device=device)

gte_model = GTEModel().to(device)
freeze_all_but_last_layers(gte_model.model, n_last_layers=2)

decoder = Decoder(emb_size=GTE_LARGE_EMBED_SIZE, out_size=train["goal"].shape[1]).to(device)
params = list(filter(lambda p: p.requires_grad, gte_model.parameters())) + list(decoder.parameters())
optimizer = optim.AdamW(params, lr=1e-5)

pbar = tqdm.tqdm(total=epochs * train["goal"].shape[0] // batch_size)
import os

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Optional: load checkpoint if exists
# checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_checkpoint.pt")
# if os.path.exists(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path)
#     decoder.load_state_dict(checkpoint['decoder_state_dict'])
#     gte_model.model.load_state_dict(checkpoint['gte_model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch'] + 1
#     best_val_loss = checkpoint['best_val_loss']
# else:
#     start_epoch = 0
#     best_val_loss = float('inf')

# Replace this with the line below if using resume logic
best_val_loss = float('inf')

for epoch in range(epochs):  # replace with `range(start_epoch, epochs)` if resuming
    for i in range(train["goal"].shape[0] // batch_size):

        text_batch = train["task_text"][i * batch_size : (i + 1) * batch_size]
        goal = torch.tensor(train["goal"][i * batch_size : (i + 1) * batch_size], dtype=torch.float32).to(device)
        
        emb = gte_model(text_batch)  # returns tensor
        pred = decoder(emb)
        optimizer.zero_grad()
        loss = torch.mean(torch.norm(pred - goal, dim=-1))
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            v_emb = gte_model(text_batch)
            v_pred = decoder(v_emb)
            val_loss = torch.mean(torch.norm(v_pred - goal, dim=-1))

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save({
                'epoch': epoch,
                'decoder_state_dict': decoder.state_dict(),
                'gte_model_state_dict': gte_model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, os.path.join(CHECKPOINT_DIR, "best_checkpoint.pt"))

        pbar.update()
        pbar.set_description(f"Epoch: {epoch}, loss: {loss.item():0.4f}, val_loss: {val_loss.item():0.4f}, best val_loss: {best_val_loss:0.4f}")
        wandb.log({
            "epoch": epoch,
            "loss": loss.item(),
            "eval_loss": val_loss.item(),
            "best_eval_loss": best_val_loss,
        })

# Final saving
wandb.finish()
torch.save(decoder.state_dict(), f"llm_decoder_model_grid_danger&target.pth")
gte_model.model.save_pretrained("finetuned_gte_large_danger_target_patch_pair")
gte_model.tokenizer.save_pretrained("finetuned_gte_large_danger_target_patch_pair")

# Load with:
#from transformers import AutoModel, AutoTokenizer

# model = AutoModel.from_pretrained("finetuned_gte_large")
# tokenizer = AutoTokenizer.from_pretrained("finetuned_gte_large")