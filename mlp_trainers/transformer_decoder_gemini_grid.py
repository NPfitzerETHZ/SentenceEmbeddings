import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
import pandas as pd
from sentence_transformers import SentenceTransformer
from heading_ideas import Heading
from process_gemini_data import collect_train_test_data_grid, collect_train_test_data_from_embeddings

# Define available LLMs
llms = {
#    SentenceTransformer('BAAI/bge-large-en-v1.5'): "BAAI/bge-large-en-v1.5",
#    SentenceTransformer('hkunlp/instructor-large'): "hkunlp/instructor-large",
    SentenceTransformer('thenlper/gte-large'): "thenlper/gte-large"
}

json_data_file = "data/language_data_complete_danger&target.json"

class Decoder(nn.Module):
    def __init__(self, emb_size, out_size, hidden_size=256):
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


class TransformerDecoder(nn.Module):
    def __init__(self, emb_size, out_size, hidden_size=256, num_chunks=32, transformer_dim=24, num_heads=8, num_layers=1):
        super().__init__()
        
        assert emb_size % num_chunks == 0, "Embedding size must be divisible by num_chunks"
        self.chunk_size = emb_size // num_chunks

        self.input_norm = nn.LayerNorm(emb_size)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.chunk_size, 
            nhead=num_heads, 
            dim_feedforward=transformer_dim * 2, 
            dropout=0.1,
            batch_first=True  # Important: so input is (batch, seq, dim)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP Decoder
        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, out_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_norm(x)  # (B, E)
        B, E = x.shape
        x = x.view(B, -1, self.chunk_size)  # (B, num_chunks, chunk_size)
        x = self.transformer(x)             # (B, num_chunks, chunk_size)
        x = x.view(B, -1)                   # Flatten back to (B, E)
        return self.mlp(x)


# Define loss function
def loss_fn(model, emb, goal):
    pred = model(emb)
    return torch.mean(torch.norm(pred - goal, dim=-1))


results = {}
m = 0
patience_counter = 0
for llm, llm_name in llms.items():
    wandb.init(project='grid_decoder_llm', name=llm_name)
    batch_size = 128
    epochs = 2500
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    train, test = collect_train_test_data_from_embeddings(json_path=json_data_file,train_ratio=0.8,test_ratio=0.2, device=device)

    model = TransformerDecoderDecoder(
        emb_size=train["task_embedding"].shape[1],
        out_size=train["goal"].shape[1]
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    
    pbar = tqdm.tqdm(total=epochs * train["task_embedding"].shape[0] // batch_size)
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(train["task_embedding"].shape[0] // batch_size):
            emb = torch.tensor(train["task_embedding"][i * batch_size : (i + 1) * batch_size], dtype=torch.float32).to(device)
            goal = torch.tensor(train["goal"][i * batch_size : (i + 1) * batch_size], dtype=torch.float32).to(device)

            optimizer.zero_grad()
            loss = loss_fn(model, emb, goal)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.update()

        with torch.no_grad():
            v_emb = torch.tensor(test["task_embedding"], dtype=torch.float32).to(device)
            v_goal = torch.tensor(test["goal"], dtype=torch.float32).to(device)
            val_loss = loss_fn(model, v_emb, v_goal).item()

        if val_loss < best_val_loss - 1e-5:  # Small margin to avoid noise-based updates
            best_val_loss = val_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        pbar.set_description(f"LLM: {llm_name}, epoch: {epoch}, train_loss: {epoch_loss:.4f}, val_loss: {val_loss:.4f}, best: {best_val_loss:.4f}")
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "eval_loss": val_loss,
            "best_eval_loss": best_val_loss,
        })

        if epochs_since_improvement > 300:
            print(f"No improvement in 40 epochs, stopping early at epoch {epoch}.")
            break
    
    results[llm_name] = best_val_loss
    wandb.finish()
    torch.save(model.state_dict(), f"decoders/llm{m}_transformer_model_single_target.pth")
    m += 1

print(results)