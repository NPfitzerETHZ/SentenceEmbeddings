import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
import pandas as pd
from sentence_transformers import SentenceTransformer
from heading_ideas import Heading
from process_gemini_data import collect_train_test_data_grid

# Define available LLMs
llms = {
#    SentenceTransformer('BAAI/bge-large-en-v1.5'): "BAAI/bge-large-en-v1.5",
#    SentenceTransformer('hkunlp/instructor-large'): "hkunlp/instructor-large",
    SentenceTransformer('thenlper/gte-large'): "thenlper/gte-large"
}

json_data_file = "gemini_patch_dataset_grid.json"

# Define the Decoder model in PyTorch
class Decoder(nn.Module):
    def __init__(self, emb_size, out_size):
        super(Decoder, self).__init__()
        hidden_size = 256
        self.l0 = nn.Linear(emb_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, out_size)
        self.relu = nn.ReLU()

    def forward(self, embed):
        x = self.relu(self.l0(embed))
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return self.l3(x)

# Define loss function
def loss_fn(model, emb, goal):
    pred = model(emb)
    return torch.mean(torch.norm(pred - goal, dim=-1))

# def weighted_loss_fn(model, emb, goal, weights=None):

#     pred = model(emb)
#     if weights is None:
#         weights = torch.tensor([2.0, 2.0, 1.0, 1.0], device=pred.device)
#     diff = pred - goal
#     weighted_diff = diff * weights
#     weighted_loss = torch.mean(torch.norm(weighted_diff, dim=-1))
#     return weighted_loss

def loss_fn(model, emb, goal):
    
    pred = model(emb)
    diff = pred - goal
    return torch.mean(torch.norm(diff, dim=-1))

results = {}
m = 0

for llm, llm_name in llms.items():
    wandb.init(project='grid_decoder_llm', name=llm_name)
    batch_size = 16
    epochs = 130

    heading = Heading(batch_size=batch_size * 300, num_targets=2)
    train = heading._initalize_heading(eval=False, model=llm)
    test = heading._initalize_heading(eval=True, model=llm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Decoder(train["task_embedding"].shape[1],train["goal"].shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    
    pbar = tqdm.tqdm(total=epochs * train["task_embedding"].shape[0] // batch_size)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        for i in range(train["task_embedding"].shape[0] // batch_size):
            emb = torch.tensor(train["task_embedding"][i * batch_size : (i + 1) * batch_size], dtype=torch.float32).to(device)
            goal = torch.tensor(train["goal"][i * batch_size : (i + 1) * batch_size], dtype=torch.float32).to(device)

            optimizer.zero_grad()
            loss = loss_fn(model, emb, goal)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                v_emb = torch.tensor(test["task_embedding"], dtype=torch.float32).to(device)
                v_goal = torch.tensor(test["goal"], dtype=torch.float32).to(device)
                val_loss = loss_fn(model, v_emb, v_goal)
                best_val_loss = min(val_loss.item(), best_val_loss)

            pbar.update()
            pbar.set_description(f"LLM: {llm_name}, epoch: {epoch}, loss: {loss.item():0.4f}, val_loss: {val_loss.item():0.4f}, best val_loss: {best_val_loss:0.4f}")
            wandb.log({
                "epoch": epoch,
                "loss": loss.item(),
                "eval_loss": val_loss.item(),
                "best_eval_loss": best_val_loss,
            })
    
    results[llm_name] = best_val_loss
    wandb.finish()
    torch.save(model.state_dict(), f"llm{m}_decoder_model.pth")
    m += 1

print(results)