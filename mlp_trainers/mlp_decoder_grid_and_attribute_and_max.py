import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
import pandas as pd
from sentence_transformers import SentenceTransformer
from process_gemini_data import collect_train_test_data_grid, collect_train_test_data_from_embeddings_attributes_max
import torch.nn.functional as F

# Define available LLMs
llms = {
#    SentenceTransformer('BAAI/bge-large-en-v1.5'): "BAAI/bge-large-en-v1.5",
#    SentenceTransformer('hkunlp/instructor-large'): "hkunlp/instructor-large",
    SentenceTransformer('thenlper/gte-large'): "thenlper/gte-large"
}

json_data_file = "data/language_data_complete_multi_target_color_medium.json"

# Define the Decoder model in PyTorch
# class Decoder(nn.Module):
#     def __init__(self, emb_size, out_size):
#         super(Decoder, self).__init__()
#         hidden_size = 256
#         self.l0 = nn.Linear(emb_size, hidden_size)
#         self.l1 = nn.Linear(hidden_size, hidden_size)
#         self.l2 = nn.Linear(hidden_size, hidden_size)
#         self.l3 = nn.Linear(hidden_size, hidden_size)
#         self.l4 = nn.Linear(hidden_size, out_size)
#         #self.l2 = nn.Linear(hidden_size, out_size)
#         self.relu = nn.ReLU()
#         self.leakyrelu = nn.LeakyReLU(0.1)

#     def forward(self, embed):
#         x = self.leakyrelu(self.l0(embed))
#         x = self.leakyrelu(self.l1(x))
#         x = self.leakyrelu(self.l2(x))
#         x = self.leakyrelu(self.l3(x))
#         return self.l4(x)
    
class Decoder(nn.Module):
    def __init__(self, emb_size, out_size):
        super(Decoder, self).__init__()
        hidden_size = 256
        self.l0 = nn.Linear(emb_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, out_size)
        #self.l2 = nn.Linear(hidden_size, out_size)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, embed):
        x = self.leakyrelu(self.l0(embed))
        x = self.leakyrelu(self.l1(x))
        x = self.leakyrelu(self.l2(x))
        return self.l3(x)
    
class MultiHeadDecoder(nn.Module):
    def __init__(self, emb_size, goal_out_size, num_classes, num_max_targets, shared_depth=3, hidden_size=613):
        super(MultiHeadDecoder, self).__init__()
        assert 0 <= shared_depth <= 3, "shared_depth must be between 0 and 3"
        self.activation = nn.LeakyReLU(0.1)
        self.shared_depth = shared_depth
        self.hidden_size = hidden_size

        # Shared layers
        self.shared_layers = nn.ModuleList()
        self.shared_layers.append(nn.Linear(emb_size, hidden_size))  # l0
        self.shared_layers.append(nn.Linear(hidden_size, hidden_size))  # l1
        self.shared_layers.append(nn.Linear(hidden_size, hidden_size))  # l2

        # Optional input projections if no layers are shared
        if shared_depth == 0:
            self.goal_input_proj = nn.Linear(emb_size, hidden_size)
            self.class_input_proj = nn.Linear(emb_size, hidden_size)
            self.max_target_input_proj = nn.Linear(emb_size, hidden_size)

        # Task-specific branches
        self.goal_layers = nn.ModuleList()
        self.class_layers = nn.ModuleList()
        self.max_target_layers = nn.ModuleList()
        for _ in range(3 - shared_depth):
            self.goal_layers.append(nn.Linear(hidden_size, hidden_size))
            self.class_layers.append(nn.Linear(hidden_size, hidden_size))
            self.max_target_layers.append(nn.Linear(hidden_size, hidden_size))

        # Output heads
        self.goal_head = nn.Linear(hidden_size, goal_out_size)
        self.class_head = nn.Linear(hidden_size, num_classes)
        self.max_target_head = nn.Linear(hidden_size, num_max_targets)

    def forward(self, emb):
        if self.shared_depth > 0:
            x = emb
            for i in range(self.shared_depth):
                x = self.activation(self.shared_layers[i](x))
            goal_x = x
            class_x = x
            max_target_x = x
        else:
            goal_x = self.activation(self.goal_input_proj(emb))
            class_x = self.activation(self.class_input_proj(emb))
            max_target_x = self.activation(self.max_target_input_proj(emb))
        for layer in self.max_target_layers:
            max_target_x = self.activation(layer(max_target_x))
        for layer in self.goal_layers:
            goal_x = self.activation(layer(goal_x))
        for layer in self.class_layers:
            class_x = self.activation(layer(class_x))

        goal_pred = self.goal_head(goal_x)
        class_logits = self.class_head(class_x)
        max_target_logits = self.max_target_head(max_target_x)

        return goal_pred, class_logits, max_target_logits

# Define loss function
def loss_fn(model, emb, goal, class_goal):
    pred = model(emb)
    return torch.mean(torch.norm(pred - goal * class_goal.unsqueeze(1) / 11, dim=-1))
    #return torch.mean(torch.norm(pred - goal, dim=-1))

def loss_fn_class(model, emb, class_goal, alpha=0.01):
    # Get model prediction
    pred = model(emb)  # Expected shape: (batch_size, num_classes)

    # Apply softmax to get class probabilities
    prob = F.softmax(pred, dim=1)

    # Classification loss: mean squared error between predicted probs and one-hot target
    loss_main = F.mse_loss(prob, class_goal)

    # Regularization: encourage confident (low-entropy) outputs
    entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)  # shape: (batch_size,)
    loss_entropy = torch.mean(entropy)

    # Total loss
    return loss_main + alpha * loss_entropy

def combined_loss_fn(model, emb, goal, class_goal, max_target_goal, alpha=0.01, beta=1.0):
    goal_pred, class_logits, max_target_logits = model(emb)

    # Classification loss 1
    prob = F.softmax(class_logits, dim=1)
    loss_class = F.mse_loss(prob, class_goal)
    entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1).mean()
    loss_class_total = loss_class + alpha * entropy
    
    # Classification loss 2 (for max_target)
    prob_max_target = F.softmax(max_target_logits, dim=1)
    loss_max_target = F.mse_loss(prob_max_target, max_target_goal)
    entropy_max_target = -torch.sum(prob_max_target * torch.log(prob_max_target + 1e-8), dim=1).mean()
    loss_class_total += loss_max_target + alpha * entropy_max_target

    # Goal loss 
    loss_goal = torch.mean(torch.norm(goal_pred - goal, dim=-1))

    # Total loss
    total_loss = loss_goal + beta * loss_class_total
    return total_loss, loss_goal, loss_class_total


results = {}
m = 0
patience_counter = 0
for llm, llm_name in llms.items():
    wandb.init(project='grid_decoder_llm', name=llm_name)
    batch_size = 128
    epochs = 2000

    train, test = collect_train_test_data_from_embeddings_attributes_max(json_path=json_data_file,train_ratio=0.8,test_ratio=0.2, device="mps")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = MultiHeadDecoder(train["task_embedding"].shape[1],train["goal"].shape[1],num_classes=11,num_max_targets=4).to(device)
    #model = Decoder(train["task_embedding"].shape[1],11).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    
    pbar = tqdm.tqdm(total=epochs)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        for i in range(train["task_embedding"].shape[0] // batch_size):
            # Get batch
            emb = torch.tensor(train["task_embedding"][i * batch_size : (i + 1) * batch_size], dtype=torch.float32).to(device)
            goal = torch.tensor(train["goal"][i * batch_size : (i + 1) * batch_size], dtype=torch.float32).to(device)
            class_goal = torch.tensor(train["class"][i * batch_size : (i + 1) * batch_size], dtype=torch.long).to(device)
            max_target_goal = torch.tensor(train["max_targets"][i * batch_size : (i + 1) * batch_size], dtype=torch.long).to(device)

            # One-hot encoding for class targets
            class_goal_onehot = F.one_hot(class_goal, num_classes=11).float()
            max_target_goal_onehot = F.one_hot(max_target_goal, num_classes=4).float()

            # Zero gradients
            optimizer.zero_grad()

            # Compute losses
            loss, goal_loss, class_loss = combined_loss_fn(model, emb, goal, class_goal_onehot, max_target_goal_onehot)

            # Backprop
            loss.backward()
            optimizer.step()

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            v_emb = torch.tensor(test["task_embedding"], dtype=torch.float32).to(device)
            v_goal = torch.tensor(test["goal"], dtype=torch.float32).to(device)
            v_class_goal = torch.tensor(test["class"], dtype=torch.long).to(device)
            v_class_goal_onehot = F.one_hot(v_class_goal, num_classes=11).float()
            v_max_target_goal = torch.tensor(test["max_targets"], dtype=torch.long).to(device)
            v_max_target_goal_onehot = F.one_hot(v_max_target_goal, num_classes=4).float()

            v_total_loss, v_goal_loss, v_class_loss = combined_loss_fn(model, v_emb, v_goal, v_class_goal_onehot, v_max_target_goal_onehot)
            best_val_loss = min(v_total_loss.item(), best_val_loss)

            pbar.update()
            pbar.set_description(f"LLM: {llm_name}, epoch: {epoch}, loss: {loss.item():0.4f}, val_loss: {v_total_loss.item():0.4f}, best val_loss: {best_val_loss:0.4f}")
            wandb.log({
                "epoch": epoch,
                "loss": loss.item(),
                "eval_loss": v_total_loss.item(),
                "best_eval_loss": best_val_loss,
            })
    
    results[llm_name] = best_val_loss
    wandb.finish()
    torch.save(model.state_dict(), f"llm{m}_decoder_model_grid_single_target_color_class.pth")
    m += 1

print(results)