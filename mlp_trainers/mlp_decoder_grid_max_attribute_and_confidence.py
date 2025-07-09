import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
import pandas as pd
from sentence_transformers import SentenceTransformer
from process_gemini_data import collect_train_test_data_from_embeddings_confidence, collect_train_test_data_grid_attribute_confidence

# Define available LLMs
llms = {
    SentenceTransformer("WhereIsAI/UAE-Large-V1"): "WhereIsAI/UAE-Large-V1",
    SentenceTransformer("avsolatorio/GIST-large-Embedding-v0", revision=None): "avsolatorio/GIST-large-Embedding-v0",
    SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1"): "mixedbread-ai/mxbai-embed-large-v1",
    SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True): "jinaai/jina-embeddings-v3",
    SentenceTransformer('BAAI/bge-large-en-v1.5'): "BAAI/bge-large-en-v1.5",
    SentenceTransformer('hkunlp/instructor-large'): "hkunlp/instructor-large",
    SentenceTransformer('thenlper/gte-large'): "thenlper/gte-large",
}

json_data_file = "sentences/gemini_patch_dataset_multi_target_color_scale_confidence.json"
#json_data_file = "data/language_data_complete_multi_target_color_scale_confidence.json"
bce_logits = nn.BCEWithLogitsLoss()
num_confidence = 3  # Number of confidence classes
num_class = 5  # Number of classes
num_targets = 5

class Decoder(nn.Module):
    def __init__(self, emb_size, out_size, hidden_size=128):
        super().__init__()
        self.l0 = nn.Linear(emb_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, out_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.act(self.l0(x))
        return self.l1(x)

def loss_fn(model, emb, grid, confidence, class_goal, target_goal, alpha, beta, gamma, theta):
    pred = model(emb)
    pred_grid = pred[:, :grid.shape[1]]
    pred_confidence = pred[:, grid.shape[1]:grid.shape[1] + num_confidence]
    pred_class = pred[:, grid.shape[1] + num_confidence: grid.shape[1] + num_confidence + num_class]
    pred_target = pred[:, grid.shape[1] + num_confidence + num_class:]

    grid_loss = bce_logits(pred_grid, grid)
    confidence_loss = bce_logits(pred_confidence, confidence)
    class_loss = bce_logits(pred_class, class_goal)
    target_loss = bce_logits(pred_target, target_goal)

    total_loss = alpha * grid_loss + beta * confidence_loss + gamma * class_loss + theta * target_loss
    return total_loss, grid_loss.item(), confidence_loss.item(), class_loss.item(), target_loss.item()


results = {}
m = 0
patience_counter = 0
alpha = 1.0
beta = 1.0
gamma = 1.0
device = "mps"

for llm, llm_name in llms.items():
    wandb.init(project='sentence_decoder_final', name=llm_name)
    batch_size = 128
    epochs = 500

    train, test = collect_train_test_data_grid_attribute_confidence(llm=llm, json_path=json_data_file, train_ratio=0.8, test_ratio=0.2, device=device)
    #train, test = collect_train_test_data_from_embeddings_confidence(json_path=json_data_file, train_ratio=0.8, test_ratio=0.2, device=device)

    model = Decoder(train["task_embedding"].shape[1],train["goal"].shape[1]+num_class+num_confidence+num_targets).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    pbar = tqdm.tqdm(total=epochs * train["task_embedding"].shape[0] // batch_size)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        for i in range(train["task_embedding"].shape[0] // batch_size):
            
            emb = torch.tensor(train["task_embedding"][i * batch_size : (i + 1) * batch_size], dtype=torch.float32).to(device)
            goal = torch.tensor(train["goal"][i * batch_size : (i + 1) * batch_size], dtype=torch.float32).to(device)
            confidence_goal = (torch.tensor(train["confidence"][i * batch_size : (i + 1) * batch_size], dtype=torch.long).to(device) + 1)
            class_goal = (torch.tensor(train["class"][i * batch_size : (i + 1) * batch_size], dtype=torch.long).to(device) + 1)
            target_goal = (torch.tensor(train["max_targets"][i * batch_size : (i + 1) * batch_size], dtype=torch.float32).to(device))
            confidence_goal_onehot = torch.nn.functional.one_hot(confidence_goal, num_classes=num_confidence).float()
            class_goal_onehot = torch.nn.functional.one_hot(class_goal, num_classes=num_class).float()
            target_goal_onehot = torch.nn.functional.one_hot(target_goal.long(), num_classes=num_targets).float()
            

            optimizer.zero_grad()
            
            loss, grid_loss, conf_loss, class_loss, target_loss = loss_fn(model, emb, goal, confidence_goal_onehot, class_goal_onehot, target_goal_onehot, alpha=alpha, beta=beta, gamma=gamma)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                v_emb = torch.tensor(test["task_embedding"], dtype=torch.float32).to(device)
                v_goal = torch.tensor(test["goal"], dtype=torch.float32).to(device)
                v_confidence = (torch.tensor(test["confidence"], dtype=torch.long).to(device) + 1)
                v_class = (torch.tensor(test["class"], dtype=torch.long).to(device) + 1)
                v_target = (torch.tensor(test["max_targets"], dtype=torch.float32).to(device))  
                v_confidence_onehot = torch.nn.functional.one_hot(v_confidence, num_classes=num_confidence).float()
                v_class_onehot = torch.nn.functional.one_hot(v_class, num_classes=num_class).float()
                v_target_onehot = torch.nn.functional.one_hot(v_target.long(), num_classes=num_targets).float()

                val_loss, val_grid_loss, val_conf_loss, val_class_loss, val_target_loss = loss_fn(model, v_emb, v_goal, v_confidence_onehot, v_class_onehot, v_target_onehot, alpha=alpha, beta=beta, gamma=gamma)
                best_val_loss = min(val_loss.item(), best_val_loss)

            pbar.update()
            pbar.set_description(
                f"LLM: {llm_name}, epoch: {epoch}, "
                f"loss: {loss.item():.4f}, val_loss: {val_loss.item():.4f}, "
                f"grid: {grid_loss:.3f}, conf: {conf_loss:.3f}, class: {class_loss:.3f}, target: {target_loss:.3f}"
            )

            wandb.log({
                "epoch": epoch,
                "loss": loss.item(),
                "grid_loss": grid_loss,
                "confidence_loss": conf_loss,
                "class_loss": class_loss,
                "target_loss": target_loss,
                "eval_loss": val_loss.item(),
                "eval_grid_loss": val_grid_loss,
                "eval_confidence_loss": val_conf_loss,
                "eval_class_loss": val_class_loss,
                "best_eval_loss": best_val_loss,
            })

    
    results[llm_name] = best_val_loss
    wandb.finish()
    torch.save(model.state_dict(), f"llm{m}_decoder_model_grid_single_target_confidence.pth")
    m += 1

print(results)