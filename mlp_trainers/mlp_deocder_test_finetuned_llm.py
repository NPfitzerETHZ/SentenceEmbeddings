import torch
import torch.nn as nn
import tkinter as tk
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Set device
device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"
output_grid_dim = 5
GTE_LARGE_EMBED_SIZE = 1024

# Load sentence transformer
from transformers import AutoModel, AutoTokenizer

llm = AutoModel.from_pretrained("finetuned_llms/finetuned_gte_large_single_target")
tokenizer = AutoTokenizer.from_pretrained("finetuned_llms/finetuned_gte_large_single_target")

def encode(texts):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        model_output = llm(**encoded_input)
        embedding = model_output.last_hidden_state[:, 0]
    return embedding


class Decoder(nn.Module):
    def __init__(self, emb_size, out_size, hidden_size=256):
        super().__init__()
        self.l0 = nn.Linear(emb_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, out_size)
        self.act = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(self.relu(self.l0(x)))
        x = self.dropout(self.relu(self.l1(x)))
        return torch.tanh(self.l2(x))

# Load the trained decoder model
decoder_path = "decoders/llm_decoder_model_grid_single_target_finetuned.pth"  # Update this path if needed

decoder = Decoder(GTE_LARGE_EMBED_SIZE, output_grid_dim*output_grid_dim).to(device)
decoder.load_state_dict(torch.load(decoder_path, map_location=device))
decoder.eval()

# Build the UI
root = tk.Tk()
root.title("Decoder UI with Finetuned LLM")
# Widgets
tk.Label(root, text="Enter a sentence:").pack(pady=5)

# Larger multiline text input
entry = tk.Text(root, width=60, height=6, wrap="word")
entry.pack(pady=5)

# Create a placeholder for the matplotlib plot
fig, ax = plt.subplots(figsize=(4, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=10)

result_var = tk.StringVar()
tk.Label(root, textvariable=result_var, wraplength=500, justify="left").pack(pady=10)

colorbar = None

def predict():
    global colorbar
    sentence = entry.get("1.0", tk.END).strip()
    if not sentence:
        result_var.set("Please enter a sentence.")
        return
    
    embedding = encode(sentence)
    with torch.no_grad():  
        prediction = decoder(embedding).cpu().numpy()

    grid = prediction.reshape(output_grid_dim, output_grid_dim)

    # Clear entire figure
    fig.clf()

    # Create fresh axes
    ax = fig.add_subplot(111)
    im = ax.imshow(grid, cmap='viridis')
    ax.set_title(f"Decoder Output ({output_grid_dim}x{output_grid_dim} Grid)")

    # Create a fresh colorbar
    colorbar = fig.colorbar(im, ax=ax)

    canvas.draw()
    result_var.set("Prediction completed and visualized.")

tk.Button(root, text="Get Decoder Output", command=predict).pack(pady=10)

# Start the UI
root.mainloop()

