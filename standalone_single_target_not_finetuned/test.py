import torch
import torch.nn as nn
import tkinter as tk
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Set device
device = "mps" if torch.backends.mps.is_available() else "cpu"
output_grid_dim = 5

# Load sentence transformer
llm = SentenceTransformer('thenlper/gte-large', device=device)

class Decoder(nn.Module):
    def __init__(self, emb_size, out_size):
        super(Decoder, self).__init__()
        hidden_size = 256
        self.l0 = nn.Linear(emb_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, out_size)
        self.relu = nn.ReLU()

    def forward(self, embed):
        x = self.relu(self.l0(embed))
        x = self.relu(self.l1(x))
        #x = self.relu(self.l2(x))
        return torch.tanh(self.l2(x))

# Load the trained decoder model
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

model_path = f"{current_dir}/llm0_decoder_model_grid_single_target.pth"
embedding_size = llm.encode(["dummy"], device=device).shape[1]
model = Decoder(embedding_size, output_grid_dim*output_grid_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Build the UI
root = tk.Tk()
root.title("Decoder UI")

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

    with torch.no_grad():
        embedding = torch.tensor(llm.encode([sentence]), device=device).squeeze(0)
        prediction = model(embedding).cpu().numpy()

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
