import torch
import torch.nn as nn
import tkinter as tk
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Set device
device = "mps" if torch.backends.mps.is_available() else "cpu"
output_grid_dim = 10

# Load sentence transformer
llm = SentenceTransformer('thenlper/gte-large', device=device)

# class Block(nn.Module):
#     """A standard nn layer with linear, norm, and activation."""
#     def __init__(self, input_size, output_size, dropout=0.0):
#         super().__init__()
#         layers = [
#             nn.Linear(input_size, output_size),
#             nn.LayerNorm(output_size, elementwise_affine=False),
#             nn.LeakyReLU()
#         ]
#         if dropout > 0.0:
#             layers.insert(2, nn.Dropout(dropout))  # Insert dropout before activation
#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.net(x)


# class Decoder(nn.Module):
#     def __init__(self, emb_size, out_shape):
#         super().__init__()
#         hidden_size = 256
#         self.l0 = Block(emb_size, hidden_size)
#         self.l1 = Block(hidden_size, hidden_size)
#         self.l2 = Block(hidden_size, hidden_size)
#         self.l3 = nn.Linear(hidden_size, out_shape)

#     def forward(self, embed):
#         x = self.l0(embed)
#         x = self.l1(x)
#         x = self.l2(x)
#         return self.l3(x)
    

# class Decoder(nn.Module):
#     def __init__(self, emb_size, out_size):
#         super(Decoder, self).__init__()
#         hidden_size = 256
#         self.l0 = nn.Linear(emb_size, hidden_size)
#         self.l1 = nn.Linear(hidden_size, hidden_size)
#         #self.l2 = nn.Linear(hidden_size, hidden_size)
#         #self.l3 = nn.Linear(hidden_size, out_size)
#         self.l2 = nn.Linear(hidden_size, out_size)
#         self.relu = nn.ReLU()

#     def forward(self, embed):
#         x = self.relu(self.l0(embed))
#         x = self.relu(self.l1(x))
#         #x = self.relu(self.l2(x))
#         return torch.tanh(self.l2(x))
    
# #MLP on sterroids
# class Decoder_boost(nn.Module):
#     def __init__(self, emb_size, out_size, hidden_size=256):
#         super().__init__()
#         self.norm_input = nn.LayerNorm(emb_size)
#         self.l0 = nn.Linear(emb_size, hidden_size)
#         self.l1 = nn.Linear(hidden_size, hidden_size)
#         self.l2 = nn.Linear(hidden_size, out_size)
#         self.norm_hidden = nn.LayerNorm(hidden_size)
#         self.act = nn.LeakyReLU(0.1)
#         self.dropout = nn.Dropout(0.3)

#     def forward(self, x):
#         x = self.norm_input(x)
#         x = self.dropout(self.act(self.l0(x)))
#         x = self.norm_hidden(x)
#         x = self.dropout(self.act(self.l1(x)))
#         return torch.tanh(self.l2(x))

class Decoder(nn.Module):
    def __init__(self, emb_size, out_size, hidden_size=128):
        super().__init__()
        self.l0 = nn.Linear(emb_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, out_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.act(self.l0(x))
        return torch.sigmoid(self.l1(x))

# Load the trained decoder model
model_path = "decoders/llm0_decoder_model_grid_scale.pth"  # Update this path if needed
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
        prediction = model(embedding).cpu()  # shape: (100,)

        # Step 1: Min-max normalize to [0, 1]
        min_val = prediction.min()
        max_val = prediction.max()

        # Step 2: Apply fixed threshold (e.g., 0.8)
        threshold = 0.8 * (max_val - min_val) + min_val
        above_thresh = prediction >= threshold

        # Step 3: Subtract threshold *only* from values above it
        rescaled = torch.zeros_like(prediction)
        rescaled[above_thresh] = prediction[above_thresh]

        grid = rescaled.reshape(output_grid_dim, output_grid_dim).numpy()

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
