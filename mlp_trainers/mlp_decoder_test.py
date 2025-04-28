import torch
import torch.nn as nn
import tkinter as tk
from sentence_transformers import SentenceTransformer

# Set device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load sentence transformer
llm = SentenceTransformer('thenlper/gte-large', device=device)

# Define the Decoder model
class Decoder(nn.Module):
    def __init__(self, emb_size, out_size):
        super(Decoder, self).__init__()
        hidden_size = 256
        self.l0 = nn.Linear(emb_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, out_size)
        #self.l2 = nn.Linear(hidden_size, out_size)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, embed):
        x = self.leakyrelu(self.l0(embed))
        x = self.leakyrelu(self.l1(x))
        x = self.leakyrelu(self.l2(x))
        x = self.leakyrelu(self.l3(x))
        return torch.tanh(self.l4(x))


# Load the trained decoder model
model_path = "llm0_decoder_model_grid_single_target_color_class.pth"  # Update this path if needed
embedding_size = llm.encode(["dummy"], device=device).shape[1]
model = Decoder(embedding_size,11).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define function to get prediction
def predict():
    sentence = entry.get()
    if not sentence.strip():
        result_var.set("Please enter a sentence.")
        return
    with torch.no_grad():
        embedding = torch.tensor(llm.encode([sentence]), device=device).squeeze(0)
        prediction = model(embedding).cpu().numpy()
    result_var.set(f"Decoder output: {prediction}")

# Build the UI
root = tk.Tk()
root.title("Decoder UI")

# Widgets
tk.Label(root, text="Enter a sentence:").pack(pady=5)

entry = tk.Entry(root, width=60)
entry.pack(pady=5)

tk.Button(root, text="Get Decoder Output", command=predict).pack(pady=10)

result_var = tk.StringVar()
tk.Label(root, textvariable=result_var, wraplength=500, justify="left").pack(pady=10)

# Start the UI
root.mainloop()

