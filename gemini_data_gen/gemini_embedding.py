import torch
import google.generativeai as genai
from api_keys import GEMINI_API_KEY

# Configure Google AI API
API_KEY = GEMINI_API_KEY  # Replace with your actual API key
genai.configure(api_key=API_KEY)

# Select the embedding model
model = "models/gemini-embedding-exp-03-07"

# Create content to embed
content = "What is the meaning of life?"

# Generate embedding
result = genai.embed_content(
    model=model,
    content=content
)

# Convert embedding to PyTorch tensor
embedding_tensor = torch.tensor(result['embedding'], dtype=torch.float32)

# Print tensor details
print("Embedding Tensor:")
print(embedding_tensor)
print("\nTensor Shape:", embedding_tensor.shape)
print("Tensor Data Type:", embedding_tensor.dtype)