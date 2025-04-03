import json
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm

# File paths
input_file = 'sentences/gemini_patch_dataset_grid.json'
output_file = 'data/language_data_complete_single_target_finetuned.json'
device = "mps"

def predtrained_llm():
    # Load model
    llm = SentenceTransformer('thenlper/gte-large')

    # Read the full JSON (assuming it's a list of dicts)
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Extract texts to embed
    texts_to_embed = [item['gemini_response'] for item in data if 'gemini_response' in item]

    # Batch encode all texts
    embeddings = llm.encode(texts_to_embed, batch_size=32, show_progress_bar=True)

    # Map back embeddings to the data
    embed_index = 0
    for item in tqdm(data, desc="Adding embeddings"):
        if 'gemini_response' in item:
            item['embedding'] = embeddings[embed_index].tolist()
            embed_index += 1

        if 'grid' in item and isinstance(item['grid'], list):
            item['grid'] = [int(x) for x in item['grid']]

    # Write output line by line (newline-delimited JSON)
    with open(output_file, 'a') as f_out:
        for item in data:
            f_out.write(json.dumps(item) + '\n')

    print(f"Conversion complete. Results saved to {output_file}")

def fine_tuned_llm(input_file, output_file):

    llm = AutoModel.from_pretrained("finetuned_llms/finetuned_gte_large_single_target").to(device)
    tokenizer = AutoTokenizer.from_pretrained("finetuned_llms/finetuned_gte_large_single_target")

    def encode(texts, batch_size=32, show_progress_bar=False):
        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding batches")

        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                model_output = llm(**encoded_input)
                batch_embeddings = model_output.last_hidden_state[:, 0]
                all_embeddings.append(batch_embeddings)

        return torch.cat(all_embeddings, dim=0)

    # Read the full JSON (assuming it's a list of dicts)
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Extract texts to embed
    texts_to_embed = [item['gemini_response'] for item in data if 'gemini_response' in item]

    # Batch encode all texts
    embeddings = encode(texts_to_embed, batch_size=32, show_progress_bar=True)

    # Map back embeddings to the data
    embed_index = 0
    for item in tqdm(data, desc="Adding embeddings"):
        if 'gemini_response' in item:
            item['embedding'] = embeddings[embed_index].tolist()
            embed_index += 1

        if 'grid' in item and isinstance(item['grid'], list):
            item['grid'] = [int(x) for x in item['grid']]

    # Write output line by line (newline-delimited JSON)
    with open(output_file, 'a') as f_out:
        for item in data:
            f_out.write(json.dumps(item) + '\n')

    print(f"Conversion complete. Results saved to {output_file}")
    
predtrained_llm()
            