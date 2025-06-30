import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import google.generativeai as genai
from api_keys import GEMINI_API_KEY

# ── Config ─────────────────────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME   = "models/gemini-embedding-exp-03-07"
INPUT_FILE   = "sequential_tasks/sentences/gemini_patch_dataset_exploration_scale.json"
OUTPUT_FILE  = "sequential_tasks/data/language_data_complete_exploration.json"

# ── Recursive collector ────────────────────────────────────────────────
def collect(obj, sentences, owners, seen_ids):
    """
    Depth-first walk of dicts *and* lists.
    • Push every 'response' string into `sentences`
    • Remember its owning dict in `owners`
    • Normalise *any* 'grid' list to ints
    """
    if isinstance(obj, dict):
        # Fix any grid list we stumble upon
        if "grid" in obj and isinstance(obj["grid"], list):
            obj["grid"] = [int(x) for x in obj["grid"]]

        # Capture this dict if it directly owns a response
        if "response" in obj and isinstance(obj["response"], str):
            if id(obj) not in seen_ids:            # avoid dup if same dict reached twice
                sentences.append(obj["response"])
                owners.append(obj)
                seen_ids.add(id(obj))

        # Recurse into *all* values
        for v in obj.values():
            collect(v, sentences, owners, seen_ids)

    elif isinstance(obj, list):
        for item in obj:
            collect(item, sentences, owners, seen_ids)

# ── Main ───────────────────────────────────────────────────────────────
def pretrained_llm():
    llm = SentenceTransformer("thenlper/gte-large")

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)                       # list of records

    sentences, owners = [], []
    if isinstance(data, list):
        for record in data:
            collect(record, sentences, owners, set())
    else:
        for record in data.values():
            collect(record, sentences, owners, set())

    if not sentences:                             # nothing to embed → bail early
        print("No responses found; nothing written.")
        return

    # Encode
    embeddings = llm.encode(
        sentences,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,                    # SentenceTransformer supports this
    )

    # Attach
    for owner, emb in zip(owners, embeddings):
        owner["embedding"] = emb.tolist()

    # Write ND-JSON (overwrite, not append)
    with open(OUTPUT_FILE, "w") as out:
        json.dump(data, out, indent=2)

    print(f"{len(embeddings)} embeddings saved → {OUTPUT_FILE}")

# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pretrained_llm()
