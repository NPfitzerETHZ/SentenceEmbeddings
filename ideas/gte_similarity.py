"""
Compute the similarity between two sentences using GTE-Large.

Usage:
    python gte_similarity.py "Sentence one." "Sentence two."
"""

import sys
from sentence_transformers import SentenceTransformer, util

def main() -> None:
    if len(sys.argv) != 3:
        print("Provide exactly two sentences:\n"
              "python gte_similarity.py \"First sentence\" \"Second sentence\"")
        sys.exit(1)

    sent1, sent2 = sys.argv[1], sys.argv[2]

    model_name = "thenlper/gte-large"
    model = SentenceTransformer(model_name)

    # get L2-normalised embeddings
    embeds = model.encode([sent1, sent2], convert_to_tensor=True,
                          normalize_embeddings=True)

    cosine_sim = util.cos_sim(embeds[0], embeds[1]).item()
    print(f"Cosine similarity: {cosine_sim:.4f}")

if __name__ == "__main__":
    main()