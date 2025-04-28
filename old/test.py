from sentence_transformers import SentenceTransformer
from sentence_transformers import util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


sentences = ["The weather today is beautiful", "It's raining!", "Dogs are awesome"]
embeddings = model.encode(sentences)
print(embeddings.shape)

first_embedding = model.encode("Today is a sunny day")
for embedding, sentence in zip(embeddings, sentences):
    similarity = util.pytorch_cos_sim(first_embedding, embedding)
    print(similarity, sentence)