import random
import numpy as np
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
model_1 = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
model_2 = SentenceTransformer('thenlper/gte-base')

# Define word categories
target_terms = [
    "Objective", "Goal", "Mark", "Point", "Focus", "Destination", "Aim", "Spot", 
    "Site", "Position", "Location", "Zone", "Subject", "Waypoint", "Landmark", 
    "Endpoint", "Reference point"
]

start_terms = [
    "is located in", "can be found in", "is positioned in", "is situated in", 
    "lies in", "resides in", "is placed in", "is set in", "is within", 
    "exists in", "is inside", "occupies", "rests in", "stands in"
]

# position_terms = {
#     "corner": ["Edge", "corner",],
#     "center": ["center","core", "middle", "epicenter"],
#     "side": ["side","flank","boundary","border","margin"]
# }

position_terms = [
    "Edge", "corner", "side","flank","border"
]

direction_terms = {
    "left": ["West", "Western", "Left", "Westerly", "Westernmost", "Leftmost", "Leftward", "Left-hand"],
    "top": ["Upper", "Northern", "Top", "Nordic"],
    "right": ["East", "Eastern", "Right", "Easterly"],
    "bottom": ["Bottom", "Lower", "South", "Southern"]
}


direction_map = {
    "left": (-1, 0),
    "top": (0, 1),
    "right": (1, 0),
    "bottom": (0, -1)
}


# Define opposite directions for filtering
opposite_directions = {
    "left": "right",
    "right": "left",
    "top": "bottom",
    "bottom": "top"
}

proximity_terms = {
    "close": ["very close to", "near", "adjacent to", "beside", "a short distance from",
                "moderately close to", "not far from", "at the edge of", "on the outskirts of"],
    "far": ["far from", "distant from", "a considerable distance from", "nowhere near",
            "well beyond", "on the opposite side of", "separated from", "remote from", "away from"]
}

space_terms = [
    "of the area", "of the region", "of the zone", "of the territory", "of the surroundings",
    "of the environment", "of the field", "of the landscape", "of the setting", "of the domain",
    "of the sector", "of the vicinity", "of the grounds",
    "of the premises", "of the scene"
]

search_shapes = [
    "The search area is", "The search zone is", "The search boundary is", "The search space is",
    "The search perimeter is", "The search field is", "The search scope is", "The search territory is",
    "The search extent is", "The investigation area is"
]

size_terms = {
    "large": ["Vast", "Expansive", "Immense", "Enormous", "Extensive", "Broad", "Wide",
                "Colossal", "Gigantic", "Massive", "Sprawling"],
    "small": ["Tiny", "Miniature", "Compact", "Narrow", "Petite", "Minute", "Modest", "Limited",
                "Diminutive", "Micro", "Restricted"],
    "medium": ["Moderate", "Average", "Intermediate", "Mid-sized", "Balanced", "Medium-scale",
                "Midsize", "Fair-sized", "Middle-range", "Standard"]
}
size_map = {
    "large": 1.0,
    "medium": 0.66,
    "small": 0.33
}

def generate_sentence_from_vector(sentence_vector):
    
    target = random.choice(target_terms)
    inside = random.choice(start_terms)
    space_reference = random.choice(space_terms)
    search_phrase = random.choice(search_shapes)
    position = random.choice(position_terms)
    size_category = random.choice(["large", "small", "medium"])
    size = random.choice(size_terms[size_category])
    
    # Unpack provided main location vector
    dir_1 , dir_2 = sentence_vector
    
    sentence_1 = f"The {target} {inside} the {dir_1} {dir_2} {position} {space_reference}."
    sentence_2 = f"{search_phrase} {size}."
    
    return sentence_1, sentence_2

def generate_random_sentence_vector():
    dir_type_1 = random.choice(list(direction_terms.keys()))
    dir_1 = random.choice(direction_terms[dir_type_1])
    
    valid_d_type_2 = [d for d in direction_terms if d != dir_type_1 and d != opposite_directions.get(dir_type_1)]
    dir_type_2 = random.choice(valid_d_type_2)
    dir_2 = random.choice(direction_terms[dir_type_2])
    
    val = np.array(direction_map[dir_type_1], dtype=np.float64) + np.array(direction_map[dir_type_2], dtype=np.float64)
    uncertainty = (np.random.rand(2) * 0.5) * -val
    val += uncertainty
    
    return (dir_1, dir_2), val

s_1s = []
s_2s = []
l_s = []

for _ in range(10):
    sentence_vec , sentence_val = generate_random_sentence_vector()
    s1, s2 = generate_sentence_from_vector(sentence_vec)
    s_1s.append(s1)
    s_2s.append(s2)
    l_s.append(sentence_vec)

    print(s1,s2)
    print(sentence_val)

s1_embeddings = model_1.encode(s_1s)
s1_embeddings_2 = model_2.encode(s_1s)
#s2_embeddings = model.encode(s_2s)

from sklearn.metrics.pairwise import cosine_similarity 
# Compute cosine similarities
cosine_similarities_1 = cosine_similarity(s1_embeddings)
cosine_similarities_2 = cosine_similarity(s1_embeddings_2)

# Convert to DataFrame
similarity_matrix_1 = pd.DataFrame(cosine_similarities_1, index=l_s, columns=l_s)
similarity_matrix_2 = pd.DataFrame(cosine_similarities_2, index=l_s, columns=l_s)

# Set values below 0.6 to zero
filtered_similarity_matrix_1 = similarity_matrix_1.copy()
# filtered_similarity_matrix_1[filtered_similarity_matrix_1 < 0.6] = 0

filtered_similarity_matrix_2 = similarity_matrix_2.copy()
# filtered_similarity_matrix_2[filtered_similarity_matrix_2 < 0.6] = 0

# Plot heatmaps
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

sns.heatmap(filtered_similarity_matrix_1, annot=False, cmap="coolwarm", xticklabels=True, yticklabels=True, ax=axes[0])
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20, ha="right", fontsize=10)
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=45, ha="right", fontsize=10)
axes[0].set_title("Sentence Similarity Heatmap (Model 1)", fontsize=12)

sns.heatmap(filtered_similarity_matrix_2, annot=False, cmap="coolwarm", xticklabels=True, yticklabels=True, ax=axes[1])
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=20, ha="right", fontsize=10)
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=45, ha="right", fontsize=10)
axes[1].set_title("Sentence Similarity Heatmap (Model 2)", fontsize=12)

plt.tight_layout()
plt.show()
