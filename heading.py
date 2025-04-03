import random
import numpy as np
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
model = SentenceTransformer('all-MiniLM-L6-v2')

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

position_terms = {
    "corner": ["Edge", "corner",],
    "center": ["center","core", "middle", "epicenter"],
    "side": ["side","flank","boundary","border","margin"]
}

direction_terms = {
    "left": ["West", "Western", "Left", "Westerly", "Westernmost", "Leftmost", "Leftward", "Left-hand"],
    "top": ["Upper", "Northern", "Top", "Nordic"],
    "right": ["East", "Eastern", "Right", "Easterly"],
    "bottom": ["Bottom", "Foot", "Lower", "South", "Southern", "Austral"]
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
    "bottom": "top",
    "left": "left",
    "right": "right",
    "top": "top",
    "bottom": "bottom",

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
                "Colossal", "Gigantic", "Massive", "Boundless", "Sprawling"],
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
term_to_size = {term: size_map[size] for size, terms in size_terms.items() for term in terms}

def create_sentences(num_sentences=10):

    sentence_data = []
    sentences = []
    
    for _ in range(num_sentences):
        # Randomly select terms
        target = random.choice(target_terms)
        inside = random.choice(start_terms)

        # Select primary location (position + direction)
        main_position_type = random.choice(list(position_terms.keys()))
        main_position_list = position_terms[main_position_type]
        main_position_index = random.randint(0, len(main_position_list) - 1)
        main_position = main_position_list[main_position_index]

        main_direction_type = random.choice(list(direction_terms.keys()))
        main_direction_list = direction_terms[main_direction_type]
        main_direction_index = random.randint(0, len(main_direction_list) - 1)
        main_direction = main_direction_list[main_direction_index]

        # Ensure secondary location is different from primary
        secondary_position_type = random.choice([p for p in position_terms.keys() if p != main_position_type])
        secondary_position_list = position_terms[secondary_position_type]
        secondary_position_index = random.randint(0, len(secondary_position_list) - 1)
        secondary_position = secondary_position_list[secondary_position_index]

        # Exclude the opposite direction of the main direction
        valid_secondary_directions = [d for d in direction_terms.keys() if d != main_direction_type and d != opposite_directions.get(main_direction_type, None)]
        secondary_direction_type = random.choice(valid_secondary_directions)
        secondary_direction_list = direction_terms[secondary_direction_type]
        secondary_direction_index = random.randint(0, len(secondary_direction_list) - 1)
        secondary_direction = secondary_direction_list[secondary_direction_index]

        # Additional sentence components
        close = random.choice(proximity_terms["close"])
        far = random.choice(proximity_terms["far"])
        space_reference = random.choice(space_terms)
        search_phrase = random.choice(search_shapes)
        size_category = random.choice(["large", "small", "medium"])
        size = random.choice(size_terms[size_category])

        # Construct sentences
        sentence_1 = f"The {target} {inside} the {main_direction} {main_position} {space_reference}, {close} the {secondary_direction} {secondary_position}."
        sentence_2 = f"{search_phrase} {size}."

        full_sentence = f"{sentence_1} {sentence_2}"

        # Store sentence along with location index details
        sentence_data.append({
            "sentence": full_sentence,
            "main_location": ((main_position_type, main_direction_type, main_position_index, main_direction_index),
                              (secondary_position_type, secondary_direction_type, secondary_position_index, secondary_direction_index))
        })

        sentences.append(full_sentence)

    return sentence_data, sentences


def generate_sentence_from_location(main_location):
    
    target = random.choice(target_terms)
    inside = random.choice(start_terms)
    space_reference = random.choice(space_terms)
    search_phrase = random.choice(search_shapes)
    size_category = random.choice(["large", "small", "medium"])
    size = random.choice(size_terms[size_category])
    
    # Unpack provided main location vector
    (main_position_type, main_direction_type, main_position_index, main_direction_index), \
    (secondary_position_type, secondary_direction_type, secondary_position_index, secondary_direction_index) = main_location
    
    main_position = position_terms[main_position_type][main_position_index]
    main_direction = direction_terms[main_direction_type][main_direction_index]
    
    secondary_position = position_terms[secondary_position_type][secondary_position_index]
    secondary_direction = direction_terms[secondary_direction_type][secondary_direction_index]
    
    sentence_1 = f"The {target} {inside} the {main_direction} {main_position} {space_reference}, close to the {secondary_direction} {secondary_position}."
    sentence_2 = f"{search_phrase} {size}."
    
    return sentence_1, sentence_2

# data, sentences = create_sentences(10)

# # Encode sentences into embeddings
# embeddings = model.encode(sentences)

# position_priority = {"corner": 0, "center": 1, "side": 2}
# direction_priority = {"top": 0, "bottom": 1, "left": 2, "right": 3}

# # Sort the sentences by main position type first, then by main direction type
# sorted_data = sorted(
#     data, 
#     key=lambda x: (
#         position_priority.get(x["main_location"][0][0], 99),  # Sort by position type
#         direction_priority.get(x["main_location"][0][1], 99)  # Sort by direction type
#     )
# )

import tqdm

# Example usage
main_location_vector_1 = (
    ("corner", "left"),  # Main position: 'Boundary', Main direction: 'West'
    ("center", "top")     # Secondary position: 'Nucleus', Secondary direction: 'Pinnacle'
)

main_location_vector_2 = (
    ("corner", "left"),  # Main position: 'Boundary', Main direction: 'West'
    ("center", "left")     # Secondary position: 'Nucleus', Secondary direction: 'Pinnacle'
)
# Define matrix sizes
A = len(position_terms.keys())
B = len(direction_terms.keys())

# Initialize empty matrices
label_matrix = np.empty((A, B, A, B), dtype=object)
embedding_matrix = np.empty((A, B, A, B), dtype=object)

# Total iterations for tqdm progress bar
total_iterations = A * B * A * B

# Progress bar
#pbar = tqdm(total=total_iterations, desc="Generating labels and embeddings")

for i, mp in enumerate(position_terms.keys()):
    for j, md in enumerate(direction_terms.keys()):
        for k, sp in enumerate(position_terms.keys()):
            for h, sd in enumerate(direction_terms.keys()):
                
                # Sample random positions
                m_p = random.randint(0, len(position_terms[mp]) - 1)
                m_d = random.randint(0, len(direction_terms[md]) - 1)
                s_p = random.randint(0, len(position_terms[sp]) - 1)
                s_d = random.randint(0, len(direction_terms[sd]) - 1)

                # Create label string
                label = (mp, md, m_p, m_d), (sp, sd, s_p, s_d)
                label_matrix[i, j, k, h] = label

                # Generate 10 sentences
                sentences = [
                    generate_sentence_from_location(((mp, md, m_p, m_d), (sp, sd, s_p, s_d)))[0]
                    for _ in range(10)
                ]

                # Compute mean embedding
                sentence_embeddings = model.encode(sentences)  # Shape: (10, embedding_dim)
                mean_embedding = np.mean(sentence_embeddings, axis=0)  # Shape: (embedding_dim,)

                # Store embedding
                embedding_matrix[i, j, k, h] = mean_embedding

                # Update progress bar
               # pbar.update(1)

# Sample 10 configurations randomly
labels = []
embeddings = []
seen_indices = set()

all_indices = [(i, j, k, h) for i in range(A) for j in range(B) for k in range(A) for h in range(B)]

# Randomly sample 10 unique indices
sampled_indices = random.sample(all_indices, 10)

# Sort the sampled indices
sampled_indices.sort()

# Retrieve labels and embeddings in sorted order
labels = [label_matrix[i, j, k, h] for i, j, k, h in sampled_indices]
embeddings = [embedding_matrix[i, j, k, h] for i, j, k, h in sampled_indices]


# Convert embeddings to tensor
embeddings_array = np.array(embeddings)

# Compute cosine similarity
from sklearn.metrics.pairwise import cosine_similarity 
cosine_similarities = cosine_similarity(embeddings_array)

# Convert to Pandas DataFrame
similarity_matrix = pd.DataFrame(
    cosine_similarities, 
    index=labels, 
    columns=labels
)

for i in range(10):
    s, _ = generate_sentence_from_location(labels[i])
    print(labels[i],s)

# Plot heatmap for sorted results
plt.figure(figsize=(12, 10))
sns.heatmap(similarity_matrix, annot=False, cmap="coolwarm", xticklabels=True, yticklabels=True)
plt.xticks(rotation=20, ha="right", fontsize=5)  # Adjust x-axis labels
plt.yticks(rotation=45, ha="right", fontsize=5)  # Adjust y-axis labels
plt.title("Sentence Similarity Heatmap", fontsize=12)
plt.show()

    


# # # Generate and print 10 random sentences with tracked locations
# # for data in create_sentences(10):
# #     print(f"Sentence: {data['sentence']}")
# #     print(f"Main Location (Vector): {data['main_location']}")
# #     print("-" * 80)

# Generate sentences

# sentences_1 = []
# labels_1 = []

# for i in range(10):
#     m_p = random.randint(0, len(position_terms[main_location_vector_1[0][0]]) - 1)
#     m_d = random.randint(0, len(direction_terms[main_location_vector_1[0][1]]) - 1)
#     s_p = random.randint(0, len(position_terms[main_location_vector_1[1][0]]) - 1)
#     s_d = random.randint(0, len(direction_terms[main_location_vector_1[1][1]]) - 1)
    
#     sentence_1 = generate_sentence_from_location(((main_location_vector_1[0][0], main_location_vector_1[0][1], m_p, m_d),
#                                                   (main_location_vector_1[1][0], main_location_vector_1[1][1], s_p, s_d)))
#     sentences_1.append(sentence_1)
#     labels_1.append(f"{(main_location_vector_1[0][0], main_location_vector_1[0][1], m_p, m_d), (main_location_vector_1[1][0], main_location_vector_1[1][1], s_p, s_d)}")

# sentences_2 = []
# labels_2 = []

# for i in range(10):
#     m_p = random.randint(0, len(position_terms[main_location_vector_2[0][0]]) - 1)
#     m_d = random.randint(0, len(direction_terms[main_location_vector_2[0][1]]) - 1)
#     s_p = random.randint(0, len(position_terms[main_location_vector_2[1][0]]) - 1)
#     s_d = random.randint(0, len(direction_terms[main_location_vector_2[1][1]]) - 1)
    
#     sentence_1 = generate_sentence_from_location(((main_location_vector_2[0][0], main_location_vector_2[0][1], m_p, m_d),
#                                                   (main_location_vector_2[1][0], main_location_vector_2[1][1], s_p, s_d)))
#     sentences_2.append(sentence_1)
#     labels_2.append(f"{(main_location_vector_2[0][0], main_location_vector_2[0][1], m_p, m_d), (main_location_vector_2[1][0], main_location_vector_2[1][1], s_p, s_d)}")

# embeddings_1 = model.encode(sentences_1)
# embeddings_2 = model.encode(sentences_2)
# cosine_similarities = util.pytorch_cos_sim(embeddings_1, embeddings_2).numpy()

