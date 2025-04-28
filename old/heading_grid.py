import random
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# Train

# Define word categories
target_terms = [
    "Objective", "Goal", "Mark", "Point", "Focus", "Destination", "Aim", "Spot", 
    "Site", "Position", "Location", "Zone", "Subject", "Waypoint"
]

start_terms = [
    "is located in", "can be found in", "is positioned in", "is situated in", 
    "lies in", "resides in", "is placed in", "is set in", "is within", 
    "exists in", "is inside"
]

# position_terms = {
#     "corner": ["Edge", "corner",],
#     "center": ["center","core", "middle", "epicenter"],
#     "side": ["side","flank","boundary","border","margin"]
# }

position_terms = [
    "Edge", "corner","flank"
]

direction_terms = {
    "left": ["West", "Left", "Westerly", "Westernmost", "Leftmost", "Left-hand"],
    "top": ["Upper", "Northern", "Nordic"],
    "right": ["East", "Eastern", "Right", "Easterly", "Rightmost", "Rightward"],
    "bottom": ["Bottom", "South", "Southern"]
}


direction_map = {
    "left": (0., 0.),
    "top": (0., 1.),
    "right": (1., 0.),
    "bottom": (0., 0.)
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
                "moderately close to"],
    "far": ["far from", "distant from", "a considerable distance from", "nowhere near",
            "well beyond", "on the opposite side of"]
}

space_terms = [
    "of the area", "of the region", "of the zone", "of the territory", "of the surroundings",
    "of the environment", "of the field", "of the landscape", "of the setting", "of the domain"
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

enumerations = [
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth"
    "seventh",
    "eighth"
]

# Eval

# Define word categories
eval_target_terms = [
    "Landmark", "Endpoint", "Reference point"
]

eval_start_terms = [
    "occupies", "rests in", "stands in"
]

eval_position_terms = [
    "side","border"
]

eval_direction_terms = {
    "left": ["Western","Leftward"],
    "top": ["Top"],
    "right": ["Easternmost", "Right-hand"],
    "bottom": ["Lower"]
}

eval_space_terms = [
    "of the sector", "of the vicinity", "of the grounds",
    "of the premises", "of the scene"
]



class Heading():

    def __init__(self, batch_size, num_targets, grid_size = 5):

        self.batch_size = batch_size
        self.grid_width = 1.
        self.grid_height = 1.
        self.grid_size = grid_size
        self.cell_width = self.grid_width / self.grid_size
        self.cell_height = self.grid_height / self.grid_size
        self.mini_grid_radius = 1
        self.device = "mps"
        self.eval = False
        self.model = None
        self.num_targets = num_targets

    def update_sentence_from_vector(self,idx,target_activation):
        
        string = ""
        
        heading_count = 0
        
        for i in range(self.num_targets):
            
            if i < 1 or target_activation[i] < 0.5:

                targets = random.choices(target_terms, k=1)
                insides = random.choices(start_terms, k=1)
                spaces = random.choices(space_terms, k=1)
                searches = random.choices(search_shapes, k=1)
                positions = random.choices(position_terms, k=1)
                size_categories = random.choices(["large", "small", "medium"], k=1)
                sizes = [random.choice(size_terms[cat]) for cat in size_categories]

                if self.eval:
                    targets = random.choices(eval_target_terms, k=1)
                    insides = random.choices(eval_start_terms, k=1)
                    spaces = random.choices(eval_space_terms, k=1)
                    searches = random.choices(search_shapes, k=1)
                    positions = random.choices(eval_position_terms, k=1)
                    size_categories = random.choices(["large", "small", "medium"], k=1)
                    sizes = [random.choice(size_terms[cat]) for cat in size_categories]

                # Extracting keywords (lists)
                dir_1 = self.heading_keywords[idx][i][0]
                dir_2 = self.heading_keywords[idx][i][1]
                
                string += f"The {enumerations[heading_count]} {targets[0]} {insides[0]} the {dir_1} {dir_2} {positions[0]} {spaces[0]}."
                heading_count +=1 

        # Store results in lists instead of tensors
        self.headings_pos_string[idx] = string
        
        
        self.heading_size_string[idx] = [
            f"{searches[0]} {sizes[0]}."
        ]

    def update_sentence_vectors(self,idx,target_activation):
        
        
        for i in range(self.num_targets):
            
            if i < 1 or target_activation[i] < 0.5:
                
                dir_types_1 = random.choices(list(direction_terms.keys()), k=1)
                dir_1 = [random.choice(direction_terms[d]) for d in dir_types_1]

                valid_d_types_2 = [
                    random.choice([d for d in direction_terms if d != d1 and d != opposite_directions[d1]])
                    for d1 in dir_types_1
                ]
                dir_2 = [random.choice(direction_terms[d]) for d in valid_d_types_2]

                if self.eval: 
                    dir_types_1 = random.choices(list(eval_direction_terms.keys()), k=1)
                    dir_1 = [random.choice(eval_direction_terms[d]) for d in dir_types_1]

                    valid_d_types_2 = [
                        random.choice([d for d in eval_direction_terms if d != d1 and d != opposite_directions[d1]])
                        for d1 in dir_types_1
                    ]
                
                    dir_2 = [random.choice(eval_direction_terms[d]) for d in valid_d_types_2]

                # Convert to NumPy array before tensor
                val_tensor = np.clip(
                    np.array([direction_map[d1] for d1 in dir_types_1]) + np.array([direction_map[d2] for d2 in valid_d_types_2]),
                    a_min=0.0, a_max=1.0
                )
                uncertainty = (np.random.rand(1, 2) * 0.2) * (- val_tensor)
                val_tensor += uncertainty

                # Convert to PyTorch tensor
                val_tensor = torch.tensor(val_tensor, dtype=torch.float32, device=self.device)
                x_coords = (val_tensor[:, 0] * self.grid_width // self.cell_width - self.mini_grid_radius).clamp(0, self.grid_size - 1)
                y_coords = (val_tensor[:, 1] * self.grid_height // self.cell_height - self.mini_grid_radius).clamp(0, self.grid_size - 1)

                # Expand 1D coordinates into full mini-grid ranges
                x_range = (torch.arange(0, 2 * self.mini_grid_radius + 1, device=self.device) + x_coords).squeeze()
                y_range = (torch.arange(0, 2 * self.mini_grid_radius + 1, device=self.device) + y_coords).squeeze()

                # Clamp ranges to grid size
                x_range = x_range.clamp(0, self.grid_size - 1)
                y_range = y_range.clamp(0, self.grid_size - 1)

                # Create meshgrid for 2D indexing
                yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')  # y first, then x
                yy = yy.long()
                xx = xx.long()
                # Correct index order: [batch, y, x]
                self.heading_pos_grid[idx, yy, xx] = 1.

                # Store keywords
                self.heading_keywords[idx][i] = [dir_1[0], dir_2[0]]


    def collect_embedding(self,idx):

            self.heading_embedding[idx] = torch.tensor(self.model.encode(self.headings_pos_string[idx]), device = self.device)

    def _initalize_heading(self,eval, model):

        self.eval = eval
        self.model = model
        self.embedding_size = self.model.encode(["test sentence"], device = self.device).shape[1]

        if eval:
             self.batch_size = self.batch_size // 10

        # Handle strings with lists, not tensors
        self.headings_pos_string = [""] * self.batch_size
        self.heading_size_string = [""] * self.batch_size
        self.heading_keywords = [["", ""] * self.num_targets] * self.batch_size

        # Use proper tensor initialization for numerical values
        self.heading_pos_grid = torch.zeros((self.batch_size, self.grid_size, self.grid_size), device=self.device)

        # Example embedding size (define EMBEDDING_SIZE in your context)
        self.heading_embedding = torch.zeros((self.batch_size, self.embedding_size), device=self.device)

        for i in range(self.batch_size):
            target_activation = torch.rand(self.num_targets, device=self.device)
            self.update_sentence_vectors(i, target_activation)
            self.update_sentence_from_vector(i, target_activation)
            
        with torch.no_grad():
            embeddings = torch.tensor(
                self.model.encode(self.headings_pos_string, device=self.device),
                device=self.device
            )
        self.heading_embedding = embeddings.view(self.batch_size, -1)
        
        return {
            "task_embedding": self.heading_embedding.cpu().numpy(),
            "goal": self.heading_pos_grid.flatten(start_dim=1,end_dim=-1).cpu().numpy()
        }

