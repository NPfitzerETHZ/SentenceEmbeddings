import os
import time
import json
import random
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from io import BytesIO
from PIL import Image
from tqdm import tqdm
from google import genai

from api_keys import GEMINI_API_KEY_2

# ========================== Configuration ==========================

# Gemini API Key
genai_client = genai.Client(api_key=GEMINI_API_KEY_2)  # 🔐 Replace with your actual key

# Dataset generation parameters
NUM_DATA_TARGETS = 1500
API_DELAY_SECONDS = 0.75
grid_size = 5
min_patch_size = 2
max_patch_size = 5
min_std = 0.1
max_std = 1.5
max_num_patches = 2
multipatch_prob = 0.5
no_patch_prob = 0.1
danger_zone_prob = 1.0
BATCH_SIZE = 10
# ========================== Vocabulary ==========================

direction_terms = {
    "ern": ["eastern", "western", "southern", "northern", "center"],
    "cardinal": ["east", "west", "south", "north", "middle"],
    "erly": ["easterly", "westerly", "southerly", "norhterly", "center"],
    "dunno": ["upper", "lower", "leftmost", "rightmost", "middle"],
    "dunno_2": ["top", "bottom", "left", "right", "center"]
}

size_terms = {
    "cat_1": ["vast", "moderate", "tiny"],
    "cat_2": ["expansive", "average", "miniature"],
    "cat_3": ["immense", "intermediate", "compact"],
    "cat_4": ["enormous", "mid-sized", "narrow"],
    "cat_5": ["extensive", "medium-scale", "petite"],
    "cat_6": ["broad", "midsize", "modest"],
    "cat_7": ["wide", "standard", "limited"],
    "cat_8": ["colossal", "fair-sized", "restricted"],
    "cat_9": ["large", "medium", "small"]
}

environment_terms = [
    *["area", "region", "zone", "territory", "surroundings", "environment", "field", "landscape", "setting", "domain"],
    *[f"{prefix} {term}" for prefix in ["search", "exploration", "reconnaissance", "investigation"]
      for term in ["area", "region", "zone", "territory", "surroundings", "environment", "field", "landscape", "setting", "domain"]]
]

danger_zone_terms = [
    "danger zone", "danger region", "hot zone", "hot region", "red zone", "red region",
    "hazard area", "hazard region", "restricted area", "restricted region", "no-go zone",
    "no-go region", "kill zone", "kill region", "combat zone", "combat region", "war zone",
    "war region", "exclusion zone", "exclusion region", "critical zone", "critical region",
    "unsafe area", "unsafe region", "high-risk area", "high-risk region", "death zone",
    "death region", "threat zone", "threat region"
]

target_terms = [
    "Landmark", "Endpoint", "Reference point", "Objective", "Goal", "Mark", "Point", "Focus",
    "Destination", "Aim", "Spot", "Site", "Position", "Location", "Zone", "Subject", "Waypoint"
]

# ========================== Utility Functions ==========================

def get_neighbors(cell, grid_size):
    r, c = cell
    neighbors = []
    if r > 0: neighbors.append((r-1, c))
    if r < grid_size-1: neighbors.append((r+1, c))
    if c > 0: neighbors.append((r, c-1))
    if c < grid_size-1: neighbors.append((r, c+1))
    return neighbors

def generate_gaussian_prob_map(center, std_x, std_y, grid_size):
    cx, cy = center
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    gauss = np.exp(-((xx - cx) ** 2) / (2 * std_x ** 2) - ((yy - cy) ** 2) / (2 * std_y ** 2))
    return gauss / gauss.sum()

def plot_grid(grid):
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='binary')
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_color_grid(grid):
    # Define custom colors: -1 (blue), 0 (white), 1 (red)
    colors = ['blue', 'white', 'red']
    cmap = ListedColormap(colors)

    # Shift grid values from [-1, 0, 1] to [0, 1, 2] to index into the colormap
    color_index_grid = (np.array(grid) + 1).astype(int)

    fig, ax = plt.subplots()
    ax.imshow(color_index_grid, cmap=cmap)
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return buf

# ========================== Core Functions ==========================

# This function generates target and danger patch pairs
def generate_grid_target_danger_pair():
    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    used_cells = set()
    target_flag = True
    
    for patch_idx in range(max_num_patches):
        
        if patch_idx % 2 == 1: target_flag = False
        
        candidates = [(random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)) for _ in range(10)]
        best_start = max(candidates, key=lambda cell: min([np.linalg.norm(np.subtract(cell, u)) for u in used_cells], default=0))
        
        if best_start in used_cells:
                    continue

        prob_map = generate_gaussian_prob_map(
            center=best_start,
            std_x=random.uniform(min_std, max_std),
            std_y=random.uniform(min_std, max_std),
            grid_size=grid_size
        )

        patch = {best_start}
        frontier = set(get_neighbors(best_start, grid_size))
        target_size = random.randint(min_patch_size, max_patch_size)

        while len(patch) < target_size and frontier:
            probs = np.array([prob_map[r, c] if (r, c) not in used_cells else 0. for r, c in frontier])
            if probs.sum() == 0: break
            next_cell = random.choices(list(frontier), weights=probs, k=1)[0]
            patch.add(next_cell)
            frontier.remove(next_cell)
            frontier.update({n for n in get_neighbors(next_cell, grid_size) if n not in patch and n not in used_cells})

        value = 1 if target_flag else -1
        for r, c in patch:
            grid[r, c] = value
        used_cells.update(patch)
        
    return grid
    

# This function generates either a set of danger zones or a set of target zones
def generate_grid_and_patches():
    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    used_cells = set()
    num_patches = 0
    target_flag = random.random() > danger_zone_prob

    if random.random() > no_patch_prob:
        for patch_idx in range(max_num_patches):
            if patch_idx == 0 or random.random() < multipatch_prob:
                num_patches += 1
                candidates = [(random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)) for _ in range(10)]
                best_start = max(candidates, key=lambda cell: min([np.linalg.norm(np.subtract(cell, u)) for u in used_cells], default=0))

                if best_start in used_cells:
                    continue

                prob_map = generate_gaussian_prob_map(
                    center=best_start,
                    std_x=random.uniform(min_std, max_std),
                    std_y=random.uniform(min_std, max_std),
                    grid_size=grid_size
                )

                patch = {best_start}
                frontier = set(get_neighbors(best_start, grid_size))
                target_size = random.randint(min_patch_size, max_patch_size)

                while len(patch) < target_size and frontier:
                    probs = np.array([prob_map[r, c] if (r, c) not in used_cells else 0. for r, c in frontier])
                    if probs.sum() == 0: break
                    next_cell = random.choices(list(frontier), weights=probs, k=1)[0]
                    patch.add(next_cell)
                    frontier.remove(next_cell)
                    frontier.update({n for n in get_neighbors(next_cell, grid_size) if n not in patch and n not in used_cells})

                value = 1 if target_flag else -1
                for r, c in patch:
                    grid[r, c] = value
                used_cells.update(patch)

    return grid, num_patches, target_flag

def describe_image(image_buf):
    
    prompt = f"""You are leading a team of robots.
    Describe the location and size of each patch with respect to the environment. Red patch is the target. Blue patch is a danger zone. Use sentences. Do not mention color or the grid.
    The target patch must be reached". "The danger patch must be avoided.
    """
    if random.random() < 0.5:
        prompt += f"""For the target use the term {random.choice(target_terms)}.
    For the danger zone use the term {random.choice(danger_zone_terms)}.
    For the exploration area use the term {random.choice(environment_terms)}.
    For the location of each patch use the following adjectives: {random.choice(list(direction_terms.values()))}.
    For the size of each patch use at least one of the following adjectives: {random.choice(list(size_terms.values()))}.
    """
    # objective = "target" if target_flag else "danger"
    # term = random.choice(target_terms if target_flag else danger_zone_terms)

    # prompt = f"""You are leading a team of robots.
    # Describe the location and size of each {objective} patch with respect to the environment.
    # There is {num_patches} {objective} patch{'es' if num_patches > 1 else ''}. Use sentences. Do not mention color or the grid.
    # """
    # if random.random() < 0.5:
    #     prompt += f"""Use the term "{term}".
    #     Refer to the environment as "{random.choice(environment_terms)}".
    #     Use directional terms like {random.choice(list(direction_terms.values()))}.
    #     Use size terms like {random.choice(list(size_terms.values()))}.
    #     """
    img = Image.open(image_buf)
    response = genai_client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[prompt, img]
    )
    return response.text

def describe_image_with_timeout(image_buf, timeout=10):
    image_buf.seek(0)
    buf_copy = BytesIO(image_buf.read())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(describe_image, buf_copy)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print("⚠️ Gemini took too long, skipping.")
        except Exception as e:
            print(f"⚠️ Gemini error: {e}")
        return None

# ========================== Main Generation Loop ==========================
def danger_zones():
    output_file = "gemini_patch_dataset_grid_danger.jsonl"
    json_output_file = "gemini_patch_dataset_grid_danger.json"
    start_index = 0
    buffer = []

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_lines = f.readlines()
            start_index = len(existing_lines)
            print(f"📂 Resuming from index {start_index}")

    with open(output_file, "a") as f:
        for i in tqdm(range(start_index, NUM_DATA_TARGETS), desc="Generating Data"):
            grid, num_patches, target_flag = generate_grid_and_patches()
            image_buf = plot_grid(grid if target_flag else -grid)
            description = describe_image_with_timeout(image_buf, num_patches, target_flag)

            if description is None:
                continue

            time.sleep(API_DELAY_SECONDS)

            buffer.append({
                "grid": grid.flatten().tolist(),
                "gemini_response": description
            })

            if len(buffer) >= BATCH_SIZE:
                for entry in buffer:
                    f.write(json.dumps(entry) + "\n")
                buffer.clear()

        for entry in buffer:
            f.write(json.dumps(entry) + "\n")

    # Save full dataset as JSON for readability
    with open(output_file) as f:
        data = [json.loads(line) for line in f]
    with open(json_output_file, "w") as f:
        json.dump(data, f, indent=2)

def target_danger_pairs():
    
    output_file = "gemini_patch_dataset_grid_target_danger.jsonl"
    json_output_file = "gemini_patch_dataset_grid_target_danger.json"
    start_index = 0
    buffer = []

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_lines = f.readlines()
            start_index = len(existing_lines)
            print(f"📂 Resuming from index {start_index}")

    with open(output_file, "a") as f:
        for i in tqdm(range(start_index, NUM_DATA_TARGETS), desc="Generating Data"):
            grid = generate_grid_target_danger_pair()
            image_buf = plot_color_grid(grid)
            description = describe_image_with_timeout(image_buf)

            if description is None:
                continue

            time.sleep(API_DELAY_SECONDS)

            buffer.append({
                "grid": grid.flatten().tolist(),
                "gemini_response": description
            })

            if len(buffer) >= BATCH_SIZE:
                for entry in buffer:
                    f.write(json.dumps(entry) + "\n")
                buffer.clear()

        for entry in buffer:
            f.write(json.dumps(entry) + "\n")

    # Save full dataset as JSON for readability
    with open(output_file) as f:
        data = [json.loads(line) for line in f]
    with open(json_output_file, "w") as f:
        json.dump(data, f, indent=2)
        
target_danger_pairs()

    










# import numpy as np
# import random
# from io import BytesIO
# from PIL import Image
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from google import genai
# import time
# import json
# import os

# direction_terms = {
#     "ern": "eastern, western, southern, northern, center",
#     "cardinal": "east, west, south, north, middle",
#     "erly": "easterly, westerly, southerly, norhterly, center",
#     "dunno": "upper, lower, leftmost, rightmost, middle",
#     "dunno_2": "top, bottom, left, right, center"
# }

# environment_terms = [
#     "area", "region", "zone", "territory" , "surroundings", "environment", "field", "landscape", "setting", "domain", 
#     "search area", "search region", "search zone", "search territory" , "search surroundings", "search environment", "search field", "search landscape", "search setting", "search domain"
#     "exploration area", "exploration region", "exploration zone", "exploration territory" , "exploration surroundings", "exploration environment", "exploration field", "exploration landscape", "exploration setting", "exploration domain",
#     "reconnaissance area", "reconnaissance region", "reconnaissance zone", "reconnaissance territory" , "reconnaissance surroundings", "reconnaissance environment", "reconnaissance field", "reconnaissance landscape", "reconnaissance setting", "reconnaissance domain", 
#     "investigation area", "investigation region", "investigation zone", "investigation territory" , "investigation surroundings", "investigation environment", "investigation field", "investigation landscape", "investigation setting", "investigation domain"]

# danger_zone_terms = ["danger zone", "danger region", "hot zone", "hot region", "red zone", "red region", "hazard area", "hazard region", "restricted area",
#                      "restricted region", "no-go zone", "no-go region", "kill zone", "kill region", "combat zone", "combat region", "war zone", "war region",
#                      "exclusion zone", "exclusion region", "critical zone", "critical region", "unsafe area", "unsafe region", "high-risk area", "high-risk region",
#                      "death zone", "death region", "threat zone", "threat region"]

# size_terms = {
#     "cat_1": "vast, moderate, tiny",
#     "cat_2": "expansive, average, miniature",
#     "cat_3": "immense, intermediate, compact",
#     "cat_4": "enormous, mid-sized, narrow",
#     "cat_5": "extensive, medium-scale, petite",
#     "cat_6": "broad, midsize, modest",
#     "cat_7": "wide, standard, limited",
#     "cat_8": "colossal, fair-sized, restricted",
#     "cat_9": "large, medium, small"
# }

# target_terms = ["Landmark", "Endpoint", "Reference point", "Objective", "Goal", "Mark", "Point", "Focus", "Destination", "Aim", "Spot", 
#     "Site", "Position", "Location", "Zone", "Subject", "Waypoint"]

# # Set your Gemini API key
# genai_client = genai.Client(api_key="AIzaSyCychUuEpP32zRrX92jzep-1FqfYOASMGU")

# # Parameters
# NUM_DATA_TARGETS = 1500
# API_DELAY_SECONDS = 1.
# grid_size = 5
# min_patch_size = 1
# max_patch_size = 6
# min_std = 0.1
# max_std = 1.5
# max_num_patches = 1
# multipatch_prob = 0.5
# no_patch_prob = 0.1
# danger_zone_prob = 1.0
# BATCH_SIZE = 50 # Save every 50 entries

# def get_neighbors(cell, grid_size):
#     r, c = cell
#     neighbors = []
#     if r > 0: neighbors.append((r-1, c))
#     if r < grid_size-1: neighbors.append((r+1, c))
#     if c > 0: neighbors.append((r, c-1))
#     if c < grid_size-1: neighbors.append((r, c+1))
#     return neighbors

# def generate_gaussian_prob_map(center, std_x, std_y, grid_size):
#     cx, cy = center
#     x = np.arange(grid_size)
#     y = np.arange(grid_size)
#     xx, yy = np.meshgrid(x, y, indexing='ij')
#     gauss = np.exp(-((xx - cx) ** 2) / (2 * std_x ** 2) - ((yy - cy) ** 2) / (2 * std_y ** 2))
#     gauss /= gauss.sum()
#     return gauss

# def compute_patch_spread(patch):
#     coords = np.array(list(patch))
#     if len(coords) == 0:
#         return 0.0, 0.0
#     std_y = np.std(coords[:, 0])  # rows
#     std_x = np.std(coords[:, 1])  # cols
#     return std_x, std_y

# def generate_grid_and_patches():
#     grid = np.zeros((grid_size, grid_size), dtype=np.int8)  # Empty cells are 0, patches are 1
#     used_cells = set()
#     num_patches = 0
    
#     target_flag = random.random() > danger_zone_prob

#     if random.random() > no_patch_prob:  # 85% chance to place at least one patch
#         for patch_idx in range(max_num_patches):
#             if patch_idx == 0 or random.random() < multipatch_prob:  # Always place first, 50% chance for others
#                 num_patches += 1

#                 # Sample multiple start candidates and choose the farthest one from used_cells
#                 candidate_starts = [(random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)) for _ in range(10)]
#                 best_start = None
#                 max_min_dist = -1

#                 for start in candidate_starts:
#                     if start in used_cells:
#                         continue
#                     if not used_cells:  # If first patch, just take any valid start
#                         best_start = start
#                         break
#                     # Compute minimum distance to existing patches
#                     min_dist = min(np.linalg.norm(np.array(start) - np.array(cell)) for cell in used_cells)
#                     if min_dist > max_min_dist:
#                         max_min_dist = min_dist
#                         best_start = start

#                 if best_start is None:
#                     continue  # Skip if no valid start found

#                 start_r, start_c = best_start
#                 start = (start_r, start_c)
#                 init_std_x = random.uniform(min_std, max_std)
#                 init_std_y = random.uniform(min_std, max_std)
#                 prob_map = generate_gaussian_prob_map(start, init_std_x, init_std_y, grid_size)

#                 patch = set([start])
#                 frontier = set(get_neighbors(start, grid_size))
#                 target_size = random.randint(min_patch_size, max_patch_size)

#                 while len(patch) < target_size and frontier:
#                     probs = np.array([prob_map[r, c] if (r, c) not in used_cells else 0.0 for r, c in frontier])
#                     if probs.sum() == 0:
#                         break
#                     probs /= probs.sum()
#                     next_cell = random.choices(list(frontier), weights=probs, k=1)[0]

#                     patch.add(next_cell)
#                     frontier.remove(next_cell)

#                     for neighbor in get_neighbors(next_cell, grid_size):
#                         if neighbor not in patch and neighbor not in used_cells:
#                             frontier.add(neighbor)

#                 # Mark patch in grid
#                 if target_flag:
#                     for r, c in patch:
#                         grid[r, c] = 1  # Mark full cells as 1
#                 else:
#                     for r, c in patch:
#                         grid[r, c] = -1  # Mark full cells as 1

#                 used_cells.update(patch)

#     return grid, num_patches, target_flag

# def plot_grid(grid):
#     fig, ax = plt.subplots()
#     ax.imshow(grid, cmap='binary')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.axis("off")
    
#     buf = BytesIO()
#     plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
#     buf.seek(0)
#     plt.close(fig)
#     return buf

# def describe_image(image_buf, num_patches, target_flag):
    
#     objective = "target" if target_flag else "danger"
#     objective_term = random.choice(target_terms) if target_flag else random.choice(danger_zone_terms)
    
#     img = Image.open(image_buf)
#     prompt = f"""You are leading a team of robots.
#     Describe the location and size of each {objective} patch with respect to the environment.
#     There is {num_patches} {objective} patche{"s" if num_patches > 1 else ""}. Use sentences. Do not mention color or the grid.
#     """
#     if random.random() < 0.5:
#         prompt += f"""For the {objective} use the term {objective_term}.
#     For the exploration area use the term {random.choice(environment_terms)}.
#     For the location consider using some of the following adjectives: {random.choice(list(direction_terms.values()))}.
#     For the size consider using at least one of the following adjectives: {random.choice(list(size_terms.values()))}.
#     """           
#     response = genai_client.models.generate_content(
#         model="gemini-2.0-flash-lite",
#         contents=[prompt,img])
#     return response.text

# import concurrent.futures

# def describe_image_with_timeout(image_buf, num_patches, timeout=10):
#     # Create a copy of the BytesIO buffer to avoid issues across threads
#     image_buf.seek(0)
#     buf_copy = BytesIO(image_buf.read())
    
#     executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
#     future = executor.submit(describe_image, buf_copy, num_patches)
#     try:
#         return future.result(timeout=timeout)
#     except concurrent.futures.TimeoutError:
#         print("⚠️ Gemini took too long, skipping this sample.")
#         return None
#     except Exception as e:
#         print(f"⚠️ Gemini failed with error: {e}")
#         return None
#     finally:
#         executor.shutdown(wait=False)

# output_file = "gemini_patch_dataset_grid_3.jsonl"
# json_output_file = "gemini_patch_dataset_grid_3.json"

# # Resume from existing file if present
# start_index = 0
# buffer = []

# if os.path.exists(output_file):
#     with open(output_file, "r") as f:
#         existing_lines = f.readlines()
#         start_index = len(existing_lines)
#         print(f"📂 Resuming from index {start_index}, found existing file with {start_index} entries.")

# # Append mode so we don't erase existing content
# with open(output_file, "a") as f:
#     for i in tqdm(range(start_index, NUM_DATA_TARGETS), desc="Generating Data"):
#         grid, num_patches, target_flag = generate_grid_and_patches()
#         if target_flag:
#             image_buf = plot_grid(grid)
#         else: 
#             image_buf = plot_grid(-grid)
#         description = describe_image_with_timeout(image_buf, num_patches, timeout=10)
        
#         if description is None:
#             continue  # Skip if Gemini timed out or failed

#         time.sleep(API_DELAY_SECONDS)

#         data_point = {
#             "grid": grid.flatten().tolist(),  # Flatten the grid to a list for JSON serialization
#             "gemini_response": description
#         }
#         buffer.append(data_point)

#         if len(buffer) >= BATCH_SIZE:
#             for entry in buffer:
#                 f.write(json.dumps(entry) + "\n")
#             buffer = []

#     # Write any leftovers
#     for entry in buffer:
#         f.write(json.dumps(entry) + "\n")

# # Convert JSONL to JSON for easier reading
# with open(output_file) as f:
#     data = [json.loads(line) for line in f]
# with open(json_output_file, "a") as f:
#     json.dump(data, f, indent=2)