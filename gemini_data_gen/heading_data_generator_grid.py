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

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api_keys import GEMINI_API_KEY

# ========================== Configuration ==========================

# Gemini API Key
genai_client = genai.Client(api_key=GEMINI_API_KEY)  # 🔐 Replace with your actual key

# Dataset generation parameters
NUM_DATA_TARGETS = 5000
API_DELAY_SECONDS = 0.
grid_size = 10
min_patch_size = 2
max_patch_size = 8
min_std = 0.1
max_std = 1.5
max_num_patches = 1
multipatch_prob = 0.5
no_patch_prob = 0.0
danger_zone_prob = 0.0
BATCH_SIZE = 1
# ========================== Vocabulary ==========================

color_dict = {
    "red":      {"rgb": [1.0, 0.0, 0.0], "index": 0},
    "green":    {"rgb": [0.0, 1.0, 0.0], "index": 1},
    "blue":     {"rgb": [0.0, 0.0, 1.0], "index": 2},
    "yellow":   {"rgb": [1.0, 1.0, 0.0], "index": 3},
    "orange":   {"rgb": [1.0, 0.5, 0.0], "index": 4},
    # "cyan":     {"rgb": [0.0, 1.0, 1.0], "index": 5},
    # "magenta":  {"rgb": [1.0, 0.0, 1.0], "index": 6},
    # "purple":   {"rgb": [0.5, 0.0, 0.5], "index": 7},
    # "pink":     {"rgb": [1.0, 0.75, 0.8], "index":8},
    # "brown":    {"rgb": [0.6, 0.4, 0.2], "index": 9},
    # "gray":     {"rgb": [0.5, 0.5, 0.5], "index": 10}
}

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

def plot_grid(grid, rgb):
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='binary')
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return buf

from matplotlib.patches import Rectangle

def plot_grid_with_rgb(grid, rgb):
    # Create an RGB image with all white
    H, W = grid.shape
    color_img = np.ones((H, W, 3), dtype=np.float32)

    # Apply the rgb color to the positions where grid == 1
    for i in range(3):
        color_img[:, :, i] = np.where(grid == 1, rgb[i], 1.0)

    # Plot
    fig, ax = plt.subplots()
    ax.imshow(color_img)
    
    grid_size = grid.shape[0]
    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    #ax.grid(which='minor', color='black', linewidth=1)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Add outer boundary rectangle
    rect = Rectangle((-0.45,-0.45), grid_size-0.15, grid_size-0.1,
                     linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # Save to buffer
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
    
    color_info = []
    
    num_targets = random.randint(1, 4)

    if random.random() > no_patch_prob:
        for patch_idx in range(max_num_patches):
            
            color_name, _color_info = random.choice(list(color_dict.items()))
            color_index = _color_info["index"]
            _color = _color_info["rgb"]
            color_info.append({"name":color_name,"index": color_index, "rgb": _color, "num_targets": num_targets})
            
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

    return grid, num_patches, target_flag, color_info

def describe_image(image_buf, color_name, num_targets):
    
    plural = "s" if num_targets > 1 else ""
    
    prompt = f"""You are leading a team of robots and you need to help them find {num_targets} target{plural}.
    The image is a simplification of the environment, showing a rough estimate of where the team should look. It does not show the targets, rather a region of high interest.
    Guide the team and describe the location and size of the {color_name} region with respect to the environment. The more precise you are, the better chance they have to find it.
    Use sentences and be precise. The color and number of targets must be part of the instruction.
    """
    if random.random() < 0.5:
        prompt += f"""For the target use the term {random.choice(target_terms)}.
    For the exploration area use the term {random.choice(environment_terms)}.
    For the location of each patch use the following adjectives: {random.choice(list(direction_terms.values()))}.
    For the size of each patch use at least one of the following adjectives: {random.choice(list(size_terms.values()))}.
    Be creative in how you address the team.
    """

    img = Image.open(image_buf)
    response = genai_client.models.generate_content(
        model="gemini-2.0-flash",
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

def describe_image_with_timeout(image_buf, color_name, num_targets, timeout=10):
    image_buf.seek(0)
    buf_copy = BytesIO(image_buf.read())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(describe_image, buf_copy, color_name, num_targets)
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
        
def target_and_color():
    
    output_file = "sentences/gemini_patch_dataset_multi_target_color_medium_.jsonl"
    json_output_file = "sentences/gemini_patch_dataset_multi_target_color_medium.json"
    start_index = 0
    buffer = []

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_lines = f.readlines()
            start_index = len(existing_lines)
            print(f"📂 Resuming from index {start_index}")

    with open(output_file, "a") as f:
        for i in tqdm(range(start_index, NUM_DATA_TARGETS), desc="Generating Data"):
            grid, _ , _ , color_info  = generate_grid_and_patches()
            image_buf = plot_grid_with_rgb(grid, color_info[0]["rgb"])
            description = describe_image_with_timeout(image_buf, color_info[0]["name"], color_info[0]["num_targets"])

            if description is None:
                continue

            time.sleep(API_DELAY_SECONDS)

            buffer.append({
                "grid": grid.flatten().tolist(),
                "class": color_info[0]["index"],
                "max_targets": color_info[0]["num_targets"],
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
        
target_and_color()

# Old prompts
#==================
if False:
    
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
    
    objective = "target" if target_flag else "danger"
    term = random.choice(target_terms if target_flag else danger_zone_terms)

    prompt = f"""You are leading a team of robots.
    Describe the location and size of each {objective} patch with respect to the environment.
    There is {num_patches} {objective} patch{'es' if num_patches > 1 else ''}. Use sentences. Do not mention color or the grid.
    """
    if random.random() < 0.5:
        prompt += f"""Use the term "{term}".
        Refer to the environment as "{random.choice(environment_terms)}".
        Use directional terms like {random.choice(list(direction_terms.values()))}.
        Use size terms like {random.choice(list(size_terms.values()))}.
        """
    
    