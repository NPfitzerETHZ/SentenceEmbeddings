import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
from io import BytesIO
from PIL import Image
from google import genai
from api_keys import GEMINI_API_KEY_3

# Set your Gemini API key
genai_client = genai.Client(api_key=GEMINI_API_KEY_3)

direction_terms = {
    "ern": "eastern, western, southern, northern, center",
    "cardinal": "east, west, south, north, middle",
    "erly": "easterly, westerly, southerly, norhterly, center",
    "dunno": "upper, lower, leftmost, rightmost, middle",
    "dunno_2": "top, bottom, left, right, center"
}

environment_terms =["area", "region", "zone", "territory" , "surroundings", "environment", "field", "landscape", "setting", "domain", 
                    "search area", "search region", "search zone", "search territory" , "search surroundings", "search environment", "search field", "search landscape", "search setting", "search domain"
                    "exploration area", "exploration region", "exploration zone", "exploration territory" , "exploration surroundings", "exploration environment", "exploration field", "exploration landscape", "exploration setting", "exploration domain",
                    "reconnaissance area", "reconnaissance region", "reconnaissance zone", "reconnaissance territory" , "reconnaissance surroundings", "reconnaissance environment", "reconnaissance field", "reconnaissance landscape", "reconnaissance setting", "reconnaissance domain", 
                    "investigation area", "investigation region", "investigation zone", "investigation territory" , "investigation surroundings", "investigation environment", "investigation field", "investigation landscape", "investigation setting", "investigation domain"]

size_terms = {
    "cat_1": "vast, moderate, tiny",
    "cat_2": "expansive, average, miniature",
    "cat_3": "immense, intermediate, compact",
    "cat_4": "enormous, mid-sized, narrow",
    "cat_5": "extensive, medium-scale, petite",
    "cat_6": "broad, midsize, modest",
    "cat_7": "wide, standard, limited",
    "cat_8": "colossal, fair-sized, restricted",
    "cat_9": "large, medium, small"
}

target_terms = ["Landmark", "Endpoint", "Reference point", "Objective", "Goal", "Mark", "Point", "Focus", "Destination", "Aim", "Spot", 
    "Site", "Position", "Location", "Zone", "Subject", "Waypoint"]

danger_zone_terms = ["danger zone", "danger region", "hot zone", "hot region", "red zone", "red region", "hazard area", "hazard region", "restricted area",
                     "restricted region", "no-go zone", "no-go region", "kill zone", "kill region", "combat zone", "combat region", "war zone", "war region",
                     "exclusion zone", "exclusion region", "critical zone", "critical region", "unsafe area", "unsafe region", "high-risk area", "high-risk region",
                     "death zone", "death region", "threat zone", "threat region"]


# Parameters
grid_size = 5
min_patch_size = 2
max_patch_size = 5
min_std = 0.5
max_std = 1.5
max_num_patches = 2
multipatch_prob = 1.0
no_patch_prob = 0.0
danger_zone_prob = 0.5

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
    gauss /= gauss.sum()  # normalize
    return gauss

def compute_patch_spread(patch):
    coords = np.array(list(patch))
    if len(coords) == 0:
        return 0.0, 0.0
    std_y = np.std(coords[:, 0])  # rows
    std_x = np.std(coords[:, 1])  # cols
    return std_x, std_y

def generate_grid_one_target_and_one_danger():
    
    grid = np.ones((grid_size, grid_size, 3))
    used_cells = set()
    patch_info = []
    num_patches = 0
    target = True

    for patch_idx in range(max_num_patches):
        
        if patch_idx == 1:
            target = False
        num_patches += 1

        # Sample multiple start candidates and choose the farthest one from used_cells
        candidate_starts = [(random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)) for _ in range(10)]
        best_start = None
        max_min_dist = -1

        for start in candidate_starts:
            if start in used_cells:
                continue
            if not used_cells:
                best_start = start
                break
            # Compute minimum distance to used cells
            dists = [np.linalg.norm(np.array(start) - np.array(cell)) for cell in used_cells]
            min_dist = min(dists)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_start = start

        if best_start is None:
            continue  # Skip if all candidates are used

        start_r, start_c = best_start
        start = (start_r, start_c)
        init_std_x = random.uniform(min_std, max_std)
        init_std_y = random.uniform(min_std, max_std)
        prob_map = generate_gaussian_prob_map(start, init_std_x, init_std_y, grid_size)

        patch = set([start])
        frontier = set(get_neighbors(start, grid_size))
        target_size = random.randint(min_patch_size, max_patch_size)

        while len(patch) < target_size and frontier:
            probs = np.array([prob_map[r, c] if (r, c) not in used_cells else 0.0 for r, c in frontier])
            if probs.sum() == 0:
                break
            probs /= probs.sum()
            next_cell = random.choices(list(frontier), weights=probs, k=1)[0]

            patch.add(next_cell)
            frontier.remove(next_cell)

            for neighbor in get_neighbors(next_cell, grid_size):
                if neighbor not in patch and neighbor not in used_cells:
                    frontier.add(neighbor)

            for r, c in patch:
                if target:
                    grid[r, c] = [1, 0, 0]
                else:
                    grid[r, c] = [0, 0, 1]
            
        used_cells.update(patch)

        coords = np.array(list(patch))
        mean_r = coords[:, 0].mean()
        mean_c = coords[:, 1].mean()
        norm_center = (mean_r / (grid_size - 1), mean_c / (grid_size - 1))
        std_x_actual, std_y_actual = compute_patch_spread(patch)
        norm_spread_x = std_x_actual * 2 / (grid_size - 1)
        norm_spread_y = std_y_actual * 2 / (grid_size - 1)

        patch_info.append((norm_center, norm_spread_x, norm_spread_y))

    return grid, patch_info, num_patches

def generate_grid_and_two_patches():
    
    grid = np.ones((grid_size, grid_size, 3))
    used_cells = set()
    patch_info = []
    num_patches = 0

    if random.random() > no_patch_prob:  # 85% chance to place at least one patch
        for patch_idx in range(max_num_patches):
            if patch_idx == 0 or random.random() < multipatch_prob:  # Always place first, 50% chance for others
                num_patches += 1

                # Sample multiple start candidates and choose the farthest one from used_cells
                candidate_starts = [(random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)) for _ in range(10)]
                best_start = None
                max_min_dist = -1

                for start in candidate_starts:
                    if start in used_cells:
                        continue
                    if not used_cells:
                        best_start = start
                        break
                    # Compute minimum distance to used cells
                    dists = [np.linalg.norm(np.array(start) - np.array(cell)) for cell in used_cells]
                    min_dist = min(dists)
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_start = start

                if best_start is None:
                    continue  # Skip if all candidates are used

                start_r, start_c = best_start
                start = (start_r, start_c)
                init_std_x = random.uniform(min_std, max_std)
                init_std_y = random.uniform(min_std, max_std)
                prob_map = generate_gaussian_prob_map(start, init_std_x, init_std_y, grid_size)

                patch = set([start])
                frontier = set(get_neighbors(start, grid_size))
                target_size = random.randint(min_patch_size, max_patch_size)

                while len(patch) < target_size and frontier:
                    probs = np.array([prob_map[r, c] if (r, c) not in used_cells else 0.0 for r, c in frontier])
                    if probs.sum() == 0:
                        break
                    probs /= probs.sum()
                    next_cell = random.choices(list(frontier), weights=probs, k=1)[0]

                    patch.add(next_cell)
                    frontier.remove(next_cell)

                    for neighbor in get_neighbors(next_cell, grid_size):
                        if neighbor not in patch and neighbor not in used_cells:
                            frontier.add(neighbor)

                    for r, c in patch:
                        grid[r, c] = [1, 0, 0]

                used_cells.update(patch)

                coords = np.array(list(patch))
                mean_r = coords[:, 0].mean()
                mean_c = coords[:, 1].mean()
                norm_center = (mean_r / (grid_size - 1), mean_c / (grid_size - 1))
                std_x_actual, std_y_actual = compute_patch_spread(patch)
                norm_spread_x = std_x_actual * 2 / (grid_size - 1)
                norm_spread_y = std_y_actual * 2 / (grid_size - 1)

                patch_info.append((norm_center, norm_spread_x, norm_spread_y))

    return grid, patch_info, num_patches

# def generate_grid_and_two_patches():
#     grid = np.ones((grid_size, grid_size, 3))
#     used_cells = set()
#     patch_info = []
#     num_patches = 0

#     if random.random() < 0.85: # Patch or no Patch
#         for patch_idx in range(max_num_patches):  # 🔧 generate multiple patches
#             if (patch_idx == 0 or random.random() < 1.0):
#                 num_patches += 1
#                 while True:
#                     start_r = random.randint(0, grid_size - 1)
#                     start_c = random.randint(0, grid_size - 1)
#                     if (start_r, start_c) not in used_cells:
#                         break

#                 start = (start_r, start_c)
#                 init_std_x = random.uniform(min_std, max_std)
#                 init_std_y = random.uniform(min_std, max_std)
#                 prob_map = generate_gaussian_prob_map((start_r, start_c), init_std_x, init_std_y, grid_size)

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
#                 for r, c in patch:
#                     grid[r, c] = [1, 0, 0]  # 🔧 still red for both

#                 used_cells.update(patch)

#                 coords = np.array(list(patch))
#                 mean_r = coords[:, 0].mean()
#                 mean_c = coords[:, 1].mean()
#                 norm_center = (mean_r / (grid_size - 1), mean_c / (grid_size - 1))
#                 std_x_actual, std_y_actual = compute_patch_spread(patch)
#                 norm_spread_x = std_x_actual * 2 / (grid_size - 1)
#                 norm_spread_y = std_y_actual * 2 / (grid_size - 1)

#                 patch_info.append((norm_center, norm_spread_x, norm_spread_y))

#     return grid, patch_info, num_patches


def plot_grid(grid):
    fig, ax = plt.subplots()
    ax.imshow(grid)
    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=1)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def describe_image(image_buf):
    
    img = Image.open(image_buf)
    
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
                  
    response = genai_client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[prompt,img])
    return response.text

# prompt = f"""You are leading a team of robots.
#     Describe the location and size of each {objective} patch with respect to the environment.
#     There is {num_patches} target patch{"es" if num_patches > 1 else ""}. Use sentences. Do not mention color or the grid.
#     """
#     if random.random() < 0.5:
#         prompt += f"""For the {objective} use the term {objective_term}.
#     For the exploration area use the term {random.choice(environment_terms)}.
#     For the location use the following adjectives: {random.choice(list(direction_terms.values()))}.
#     For the size use at least one of the following adjectives: {random.choice(list(size_terms.values()))}.
#     """

# Streamlit UI
st.title("🗺️ Random Map Generator with Gemini")
st.write("Click the button to generate a new map and get a description of the red patch.")

if st.button("Generate New Map"):
    grid, patch_info, num_patches = generate_grid_one_target_and_one_danger()
    image_buf = plot_grid(grid)
    st.image(image_buf, caption="Generated Map with Two Patches", use_column_width=True)

    for i, (norm_start, norm_std_x, norm_std_y) in enumerate(patch_info):
        st.subheader(f"📌 Normalized Patch {i+1} Details:")
        st.markdown(f"- **Start location** (normalized): ({norm_start[0]:.2f}, {norm_start[1]:.2f})")
        st.markdown(f"- **Standard deviations** (normalized): σₓ = {norm_std_x:.2f}, σᵧ = {norm_std_y:.2f}")

    with st.spinner("Gemini is thinking..."):
        description = describe_image(image_buf)
    st.subheader("🔍 Gemini's Description:")
    st.write(description)

