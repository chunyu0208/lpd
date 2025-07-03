import numpy as np
import random

from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean


def lpd_order_schedule(group_sizes=None, grid_size=16, proximity_threshold=1, repulsion_threshold=1):
    """
    Generate a linear order of grid coords based on local proximity and repulsion constraints.

    Args:
        group_sizes (list): List of group sizes for each step. Default is [1]*256.
        grid_size (int): Size of the grid (grid_size x grid_size).
        proximity_threshold (float): Minimum proximity score to prioritize candidates.
        repulsion_threshold (float): Minimum distance to avoid selecting too-close points in a group.

    Returns:
        list: Linear indices of selected grid coords.
    """
    if group_sizes is None:
        group_sizes = [1] * (grid_size * grid_size)

    grid_coords = [[i, j] for i in range(grid_size) for j in range(grid_size)]
    selected_coords = []
    
    for step, group_size in enumerate(group_sizes):
        if step == 0:
            # For the first step, select a random coord. We always assume the group size for the first step is 1.
            selected_coords.append(random.choice(grid_coords))
            continue
        
        # Calculate the proximity score for all remaining grid coords
        candidates = []
        for coord in grid_coords:
            if coord in selected_coords:
                continue
                
            # Calculate the proximity score based on euclidean distance to already selected grid coords
            proximity_score = 0
            for selected_coord in selected_coords:
                if abs(coord[0] - selected_coord[0]) <= 1 and abs(coord[1] - selected_coord[1]) <= 1:
                    distance = euclidean(coord, selected_coord)
                    if distance > 0:
                        proximity_score += 1.0 / distance
            candidates.append([proximity_score, coord])
        
        # Shuffle candidates so that grid coords with the same proximity score are randomly ordered
        random.shuffle(candidates)
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates1 = [item[1] for item in candidates if item[0] >= proximity_threshold]
        candidates2 = [item[1] for item in candidates if item[0] < proximity_threshold]

        step_selected = []
        step_filtered = []

        # Proximity-based selection
        while len(step_selected) < group_size and candidates1:
            candidate = candidates1.pop(0)
            too_close = False
            for selected in step_selected:
                if abs(candidate[0] - selected[0]) <= repulsion_threshold and abs(candidate[1] - selected[1]) <= repulsion_threshold:
                    too_close = True
                    step_filtered.append(candidate)
                    break
                
            if not too_close:
                step_selected.append(candidate)
        
        step_filtered.extend(candidates1)
        candidates2.extend(step_filtered)

        # Low-dependency selection
        remaining = group_size - len(step_selected)
        if remaining > 0:
            step_selected.extend(farthest_point_sampling(step_selected, candidates2, remaining))
        
        selected_coords.extend(step_selected)

    return np.ravel_multi_index(np.array(selected_coords).T, (grid_size, grid_size)).tolist()


def farthest_point_sampling(existing_points, candidate_points, num_to_select):
    """Select points using farthest point sampling algorithm.
    
    Args:
        existing_points: List of already selected points
        candidate_points: List of candidate points to select from
        num_to_select: Number of points to select

    Returns:
        List of selected points from candidate_points
    """
    if len(candidate_points) <= num_to_select:
        return candidate_points
    
    # Convert to numpy arrays for efficient computation
    existing_np = np.array(existing_points)
    candidates_np = np.array(candidate_points)
    
    # Initialize with existing points
    selected_np = existing_np.copy()
    selected_indices = []
    
    for _ in range(num_to_select):
        if len(selected_np) == 0:
            # If no existing points, select randomly
            idx = np.random.randint(len(candidates_np))
            selected_np = candidates_np[idx][np.newaxis, :]
        else:
            # Calculate distances from all candidates to selected points
            distances = cdist(candidates_np, selected_np)
            min_distances = np.min(distances, axis=1)
            
            # Set already selected candidates to 0 distance
            min_distances[selected_indices] = 0
            
            # Select the candidate with maximum minimum distance
            idx = np.argmax(min_distances)
            selected_np = np.vstack([selected_np, candidates_np[idx]])

        selected_indices.append(idx)

    return [candidate_points[i] for i in selected_indices]


def run_lpd_order_schedule(num_runs=10, group_sizes=[1]*256, grid_size=16, proximity_threshold=1, repulsion_threshold=1, max_workers=8):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(lpd_order_schedule, group_sizes, grid_size, proximity_threshold, repulsion_threshold) for _ in range(num_runs)]
        results = []
        for future in futures:
            results.append(future.result())
    return results