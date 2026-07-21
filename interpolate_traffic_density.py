import pandas as pd
import numpy as np


def redistribute_counts(counts, variance, directions=None):
    """Redistribute counts by a variance in [0, 1], preserving the group's total sum exactly."""
    counts = np.array(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return counts.astype(int).tolist()

    n = len(counts)
    # Calculate original proportions
    original_props = counts / total

    # Generate random proportions favoring a whole direction (e.g. North)
    random_props = np.zeros(n)
    
    if directions is not None and len(set(directions)) > 1:
        unique_dirs = list(set(directions))
        favored_dir = np.random.choice(unique_dirs)
        
        favored_idx = [i for i, d in enumerate(directions) if d == favored_dir]
        other_idx = [i for i, d in enumerate(directions) if d != favored_dir]
        
        if favored_idx and other_idx:
            # Distribute 70% among favored direction lanes
            random_props[favored_idx] = np.random.dirichlet([1] * len(favored_idx)) * 0.85
            # Distribute 30% among other direction lanes
            random_props[other_idx] = np.random.dirichlet([1] * len(other_idx)) * 0.15
        else:
            random_props = np.random.dirichlet([1] * n)
    elif n > 3:
        # Fallback to random 3 lanes if no directions provided
        top_idx = np.random.choice(n, min(3, n-1), replace=False)
        other_idx = np.setdiff1d(np.arange(n), top_idx)
        
        # Distribute using Dirichlet
        random_props[top_idx] = np.random.dirichlet([1] * len(top_idx)) * 0.85
        random_props[other_idx] = np.random.dirichlet([1] * len(other_idx)) * 0.15
    else:
        # If 3 or fewer lanes, just distribute uniformly
        random_props = np.random.dirichlet([1] * n)

    # Interpolate between original and random
    new_props = (1 - variance) * original_props + variance * random_props

    # Scale to total and round using the Largest Remainder Method
    new_counts = new_props * total
    final_counts = np.floor(new_counts).astype(int)

    # Correct rounding drift to keep the sum exactly the same
    remainder = int(round(total - final_counts.sum()))
    if remainder > 0:
        # Distribute remaining units to categories with the largest decimals
        indices = np.argsort(new_counts - final_counts)[-remainder:]
        final_counts[indices] += 1

    return final_counts.tolist()


def process_traffic_data(file_path, variance, output_path):
    df = pd.read_csv(file_path, header=None)

    # Row 1 contains the directions (N, E, S, W)
    directions_row = df.iloc[1].values
    # Row 3 contains the class labels (C, B, A, H, T, P)
    class_labels = df.iloc[3].values

    # Group column indices by class (ignoring the first two metadata columns)
    groups = {}
    for i in range(2, len(class_labels)):
        cls = class_labels[i]
        if pd.isna(cls): continue
        if cls not in groups:
            groups[cls] = []
        groups[cls].append(i)

    # Process each data row (starting from index 4)
    for idx in range(4, len(df)):
        row = df.iloc[idx]

        # Skip rows without a timestamp in column 1
        if pd.isna(row[1]) or str(row[1]).strip() == "":
            continue

        # Apply redistribution independently for each class group
        for cls, indices in groups.items():
            try:
                original_vals = [float(row[i]) if not pd.isna(row[i]) else 0 for i in indices]
                dirs = [str(directions_row[i]).strip() for i in indices]
                new_vals = redistribute_counts(original_vals, variance, dirs)

                # Write back the varied counts
                for i, val in zip(indices, new_vals):
                    df.iloc[idx, i] = val
            except (ValueError, TypeError):
                continue

    df.to_csv(output_path, index=False, header=False)
    print(f"Processed: {output_path} (Variance: {variance})")

# Example Usage:
# for v in [0.3, 0.5, 0.8, 1.0]:
#     process_traffic_data('Mock.csv', v, f'Mock_ClassVaried_{v}.csv')