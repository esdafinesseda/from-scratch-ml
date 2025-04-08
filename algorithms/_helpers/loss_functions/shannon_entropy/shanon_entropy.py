import numpy as np
from typing import Literal


def entropy_criterion(
    labels: np.ndarray, entropy_type: Literal["bit", "nat"] = "nat"
) -> float:
    # Handle empty input
    if len(labels) == 0:
        return 0.0

    # Convert input to np array
    labels = np.asarray(labels)

    # Get total number of samples
    n_samples = len(labels)

    # Label counts
    _, label_counts = np.unique(labels, return_counts=True)

    entropy_value = 0.0

    for count in label_counts:
        # Avoid log zero errors
        if count == 0:
            continue

        p = count / n_samples  # label probability

        print(entropy_type)

        # Apply correct log base
        if entropy_type == "bit":
            entropy_value -= p * np.log2(p)

        elif entropy_type == "nat":
            entropy_value -= p * np.log(p)

        else:
            raise ValueError("Entropy type must be 'bit' or 'nat'")

    return entropy_value
