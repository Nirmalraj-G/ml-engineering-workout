import numpy as np


# Gini Impurity
def gini(y):

    # Empty array
    if len(y) == 0:
        return 0.0

    # Get class counts
    _, counts = np.unique(y, return_counts=True)

    # Convert counts to probabilities
    probabilities = counts / len(y)

    # Gini = 1 - sum(p^2)
    return 1 - np.sum(probabilities ** 2)


# Find Best Decision Tree Split
def best_split(x, y):

    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    if len(x) < 2:
        raise ValueError("At least two samples are required")

    # Sort unique feature values
    unique_values = np.sort(np.unique(x))

    # Need at least two unique values
    if len(unique_values) < 2:
        return None, None, None, None

    # Candidate thresholds
    # Midpoints between neighboring values
    thresholds = (
        unique_values[:-1] + unique_values[1:]
    ) / 2

    best_threshold = None
    best_impurity = np.inf

    best_left_indices = None
    best_right_indices = None

    # Test every threshold
    for threshold in thresholds:

        # Left: x <= threshold
        left_indices = np.where(x <= threshold)[0]

        # Right: x > threshold
        right_indices = np.where(x > threshold)[0]

        # Skip empty sides
        if len(left_indices) == 0 or len(right_indices) == 0:
            continue

        # Get labels
        y_left = y[left_indices]
        y_right = y[right_indices]

        # Calculate Gini
        gini_left = gini(y_left)
        gini_right = gini(y_right)

        # Weighted Gini impurity
        n = len(y)

        weighted_gini = (
            (len(y_left) / n) * gini_left
            + (len(y_right) / n) * gini_right
        )

        # Keep the best split
        if weighted_gini < best_impurity:
            best_impurity = weighted_gini
            best_threshold = threshold
            best_left_indices = left_indices
            best_right_indices = right_indices

    return (
        best_threshold,
        best_impurity,
        best_left_indices,
        best_right_indices
    )


# Test Dataset
x = np.array([1., 2., 3., 4., 5., 6.])

y = np.array([
    0,
    0,
    0,
    1,
    1,
    1
])


# Find Best Split
threshold, impurity, left_indices, right_indices = best_split(x, y)

print("Best Threshold:", threshold)

print("Best Weighted Gini:", impurity)

print("Left Indices:", left_indices)

print("Right Indices:", right_indices)

print("Left Values:", x[left_indices])

print("Right Values:", x[right_indices])

print("Left Labels:", y[left_indices])

print("Right Labels:", y[right_indices])

# Assertions
assert threshold == 3.5
assert np.isclose(impurity, 0.0)

assert np.array_equal(
    left_indices,
    np.array([0, 1, 2])
)

assert np.array_equal(
    right_indices,
    np.array([3, 4, 5])
)

print("\nAll tests passed!")
