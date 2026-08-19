import numpy as np


def knn_predict(X_train, y_train, X_test, k=3):

    # Validate k
    if k < 1 or k > len(X_train):
        raise ValueError("k must be between 1 and len(X_train)")

    predictions = []
    neighbor_indices = []

    for x in X_test:

        # 1. Calculate Euclidean distances
        distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))

        # 2. Get indices of k nearest points
        nearest_indices = np.argsort(distances)[:k]

        # 3. Get labels of nearest points
        nearest_labels = y_train[nearest_indices]

        # 4. Majority vote
        labels, counts = np.unique(
            nearest_labels,
            return_counts=True
        )

        # np.unique sorts labels.
        # np.argmax returns the first maximum,
        # so ties automatically choose the smaller label.
        prediction = labels[np.argmax(counts)]

        predictions.append(prediction)
        neighbor_indices.append(nearest_indices)

    return np.array(predictions), np.array(neighbor_indices)


# Create a 2D Dataset

X_train = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    [8, 8],
    [8, 9],
    [9, 8],
    [9, 9]
])

y_train = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
])


# Test Data
X_test = np.array([
    [1.5, 1.5],
    [8.5, 8.5],
    [5, 5]
])


# Make Predictions
predictions, neighbor_indices = knn_predict(
    X_train, y_train, X_test, k=3
)


print("Predictions:")
print(predictions)

print("\nNeighbor Indices:")
print(neighbor_indices)


# Display Neighbor Details
for i in range(len(X_test)):

    print("\nTest point:", X_test[i])

    print("Nearest neighbor indices:",
          neighbor_indices[i])

    print("Nearest neighbor labels:",
          y_train[neighbor_indices[i]])

    print("Prediction:",
          predictions[i])
