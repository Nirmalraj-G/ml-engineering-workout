import numpy as np

# K-Means
def kmeans(X, k, max_iter=100, tol=1e-4, seed=0):

    X = np.asarray(X, dtype=float)

    # Validate input
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if k < 1:
        raise ValueError("k must be at least 1")

    if k > len(X):
        raise ValueError(
            "k cannot be greater than number of samples"
        )

    # Random generator
    rng = np.random.default_rng(seed)

    # Initialize centroids
    # Choose k random data points
    initial_indices = rng.choice(
        len(X),
        size=k,
        replace=False
    )

    centroids = X[initial_indices].copy()

    # Main K-Means loop
    for iteration in range(max_iter):

        # 1. Calculate squared Euclidean distances
        distances = np.sum(
            ( X[:, None, :] - centroids[None, :, :] ) ** 2, axis=2
        )

        # 2. Assign each point to nearest centroid
        labels = np.argmin(
            distances,
            axis=1
        )

        # 3. Recalculate centroids
        new_centroids = np.zeros_like(
            centroids
        )

        for j in range(k):

            cluster_points = X[
                labels == j
            ]

            # Empty cluster strategy:
            # Keep previous centroid
            if len(cluster_points) == 0:

                new_centroids[j] = centroids[j]

            else:

                new_centroids[j] = np.mean(
                    cluster_points,
                    axis=0
                )

        # 4. Check centroid movement
        movement = np.max(
            np.linalg.norm(
                new_centroids - centroids,
                axis=1
            )
        )

        # Update centroids
        centroids = new_centroids

        # 5. Check convergence

        if movement < tol:
            break

    # Calculate final distances
    distances = np.sum(
        (
            X[:, None, :]
            - centroids[None, :, :]
        ) ** 2,
        axis=2
    )

    # Final labels
    labels = np.argmin(
        distances,
        axis=1
    )

    # Inertia
    inertia = np.sum(
        np.min(
            distances,
            axis=1
        )
    )

    return centroids, labels, float(inertia)

# Create Three Synthetic Blobs
rng = np.random.default_rng(42)

cluster_1 = rng.normal(
    loc=[2, 2],
    scale=0.5,
    size=(50, 2)
)

cluster_2 = rng.normal(
    loc=[8, 8],
    scale=0.5,
    size=(50, 2)
)

cluster_3 = rng.normal(
    loc=[2, 8],
    scale=0.5,
    size=(50, 2)
)

X = np.vstack([
    cluster_1,
    cluster_2,
    cluster_3
])

# Run K-Means
centroids, labels, inertia = kmeans(
    X,
    k=3,
    max_iter=100,
    tol=1e-4,
    seed=0
)

# Display Results
print("Final Centroids:")
print(centroids)

print("\nFirst 20 Labels:")
print(labels[:20])

print("\nInertia:")
print(inertia)

print("\nNumber of points in each cluster:")

for cluster in range(3):

    count = np.sum(
        labels == cluster
    )

    print(
        f"Cluster {cluster}: {count} points"
    )

# Basic Checks
assert centroids.shape == (3, 2)

assert labels.shape == (150,)

assert np.all(
    labels >= 0
)

assert np.all(
    labels < 3
)

assert np.isscalar(inertia)

assert inertia >= 0

print("\nAll tests passed!")
