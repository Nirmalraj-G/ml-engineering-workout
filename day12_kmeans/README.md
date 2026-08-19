# Day 13 - K-Means Clustering

## Objective

Implement K-Means clustering from scratch using NumPy.

## Concepts Covered

- Unsupervised learning
- K-Means clustering
- Centroids
- Squared Euclidean distance
- Cluster assignment
- Centroid updates
- Convergence
- Inertia

## Algorithm

Initialize centroids
       ↓
Calculate distances
       ↓
Assign points to clusters
       ↓
Recalculate centroids
       ↓
Check convergence
       ↓
Repeat

Distance

Squared Euclidean distance is used:

distance = sum((x - centroid)²)
Empty Clusters

If a cluster becomes empty, the implementation keeps its previous centroid.

Inertia

Inertia is calculated as the sum of the squared distances between each point and its assigned centroid.

Lower inertia indicates that points are closer to their cluster centers.

Test

The implementation is tested on three synthetic clusters and can visualize the resulting labels using Matplotlib.

Technologies
Python
NumPy
Matplotlib
