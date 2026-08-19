# Day 3 - KNN Classifier

## Objective

Implement a binary K-Nearest Neighbors classifier from scratch using NumPy.

## Concepts Covered

- K-Nearest Neighbors
- Euclidean distance
- Distance-based classification
- Nearest-neighbor selection
- Majority voting
- Tie handling

## Implementation

For every test point:

1. Calculate its distance from every training point.
2. Select the k nearest points.
3. Check their labels.
4. Select the majority label.
5. Resolve ties by choosing the smaller label.

## Validation

The implementation validates that:

- k is at least 1
- k does not exceed the number of training samples
- Predictions have the expected shape

## Extension

The implementation can also return the indices of the nearest neighbors.

## Technologies

- Python
- NumPy
