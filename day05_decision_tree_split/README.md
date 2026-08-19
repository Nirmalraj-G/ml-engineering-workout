# Day 5 - Decision Tree Split Search

## Objective

Find the best threshold for splitting a numerical feature using weighted Gini impurity.

## Concepts Covered

- Decision trees
- Gini impurity
- Binary classification
- Candidate split thresholds
- Weighted impurity
- Numerical features

## Gini Impurity

For class probabilities p:

Gini = 1 - sum(p²)

A pure node has Gini impurity of 0.

## Implementation

The algorithm:

1. Sorts the feature values.
2. Finds candidate thresholds between unique values.
3. Splits the dataset into left and right groups.
4. Calculates Gini impurity for both groups.
5. Calculates weighted impurity.
6. Selects the threshold with the lowest impurity.

## Edge Cases

Thresholds producing an empty left or right side are ignored.

## Extension

The implementation can return the row indices belonging to the left and right partitions.

## Technologies

- Python
- NumPy
