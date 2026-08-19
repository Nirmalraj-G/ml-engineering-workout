# Day 4 - Logistic Regression Metrics

## Objective

Implement the main evaluation components used for binary classification.

## Concepts Covered

- Sigmoid function
- Binary cross-entropy
- Classification threshold
- Accuracy
- Precision
- Recall
- F1 score
- Numerical stability

## Implementation

Implemented:

- Numerically safe sigmoid
- Binary cross-entropy
- Classification predictions using a configurable threshold
- Accuracy
- Precision
- Recall
- F1 score

## Threshold Experiment

The classifier can be evaluated using different thresholds:

- 0.3
- 0.5
- 0.7

A lower threshold generally increases recall while potentially reducing precision.
A higher threshold generally increases precision while potentially reducing recall.

## Edge Cases

The implementation handles zero denominators for precision and recall without producing errors.

## Technologies

- Python
- NumPy
