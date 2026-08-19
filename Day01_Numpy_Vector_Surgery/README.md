# Day 1 - NumPy Vector Surgery

## Objective

Implement common feature-scaling techniques using NumPy without explicit Python loops.

## Concepts Covered

- NumPy arrays
- Vectorization
- Min-Max scaling
- Z-score standardization
- Mean and standard deviation
- Broadcasting
- Handling zero-variance columns

## Implementation

Implemented:

- Min-Max scaler
- Z-score standardizer

The implementation avoids mutating the original input array.

## Validation

The implementation checks:

- Output shape
- Min-Max values are within the expected range
- Z-score output has mean approximately 0
- Z-score output has standard deviation approximately 1
- Constant columns do not produce NaN or infinity

## Technologies

- Python
- NumPy
