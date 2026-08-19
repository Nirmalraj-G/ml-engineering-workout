# Day 2 - Linear Regression from Scratch

## Objective

Implement linear regression using batch gradient descent without using a machine learning library.

## Problem

Fit the equation:

y = 3x + 2

using gradient descent.

## Concepts Covered

- Linear regression
- Mean Squared Error
- Gradient descent
- Weight and bias
- Analytical gradients
- Iterative optimization
- Loss tracking

## Implementation

The model starts with:

- weight = 0
- bias = 0

The parameters are updated using the gradients of the MSE loss.

## Expected Result

The learned parameters should approach:

- Weight ≈ 3
- Bias ≈ 2

The final MSE should be very close to zero.

## Technologies

- Python
- NumPy
