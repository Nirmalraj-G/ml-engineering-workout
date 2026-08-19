# Day 9 - Backpropagation

## Objective

Implement backpropagation for the two-layer neural network from Day 8.

## Concepts Covered

- Backpropagation
- Chain rule
- Binary cross-entropy
- Gradients
- ReLU derivative
- Parameter gradients
- Numerical gradient checking

## Gradients

The implementation calculates gradients for:

- W1
- b1
- W2
- b2

## Numerical Gradient Check

A finite-difference approximation is used to validate analytical gradients.

Central difference:

```text
f(x + epsilon) - f(x - epsilon)
--------------------------------
             2epsilon
