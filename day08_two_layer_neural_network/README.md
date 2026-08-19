# Day 8 - Two-Layer Neural Network

## Objective

Implement forward propagation for a small two-layer neural network using NumPy.

## Architecture

Input
  ↓
Hidden Layer
  ↓
ReLU
  ↓
Output Layer
  ↓
Sigmoid
Concepts Covered
Neural networks
Matrix multiplication
Forward propagation
ReLU activation
Sigmoid activation
Bias broadcasting
Parameter shapes
Probability output
Shapes

The network uses:

X  → (n, d)
W1 → (d, h)
W2 → (h, 1)

The final sigmoid output produces a probability between 0 and 1.

Cache

The forward function stores intermediate values required later by backpropagation.

Extension

A softmax-based output can be added for multi-class classification.

Technologies
Python
NumPy
