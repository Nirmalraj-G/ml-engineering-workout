# Day 11 - CNN Shape Detective

## Objective

Build a small convolutional neural network in PyTorch and track tensor shapes after every layer.

## Input

The model accepts grayscale images of size:
28 × 28

A dummy batch is used with shape:

(4, 1, 28, 28)
Architecture
Input
  ↓
Conv2D
  ↓
ReLU
  ↓
MaxPool
  ↓
Conv2D
  ↓
ReLU
  ↓
MaxPool
  ↓
Flatten
  ↓
Linear

Shape Flow
(4, 1, 28, 28)
        ↓
(4, 8, 28, 28)
        ↓
(4, 8, 14, 14)
        ↓
(4, 16, 14, 14)
        ↓
(4, 16, 7, 7)
        ↓
(4, 784)
        ↓
(4, 10)
Concepts Covered
Convolution
Feature maps
ReLU
Max pooling
Flattening
Fully connected layers
Tensor dimensions
CNN architecture
Output

The final layer produces 10 logits for digit classification.
Softmax is not included in the final layer.

Technologies
Python
PyTorch

