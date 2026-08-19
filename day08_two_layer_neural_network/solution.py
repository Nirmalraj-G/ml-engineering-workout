import numpy as np

# Activation Functions
def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Forward Propagation
def forward(X, params):

    W1 = params["W1"]
    b1 = params["b1"]

    W2 = params["W2"]
    b2 = params["b2"]

    # Layer 1
    Z1 = X @ W1 + b1

    # ReLU activation
    A1 = relu(Z1)
    
    # Layer 2
    Z2 = A1 @ W2 + b2

    # Sigmoid activation
    A2 = sigmoid(Z2)

    # Cache for backpropagation
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    # Check output probability
    assert np.all(A2 >= 0)
    assert np.all(A2 <= 1)

    return A2, cache

# Create Example Data
np.random.seed(42)

# 5 samples
# 3 input features

X = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 1.0, 3.0],
    [3.0, 2.0, 1.0],
    [1.0, 3.0, 2.0],
    [2.0, 3.0, 1.0]
])


# Initialize Parameters
input_features = 3
hidden_units = 4

params = {

    # W1: (input_features, hidden_units)
    "W1": np.random.randn(
        input_features,
        hidden_units
    ) * 0.1,

    # b1: (hidden_units,)
    "b1": np.zeros(hidden_units),

    # W2: (hidden_units, 1)
    "W2": np.random.randn(
        hidden_units,
        1
    ) * 0.1,

    # b2: (1,)
    "b2": np.zeros(1)
}

# Forward Pass
A2, cache = forward(X, params)

# Display Results
print("Input X:")
print(X)

print("\nW1:")
print(params["W1"])

print("\nb1:")
print(params["b1"])

print("\nW2:")
print(params["W2"])

print("\nb2:")
print(params["b2"])

print("\nZ1:")
print(cache["Z1"])

print("\nA1 after ReLU:")
print(cache["A1"])

print("\nZ2:")
print(cache["Z2"])

print("\nFinal probabilities A2:")
print(A2)

# Check Shapes
print("\nShapes:")

print("X :", X.shape)
print("W1:", params["W1"].shape)
print("b1:", params["b1"].shape)
print("Z1:", cache["Z1"].shape)
print("A1:", cache["A1"].shape)
print("W2:", params["W2"].shape)
print("b2:", params["b2"].shape)
print("Z2:", cache["Z2"].shape)
print("A2:", A2.shape)

# Assertions
assert params["W1"].shape == (3, 4)
assert params["W2"].shape == (4, 1)

assert cache["Z1"].shape == (5, 4)
assert cache["A1"].shape == (5, 4)

assert cache["Z2"].shape == (5, 1)
assert A2.shape == (5, 1)

assert np.all(A2 >= 0)
assert np.all(A2 <= 1)

print("\nAll tests passed!")
