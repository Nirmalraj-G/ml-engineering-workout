import numpy as np

# Activation Functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)

# Forward Propagation
def forward(X, params):

    W1 = params["W1"]
    b1 = params["b1"]

    W2 = params["W2"]
    b2 = params["b2"]

    # First layer
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    # Second layer
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    # Cache values needed for backpropagation
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache

# Binary Cross-Entropy
def binary_cross_entropy(y, prob):

    eps = 1e-15

    prob = np.clip(
        prob,
        eps,
        1 - eps
    )

    loss = -np.mean(
        y * np.log(prob)
        + (1 - y) * np.log(1 - prob)
    )

    return loss

# Backpropagation
def backward(X, y, params, cache):

    W2 = params["W2"]

    Z1 = cache["Z1"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Make sure y has shape (n, 1)
    y = y.reshape(-1, 1)

    n = X.shape[0]

    # Output layer
    # Sigmoid + binary cross entropy
    dZ2 = A2 - y

    # Gradient of W2
    dW2 = (A1.T @ dZ2) / n

    # Gradient of b2
    db2 = np.sum(dZ2, axis=0, keepdims=True) / n

    # Hidden layer
    dA1 = dZ2 @ W2.T

    # ReLU derivative
    dZ1 = dA1 * (Z1 > 0)

    # Gradient of W1
    dW1 = (X.T @ dZ1) / n

    # Gradient of b1
    db1 = np.sum(dZ1, axis=0, keepdims=True) / n

    # Return gradients
    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return grads

# Numerical Gradient
def numerical_grad(loss_fn, params, key, idx, eps=1e-5):

    # Save original value
    original_value = params[key][idx]

    # f(x + epsilon)
    params[key][idx] = original_value + eps

    loss_plus = loss_fn()

    # f(x - epsilon)
    params[key][idx] = original_value - eps

    loss_minus = loss_fn()

    # Restore original parameter
    params[key][idx] = original_value

    # Central finite difference
    numerical_gradient = (
        loss_plus - loss_minus
    ) / (2 * eps)

    return numerical_gradient

# Relative Error
def relative_error(analytical, numerical):

    numerator = abs(
        analytical - numerical
    )

    denominator = max(
        1e-8,
        abs(analytical) + abs(numerical)
    )

    return numerator / denominator

# Create Dataset
np.random.seed(42)

X = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [3.0, 2.0],
    [1.0, 3.0],
    [2.0, 3.0]
])

# IMPORTANT:
# Shape is (n, 1)
y = np.array([
    [0.],
    [0.],
    [1.],
    [1.],
    [1.]
])

# Initialize Parameters
input_features = 2
hidden_units = 3

params = {

    "W1": np.random.randn(
        input_features,
        hidden_units
    ) * 0.1,

    "b1": np.zeros(
        (1, hidden_units)
    ),

    "W2": np.random.randn(
        hidden_units,
        1
    ) * 0.1,

    "b2": np.zeros(
        (1, 1)
    )
}

# Forward Pass
prob, cache = forward(
    X,
    params
)

# Calculate Loss
loss = binary_cross_entropy(
    y,
    prob
)

print("Loss:", loss)

# Analytical Gradients
grads = backward(
    X,
    y,
    params,
    cache
)


print("\nAnalytical Gradients:")

print("\ndW1:")
print(grads["dW1"])

print("\ndb1:")
print(grads["db1"])

print("\ndW2:")
print(grads["dW2"])

print("\ndb2:")
print(grads["db2"])

# Numerical Gradient Function
def loss_function():

    prob, _ = forward(
        X,
        params
    )

    return binary_cross_entropy(
        y,
        prob
    )

# Check W1 Gradient
w1_index = (0, 0)

numerical_w1 = numerical_grad(
    loss_function,
    params,
    "W1",
    w1_index
)

analytical_w1 = grads["dW1"][w1_index]

error_w1 = relative_error(
    analytical_w1,
    numerical_w1
)


print("\n================================")
print("W1 Gradient Check")
print("================================")

print("Analytical gradient:",
      analytical_w1)

print("Numerical gradient:",
      numerical_w1)

print("Relative error:",
      error_w1)

# Check W2 Gradient
w2_index = (0, 0)

numerical_w2 = numerical_grad(
    loss_function,
    params,
    "W2",
    w2_index
)

analytical_w2 = grads["dW2"][w2_index]

error_w2 = relative_error(
    analytical_w2,
    numerical_w2
)


print("\n================================")
print("W2 Gradient Check")
print("================================")

print("Analytical gradient:",
      analytical_w2)

print("Numerical gradient:",
      numerical_w2)

print("Relative error:",
      error_w2)

# Final Assertions
assert error_w1 < 1e-4
assert error_w2 < 1e-4

print("\nAll gradient checks passed!")
