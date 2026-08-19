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

    # Layer 1
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    # Layer 2
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache

# Binary Cross Entropy
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

    y = y.reshape(-1, 1)

    n = X.shape[0]

    # Output layer
    dZ2 = A2 - y

    dW2 = (A1.T @ dZ2) / n

    db2 = np.sum(
        dZ2,
        axis=0,
        keepdims=True
    ) / n

    # Hidden layer
    dA1 = dZ2 @ W2.T

    # ReLU derivative
    dZ1 = dA1 * (Z1 > 0)

    dW1 = (X.T @ dZ1) / n

    db1 = np.sum(
        dZ1,
        axis=0,
        keepdims=True
    ) / n

    return {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

# Parameter Update
def update_params(params, grads, lr):

    params["W1"] -= lr * grads["dW1"]
    params["b1"] -= lr * grads["db1"]

    params["W2"] -= lr * grads["dW2"]
    params["b2"] -= lr * grads["db2"]

    return params

# Mini-Batch Training
def train(X, y,params, epochs=50, batch_size=32,lr=0.01,seed=0,patience=5):

    rng = np.random.default_rng(seed)

    losses = []

    # Early stopping variables
    best_loss = np.inf
    patience_counter = 0

    n = X.shape[0]

    for epoch in range(epochs):

        # 1. Shuffle rows
        indices = rng.permutation(n)

        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # 2. Mini-batches
        for start in range(0, n, batch_size):

            end = start + batch_size

            # This also includes the final short batch
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # 3. Forward propagation
            predictions, cache = forward(
                X_batch,
                params
            )

            # 4. Backpropagation
            grads = backward(
                X_batch,
                y_batch,
                params,
                cache
            )

            # 5. SGD parameter update
            update_params(
                params,
                grads,
                lr
            )

        # Calculate loss on the COMPLETE dataset
        predictions, _ = forward(
            X,
            params
        )

        epoch_loss = binary_cross_entropy(
            y,
            predictions
        )

        losses.append(epoch_loss)

        # Print progress
        print(
            f"Epoch {epoch + 1:3d}/{epochs} "
            f"- Loss: {epoch_loss:.6f}"
        )

        # Early stopping
        if epoch_loss < best_loss:

            best_loss = epoch_loss
            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= patience:

            print(
                f"\nEarly stopping at epoch "
                f"{epoch + 1}"
            )

            break

    return params, losses

# Create Toy Dataset
np.random.seed(42)

X = np.array([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.],
    [0., 2.],
    [2., 0.],
    [2., 2.],
    [3., 0.],
    [0., 3.],
    [3., 3.]
])

# Simple separable labels
y = np.array([
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    1.,
    1.,
    1.,
    1.
])

# Initialize Neural Network
input_features = 2
hidden_units = 4

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

# Train
params, losses = train(
    X,
    y,
    params,
    epochs=50,
    batch_size=3,
    lr=0.1,
    seed=0,
    patience=5
)

# Results
print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")

print("Number of epochs:", len(losses))

print("Initial loss:", losses[0])

print("Final loss:", losses[-1])

print("\nLearned W1:")
print(params["W1"])

print("\nLearned b1:")
print(params["b1"])

print("\nLearned W2:")
print(params["W2"])

print("\nLearned b2:")
print(params["b2"])

# Verify Loss Trend
print("\nLoss decreased:",losses[-1] < losses[0])
