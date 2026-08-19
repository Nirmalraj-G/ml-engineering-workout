import numpy as np

# Dataset
x = np.array([0., 1., 2., 3.])
y = 3 * x + 2


def fit_linear(x, y, lr=0.1, steps=200):

    # Initial values
    w = 0.0
    b = 0.0

    # Store loss values
    history = []

    n = len(x)

    for i in range(steps):

        # 1. Prediction
        y_pred = w * x + b

        # 2. Error
        error = y_pred - y

        # 3. Mean Squared Error
        mse = np.mean(error ** 2)

        # Store loss
        history.append(mse)

        # 4. Gradients
        dw = (2 / n) * np.sum(error * x)
        db = (2 / n) * np.sum(error)

        # 5. Update parameters
        w = w - lr * dw
        b = b - lr * db

    return w, b, history


# Train the model
w, b, history = fit_linear(x, y)

# Results
print("Learned weight:", w)
print("Learned bias:", b)

print("First loss:", history[0])
print("Final loss:", history[-1])

# Check success condition
assert history[-1] < 1e-4
assert np.isclose(w, 3, atol=1e-2)
assert np.isclose(b, 2, atol=1e-2)

print("\nTraining successful!")
