import numpy as np


# Sigmoid Function
def sigmoid(z):
    """
    Numerically safe sigmoid function.
    """

    z = np.asarray(z, dtype=float)

    result = np.empty_like(z)

    # For positive values
    positive = z >= 0
    result[positive] = 1 / (1 + np.exp(-z[positive]))

    # For negative values
    negative = ~positive
    exp_z = np.exp(z[negative])
    result[negative] = exp_z / (1 + exp_z)

    return result


# Classification Report
def classification_report(y_true, prob, threshold=0.5):

    y_true = np.asarray(y_true)
    prob = np.asarray(prob)

    # Validate inputs
    if len(y_true) != len(prob):
        raise ValueError("y_true and prob must have the same length")

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    # Clip probabilities before log
    eps = 1e-15

    prob_clipped = np.clip(prob, eps, 1 - eps)

    # Binary Cross Entropy
    bce = -np.mean(
        y_true * np.log(prob_clipped)
        + (1 - y_true) * np.log(1 - prob_clipped)
    )

    # Convert probabilities to predictions
    y_pred = (prob >= threshold).astype(int)

    # Confusion Matrix values
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Accuracy
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Precision
    precision_denominator = tp + fp

    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = tp / precision_denominator

    # Recall
    recall_denominator = tp + fn

    if recall_denominator == 0:
        recall = 0.0
    else:
        recall = tp / recall_denominator

    # F1 Score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    # Return five metrics
    return {
        "bce": bce,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# Test Data
# Imbalanced label vector
y_true = np.array([ 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

# Predicted probabilities
prob = np.array([
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.45,
    0.60,
    0.65,
    0.80,
    0.90
])


# Test Sigmoid
z = np.array([-2., -1., 0., 1., 2.])

print("Sigmoid:")
print(sigmoid(z))


# Compare Different Thresholds
for threshold in [0.3, 0.5, 0.7]:

    metrics = classification_report(
        y_true,
        prob,
        threshold=threshold
    )

    print("\n-----------------------------")
    print("Threshold:", threshold)
    print("-----------------------------")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


# Lower threshold usually increases recall but can reduce precision,
# while a higher threshold usually increases precision but can reduce recall.


# Test Constant / Edge Cases
print("\n==============================")
print("Edge Case Test")
print("==============================")


# No positive predictions
y_true_edge = np.array([0, 0, 0])
prob_edge = np.array([0.1, 0.2, 0.3])

result = classification_report(
    y_true_edge,
    prob_edge,
    threshold=0.5
)

print(result)


# Assertions

# Sigmoid should always be between 0 and 1
sigmoid_values = sigmoid(np.array([-1000., 0., 1000.]))

assert np.all(sigmoid_values >= 0)
assert np.all(sigmoid_values <= 1)

# Check all required metrics exist
metrics = classification_report(
    y_true,
    prob,
    threshold=0.5
)

assert "bce" in metrics
assert "accuracy" in metrics
assert "precision" in metrics
assert "recall" in metrics
assert "f1" in metrics

# All classification metrics should be between 0 and 1
assert 0 <= metrics["accuracy"] <= 1
assert 0 <= metrics["precision"] <= 1
assert 0 <= metrics["recall"] <= 1
assert 0 <= metrics["f1"] <= 1

print("\nAll tests passed!")
