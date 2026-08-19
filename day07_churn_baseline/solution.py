import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# 1. Generate Synthetic Churn Dataset
def make_churn_data(n=500, seed=7):

    rng = np.random.default_rng(seed)

    # Generate customer features
    # Customer tenure in months
    tenure = rng.integers(
        low=1,
        high=61,
        size=n
    )

    # Monthly spending
    monthly_spend = rng.normal(
        loc=1000,
        scale=250,
        size=n
    )

    # Make sure spending is positive
    monthly_spend = np.clip(
        monthly_spend,
        200,
        None
    )

    # Number of support tickets
    support_tickets = rng.poisson(
        lam=3,
        size=n
    )

    # Create churn probability score
    noise = rng.normal(
        loc=0,
        scale=1,
        size=n
    )

    score = (
        -0.06 * tenure
        + 0.002 * monthly_spend
        + 0.45 * support_tickets
        + noise
        - 2.5
    )

    # Convert score to churn label
    churn = (score > 0).astype(int)

    # Create DataFrame
    data = pd.DataFrame({
        "tenure": tenure,
        "monthly_spend": monthly_spend,
        "support_tickets": support_tickets,
        "churn": churn
    })

    return data

# 2. Generate Data
df = make_churn_data(
    n=500,
    seed=7
)

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nChurn distribution:")
print(df["churn"].value_counts())

# 3. Separate Features and Target
X = df[
    [
        "tenure",
        "monthly_spend",
        "support_tickets"
    ]
]

y = df["churn"]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=7,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 5. Train Logistic Regression
model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)

# 6. Make Predictions
y_pred = model.predict(X_test)

# 7. Calculate Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

# 8. Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred
)

# 9. Print Metric Summary
print("\n==============================")
print("CHURN BASELINE RESULTS")
print("==============================")

print(f"Holdout Accuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
print(cm)

# 10. Inspect Errors
errors = X_test[
    y_test != y_pred
].copy()

errors["actual"] = y_test[
    y_test != y_pred
]

errors["predicted"] = y_pred[
    y_test != y_pred
]

print("\nNumber of errors:")
print(len(errors))

print("\nFirst 10 errors:")
print(errors.head(10))

# 11. Model Coefficients
print("\nModel Coefficients:")

for feature, coefficient in zip(
    X.columns,
    model.coef_[0]
):
    print(
        f"{feature}: {coefficient:.4f}"
    )

print("\nIntercept:")
print(model.intercept_[0])
