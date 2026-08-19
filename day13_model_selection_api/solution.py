import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


# Model Selection Experiment
def run_experiment(X, y, seed=42):

    # ------------------------------------------------------
    # 1. Train / Validation / Test Split
    # ------------------------------------------------------

    # First split: 80% temporary data, 20% final test data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y
    )

    # Second split:75% of temporary = 60% overall training, 25% of temporary = 20% overall validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=seed,
        stratify=y_temp
    )

    # Final proportions: Train      = 60%, Validation = 20%, Test       = 20%

    # ------------------------------------------------------
    # 2. Candidate Hyperparameters
    # ------------------------------------------------------

    logistic_C_values = [
        0.01,
        0.1,
        1.0,
        10.0
    ]

    forest_depth_values = [
        2,
        4,
        6,
        8,
        None
    ]

    # Store all validation experiments
    experiments = []

    # ------------------------------------------------------
    # 3. Tune Logistic Regression
    # ------------------------------------------------------

    for C in logistic_C_values:

        model = LogisticRegression(
            C=C,
            max_iter=1000,
            random_state=seed
        )

        # Train ONLY on training data
        model.fit(
            X_train,
            y_train
        )

        # Evaluate ONLY on validation data
        valid_pred = model.predict(
            X_valid
        )

        valid_f1 = f1_score(
            y_valid,
            valid_pred,
            zero_division=0
        )

        experiments.append({
            "model": "LogisticRegression",
            "param": {
                "C": C
            },
            "validation_f1": float(valid_f1)
        })

    # ------------------------------------------------------
    # 4. Tune Random Forest
    # ------------------------------------------------------

    for max_depth in forest_depth_values:

        model = RandomForestClassifier(
            max_depth=max_depth,
            n_estimators=100,
            random_state=seed
        )

        # Train ONLY on training data
        model.fit(
            X_train,
            y_train
        )

        # Evaluate ONLY on validation data
        valid_pred = model.predict(
            X_valid
        )

        valid_f1 = f1_score(
            y_valid,
            valid_pred,
            zero_division=0
        )

        experiments.append({
            "model": "RandomForestClassifier",
            "param": {
                "max_depth": max_depth
            },
            "validation_f1": float(valid_f1)
        })

    # ------------------------------------------------------
    # 5. Select Best Model
    # ------------------------------------------------------

    best_result = max(
        experiments,
        key=lambda result: result["validation_f1"]
    )

    best_model_name = best_result["model"]
    best_params = best_result["param"]
    best_validation_f1 = best_result["validation_f1"]

    # ------------------------------------------------------
    # 6. Combine Training + Validation Data
    # ------------------------------------------------------

    X_train_final = np.concatenate(
        [X_train, X_valid],
        axis=0
    )

    y_train_final = np.concatenate(
        [y_train, y_valid],
        axis=0
    )

    # ------------------------------------------------------
    # 7. Create Selected Model
    # ------------------------------------------------------

    if best_model_name == "LogisticRegression":

        final_model = LogisticRegression(
            C=best_params["C"],
            max_iter=1000,
            random_state=seed
        )

    else:

        final_model = RandomForestClassifier(
            max_depth=best_params["max_depth"],
            n_estimators=100,
            random_state=seed
        )

    # ------------------------------------------------------
    # 8. Retrain Selected Model
    # ------------------------------------------------------

    final_model.fit(
        X_train_final,
        y_train_final
    )

    # ------------------------------------------------------
    # 9. Evaluate ONCE on Test Set
    # ------------------------------------------------------

    test_pred = final_model.predict(
        X_test
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        zero_division=0
    )

    # ------------------------------------------------------
    # 10. Return JSON-Serializable Dictionary
    # ------------------------------------------------------

    results = {
        "model": best_model_name,
        "best_params": best_params,
        "validation_f1": float(best_validation_f1),
        "test_f1": float(test_f1),
        "seed": int(seed)
    }

    return results

# Create Example Dataset
from sklearn.datasets import make_classification


X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    weights=[0.65, 0.35],
    random_state=42
)

# Run Experiment
results = run_experiment(
    X,
    y,
    seed=42
)

# Print Results
print("===================================")
print("MODEL SELECTION RESULTS")
print("===================================")

print(
    "Selected model:",
    results["model"]
)

print(
    "Best parameters:",
    results["best_params"]
)

print(
    "Validation F1:",
    results["validation_f1"]
)

print(
    "Test F1:",
    results["test_f1"]
)

print(
    "Seed:",
    results["seed"]
)

# JSON Serialization Test
json_result = json.dumps(
    results,
    indent=4
)

print("\nJSON Result:")
print(json_result)

# Basic Assertions
assert "model" in results
assert "best_params" in results
assert "validation_f1" in results
assert "test_f1" in results
assert "seed" in results

assert results["model"] in [
    "LogisticRegression",
    "RandomForestClassifier"
]

assert 0 <= results["validation_f1"] <= 1
assert 0 <= results["test_f1"] <= 1

assert isinstance(
    json_result,
    str
)

print("\nAll tests passed!")
