# Day 14 - Model Selection API

## Objective

Build a reusable experiment function that compares two machine learning models and selects the best configuration using validation F1.

## Models

The experiment compares:

1. Logistic Regression
2. Random Forest Classifier

## Workflow

Dataset
   ↓
Train / Validation / Test split
   ↓
Tune Logistic Regression
   ↓
Tune Random Forest
   ↓
Compare validation F1
   ↓
Select best model
   ↓
Retrain selected model
   ↓
Evaluate once on test data

Hyperparameters
Logistic Regression

The C parameter is tested with multiple values.

Random Forest

Different max_depth values are tested.

Model Selection

The model and hyperparameters are selected using validation F1.

The test set is not used for hyperparameter selection.

Final Evaluation

After selecting the best model, it is retrained using the training and validation data.

The final model is then evaluated once on the held-out test set.

Result

The API returns a JSON-serializable dictionary containing:

model
best_params
validation_f1
test_f1
seed

Concepts Covered
Train/validation/test split
Stratification
Hyperparameter tuning
F1 score
Model selection
Test-set isolation
Reproducibility
JSON serialization
Technologies
Python
NumPy
Scikit-learn
