# Day 6 - Feature Preprocessing Pipeline

## Objective

Build a reusable preprocessing pipeline using pandas.

## Tasks

The pipeline performs:

1. Numeric missing-value imputation using the training median.
2. One-hot encoding of a categorical feature.
3. Standardization of numeric features.

## Concepts Covered

- Missing-value imputation
- Median
- One-hot encoding
- Standardization
- Train/test preprocessing
- Feature consistency
- Data leakage prevention

## Important Rule

All preprocessing statistics are fitted using training data only.

The test data must never influence:

- Numeric medians
- Scaling statistics
- Category mappings

## Unknown Categories

Unknown categories in test data are handled without breaking the transformation.

## Validation

Training and test transformations produce identical feature columns and ordering.

## Technologies

- Python
- Pandas
- NumPy
