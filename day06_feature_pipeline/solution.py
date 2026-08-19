import pandas as pd
import numpy as np

# Training Data
train = pd.DataFrame({
    'age': [24, None, 40],
    'city': ['Pune', 'Delhi', 'Pune'],
    'bought': [0, 1, 1]
})

# FIT + TRANSFORM
def fit_transform(train):

    # Target column
    target = 'bought'

    # Feature columns
    numeric_cols = ['age']
    categorical_cols = ['city']

    # Copy data
    data = train.copy()

    # 1. Calculate numeric median
    numeric_medians = data[numeric_cols].median()

    # Fill missing numeric values
    data[numeric_cols] = data[numeric_cols].fillna(numeric_medians)

    # 2. Calculate numeric mean and std
    numeric_means = data[numeric_cols].mean()
    numeric_stds = data[numeric_cols].std(ddof=0)

    # Prevent division by zero
    numeric_stds = numeric_stds.replace(0, 1)

    # 3. Find training categories
    categories = {}

    for col in categorical_cols:
        categories[col] = sorted(data[col].dropna().unique())

    # 4. One-hot encode categorical columns
    encoded = pd.DataFrame(index=data.index)

    for col in categorical_cols:

        for category in categories[col]:

            column_name = f"{col}_{category}"

            encoded[column_name] = (
                data[col] == category
            ).astype(float)
            
    # 5. Standardize numerical columns
    numeric_data = (
        data[numeric_cols] - numeric_means
    ) / numeric_stds

    # 6. Combine numerical + categorical
    X = pd.concat(
        [numeric_data, encoded],
        axis=1
    )

    # Metadata needed for test transformation
    metadata = {
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'numeric_medians': numeric_medians,
        'numeric_means': numeric_means,
        'numeric_stds': numeric_stds,
        'categories': categories,
        'feature_columns': X.columns.tolist()
    }

    return X, metadata

# TRANSFORM TEST DATA

def transform(test, metadata):

    # Read metadata
    numeric_cols = metadata['numeric_cols']
    categorical_cols = metadata['categorical_cols']

    numeric_medians = metadata['numeric_medians']
    numeric_means = metadata['numeric_means']
    numeric_stds = metadata['numeric_stds']

    categories = metadata['categories']
    feature_columns = metadata['feature_columns']

    # Copy test data
    data = test.copy()

    # 1. Fill missing numeric values
    # Using TRAIN median
    data[numeric_cols] = data[numeric_cols].fillna(
        numeric_medians
    )

    # 2. Standardize numeric columns
    # Using TRAIN mean/std
    numeric_data = (
        data[numeric_cols] - numeric_means
    ) / numeric_stds

    # 3. One-hot encode categorical columns
    # Using TRAIN categories
    encoded = pd.DataFrame(index=data.index)

    for col in categorical_cols:

        for category in categories[col]:

            column_name = f"{col}_{category}"

            encoded[column_name] = (
                data[col] == category
            ).astype(float)

    # 4. Combine features
    X = pd.concat(
        [numeric_data, encoded],
        axis=1
    )

    # 5. EXACT same column order as training
    X = X.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return X


# FIT ON TRAINING DATA
X_train, metadata = fit_transform(train)


print("Original Training Data:")
print(train)

print("\nTransformed Training Data:")
print(X_train)

print("\nTraining Feature Columns:")
print(X_train.columns.tolist())

# TEST DATA
test = pd.DataFrame({
    'age': [30, None],
    'city': ['Pune', 'Mumbai']
})

# TRANSFORM TEST DATA
X_test = transform(test, metadata)


print("\nOriginal Test Data:")
print(test)

print("\nTransformed Test Data:")
print(X_test)

print("\nTest Feature Columns:")
print(X_test.columns.tolist())

# CHECKS
# Train and test must have identical column order
assert list(X_train.columns) == list(X_test.columns)

# No NaN values
assert not X_train.isna().any().any()
assert not X_test.isna().any().any()

# Target must not be present
assert 'bought' not in X_train.columns
assert 'bought' not in X_test.columns

print("\nAll tests passed!")
