import numpy as np

# Dataset
X = np.array([
    [2., 10.],
    [4., 20.],
    [6., 30.]
])

# Min-Max Scaling
def minmax_scale(X):
    X_copy = X.copy()

                    #axis=0 -- Column-wise minimum, keepdims=True -- Keeping same dimensions
    min_val = np.min(X_copy, axis=0, keepdims=True) 
    max_val = np.max(X_copy, axis=0, keepdims=True)

    eps = 1e-8
    range_safe = np.where((max_val - min_val) < eps, 1.0, (max_val - min_val))

    scaled = (X_copy - min_val) / range_safe

    assert scaled.shape == X_copy.shape
    assert np.all(scaled >= -1e-8)
    assert np.all(scaled <= 1 + 1e-8)

    return scaled


# Z-Score Standardization
def zscore(X):
    X_copy = X.copy()

    mean = np.mean(X_copy, axis=0, keepdims=True)
    std = np.std(X_copy, axis=0, keepdims=True)

    eps = 1e-8
    std_safe = np.where(std < eps, 1.0, std)

    standardized = (X_copy - mean) / std_safe

    assert standardized.shape == X_copy.shape

    mask = std.flatten() > eps

    if np.any(mask):
        assert np.allclose(
            np.mean(standardized[:, mask], axis=0), 0, atol=1e-7)

        assert np.allclose(
            np.std(standardized[:, mask], axis=0), 1, atol=1e-7)

    return standardized


# Test 1

print("Original Dataset:")
print(X)

scaled = minmax_scale(X)
print("\nMin-Max Scaled:")
print(scaled)

z = zscore(X)
print("\nZ-Score Standardized:")
print(z)

print("\nMean:")
print(np.mean(z, axis=0))

print("\nStd:")
print(np.std(z, axis=0))


# Test 2 (Constant Column)

X2 = np.array([
    [2., 10., 5.],
    [4., 20., 5.],
    [6., 30., 5.]
])

print("\n==============================")
print("Dataset with Constant Column")
print("==============================")

print(X2)

scaled2 = minmax_scale(X2)
print("\nMin-Max Scaled:")
print(scaled2)

z2 = zscore(X2)
print("\nZ-Score Standardized:")
print(z2)

print("\nMean:")
print(np.mean(z2, axis=0))

print("\nStd:")
print(np.std(z2, axis=0))
