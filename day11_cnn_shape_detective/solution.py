import torch
import torch.nn as nn

# CNN Model
class SmallCNN(nn.Module):

    def __init__(self):

        super().__init__()
        
        # First CNN block
        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # Second CNN block
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        # Dynamically calculate flatten size
        dummy = torch.zeros(
            1, 1, 28, 28
        )

        with torch.no_grad():

            feature_output = self.features(dummy)

        flatten_size = feature_output.view(
            1, -1
        ).shape[1]

        print("Flatten size:", flatten_size)

        # Classifier
        self.classifier = nn.Linear(
            flatten_size,
            10
        )

    def forward(self, x):

        # Print shape after every layer
        for layer in self.features:

            x = layer(x)

            print(
                f"{layer.__class__.__name__:15s} -> {tuple(x.shape)}"
            )

        # Flatten
        x = torch.flatten(
            x,
            start_dim=1
        )

        print(
            f"{'Flatten':15s} -> {tuple(x.shape)}"
        )

        # Classifier
        x = self.classifier(x)

        print(
            f"{'Linear':15s} -> {tuple(x.shape)}"
        )

        return x

# Create Model
model = SmallCNN()

print("\nModel:")
print(model)

# Dummy Batch
X = torch.randn(
    4,
    1,
    28,
    28
)

# Forward Pass
print("\n==============================")
print("TENSOR SHAPES")
print("==============================")

output = model(X)

# Final Result
print("\nFinal output shape:")
print(output.shape)

# Assertions
assert output.shape == (4, 10)

print("\nSuccess!")
print("Forward pass returned shape (4, 10).")
