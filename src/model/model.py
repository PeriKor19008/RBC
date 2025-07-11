import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [B, 16, 50, 50]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 16, 25, 25]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [B, 32, 25, 25]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 12, 12]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, 12, 12]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 6, 6]
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Output: diameter, thickness, ratio, ref_index
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

class SimpleModel(nn.Module):
    def __init__(self, layers):
        super(SimpleModel, self).__init__()

        # Define a list to store layers
        self.layers = nn.ModuleList()

        # Create layers dynamically from the 'layers' argument
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))  # Input to output size

        # Output layer (assuming 3 outputs)
        self.output_layer = nn.Linear(layers[-1], 4)  # Final output layer to predict 3 values

    def forward(self, x):
        # Flatten the input image to a 1D vector (for fully connected layers)
        x = x.view(x.size(0), -1)  # Flatten the image

        # Forward pass through the hidden layers with ReLU activations
        for layer in self.layers:
            x = torch.relu(layer(x))

        # Final output layer
        x = self.output_layer(x)
        return x






