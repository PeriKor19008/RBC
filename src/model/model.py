import torch
import torch.nn as nn


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






