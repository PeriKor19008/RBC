import torch
import torch.nn as nn


import torch
import torch.nn as nn

class FCAutoencoder(nn.Module):
    def __init__(self, input_dim=2500, latent_dim=64, hidden_dims=[1024, 512, 128]):
        super(FCAutoencoder, self).__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (reverse of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())  # output in range [0,1]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # Flatten input image [B, 1, 50, 50] → [B, 2500]
        x = x.view(x.size(0), -1)

        # Encode → latent → decode
        z = self.encoder(x)
        x_recon = self.decoder(z)

        # Reshape back to image [B, 1, 50, 50]
        return x_recon.view(x.size(0), 1, 50, 50)



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


class FlexibleCNN(nn.Module):
    def __init__(self,  conv_config=[("conv", 16), ("conv", 32)], fc_config=[128]):
        super(FlexibleCNN, self).__init__()
        input_shape = (1, 50, 50)
        output_dim = 4
        self.conv_layers = nn.Sequential()
        in_channels = input_shape[0]
        h, w = input_shape[1], input_shape[2]

        # === Build convolutional layers ===
        for idx, (layer_type, out_channels) in enumerate(conv_config):
            if layer_type == "conv":
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                self.conv_layers.add_module(f"conv{idx}", conv)
                self.conv_layers.add_module(f"relu{idx}", nn.ReLU())
                self.conv_layers.add_module(f"pool{idx}", nn.MaxPool2d(2))
                in_channels = out_channels
                h, w = h // 2, w // 2  # track output size
            else:
                raise ValueError(f"Unsupported conv layer type: {layer_type}")

        # === Flatten layer before FC ===
        flat_size = in_channels * h * w

        # === Build fully connected layers ===
        fc_layers = []
        fc_in = flat_size
        for idx, fc_out in enumerate(fc_config):
            fc_layers.append(nn.Linear(fc_in, fc_out))
            fc_layers.append(nn.ReLU())
            fc_in = fc_out

        fc_layers.append(nn.Linear(fc_in, output_dim))  # Final output layer

        self.fc_layers = nn.Sequential(*fc_layers)

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






