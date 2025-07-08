import torch
from torch import nn
import copy
import os
import matplotlib.pyplot as plt


# Function to plot the loss graph and save it
def plot_and_save_loss_graph(epoch_losses, run_number, num_epochs, learning_rate, batch_size, layers, save_dir='graphs'):
    # Create the graphs directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plotting the loss graph
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', color='b', label='Loss')
    plt.title(f'Training Loss for Run #{run_number}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Add the training parameters to the graph
    text = f"Epochs: {num_epochs}\nLR: {learning_rate}\nBatch Size: {batch_size}\nLayers: {layers}"
    plt.text(0.95, 0.95, text, ha='right', va='top', transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

    # Save the graph to the 'graphs' folder with filename 'loss_graph_runX.png'
    plt.savefig(os.path.join(save_dir, f'loss_graph_run{run_number}.png'))
    plt.close()

def get_next_run_number(log_file='mock_run_observations.txt'):
    # Check if the log file exists and read its contents
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            # Read all lines and count how many "run" sections exist
            lines = f.readlines()
            # Count the number of "run #" occurrences
            run_count = sum(1 for line in lines if line.startswith("run #"))
            return run_count + 1  # Return next run number
    else:
        # If the file doesn't exist, start with run #1
        return 1


def log_run_details(num_epochs, learning_rate, batch_size, layers, final_loss, device, epoch_losses):
    # Get the next run number
    run_number = get_next_run_number()

    # Open the file in append mode to ensure we don't overwrite previous runs
    with open('mock_run_observations.txt', 'a') as f:
        # Write the details with the incremented run number
        f.write(f"\nrun #{run_number}:\n")
        f.write(f"Hidden Layers: {layers[0]} --> {layers[1]} --> {layers[2]}\n")
        f.write(f"Number of Epochs: {num_epochs}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Batch Size: {batch_size}\n")

        # Log training progress
        f.write(f"Using device: {device}\n")
        for i in range(num_epochs):
            f.write(f"Epoch {i + 1} complete. Loss: {epoch_losses[i]:.4f}\n")
        f.write(f"Finished Training and saved the model.\n")
        f.write(f"---------------------------------------------------------\n")

        plot_and_save_loss_graph(epoch_losses, run_number, num_epochs, learning_rate, batch_size, layers,save_dir='graphs')


def train_model(model, dataloaders, criterion, optimizer, num_epochs, batch_size, learning_rate, layers):
    # Device configuration (set device to GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move the model to the selected device (GPU or CPU)
    model.to(device)

    epoch_losses = []  # to store losses for each epoch
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        # Each epoch code here
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloaders['train'])
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    final_loss = epoch_losses[-1]  # Final loss after the last epoch
    print(f"Finished Training. Final Loss: {final_loss:.4f}")

    # Save trained model weights
    os.makedirs("models", exist_ok=True)
    model_save_path = f"models/model_run{get_next_run_number() - 1}.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Log run details after training
    log_run_details(num_epochs, learning_rate, batch_size, layers, final_loss, device, epoch_losses)
    return epoch_losses
