import matplotlib.pyplot as plt
import os

def plot_all_val_losses(all_val_losses, filename="all_val_losses.png"):
    import matplotlib.pyplot as plt
    import os

    plt.figure(figsize=(10, 6))
    for label, val_loss in all_val_losses.items():
        plt.plot(range(1, len(val_loss) + 1), val_loss, label=label)

    plt.title("Validation Loss Across CNN Configurations")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)

    os.makedirs("plots", exist_ok=True)
    path = os.path.join("plots", filename)
    plt.savefig(path)
    plt.close()
    print(f"All validation losses plot saved to: {path}")

def plot_loss_graphs(train_losses, val_losses, run_number, num_epochs, learning_rate, batch_size, layers, filename="graphs"):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # Add the training parameters to the graph
    text = f"Epochs: {num_epochs}\nLR: {learning_rate}\nBatch Size: {batch_size}\nLayers: {layers}"
    plt.text(0.95, 0.95, text, ha='right', va='top', transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

    # Save the graph into the provided folder (now points to this run's figs/)
    plt.savefig(os.path.join(filename, f'loss_graph_run{layers}_{run_number}.png'))
    plt.close()


def plot_and_save_loss_graph(epoch_losses, run_number, num_epochs, learning_rate, batch_size, layers, save_dir='graphs_Simple'):
    # Create the directory if it doesn't exist (now it's the run's figs/)
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

    # Save the graph into the provided folder (now points to this run's figs/)
    plt.savefig(os.path.join(save_dir, f'loss_graph_run{run_number}.png'))
    plt.close()
