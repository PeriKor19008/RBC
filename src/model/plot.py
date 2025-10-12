# src/model/plot.py
import matplotlib.pyplot as plt
import os

def plot_all_val_losses(all_val_losses, out_dir, filename="all_val_losses.png"):
    """
    Overlays multiple validation-loss curves and saves ONE figure into a comparison folder.
    REQUIRED: out_dir (comparison run folder).
    """
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for label, val_loss in all_val_losses.items():
        plt.plot(range(1, len(val_loss) + 1), val_loss, label=str(label))

    plt.title("Validation Loss Across Configurations")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)

    path = os.path.join(out_dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")

def plot_loss_graphs(train_losses, val_losses, run_number, num_epochs, learning_rate, batch_size, layers, out_dir,lr_tag):
    """
    Per-run: train + val loss. REQUIRED: out_dir (usually the run's figs/).
    """
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    if val_losses:
        plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend(loc="upper right")
    plt.grid(True)
    lr_line = f"{learning_rate}" + (f" ({lr_tag})" if lr_tag else "")
    text = f"Epochs: {num_epochs}\nLR: {lr_line}\nBatch Size: {batch_size}\nLayers: {layers}"
    plt.text(0.95, 0.95, text, ha='right', va='top', transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

    plt.savefig(os.path.join(out_dir, f'loss_graph_run{run_number}.png'))
    plt.close()

def plot_and_save_loss_graph(epoch_losses, run_number, num_epochs, learning_rate, batch_size, layers, out_dir, lr_tag):
    """
    Per-run: train-only loss. REQUIRED: out_dir (run's figs/).
    """
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label='Loss')
    plt.title(f'Training Loss for Run #{run_number}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc="upper right")

    lr_line = f"Learning Rate: {learning_rate}"+ (f"({lr_tag})" if lr_tag else "")
    text = f"Epochs: {num_epochs}\nLR: {lr_line}\nBatch Size: {batch_size}\nLayers: {layers}"
    plt.text(0.95, 0.95, text, ha='right', va='top', transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

    plt.savefig(os.path.join(out_dir, f'loss_graph_run{run_number}.png'))
    plt.close()
