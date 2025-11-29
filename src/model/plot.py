import matplotlib.pyplot as plt
import os


def plot_loss_graphs(train_losses, val_losses, run_number, num_epochs, learning_rate, batch_size, layers, out_dir,lr_tag):

    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
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
    #plt.show()
    plt.close()



def plot_lr_graph(epoch_lrs, run_number, num_epochs, learning_rate,
                  batch_size, layers, out_dir, lr_tag=None):
    """
    Per-run: Learning Rate vs Epoch. Saves into the same figs/ dir as the loss graph.
    """
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(epoch_lrs) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, epoch_lrs, marker='o', label='LR')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    title = 'Learning Rate over Epochs'
    if lr_tag:
        title += f' ({lr_tag})'
    plt.title(title)
    plt.grid(True)
    plt.legend(loc="upper right")

    lr_line = f"{learning_rate}" + (f" ({lr_tag})" if lr_tag else "")
    text = f"Epochs: {num_epochs}\nLR: {lr_line}\nBatch Size: {batch_size}\nLayers: {layers}"
    plt.text(0.95, 0.95, text, ha='right', va='top', transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

    plt.savefig(os.path.join(out_dir, f'lr_graph_run{run_number}.png'))
    plt.close()
