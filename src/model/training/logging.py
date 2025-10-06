import os
from src.model.plot import plot_and_save_loss_graph


def log_run_details(num_epochs, learning_rate, batch_size, layers,
                    final_loss, device, epoch_losses,  # train losses
                    val_losses=None,                    # <-- NEW (optional)
                    run_log_path='mock_run_observations.txt',
                    figs_dir='graphs_SimpleNN'):

    run_number = get_next_run_number(run_log_path) if os.path.exists(run_log_path) else 1

    with open(run_log_path, 'a') as f:
        f.write(f"\nrun #{run_number}:\n")
        if isinstance(layers, (list, tuple)) and len(layers) >= 3:
            f.write(f"Hidden Layers: {layers[0]} --> {layers[1]} --> {layers[2]}\n")
        else:
            f.write(f"Layers: {layers}\n")
        f.write(f"Number of Epochs: {num_epochs}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Using device: {device}\n")
        for i in range(num_epochs):
            # Train
            f.write(f"Epoch {i + 1} train: {epoch_losses[i]:.6f}\n")
            # Val (if provided)
            if val_losses is not None and i < len(val_losses):
                f.write(f"Epoch {i + 1} val: {val_losses[i]:.6f}\n")
        f.write(f"Finished Training and saved the model.\n")
        f.write(f"---------------------------------------------------------\n")

    # keep your old plot, saved into this run’s figs/
    plot_and_save_loss_graph(epoch_losses, run_number, num_epochs, learning_rate,
                             batch_size, layers, out_dir=figs_dir)


def get_next_run_number(log_file='mock_run_observations.txt'):
    # Check if the log file exists and read its contents
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            run_count = sum(1 for line in lines if line.startswith("run #"))
            return run_count + 1
    else:
        return 1

