# src/model/train.py
import torch
from torch import nn
import copy
import os
import matplotlib.pyplot as plt

# === NEW: import tiny helpers for per-run folders ===
from src.utils.run_utils import RunConfig, make_run_id, ensure_run_dir, write_config


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


def get_next_run_number(log_file='mock_run_observations.txt'):
    # Check if the log file exists and read its contents
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            run_count = sum(1 for line in lines if line.startswith("run #"))
            return run_count + 1
    else:
        return 1


# === NEW: small helper to create a per-run folder (and a figs/ subfolder) ===
def _make_run_dirs(num_epochs, learning_rate, batch_size, layers, base_models_dir="models"):
    # Build a simple config (just for naming reproducibility)
    cfg = RunConfig(
        arch="SimpleNN",
        epochs=num_epochs,
        lr=learning_rate,
        batch_size=batch_size,
        notes=f"layers={layers}"
    )
    # Use helper to create a unique run id + directory
    run_id = make_run_id(cfg, ds_fingerprint="manual")  # keep it simple; no dataset fingerprint needed here
    run_dir = ensure_run_dir(base_models_dir, run_id)
    figs_dir = os.path.join(run_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # Save a tiny config json (handy later)
    write_config(run_dir, cfg, ds_fingerprint="manual", extra={"layers": layers})

    return run_dir, figs_dir, run_id


def log_run_details(num_epochs, learning_rate, batch_size, layers, final_loss, device, epoch_losses,
                    run_log_path='mock_run_observations.txt', figs_dir='graphs_SimpleNN'):

    # Get the next run number (based on this log file)
    run_number = get_next_run_number(run_log_path)

    # Append to the (now per-run) log file
    with open(run_log_path, 'a') as f:
        f.write(f"\nrun #{run_number}:\n")
        f.write(f"Hidden Layers: {layers[0]} --> {layers[1]} --> {layers[2]}\n")
        f.write(f"Number of Epochs: {num_epochs}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Using device: {device}\n")
        for i in range(num_epochs):
            f.write(f"Epoch {i + 1} complete. Loss: {epoch_losses[i]:.4f}\n")
        f.write(f"Finished Training and saved the model.\n")
        f.write(f"---------------------------------------------------------\n")

    # Keep your old plot, but now save into this run's figs/
    plot_and_save_loss_graph(epoch_losses, get_next_run_number(run_log_path), num_epochs,
                             learning_rate, batch_size, layers, save_dir=figs_dir)


def train_model(model, dataloaders, criterion, optimizer, num_epochs, batch_size, learning_rate, layers):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # === NEW: create this run's folders ===
    run_dir, figs_dir, run_id, arch_name = _start_run(
        model, num_epochs, learning_rate, batch_size, layers
    )
    epoch_losses = []
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloaders['train'])
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    final_loss = epoch_losses[-1]
    print(f"Finished Training. Final Loss: {final_loss:.4f}")

    # === CHANGED: save model inside this run's folder (informative name) ===
    model_filename = f"SimpleNN_e{num_epochs}_lr{learning_rate}_bs{batch_size}_final{final_loss:.3f}.pt"
    model_save_path = os.path.join(run_dir, model_filename)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # === CHANGED: write a per-run log file and plots inside this run ===
    run_log_path = os.path.join(run_dir, "run_log.txt")
    log_run_details(num_epochs, learning_rate, batch_size, layers, final_loss, device, epoch_losses,
                    run_log_path=run_log_path, figs_dir=figs_dir)
    return epoch_losses


def train_model_val_loss(model, dataloaders, criterion, optimizer, num_epochs, batch_size, learning_rate, layers=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # === NEW: create this run's folders ===
    run_dir, figs_dir, run_id, arch_name = _start_run(
        model, num_epochs, learning_rate, batch_size, layers
    )
    epoch_losses = []
    val_losses = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # === Training Phase ===
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloaders['train'])
        epoch_losses.append(epoch_loss)

        # === Validation Phase ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in dataloaders.get('val', []):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= max(1, len(dataloaders.get('val', [])))
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Finished Training. Final Val Loss: {best_val_loss:.4f}")

    # === CHANGED: save best model inside this run's folder (informative name) ===
    model_filename = f"SimpleNN_e{num_epochs}_lr{learning_rate}_bs{batch_size}_val{best_val_loss:.3f}.pt"
    model_save_path = os.path.join(run_dir, model_filename)
    torch.save(best_model_wts, model_save_path)
    print(f"Model saved to {model_save_path}")

    # Also save a simple combined loss plot using your existing function
    # (we pass figs_dir so it lands in this run's folder)
    plot_loss_graphs(epoch_losses, val_losses, run_number=1, num_epochs=num_epochs,
                     learning_rate=learning_rate, batch_size=batch_size, layers=layers, filename=figs_dir)

    # And log a small per-run text file
    run_log_path = os.path.join(run_dir, "run_log.txt")
    log_run_details(num_epochs, learning_rate, batch_size, layers, best_val_loss, device, epoch_losses,
                    run_log_path=run_log_path, figs_dir=figs_dir)

    return epoch_losses, val_losses


def train_autoencoder(model, dataloaders, criterion, optimizer, num_epochs, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # === NEW: create this run's folders ===
    layers = ["AE", "AE", "AE"]  # placeholder to keep the same function signatures
    run_dir, figs_dir, run_id, arch_name = _start_run(
        model, num_epochs, learning_rate, batch_size, layers
    )
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, _ in dataloaders['train']:
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss / len(dataloaders['train'])
        train_losses.append(epoch_train_loss)

        # --- Validation loss ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, _ in dataloaders['val']:
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()
        epoch_val_loss = val_loss / len(dataloaders['val'])
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    # --- Save the trained model into this run's folder ---
    model_path = os.path.join(run_dir, "autoencoder_final.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save a quick plot of AE train loss using your old function
    plot_and_save_loss_graph(train_losses, run_number=1, num_epochs=num_epochs,
                             learning_rate=learning_rate, batch_size=batch_size, layers=layers, save_dir=figs_dir)

    return train_losses, val_losses


def _start_run(model, num_epochs, learning_rate, batch_size, layers, base_models_dir="models"):
    """
    Create a per-run folder grouped by architecture name and return paths.
    Example structure:
      models/FlexibleCNN/<run_id>/
        figs/
        config.json
        (your checkpoints and logs)
    """
    arch_name = type(model).__name__                # infer name: SimpleModel, FlexibleCNN, etc.
    cfg = RunConfig(
        arch=arch_name,
        epochs=num_epochs,
        lr=learning_rate,
        batch_size=batch_size,
        notes=f"layers={layers}"
    )

    # Simple run id; no dataset fingerprint needed -> use a constant
    run_id = make_run_id(cfg, ds_fingerprint="manual")

    # Group runs under the arch
    arch_base = os.path.join(base_models_dir, arch_name)
    run_dir = ensure_run_dir(arch_base, run_id)
    figs_dir = os.path.join(run_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # Save a tiny config for breadcrumbs
    write_config(run_dir, cfg, ds_fingerprint="manual", extra={"layers": layers})

    return run_dir, figs_dir, run_id, arch_name