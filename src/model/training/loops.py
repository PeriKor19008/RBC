import os
import torch
from src.model.training.logging import *
from src.model.plot import *
from src.model.training.run_dirs import *
import copy

def train_model_val_loss(model, dataloaders, criterion, optimizer, num_epochs, batch_size, learning_rate, layers=None, conv_config=None, fc_config=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # === NEW: create this run's folders ===
    run_dir, figs_dir, run_id, arch_name = start_run(
        model, num_epochs, learning_rate, batch_size, layers,
        extra_info={"conv_config": conv_config, "fc_config": fc_config}
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
    log_run_details(num_epochs, learning_rate, batch_size, layers,
                    best_val_loss, device, epoch_losses,
                    val_losses=val_losses,
                    run_log_path=run_log_path, figs_dir=figs_dir)

    return epoch_losses, val_losses, run_dir


def train_autoencoder(model, dataloaders, criterion, optimizer, num_epochs, batch_size, learning_rate, layers=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # === NEW: create this run's folders ===
    if layers is None:
        layers = ["AE", "AE", "AE"]  # placeholder to keep the same function signatures

    run_dir, figs_dir, run_id, arch_name = start_run(
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
    run_log_path = os.path.join(run_dir, "run_log.txt")
    log_run_details(num_epochs, learning_rate, batch_size, layers,
                    epoch_val_loss, device, train_losses,
                    val_losses=val_losses,
                    run_log_path=run_log_path, figs_dir=figs_dir)

    return train_losses, val_losses, run_dir
