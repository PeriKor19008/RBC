# src/model/train.py
import torch
from torch import nn
import copy
import os

from plot import *

# === NEW: import tiny helpers for per-run folders ===
from src.utils.run_utils import RunConfig, make_run_id, ensure_run_dir, write_config

#

#training/logging.py
# def get_next_run_number(log_file='mock_run_observations.txt'):
#     # Check if the log file exists and read its contents
#     if os.path.exists(log_file):
#         with open(log_file, 'r') as f:
#             lines = f.readlines()
#             run_count = sum(1 for line in lines if line.startswith("run #"))
#             return run_count + 1
#     else:
#         return 1
# #training/logging.py
# def log_run_details(num_epochs, learning_rate, batch_size, layers,
#                     final_loss, device, epoch_losses,  # train losses
#                     val_losses=None,                    # <-- NEW (optional)
#                     run_log_path='mock_run_observations.txt',
#                     figs_dir='graphs_SimpleNN'):
#
#     run_number = get_next_run_number(run_log_path) if os.path.exists(run_log_path) else 1
#
#     with open(run_log_path, 'a') as f:
#         f.write(f"\nrun #{run_number}:\n")
#         if isinstance(layers, (list, tuple)) and len(layers) >= 3:
#             f.write(f"Hidden Layers: {layers[0]} --> {layers[1]} --> {layers[2]}\n")
#         else:
#             f.write(f"Layers: {layers}\n")
#         f.write(f"Number of Epochs: {num_epochs}\n")
#         f.write(f"Learning Rate: {learning_rate}\n")
#         f.write(f"Batch Size: {batch_size}\n")
#         f.write(f"Using device: {device}\n")
#         for i in range(num_epochs):
#             # Train
#             f.write(f"Epoch {i + 1} train: {epoch_losses[i]:.6f}\n")
#             # Val (if provided)
#             if val_losses is not None and i < len(val_losses):
#                 f.write(f"Epoch {i + 1} val: {val_losses[i]:.6f}\n")
#         f.write(f"Finished Training and saved the model.\n")
#         f.write(f"---------------------------------------------------------\n")
#
#     # keep your old plot, saved into this run’s figs/
#     plot_and_save_loss_graph(epoch_losses, run_number, num_epochs, learning_rate,
#                              batch_size, layers, save_dir=figs_dir)
#
# #training/run_dirs.py
# def _start_run(model, num_epochs, learning_rate, batch_size, layers,
#                base_models_dir="models", extra_info=None):
#     """
#     Create a per-run folder grouped by architecture name and return paths.
#     """
#     arch_name = type(model).__name__
#     cfg = RunConfig(
#         arch=arch_name,
#         epochs=num_epochs,
#         lr=learning_rate,
#         batch_size=batch_size,
#         notes=f"layers={layers}"
#     )
#
#     run_id = make_run_id(cfg, ds_fingerprint="manual")
#     arch_base = os.path.join(base_models_dir, arch_name)
#     run_dir = ensure_run_dir(arch_base, run_id)
#     figs_dir = os.path.join(run_dir, "figs")
#     os.makedirs(figs_dir, exist_ok=True)
#
#     # NEW: store layers + any extra info (like conv_config/fc_config) into config.json
#     extra = {"layers": layers}
#     if extra_info:
#         extra.update(extra_info)
#     write_config(run_dir, cfg, ds_fingerprint="manual", extra=extra)
#
#     return run_dir, figs_dir, run_id, arch_name
#
#
#
#
# # === NEW: small helper to create a per-run folder (and a figs/ subfolder) ===
# def _make_run_dirs(num_epochs, learning_rate, batch_size, layers, base_models_dir="models"):
#     # Build a simple config (just for naming reproducibility)
#     cfg = RunConfig(
#         arch="SimpleNN",
#         epochs=num_epochs,
#         lr=learning_rate,
#         batch_size=batch_size,
#         notes=f"layers={layers}"
#     )
#     # Use helper to create a unique run id + directory
#     run_id = make_run_id(cfg, ds_fingerprint="manual")  # keep it simple; no dataset fingerprint needed here
#     run_dir = ensure_run_dir(base_models_dir, run_id)
#     figs_dir = os.path.join(run_dir, "figs")
#     os.makedirs(figs_dir, exist_ok=True)
#
#     # Save a tiny config json (handy later)
#     write_config(run_dir, cfg, ds_fingerprint="manual", extra={"layers": layers})
#
#     return run_dir, figs_dir, run_id
#
#
#
#
#
#
# #training/loops.py
# def train_model_val_loss(model, dataloaders, criterion, optimizer, num_epochs, batch_size, learning_rate, layers=None, conv_config=None, fc_config=None):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     model.to(device)
#
#     # === NEW: create this run's folders ===
#     run_dir, figs_dir, run_id, arch_name = _start_run(
#         model, num_epochs, learning_rate, batch_size, layers,
#         extra_info={"conv_config": conv_config, "fc_config": fc_config}
#     )
#     epoch_losses = []
#     val_losses = []
#
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_val_loss = float('inf')
#
#     for epoch in range(num_epochs):
#         # === Training Phase ===
#         model.train()
#         running_loss = 0.0
#         for inputs, labels in dataloaders['train']:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#
#         epoch_loss = running_loss / len(dataloaders['train'])
#         epoch_losses.append(epoch_loss)
#
#         # === Validation Phase ===
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for inputs, labels in dataloaders.get('val', []):
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#         val_loss /= max(1, len(dataloaders.get('val', [])))
#         val_losses.append(val_loss)
#
#         print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
#
#         # Save best weights
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_model_wts = copy.deepcopy(model.state_dict())
#
#     print(f"Finished Training. Final Val Loss: {best_val_loss:.4f}")
#
#     # === CHANGED: save best model inside this run's folder (informative name) ===
#     model_filename = f"SimpleNN_e{num_epochs}_lr{learning_rate}_bs{batch_size}_val{best_val_loss:.3f}.pt"
#     model_save_path = os.path.join(run_dir, model_filename)
#     torch.save(best_model_wts, model_save_path)
#     print(f"Model saved to {model_save_path}")
#
#     # Also save a simple combined loss plot using your existing function
#     # (we pass figs_dir so it lands in this run's folder)
#     plot_loss_graphs(epoch_losses, val_losses, run_number=1, num_epochs=num_epochs,
#                      learning_rate=learning_rate, batch_size=batch_size, layers=layers, filename=figs_dir)
#
#     # And log a small per-run text file
#     run_log_path = os.path.join(run_dir, "run_log.txt")
#     log_run_details(num_epochs, learning_rate, batch_size, layers,
#                     best_val_loss, device, epoch_losses,
#                     val_losses=val_losses,
#                     run_log_path=run_log_path, figs_dir=figs_dir)
#
#     return epoch_losses, val_losses, run_dir
#
# #training/loops.py
# def train_autoencoder(model, dataloaders, criterion, optimizer, num_epochs, batch_size, learning_rate, layers=None):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     model.to(device)
#
#     # === NEW: create this run's folders ===
#     if layers is None:
#         layers = ["AE", "AE", "AE"]  # placeholder to keep the same function signatures
#
#     run_dir, figs_dir, run_id, arch_name = _start_run(
#         model, num_epochs, learning_rate, batch_size, layers
#     )
#     train_losses = []
#     val_losses = []
#
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for inputs, _ in dataloaders['train']:
#             inputs = inputs.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, inputs)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#
#         epoch_train_loss = running_loss / len(dataloaders['train'])
#         train_losses.append(epoch_train_loss)
#
#         # --- Validation loss ---
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for inputs, _ in dataloaders['val']:
#                 inputs = inputs.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, inputs)
#                 val_loss += loss.item()
#         epoch_val_loss = val_loss / len(dataloaders['val'])
#         val_losses.append(epoch_val_loss)
#
#         print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
#
#     # --- Save the trained model into this run's folder ---
#     model_path = os.path.join(run_dir, "autoencoder_final.pt")
#     torch.save(model.state_dict(), model_path)
#     print(f"Model saved to {model_path}")
#
#     # Save a quick plot of AE train loss using your old function
#     run_log_path = os.path.join(run_dir, "run_log.txt")
#     log_run_details(num_epochs, learning_rate, batch_size, layers,
#                     epoch_val_loss, device, train_losses,
#                     val_losses=val_losses,
#                     run_log_path=run_log_path, figs_dir=figs_dir)
#
#     return train_losses, val_losses, run_dir





# def train_model(model, dataloaders, criterion, optimizer, num_epochs, batch_size, learning_rate, layers):
#     # Device configuration
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     model.to(device)
#
#     # === NEW: create this run's folders ===
#     run_dir, figs_dir, run_id, arch_name = _start_run(
#         model, num_epochs, learning_rate, batch_size, layers
#     )
#     epoch_losses = []
#     best_model_wts = copy.deepcopy(model.state_dict())
#
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for inputs, labels in dataloaders['train']:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#
#         epoch_loss = running_loss / len(dataloaders['train'])
#         epoch_losses.append(epoch_loss)
#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
#
#     final_loss = epoch_losses[-1]
#     print(f"Finished Training. Final Loss: {final_loss:.4f}")
#
#     # === CHANGED: save model inside this run's folder (informative name) ===
#     model_filename = f"SimpleNN_e{num_epochs}_lr{learning_rate}_bs{batch_size}_final{final_loss:.3f}.pt"
#     model_save_path = os.path.join(run_dir, model_filename)
#     torch.save(model.state_dict(), model_save_path)
#     print(f"Model saved to {model_save_path}")
#
#     # === CHANGED: write a per-run log file and plots inside this run ===
#     run_log_path = os.path.join(run_dir, "run_log.txt")
#     log_run_details(num_epochs, learning_rate, batch_size, layers, final_loss, device, epoch_losses,
#                     run_log_path=run_log_path, figs_dir=figs_dir)
#     return epoch_losses, run_dir
