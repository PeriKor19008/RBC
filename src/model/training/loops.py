import os
import torch
from src.model.training.logging import *
from src.model.plot import *
from src.model.training.run_dirs import *
import copy

def train_model_val_loss(model, dataloaders, criterion, optimizer,
                         num_epochs, batch_size, learning_rate, layers=None,
                         conv_config=None, fc_config=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    run_dir, figs_dir, run_id, arch_name = start_run(
        model, num_epochs, learning_rate, batch_size, layers,
        extra_info={"conv_config": conv_config, "fc_config": fc_config}
    )
    print("using device:", device)
    epoch_losses, val_losses = [], []
    best_wts = copy.deepcopy(model.state_dict())
    best_val = float("inf")

    for epoch in range(num_epochs):
        # train
        model.train()
        running = 0.0
        for x, y in dataloaders['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        epoch_loss = running / len(dataloaders['train'])
        epoch_losses.append(epoch_loss)

        # val
        model.eval()
        v = 0.0
        with torch.no_grad():
            for x, y in dataloaders.get('val', []):
                x, y = x.to(device), y.to(device)
                v += criterion(model(x), y).item()
        v /= max(1, len(dataloaders.get('val', [])))
        val_losses.append(v)

        if v < best_val:
            best_val = v
            best_wts = copy.deepcopy(model.state_dict())
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[{epoch + 1}/{num_epochs}] "
              f"train={epoch_loss:.6f}  val={v:.6f}  lr={current_lr:.2e}", flush=True)

    # save best to this run
    ckpt = os.path.join(run_dir, f"{arch_name}_e{num_epochs}_lr{learning_rate}_bs{batch_size}_val{best_val:.3f}.pt")
    torch.save(best_wts, ckpt)

    # per-run combined plot → run_dir/figs/
    plot_loss_graphs(epoch_losses, val_losses, run_number=1, num_epochs=num_epochs,
                     learning_rate=learning_rate, batch_size=batch_size,
                     layers=layers, out_dir=figs_dir)

    # per-run log → run_dir/run_log.txt
    run_log_path = os.path.join(run_dir, "run_log.txt")
    log_run_details(num_epochs, learning_rate, batch_size, layers,
                    final_loss=best_val, device=device,
                    epoch_losses=epoch_losses, val_losses=val_losses,
                    run_log_path=run_log_path, figs_dir=figs_dir)


    return epoch_losses, val_losses, run_dir



def train_autoencoder(model, dataloaders, criterion, optimizer,
                      num_epochs, batch_size, learning_rate, layers=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    layers = layers if layers is not None else "AE"

    # (optional banner)
    if device.type == "cuda":
        try:
            name = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            name = "CUDA device"
        print(f"Using device: {device} ({name})")
    else:
        print("Using device: cpu")

    run_dir, figs_dir, run_id, arch_name = start_run(
        model, num_epochs, learning_rate, batch_size, layers
    )

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        running = 0.0
        for x, _ in dataloaders['train']:
            x = x.to(device)
            optimizer.zero_grad()
            rec = model(x)
            loss = criterion(rec, x)
            loss.backward()
            optimizer.step()
            running += loss.item()

        epoch_train_loss = running / len(dataloaders['train'])
        train_losses.append(epoch_train_loss)

        # ---- val ----
        model.eval()
        v = 0.0
        with torch.no_grad():
            for x, _ in dataloaders['val']:
                x = x.to(device)
                v += criterion(model(x), x).item()

        epoch_val_loss = v / len(dataloaders['val'])
        val_losses.append(epoch_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[{epoch+1}/{num_epochs}] "
              f"train={epoch_train_loss:.6f}  val={epoch_val_loss:.6f}  lr={current_lr:.2e}",
              flush=True)

    # save into this run
    ckpt = os.path.join(run_dir, "autoencoder_final.pt")
    torch.save(model.state_dict(), ckpt)

    # per-run combined plot → run_dir/figs/
    plot_loss_graphs(train_losses, val_losses, run_number=1, num_epochs=num_epochs,
                     learning_rate=learning_rate, batch_size=batch_size,
                     layers=layers, out_dir=figs_dir)

    # per-run log → run_dir/run_log.txt
    run_log_path = os.path.join(run_dir, "run_log.txt")
    log_run_details(num_epochs, learning_rate, batch_size, layers,
                    final_loss=val_losses[-1], device=device,
                    epoch_losses=train_losses, val_losses=val_losses,
                    run_log_path=run_log_path, figs_dir=figs_dir)

    return train_losses, val_losses, run_dir


