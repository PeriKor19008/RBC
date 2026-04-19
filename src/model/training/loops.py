import torch.nn as nn
import torch
from src.model.training.logging import *
from src.model.plot import *
from src.model.training.run_dirs import *
import copy
from src.model.training.schedulers import build_scheduler, step_scheduler, current_lr

def train_model_val_loss(model, dataloaders, criterion, optimizer,
                         num_epochs, batch_size, learning_rate, layers=None,
                         conv_config=None, fc_config=None,
                         scheduler_name: str | None = None, scheduler_params: dict | None = None,
                         selection="val_loss",ae: nn.Module | None = None):
                        #selection="avg_pct" or "val_loss"
    print(selection)
    steps_per_epoch = len(dataloaders['train'])
    scheduler, sched_mode = build_scheduler(
        optimizer,
        scheduler_name,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        base_lr=learning_rate,
        **(scheduler_params or {})
    )

    if scheduler_name:
        mx = (scheduler_params or {}).get("max_lr", None)
        lr_tag = f"{scheduler_name}, max={mx:.2e}" if mx is not None else str(scheduler_name)
    else:
        lr_tag = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if ae is not None:
        ae.to(device)
        ae.eval()


    run_dir, figs_dir, run_id, arch_name = start_run(
        model, num_epochs, learning_rate, batch_size, layers,
        extra_info={"conv_config": conv_config, "fc_config": fc_config}
    )
    print("using device:", device, "and scheduler:", scheduler_name or "none")
    epoch_losses, val_losses = [], []
    epoch_lr = []
    best_wts = copy.deepcopy(model.state_dict())

    best_val = float("inf")



    y_mean, y_std = _compute_label_stats(dataloaders['train'], device)

    y_mean = y_mean.detach()
    y_std = y_std.detach()

    for epoch in range(num_epochs):
        # train
        model.train()
        running = 0.0
        for x, y in dataloaders['train']:
            x, y = x.to(device), y.to(device)
            if ae is not None:
                with torch.no_grad():
                    x = ae(x)
            optimizer.zero_grad()
            out = model(x)


            y_norm = (y - y_mean) / y_std
            out_norm = (out - y_mean) / y_std
            loss = criterion(out_norm, y_norm)
            loss.backward()
            optimizer.step()


            step_scheduler(scheduler, sched_mode)

            running += loss.item()

        epoch_loss = running / len(dataloaders['train'])
        epoch_losses.append(epoch_loss)


        model.eval()
        v = 0.0
        with torch.no_grad():
            for x, y in dataloaders.get('val', []):
                x, y = x.to(device), y.to(device)
                if ae is not None:
                    x = ae(x)
                out = model(x)
                y_norm = (y - y_mean) / y_std
                out_norm = (out - y_mean) / y_std
                v += criterion(out_norm, y_norm).item()
        v /= max(1, len(dataloaders.get('val', [])))
        val_losses.append(v)


        if sched_mode in {"epoch", "plateau"}:
            step_scheduler(scheduler, sched_mode, val_loss=v)

        current_lr_val = optimizer.param_groups[0]["lr"]
        epoch_lr.append(current_lr_val)

        # selection: lowest val loss
        if v < best_val:
            best_val = v
            best_wts = copy.deepcopy(model.state_dict())
            print(f"[{epoch + 1}/{num_epochs}] train={epoch_loss:.6f}  val={v:.6f}  lr={current_lr_val:.2e}", flush=True)
                        # save best to this run
    model.load_state_dict(best_wts)

    # save best checkpoint
    if selection == "val_loss":
        ckpt = os.path.join(
            run_dir,
            f"{arch_name}_e{num_epochs}_lr{learning_rate}_bs{batch_size}_val{min(val_losses):.6f}.pt"
        )
    else:
        # best_metric is macro avg % error
        ckpt = os.path.join(
            run_dir,
            f"{arch_name}_e{num_epochs}_lr{learning_rate}_bs{batch_size}_pct{best_val:.3f}.pt"
        )

    model_cpu = copy.deepcopy(model).to("cpu")
    torch.save(model_cpu, ckpt)


    # per-run combined plot → run_dir/figs/
    plot_loss_graphs(epoch_losses, val_losses, run_number=1, num_epochs=num_epochs,
                     learning_rate=learning_rate, batch_size=batch_size,
                     layers=layers, out_dir=figs_dir,lr_tag=lr_tag)

    # per-run log → run_dir/run_log.txt
    run_log_path = os.path.join(run_dir, "run_log.txt")
    log_run_details(num_epochs, learning_rate, batch_size, layers,
                    final_loss=best_val, device=device,
                    epoch_losses=epoch_losses, val_losses=val_losses,
                    run_log_path=run_log_path, figs_dir=figs_dir,scheduler_name=scheduler_name,scheduler_params=scheduler_params, epoch_lrs=epoch_lr)


    return epoch_losses, val_losses, run_dir



def train_autoencoder(model, dataloaders, criterion, optimizer,
                      num_epochs, batch_size, learning_rate, layers=None,
                      scheduler_name: str | None = None,
                      scheduler_params: dict | None = None,):

    steps_per_epoch = len(dataloaders['train'])
    scheduler, sched_mode = build_scheduler(
        optimizer,
        scheduler_name,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        base_lr=learning_rate,
        **(scheduler_params or {})
    )
    lr_tag = None
    if scheduler_name:
        mx = (scheduler_params or {}).get("max_lr")
        lr_tag = f"{scheduler_name}, max={mx:.2e}" if mx else scheduler_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    layers = layers if layers is not None else "AE"

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
        for batch in dataloaders['train']:


            x_noisy, x_clean = batch
            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)
            inp = x_noisy
            target = x_clean


            optimizer.zero_grad()
            rec = model(inp)
            loss = criterion(rec, target)
            loss.backward()
            optimizer.step()
            step_scheduler(scheduler, sched_mode)
            running += loss.item()

        epoch_train_loss = running / len(dataloaders['train'])
        train_losses.append(epoch_train_loss)


        model.eval()
        v = 0.0
        with torch.no_grad():
            for batch in dataloaders['val']:

                x_noisy, x_clean = batch
                x_noisy = x_noisy.to(device)
                x_clean = x_clean.to(device)
                inp = x_noisy
                target = x_clean


                v += criterion(model(inp), target).item()

        epoch_val_loss = v / len(dataloaders['val'])
        val_losses.append(epoch_val_loss)
        if sched_mode in {"epoch", "plateau"}:
            step_scheduler(scheduler, sched_mode, val_loss=epoch_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[{epoch+1}/{num_epochs}] "
              f"train={epoch_train_loss:.6f}  val={epoch_val_loss:.6f}  lr={current_lr:.2e}",
              flush=True)

    ckpt = os.path.join(run_dir, "autoencoder_final.pt")
    model_cpu = copy.deepcopy(model).to("cpu")
    torch.save(model_cpu, ckpt)

    plot_loss_graphs(
        train_losses, val_losses, run_number=1, num_epochs=num_epochs,
        learning_rate=learning_rate, batch_size=batch_size,
        layers=layers, out_dir=figs_dir, lr_tag=lr_tag
    )

    run_log_path = os.path.join(run_dir, "run_log.txt")
    log_run_details(
        num_epochs, learning_rate, batch_size, layers,
        final_loss=val_losses[-1], device=device,
        epoch_losses=train_losses, val_losses=val_losses,
        run_log_path=run_log_path, figs_dir=figs_dir,
        scheduler_name=scheduler_name, scheduler_params=scheduler_params
    )

    return train_losses, val_losses, run_dir

def _compute_label_stats(train_loader, device):
    s = torch.zeros(4, device=device)
    ss = torch.zeros(4, device=device)
    n = 0
    with torch.no_grad():
        for _, y in train_loader:
            y = y.to(device)
            s  += y.sum(dim=0)
            ss += (y ** 2).sum(dim=0)
            n  += y.size(0)
    mean = s / n
    var  = (ss / n) - mean**2
    std  = var.clamp_min(1e-8).sqrt()
    return mean, std






