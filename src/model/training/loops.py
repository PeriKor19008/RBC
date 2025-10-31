import os
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
                         selection="val_loss"):
                        #selection="avg_pct" or "val_loss"

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
        mx = (scheduler_params or {}).get("max_lr", None)
        lr_tag = f"{scheduler_name}, max={mx:.2e}" if mx is not None else str(scheduler_name)
    else:
        lr_tag = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    run_dir, figs_dir, run_id, arch_name = start_run(
        model, num_epochs, learning_rate, batch_size, layers,
        extra_info={"conv_config": conv_config, "fc_config": fc_config}
    )
    print("using device:", device, "and scheduler:", scheduler_name or "none")
    epoch_losses, val_losses = [], []
    epoch_lr = []
    best_wts = copy.deepcopy(model.state_dict())

    best_val = float("inf")
    best_pct_per_label = None


    y_mean, y_std = _compute_label_stats(dataloaders['train'], device)

    y_mean = y_mean.detach()
    y_std = y_std.detach()

    for epoch in range(num_epochs):
        if selection == "val_loss":
            best_val, best_wts, tr, v, lr = _epoch_once_select_by_val_loss(
                model=model,
                dataloaders=dataloaders,
                device=device,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                sched_mode=sched_mode,
                y_mean=y_mean,
                y_std=y_std,
                best_val=best_val,
                best_wts=best_wts,
                epoch_losses=epoch_losses,
                val_losses=val_losses,
                epoch_lr=epoch_lr,
            )
            print(f"[{epoch + 1}/{num_epochs}] train={tr:.6f}  val={v:.6f}  lr={lr:.2e}", flush=True)

        else:  # selection == "avg_pct"
            (best_metric,
             mae_pct_per_label,
             best_wts,
             tr, v, macro_pct, lr) = _epoch_once_select_by_avg_pct(
                model=model,
                dataloaders=dataloaders,
                device=device,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                sched_mode=sched_mode,
                y_mean=y_mean,
                y_std=y_std,
                best_metric=best_val,
                best_wts=best_wts,
                epoch_losses=epoch_losses,
                val_losses=val_losses,
                epoch_lr=epoch_lr,
                pct_eps=1e-8,
            )
            # pretty print per-epoch with macro % metric
            print(
                f"[{epoch + 1}/{num_epochs}] train={tr:.6f}  val={v:.6f}  pct={macro_pct:.3f}%  lr={lr:.2e}",
                flush=True
            )

            # keep a reference to the best per-label % at the time it occurred
            if abs(macro_pct - best_metric) < 1e-12:
                best_pct_per_label = mae_pct_per_label

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
                      scheduler_name: str | None = None, scheduler_params: dict | None = None):

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
            step_scheduler(scheduler, sched_mode)
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
        if sched_mode in {"epoch", "plateau"}:
            step_scheduler(scheduler, sched_mode, val_loss=epoch_val_loss)

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
                     layers=layers, out_dir=figs_dir,lr_tag=lr_tag)

    # per-run log → run_dir/run_log.txt
    run_log_path = os.path.join(run_dir, "run_log.txt")
    log_run_details(num_epochs, learning_rate, batch_size, layers,
                    final_loss=val_losses[-1], device=device,
                    epoch_losses=train_losses, val_losses=val_losses,
                    run_log_path=run_log_path, figs_dir=figs_dir,scheduler_name=scheduler_name,scheduler_params=scheduler_params)

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



def _epoch_once_select_by_val_loss(
    *,
    model,
    dataloaders,
    device,
    optimizer,
    criterion,
    scheduler,
    sched_mode,
    y_mean,
    y_std,
    best_val,
    best_wts,
    epoch_losses,
    val_losses,
    epoch_lr
):
    # train
    model.train()
    running = 0.0
    for x, y in dataloaders['train']:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)

        # normalized loss (keep optimization invariant to label scales)
        y_norm = (y - y_mean) / y_std
        out_norm = (out - y_mean) / y_std
        loss = criterion(out_norm, y_norm)
        loss.backward()
        optimizer.step()

        # per-step schedulers (OneCycle, Cosine w/ per-step, etc.)
        step_scheduler(scheduler, sched_mode)

        running += loss.item()

    epoch_loss = running / len(dataloaders['train'])
    epoch_losses.append(epoch_loss)

    # val (normalized loss)
    model.eval()
    v = 0.0
    with torch.no_grad():
        for x, y in dataloaders.get('val', []):
            x, y = x.to(device), y.to(device)
            out = model(x)
            y_norm = (y - y_mean) / y_std
            out_norm = (out - y_mean) / y_std
            v += criterion(out_norm, y_norm).item()
    v /= max(1, len(dataloaders.get('val', [])))
    val_losses.append(v)

    # per-epoch schedulers (Cosine, Step, Plateau)
    if sched_mode in {"epoch", "plateau"}:
        step_scheduler(scheduler, sched_mode, val_loss=v)

    current_lr_val = optimizer.param_groups[0]["lr"]
    epoch_lr.append(current_lr_val)

    # selection: lowest val loss
    if v < best_val:
        best_val = v
        best_wts = copy.deepcopy(model.state_dict())

    return best_val, best_wts, epoch_loss, v, current_lr_val


def _epoch_once_select_by_avg_pct(
    *,
    model,
    dataloaders,
    device,
    optimizer,
    criterion,
    scheduler,
    sched_mode,
    y_mean,
    y_std,
    best_metric,          # best macro avg % so far (lower is better)
    best_wts,
    epoch_losses,
    val_losses,
    epoch_lr,
    pct_eps=1e-8
):

    # train
    model.train()
    running = 0.0
    for x, y in dataloaders['train']:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)

        # normalized loss (optimization)
        y_norm = (y - y_mean) / y_std
        out_norm = (out - y_mean) / y_std
        loss = criterion(out_norm, y_norm)
        loss.backward()
        optimizer.step()

        step_scheduler(scheduler, sched_mode)
        running += loss.item()

    epoch_loss = running / len(dataloaders['train'])
    epoch_losses.append(epoch_loss)

    # val (compute both: normalized loss for logging, and % error for selection)
    model.eval()
    v = 0.0
    n = 0
    sum_pct_per_label = torch.zeros((y_mean.numel(),), device=device)  # 4-D
    with torch.no_grad():
        for x, y in dataloaders.get('val', []):
            x, y = x.to(device), y.to(device)
            out = model(x)

            # normalized loss for logging
            y_norm = (y - y_mean) / y_std
            out_norm = (out - y_mean) / y_std
            v += criterion(out_norm, y_norm).item()

            # percentage error in original units
            # shape: [B, 4] -> abs error / (|target| + eps) * 100
            pct = (out - y).abs() / (y.abs() + pct_eps) * 100.0
            sum_pct_per_label += pct.sum(dim=0)
            n += y.size(0)

    v /= max(1, len(dataloaders.get('val', [])))
    val_losses.append(v)

    if sched_mode in {"epoch", "plateau"}:
        step_scheduler(scheduler, sched_mode, val_loss=v)

    current_lr_val = optimizer.param_groups[0]["lr"]
    epoch_lr.append(current_lr_val)

    # average % error per label and macro average
    mae_pct_per_label = sum_pct_per_label / max(1, n)           # [4]
    macro_pct = mae_pct_per_label.mean().item()                 # scalar

    # selection: lowest macro % error
    if macro_pct < best_metric:
        best_metric = macro_pct
        best_wts = copy.deepcopy(model.state_dict())

    return best_metric, mae_pct_per_label.detach().cpu(), best_wts, epoch_loss, v, macro_pct, current_lr_val
