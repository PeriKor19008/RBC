import torch
from torch import optim
from Data.DB_setup.db_config import DB_CONFIG
from torch.utils.data import DataLoader, random_split
from src.model.RBCDataset import RBCDatasetDB
from src.model.model import FlexibleCNN
from src.model.training.loops import train_model_val_loss
from src.model.plot import *
from src.model.training.logging import get_next_run_number
from datetime import datetime
from src.model.training.run_dirs import *
from src.model.noise import *


def train_CNN(batchSize, epochs,lr_rate, conv_config, fc_config=None,noise:bool = False):
    full_dataset = RBCDatasetDB(db_config=DB_CONFIG, use_log_image=False)

    # ---create datasets---
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataset = train_ds
    if noise:
        train_tf = nn.Sequential(
            AddGaussianNoise(std=0.3, p=0.5),
            AddSpeckleNoise(std=0.3, p=0.5),
        )
        train_dataset = WithTransform(train_ds,transform=train_tf)

    # -----create dataloader----
    batch_size = batchSize
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    }

    # defaults
    if fc_config is None:
        fc_config = [128]

    # --- build model, loss, opt ---
    model = FlexibleCNN(conv_config, fc_config)
    criterion = nn.SmoothL1Loss(beta=1.0)
    learning_rate = lr_rate
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith("bias"):
            no_decay.append(p)  # biases & LayerNorm/BatchNorm weights
        else:
            decay.append(p)  # weights to decay

    optimizer = optim.AdamW(
        [{"params": decay, "weight_decay": 1e-4},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate,
    )
    num_epochs = epochs
    # make a string label for plots/keys (do NOT use the list itself)
    label = f"conv{len(conv_config)}_fc{len(fc_config)}"

    # --- train ---
    train_losses, val_losses, run_dir = train_model_val_loss(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=epochs,
        batch_size=batchSize,
        learning_rate=learning_rate,
        layers=label,
        conv_config=conv_config,
        fc_config=fc_config,
        scheduler_name="onecycle",
        scheduler_params={
            "max_lr": learning_rate * 5.0,  # or x10
            "pct_start": 0.3,
            "div_factor": (learning_rate * 5.0) / learning_rate,  # == 5.0
            "final_div_factor": 1e4,
            "cycle_momentum": False
        }
    )



    return train_losses, val_losses, run_dir


def multi_train_CNN():
    run_dirs = []

    # === explicit CNN runs (no loops) ===
    _, _, rd = train_CNN(32, 1,0.001, [("conv", 16), ("conv", 32), ("conv", 64)], [128])
    run_dirs.append(rd)

    _, _, rd = train_CNN(32, 1, 0.0001,[("conv", 16), ("conv", 32), ("conv", 64)], [128])
    run_dirs.append(rd)


    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = out_dir = Path(__file__).resolve().parents[2] / "outputs" / "comparisons" / ts  # or "comparison" if that's your folder
    os.makedirs(out_dir, exist_ok=True)

    compare_runs_from_logs(
        run_dirs,
        os.path.join(out_dir, "cnn_manual_val.png"),
        which="val",
        title="CNN manual runs (val loss)"
    )

    compare_runs_from_logs(
        run_dirs,
        os.path.join(out_dir, "cnn_manual_train.png"),
        which="train",
        title="CNN manual runs (train loss)"
    )

    print(f"Saved comparisons in: {out_dir}")