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


def train_CNN(batchSize, epochs,lr_rate, conv_config, fc_config=None):
    full_dataset = RBCDatasetDB(db_config=DB_CONFIG, use_log_image=False)

    # ---create datasets---
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

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
    criterion = torch.nn.MSELoss()
    learning_rate = lr_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
        conv_config=conv_config,  # <--- NEW
        fc_config=fc_config  # <--- NEW
    )

    run_number = get_next_run_number()
    plot_loss_graphs(train_losses, val_losses, run_number, num_epochs,
                     learning_rate, batch_size, label)

    # if you still want the combined “all val” plot with a dict:
    all_val_losses = {label: val_losses}
    plot_all_val_losses(all_val_losses)

    return train_losses, val_losses, run_dir


def multi_train_CNN():
    run_dirs = []

    # === explicit CNN runs (no loops) ===
    _, _, rd = train_CNN(32, 40,0.001, [("conv", 16), ("conv", 32), ("conv", 64)], [128])
    run_dirs.append(rd)

    _, _, rd = train_CNN(32, 40, 0.0001,[("conv", 16), ("conv", 32), ("conv", 64)], [128])
    run_dirs.append(rd)

    _, _, rd = train_CNN(32, 40,0.001, [("conv", 16), ("conv", 32), ("conv", 64)], [128, 250])
    run_dirs.append(rd)

    _, _, rd = train_CNN(32, 40, 0.0001, [("conv", 16), ("conv", 32), ("conv", 64)], [128, 250])
    run_dirs.append(rd)

    _, _, rd = train_CNN(32, 40, 0.001, [("conv", 16), ("conv", 32), ("conv", 64), ("conv", 128)], [128])
    run_dirs.append(rd)

    _, _, rd = train_CNN(32, 40, 0.0001, [("conv", 16), ("conv", 32), ("conv", 64), ("conv", 128)], [128])
    run_dirs.append(rd)

    _, _, rd = train_CNN(32, 40, 0.001, [("conv", 16), ("conv", 32), ("conv", 64), ("conv", 128)], [128, 250])
    run_dirs.append(rd)

    _, _, rd = train_CNN(32, 40, 0.0001, [("conv", 16), ("conv", 32), ("conv", 64), ("conv", 128)], [128, 250])
    run_dirs.append(rd)

    # (add more runs exactly as you like)
    # _, _, rd = train_CNN(64, 10, [("conv",16),("conv",32),("conv",64),("conv",64)], [128,256]); run_dirs.append(rd)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("models", "comparisons", ts)  # or "comparison" if that's your folder
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