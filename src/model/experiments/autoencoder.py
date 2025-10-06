import torch
from Data.DB_setup.db_config import DB_CONFIG
from torch.utils.data import DataLoader, random_split
from src.model.RBCDataset import RBCDatasetDB
from src.model.model import FCAutoencoder
from src.model.training.loops import train_autoencoder
from datetime import datetime
from src.model.training.run_dirs import *




def run_autoencoder(batchSize, epochs,lr_rate, h_layers=None, latent_dim=None):
    full_dataset = RBCDatasetDB(db_config=DB_CONFIG, use_log_image=False)
    if h_layers is None:
        h_layers = [1024, 512, 128]
    if latent_dim is None:
        latent_dim = 64
    model = FCAutoencoder(latent_dim=latent_dim, hidden_dims=h_layers)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    batch_size = batchSize

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    }
    label = f"[{','.join(str(x) for x in h_layers)}]-{latent_dim}"
    train_losses, val_losses, run_dir  = train_autoencoder(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr_rate,
        layers=label,
    )

    return train_losses, val_losses, run_dir

def multi_train_autoencoder():
    run_dirs = []

    # === explicit AE runs (no loops) ===
    # (layers arg is unused by AE — pass anything, e.g., 0)
    _, _, rd = run_autoencoder(32, 40, 0.0001,[1024, 512, 128], 64)
    run_dirs.append(rd)
    _, _, rd = run_autoencoder(32, 40, 0.0001,[2000,1024, 512, 128], 64)
    run_dirs.append(rd)
    _, _, rd = run_autoencoder(32, 40, 0.0001,[1024, 512, 128], 100)
    run_dirs.append(rd)
    _, _, rd = run_autoencoder(32, 40, 0.0001,[2000,1024, 512, 128], 32)
    run_dirs.append(rd)

    # === timestamped comparison folder ===
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(__file__).resolve().parents[2] / "outputs" / "comparisons" / ts
    os.makedirs(out_dir, exist_ok=True)

    compare_runs_from_logs(
        run_dirs,
        os.path.join(out_dir, "ae_manual_val.png"),
        which="val",
        title="Autoencoder manual runs (val loss)"
    )
    compare_runs_from_logs(
        run_dirs,
        os.path.join(out_dir, "ae_manual_train.png"),
        which="train",
        title="Autoencoder manual runs (train loss)"
    )

    # Optional: manifest for traceability
    with open(os.path.join(out_dir, "manifest.txt"), "w") as f:
        f.write("AE runs compared:\n")
        for rd in run_dirs:
            f.write(f"- {rd}\n")

    print(f"Saved comparisons in: {out_dir}")