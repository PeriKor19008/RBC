"""
Helpers for organizing training runs of the RBC project.

Each run gets its own folder under `models/`, with:
- config.json  (hyperparameters + dataset info)
- metrics.csv  (loss values per epoch)
- summary.json (best epoch + best metric)
- model checkpoint(s)
- figs/        (all plots for this run)
"""

import os, json, csv, hashlib, time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt


# -----------------------
# Run configuration
# -----------------------

@dataclass
class RunConfig:
    arch: str = "SimpleModel"
    input_size: str = "1x50x50"
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 64
    weight_decay: float = 0.0
    seed: int = 42
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _hash(text: str, n: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def dataset_fingerprint(filepaths: List[str]) -> str:
    """Create a short fingerprint for a dataset (stable ID)."""
    fps = [os.path.normpath(p) for p in filepaths]
    key = f"{len(fps)}|" + "|".join(sorted(fps)[:100])
    return _hash(key, 10)


def make_run_id(cfg: RunConfig, ds_fingerprint: str, when: Optional[float] = None) -> str:
    t = time.gmtime(when or time.time())
    ts = time.strftime("%Y%m%d-%H%M%S", t)
    core = f"{cfg.arch}_e{cfg.epochs}_lr{cfg.lr}_bs{cfg.batch_size}_wd{cfg.weight_decay}_seed{cfg.seed}"
    return f"{ts}_{core}_ds{ds_fingerprint}"


def ensure_run_dir(base_dir: str, run_id: str) -> str:
    """Make sure run directory + figs subdir exist."""
    d = Path(base_dir) / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "figs").mkdir(exist_ok=True)
    return str(d)


# -----------------------
# Logging helpers
# -----------------------

class MetricLogger:
    def __init__(self, run_dir: str, csv_name: str = "metrics.csv"):
        self.run_dir = Path(run_dir)
        self.csv_path = self.run_dir / csv_name
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
                w.writeheader()

    def log(self, epoch: int, train_loss: float, val_loss: float):
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
            w.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})


def write_config(run_dir: str, cfg: RunConfig, ds_fingerprint: str, extra: Optional[Dict[str, Any]] = None):
    path = Path(run_dir) / "config.json"
    payload = cfg.to_dict()
    payload["dataset_fingerprint"] = ds_fingerprint
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_summary(run_dir: str, best_epoch: int, best_metric_name: str, best_metric_value: float):
    path = Path(run_dir) / "summary.json"
    payload = {
        "best_epoch": best_epoch,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# -----------------------
# Simple clean plotting
# -----------------------

def plot_losses(train_losses, val_losses, out_path: str, title: str = "Loss vs Epoch"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
    ax.plot(range(1, len(val_losses) + 1), val_losses, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
