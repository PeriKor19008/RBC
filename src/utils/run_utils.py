"""
Helpers for organizing training runs of the RBC project.

Each run gets its own folder under `models/`, with:
- config.json  (hyperparameters + dataset info)
- metrics.csv  (loss values per epoch)
- summary.json (best epoch + best metric)
- model checkpoint(s)
- figs/        (all plots for this run)
"""
import math
import os,re, json, csv, hashlib, time
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


# --- Multi-run comparison helpers ---

def _read_metrics_csv(csv_path: str):
    """Return (train_losses, val_losses) lists from a metrics.csv."""
    train, val = [], []
    try:
        import csv
        with open(csv_path, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                # be robust to missing val_loss (e.g., pure-train runs)
                train.append(float(row.get("train_loss", "nan")))
                v = row.get("val_loss", "")
                val.append(float(v) if v not in ("", None) else float("nan"))
    except Exception:
        pass
    return train, val

def _label_from_config(cfg_path: str):
    """Build a short label from config.json, e.g. FlexibleCNN lr0.001 bs32 e50."""
    try:
        import json, os
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        arch = cfg.get("arch", "model")
        lr = cfg.get("lr", "?")
        bs = cfg.get("batch_size", "?")
        e  = cfg.get("epochs", "?")
        extra = cfg.get("notes", "")
        # squeeze long notes
        if isinstance(extra, str) and len(extra) > 32:
            extra = extra[:29] + "..."
        label = f"{arch} lr{lr} bs{bs} e{e}"
        if extra:
            label += f" [{extra}]"
        return label
    except Exception:
        return "run"

def compare_runs(run_dirs, out_path: str, which: str = "val", title: str = "Run comparison"):
    """
    Overlay multiple runs on one plot.
      run_dirs: list of per-run folders (each has metrics.csv + config.json)
      which: "val" or "train"
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    any_plotted = False

    for rd in run_dirs:
        csv_path = os.path.join(rd, "metrics.csv")
        cfg_path = os.path.join(rd, "config.json")
        train_losses, val_losses = _read_metrics_csv(csv_path)
        label = _label_from_config(cfg_path)

        if which == "val" and any(not math.isnan(v) for v in val_losses):
            y = [v for v in val_losses if not math.isnan(v)]
            x = list(range(1, len(y) + 1))
            ax.plot(x, y, label=label)
            any_plotted = True
        elif which == "train" and len(train_losses) > 0:
            y = train_losses
            x = list(range(1, len(y) + 1))
            ax.plot(x, y, label=label)
            any_plotted = True

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{which.capitalize()} loss")
    ax.set_title(title)
    if any_plotted:
        ax.legend(fontsize=8)
    fig.tight_layout()

    # ensure parent exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)



def _label_from_config_or_dir(run_dir: str) -> str:
    cfg = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg):
        try:
            with open(cfg, "r") as f:
                c = json.load(f)
            arch = c.get("arch", "model")
            lr = c.get("lr", "?")
            bs = c.get("batch_size", "?")
            e  = c.get("epochs", "?")
            notes = c.get("notes", "")
            label = f"{arch} lr{lr} bs{bs} e{e}"
            if notes:
                label += f" [{notes}]"
            return label
        except Exception:
            pass
    return os.path.basename(run_dir)

def _parse_losses_from_log(log_path: str):
    """Returns (train_losses, val_losses) parsed from run_log.txt."""
    train, val = [], []
    if not os.path.exists(log_path):
        return train, val
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            # New format:
            m_train = re.match(r"^Epoch\s+(\d+)\s+train:\s*([-+eE0-9.]+)$", line)
            if m_train:
                train.append(float(m_train.group(2)))
                continue
            m_val = re.match(r"^Epoch\s+(\d+)\s+val:\s*([-+eE0-9.]+)$", line)
            if m_val:
                val.append(float(m_val.group(2)))
                continue
            # Backward compat with old lines: "Epoch X complete. Loss: Y"
            m_old = re.match(r"^Epoch\s+(\d+)\s+complete\.\s*Loss:\s*([-+eE0-9.]+)$", line)
            if m_old:
                train.append(float(m_old.group(2)))
    return train, val

def compare_runs_from_logs(run_dirs, out_path: str, which: str = "val", title: str = "Run comparison"):
    fig, ax = plt.subplots(figsize=(9, 6))
    any_plotted = False

    for rd in run_dirs:
        log_path = os.path.join(rd, "run_log.txt")
        tr, vl = _parse_losses_from_log(log_path)
        if which == "val":
            y = vl
        else:
            y = tr
        if not y:
            continue
        label = os.path.basename(os.path.normpath(rd))
        ax.plot(range(1, len(y)+1), y, label=label)
        any_plotted = True

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{which.capitalize()} loss")
    ax.set_title(title)
    if any_plotted:
        ax.legend(fontsize=8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
