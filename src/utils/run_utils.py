
import os,re, json, csv, hashlib, time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt



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



def write_config(run_dir: str, cfg: RunConfig, ds_fingerprint: str, extra: Optional[Dict[str, Any]] = None):
    path = Path(run_dir) / "config.json"
    payload = cfg.to_dict()
    payload["dataset_fingerprint"] = ds_fingerprint
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

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
