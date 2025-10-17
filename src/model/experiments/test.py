# src/model/experiments/test_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple, List
import numpy as np
import torch
from torch import nn
from src.utils.paths import rel_to_root
from src.model.model import *
from src.model.ae_heads import *
import json
from src.utils.fileName_to_params import file_name_to_params
import re
import os
import matplotlib.pyplot as plt
from src.model.noise import *
from src.model.RBCDataset import RBCDatasetDB
from Data.DB_setup.db_config import DB_CONFIG
from torch.utils.data import DataLoader, random_split


LABEL_KEYS = ["diameter", "thickness", "ratio", "ref_index"]

def _pick_device(device: str = "auto") -> torch.device:
    if device == "cuda":
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cnn_model(ckpt_path: str | Path, device: str = "auto") -> nn.Module:
    """
    Load a FlexibleCNN checkpoint saved either as a full module or as a state_dict.
    Requires a sibling config.json (written by your training runs) to rebuild the arch
    when the checkpoint is a state_dict.
    """
    ckpt_path = Path(ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    dev = _pick_device(device)
    obj = torch.load(ckpt_path, map_location="cpu")

    # If the whole module was saved, just return it.
    if isinstance(obj, nn.Module):
        return obj.to(dev).eval()

    # Otherwise reconstruct model from config.json and load state dict.
    cfg_path = ckpt_path.parent / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json next to {ckpt_path}")

    cfg = json.loads(cfg_path.read_text())
    # handle list/tuple forms like [["conv",16], ...]
    conv_cfg_raw = cfg.get("conv_config", [("conv", 16), ("conv", 32), ("conv", 64)])
    fc_cfg_raw   = cfg.get("fc_config", [128])

    conv_cfg: List[Tuple[str, int]] = [tuple(x) for x in conv_cfg_raw]
    fc_cfg:   List[int]             = [int(v) for v in fc_cfg_raw]

    model = FlexibleCNN(conv_cfg, fc_cfg)
    state_dict = obj.get("model_state", obj) if isinstance(obj, dict) else obj
    model.load_state_dict(state_dict, strict=True)
    return model.to(dev).eval()

# ---------- 2) AE-Regressor loader ----------

def _infer_ref_index_from_path(p: Path) -> float:
    """
    Find a directory segment like 'Refindx1.055' (case-insensitive) and return
    1000*(value-1), e.g. 1.055 -> 55, 1.100 -> 100. Supports '1.055' or '1_055'.
    """
    pattern = re.compile(r"(?i)refindx\s*([0-9]+(?:[._][0-9]+)?)")

    # check the path parts (fast path)
    for part in p.parts:
        m = pattern.search(part)
        if m:
            s = m.group(1).replace("_", ".")
            try:
                v = float(s)
                return float(int(round((v - 1.0) * 1000)))
            except ValueError:
                pass

    # fallback: scan parents by name
    for parent in p.parents:
        m = pattern.search(parent.name)
        if m:
            s = m.group(1).replace("_", ".")
            try:
                v = float(s)
                return float(int(round((v - 1.0) * 1000)))
            except ValueError:
                pass

    raise ValueError(f"Could not infer ref_index from path: {p}")


def load_rbc_txt_image_and_labels(
    image_path: str | Path,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a 50x50 RBC text image (Fortran 'D' exponents) and its 4 labels from the FILENAME
    using src/utils/fileName_to_params.py.

    Returns (no normalization applied):
        image_tensor : torch.FloatTensor of shape [1, 50, 50]
        labels       : torch.FloatTensor of shape [4] in order
                       [diameter, thickness, ratio, ref_index]
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    # --- 1) parse labels from filename via fileName_to_params.py ---
    import importlib
    mod = importlib.import_module("src.utils.fileName_to_params")

    ref_index = _infer_ref_index_from_path(p)
    d,t,n_decimal= file_name_to_params(p.name)
    lbl=torch.tensor([d,t,n_decimal,ref_index],dtype=torch.float32)
    if isinstance(lbl, dict):
        try:
            labels = torch.tensor(
                [
                    float(lbl["diameter"]),
                    float(lbl["thickness"]),
                    float(lbl["ratio"]),
                    float(lbl["ref_index"]),
                ],
                dtype=torch.float32,
            )
        except KeyError as ke:
            raise KeyError(
                "fileName_to_params parser must return keys: "
                "diameter, thickness, ratio, ref_index"
            ) from ke
    else:
        vals = list(lbl)
        if len(vals) != 4:
            raise ValueError(
                "fileName_to_params parser must return 4 values: "
                "(diameter, thickness, ratio, ref_index)"
            )
        labels = torch.tensor([float(v) for v in vals], dtype=torch.float32)

    # --- 2) load image (Fortran 'D' exponents) — NO normalization ---
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    if len(lines) != 2500:
        raise ValueError(f"Expected 2500 lines, got {len(lines)} in: {p}")
    vals = [float(s.replace("D", "E")) for s in lines]
    arr = np.asarray(vals, dtype=np.float32).reshape(50, 50)
    image_tensor = torch.from_numpy(arr).unsqueeze(0)  # [1,50,50]

    return image_tensor, labels

def predict_and_compare(
    model: nn.Module,
    image_path: str,noise:bool = False,
    *,
    denorm: Optional[Dict[str, List[float]]] = None,  # keep if you normalized targets; else ignore
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Load a single RBC text image (+labels) from filename, run the model, and compare.
    No image normalization is applied.
    Returns a dict with: pred, target, abs_error, diff, MAE, MSE.
    """
    # 1) load image + ground-truth labels (raw)
    x, y_true = load_rbc_txt_image_and_labels(image_path)  # x: [1,50,50], y_true: [4]
    vmin, vmax = float(x.min()), float(x.max())
    if noise:
        train_tf = nn.Sequential(
            AddGaussianNoise(std=0.02, p=1.0),
            AddSpeckleNoise(std=0.02, p=1.0),
        )
        x = train_tf(x.unsqueeze(0)).squeeze(0)  # add batch dim for transform
    plt.imshow(
        x.squeeze(0).cpu().numpy(),
        cmap='gray',
        vmin=vmin, vmax=vmax,  # <- key: disable autoscale
        interpolation='nearest'  # optional: avoid smoothing
    )
    plt.colorbar()
    plt.show()
    # 2) run model
    if x.ndim == 3:  # [1,50,50] -> [1,1,50,50]
        x = x.unsqueeze(0)
    dev = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        y_pred = model(x.to(dev)).squeeze(0).detach().cpu()  # [4]

    # 3) optional de-normalization of outputs (only if you normalized targets during training)
    if denorm and "mean" in denorm and "std" in denorm:
        mean = torch.tensor(denorm["mean"], dtype=y_pred.dtype)
        std  = torch.tensor(denorm["std"], dtype=y_pred.dtype)
        y_pred = y_pred * std + mean

    # 4) errors & metrics
    diff = y_pred - y_true
    abs_err = diff.abs()
    mae = float(abs_err.mean().item())
    mse = float((diff ** 2).mean().item())

    def as_dict(t: torch.Tensor):
        v = t.tolist()
        return {
            "diameter":  float(v[0]),
            "thickness": float(v[1]),
            "ratio":     float(v[2]),
            "ref_index": float(v[3]),
        }

    return {
        "pred": as_dict(y_pred),
        "target": as_dict(y_true),
        "abs_error": as_dict(abs_err),
        "diff": as_dict(diff),
        "MAE": mae,
        "MSE": mse,
    }
def plot_bar_pred_vs_true_single(
    result: dict,
    title: str = "Predicted vs True",
    *,
    show: bool = True,
    save_path: str | None = None,
    close: bool | None = None,   # default: close if not showing
):


    true_vals = [float(result["target"][k]) for k in LABEL_KEYS]
    pred_vals = [float(result["pred"][k])   for k in LABEL_KEYS]

    x = np.arange(len(LABEL_KEYS))
    width = 0.4

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    b1 = ax.bar(x - width/2, true_vals,  width, label="True")
    b2 = ax.bar(x + width/2, pred_vals, width, label="Predicted")

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_KEYS)
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    def _annotate(bars):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3g}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
    _annotate(b1); _annotate(b2)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    # default behavior: close only if we are NOT showing
    if close is None:
        close = not show
    if close:
        plt.close(fig)

    return fig, ax

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = "../../../outputs/models/FCAutoencoder/sched_20251012-153447_FCAutoencoder_e50_lr0.0001_bs32_wd0.0_seed42_dsmanual/20251017-071313_AERegressor_e50_lr0.001_bs32_wd0.0_seed42_dsmanual/ae_regressor_full.pt"
    # 1) load the full model (no builder/arch needed)
    #model= load_cnn_model("../../../outputs/models/FlexibleCNN/noise_20251013-214759_FlexibleCNN_e50_lr0.001_bs32_wd0.0_seed42_dsmanual/FlexibleCNN_e50_lr0.001_bs32_val374.415.pt")
    model = torch.load(path, map_location="cpu",weights_only=False).to(device).eval()  # <— full module

    # 2) load one RBC text image + labels (no normalization)
    res = predict_and_compare(model, "../../../Data/results/Refindx1.055/0450015005001a.f06",False)
    plot_bar_pred_vs_true_single(res, title="Predicted vs True for 0450015005001a.f06",show=True)

