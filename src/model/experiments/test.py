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
    conv_cfg_raw = cfg.get("conv_config", [("conv", 4),("conv", 8),("conv", 16), ("conv", 32), ("conv", 64),("conv", 120)])
    fc_cfg_raw   = cfg.get("fc_config", [128])

    conv_cfg: List[Tuple[str, int]] = [tuple(x) for x in conv_cfg_raw]
    fc_cfg:   List[int]             = [int(v) for v in fc_cfg_raw]

    model = FlexibleCNN(conv_cfg, fc_cfg)
    state_dict = obj.get("model_state", obj) if isinstance(obj, dict) else obj
    model.load_state_dict(state_dict, strict=True)
    return model.to(dev).eval()

# ---------- 2) AE-Regressor loader ----------

def legacy_infer_ref_index_from_path(p: Path) -> float:
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

def _strip_leading_id_prefix(filename: str) -> str:
    """
    If the filename starts with '<ID>_' where ID is 1..20 (optionally zero-padded),
    strip the prefix; otherwise return the filename unchanged.
    Keeps the extension.
    """
    name = Path(filename).name  # ensure just the name + extension
    return re.sub(r"^(?:0?[1-9]|1\d|20)_", "", name)





def _infer_ref_index_from_path(p: Path) -> float:
    """
    Drop-in replacement.

    NEW: If a text mapping file (ref_index_map.txt / refindex_map.txt / ri_map.txt)
    exists in the image's directory or any parent, use it to map the sample ID
    from the filename 'ID_*.f06' (ID: 1–2 digits) to the ref_index *label*.

    Mapping file format (flexible; header/comments allowed, comma/whitespace separated):
        # id   n       ref_index
        01     1.077   77
        02     1.012   12
        ...
    - If only 'id  n' is provided, compute label as round((n-1)*1000).
    - If second column looks >2, it's treated as the label directly.

    LEGACY FALLBACK: If no mapping file is found, scan path segments like
    'Refindx1.055' and return round((1.055-1)*1000) -> 55.
    """
    p = Path(p).resolve()

    # 1) Try mapping file near the image
    names = ("ref_index_map.txt", "refindex_map.txt", "ri_map.txt")
    map_file: Optional[Path] = None
    for d in [p.parent] + list(p.parents):
        for n in names:
            cand = d / n
            if cand.exists() and cand.is_file():
                map_file = cand
                break
        if map_file:
            break

    if map_file:
        # parse ID from filename: '01_payload.f06' or '1_payload.f06'
        m = re.match(r"^(?P<id>\d{1,2})_", p.stem)
        if not m:
            raise ValueError(f"Filename must start with '<ID>_': {p.name}")
        sid = int(m.group("id"))

        # build id->ref_index label map
        mapping: Dict[int, float] = {}
        for ln in map_file.read_text(encoding="utf-8").splitlines():
            line = ln.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[,\s]+", line)
            if not parts:
                continue
            # skip header rows
            if parts[0].lower() in {"id"}:
                continue
            try:
                row_id = int(parts[0])
            except Exception:
                continue
            if len(parts) < 2:
                continue
            v1 = parts[1]
            v2 = parts[2] if len(parts) >= 3 else None

            def to_f(s: str) -> float:
                return float(s.replace("_", "."))

            try:
                if v2 is not None:
                    # third column explicitly the label
                    mapping[row_id] = float(to_f(v2))
                else:
                    # single value: interpret as 'n' if <=2, else already a label
                    val = to_f(v1)
                    if 0.0 <= val <= 2.0:
                        ri = round((val - 1.0) * 1000)
                        mapping[row_id] = float(int(ri))
                    else:
                        mapping[row_id] = float(val)
            except Exception:
                continue

        if sid in mapping:
            return float(mapping[sid])
        raise KeyError(f"ID {sid} not found in mapping: {map_file}")

    # 2) Legacy fallback: scan path e.g. 'Refindx1.055' -> 55
    pattern = re.compile(r"(?i)ref(?:ind(?:x|ex)?)?\s*([0-9]+(?:[._][0-9]+)?)")
    for part in p.parts:
        m = pattern.search(part)
        if m:
            s = m.group(1).replace("_", ".")
            try:
                v = float(s)
                return float(int(round((v - 1.0) * 1000)))
            except ValueError:
                pass
    for parent in p.parents:
        m = pattern.search(parent.name)
        if m:
            s = m.group(1).replace("_", ".")
            try:
                v = float(s)
                return float(int(round((v - 1.0) * 1000)))
            except ValueError:
                pass

    raise ValueError(f"Could not infer ref_index for: {p}")


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
    payload_name = _strip_leading_id_prefix(p.name)
    d, t, n_decimal = file_name_to_params(payload_name)
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


def test_dir_avg_error(
    model: nn.Module,
    dir_path: str | Path,
    *,
    denorm: Optional[Dict[str, List[float]]] = None,
    device: Optional[torch.device] = None,
    suffixes: Tuple[str, ...] = (".f06", ".txt"),
    recursive: bool = True,
    noise: bool = False,
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
    save_path_pct: str | None = None,
    pct_epsilon: float = 1e-8,
    close: bool | None = None,   # default: close if not showing
) -> Tuple[Dict[str, float], Dict[str, float], plt.Figure, plt.Axes, plt.Figure, plt.Axes]:
    """
    Evaluate ALL images in `dir_path` and plot:
      1) Average absolute error per label
      2) Average absolute *percentage* error per label (|pred-true| / |true| * 100)
         (skips samples where |true| <= pct_epsilon for that label)

    Uses your existing `predict_and_compare`, which now relies on the UPDATED
    `_infer_ref_index_from_path` that reads `ref_index_map.txt` (with legacy fallback).

    Returns:
        avg_abs_err : dict {label -> average absolute error}
        avg_pct_err : dict {label -> average absolute percentage error (in %)}
        fig_abs, ax_abs : matplotlib fig/ax for absolute error bars
        fig_pct, ax_pct : matplotlib fig/ax for percentage error bars
    """
    dir_path = Path(dir_path).resolve()
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    # collect candidate files
    suffixes_l = tuple(s.lower() for s in suffixes)
    it = dir_path.rglob("*") if recursive else dir_path.glob("*")
    files = [p for p in it if p.is_file() and p.suffix.lower() in suffixes_l]
    files.sort()

    if not files:
        raise FileNotFoundError(f"No files with suffix {suffixes} found under: {dir_path}")

    # temporarily suppress blocking plt.show() inside predict_and_compare
    _orig_show = plt.show
    plt.show = lambda *args, **kwargs: None  # no-op during the loop

    sums_abs = {k: 0.0 for k in LABEL_KEYS}
    n_ok = 0
    # for percentage error we need separate accumulators + counts per label
    sums_pct = {k: 0.0 for k in LABEL_KEYS}
    counts_pct = {k: 0 for k in LABEL_KEYS}
    errors: List[Tuple[Path, str]] = []

    try:
        for p in files:
            try:
                res = predict_and_compare(
                    model=model,
                    image_path=str(p),
                    noise=noise,
                    denorm=denorm,
                    device=device,
                )
                # accumulate absolute errors
                for k in LABEL_KEYS:
                    e = float(res["abs_error"][k])
                    sums_abs[k] += e
                    # percentage error if denominator ok
                    denom = abs(float(res["target"][k]))
                    if denom > pct_epsilon:
                        sums_pct[k] += (e / denom) * 100.0
                        counts_pct[k] += 1
                n_ok += 1
            except Exception as e:
                errors.append((p, str(e)))
            finally:
                # ensure any figures created inside predict_and_compare are closed
                plt.close("all")
    finally:
        # restore original show behavior
        plt.show = _orig_show

    if n_ok == 0:
        raise RuntimeError(
            f"All {len(files)} files failed to evaluate. "
            f"First error: {errors[0][1] if errors else 'unknown'}"
        )

    # averages
    avg_abs_err = {k: (sums_abs[k] / n_ok) for k in LABEL_KEYS}
    avg_pct_err = {
        k: (sums_pct[k] / counts_pct[k]) if counts_pct[k] > 0 else float("nan")
        for k in LABEL_KEYS
    }

    # --- plot 1: absolute error ---
    x = np.arange(len(LABEL_KEYS))
    width = 0.6

    fig_abs = plt.figure(figsize=(7.5, 4.5))
    ax_abs = plt.gca()
    bars_abs = ax_abs.bar(x, [avg_abs_err[k] for k in LABEL_KEYS], width, label="Avg |Error|")

    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(LABEL_KEYS)
    ax_abs.set_ylabel("Average Absolute Error")
    ttl_abs = title or "Average Absolute Error"
    ax_abs.set_title(f"{ttl_abs} across {n_ok} samples\n{dir_path}")
    ax_abs.grid(axis="y", linestyle="--", alpha=0.4)

    for b in bars_abs:
        h = b.get_height()
        ax_abs.annotate(f"{h:.3g}", xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    fig_abs.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig_abs.savefig(save_path, dpi=150)

    # --- plot 2: percentage error ---
    fig_pct = plt.figure(figsize=(7.5, 4.5))
    ax_pct = plt.gca()
    vals_pct = [avg_pct_err[k] for k in LABEL_KEYS]
    # For plotting, replace NaN with 0 height but annotate as 'n/a'
    plot_vals = [0.0 if (isinstance(v, float) and np.isnan(v)) else v for v in vals_pct]
    bars_pct = ax_pct.bar(x, plot_vals, width, label="Avg |Error| (%)")

    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(LABEL_KEYS)
    ax_pct.set_ylabel("Average Absolute Percentage Error (%)")
    used_counts = ", ".join([f"{k}:{counts_pct[k]}" for k in LABEL_KEYS])
    ax_pct.set_title(
        f"Average Absolute Percentage Error across {n_ok} samples (per-label usable counts: {used_counts})\n{dir_path}"
    )
    ax_pct.grid(axis="y", linestyle="--", alpha=0.4)

    for i, b in enumerate(bars_pct):
        v = vals_pct[i]
        h = plot_vals[i]
        label = "n/a" if (isinstance(v, float) and np.isnan(v)) else f"{h:.3g}%"
        ax_pct.annotate(label, xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    fig_pct.tight_layout()
    if save_path_pct:
        os.makedirs(os.path.dirname(save_path_pct) or ".", exist_ok=True)
        fig_pct.savefig(save_path_pct, dpi=150)

    if show:
        plt.show()

    # default behavior: close only if we are NOT showing
    if close is None:
        close = not show
    if close:
        plt.close(fig_abs)
        plt.close(fig_pct)

    if errors:
        print(f"[test_dir_avg_error] Skipped {len(errors)} files due to errors. Showing first 3:")
        for p, msg in errors[:3]:
            print(f"  - {p.name}: {msg}")

    return avg_abs_err, avg_pct_err, fig_abs, ax_abs, fig_pct, ax_pct


def multi_run():
    # Choose device for loading the model; "auto" -> cuda if available, else cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths (use rel_to_root so you don't need ../../../)
    ckpt_path = rel_to_root(
        "outputs/models/FlexibleCNN/20251028-211002_FlexibleCNN_e25_lr0.001_bs32_wd0.0_seed42_dsmanual/FlexibleCNN_e25_lr0.001_bs32_val0.007.pt")
    data_dir = rel_to_root("Data/extra_runs_for_check")
    out_png = rel_to_root("outputs/test_graphs/extra_runs_avg_abs_error.png")
    out_pct = rel_to_root("outputs/test_graphs/extra_runs_avg_pct_error.png")  # <- add this

    # Load model and run the directory-level evaluation
    model = torch.load(ckpt_path, map_location="cpu",weights_only=False).to(device).eval()  # <— full module


    avg_abs_err, avg_pct_err, _, _, _, _ = test_dir_avg_error(
        model=model,
        dir_path=data_dir,
        denorm=None,  # or {"mean":[...4], "std":[...4]} if you normalized targets
        device=None,  # None -> use the model's device
        noise=False,  # keep False unless you want to add noise during eval
        title="Avg |error| on extra_runs_for_check",
        show=True,
        save_path=str(out_png),
        save_path_pct=str(out_pct),
    )

    print("Avg abs error:", avg_abs_err)
    print("Avg % error:", avg_pct_err)


def single_run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = "outputs/models/FlexibleCNN/20251027-210402_FlexibleCNN_e2_lr0.001_bs32_wd0.0_seed42_dsmanual/FlexibleCNN_e2_lr0.001_bs32_val16079.430.pt"
    # 1) load the full model (no builder/arch needed)
    #model = load_cnn_model("outputs/models/FlexibleCNN/20251027-210402_FlexibleCNN_e2_lr0.001_bs32_wd0.0_seed42_dsmanual/FlexibleCNN_e2_lr0.001_bs32_val16079.430.pt")
    model = torch.load(path, map_location="cpu",weights_only=False).to(device).eval()  # <— full module

    # 2) load one RBC text image + labels (no normalization)
    #res = predict_and_compare(model, "../../../Data/results/Refindx1.025/0450015005501a.f06", False)
    res = predict_and_compare(model, "../../../Data/extra_runs_for_check/05_0362516257251a.f06", False)
    plot_bar_pred_vs_true_single(res, title="Predicted vs True for 0450015005001a.f06", show=True)



if __name__ == "__main__":
    #print (a_infer_ref_index_from_path(Path("../../../Data/extra_runs_for_check/20_0737523754741a.f06")))
    multi_run()
    #single_run()
