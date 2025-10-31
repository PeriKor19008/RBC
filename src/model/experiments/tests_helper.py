from __future__ import annotations
from pathlib import Path
from typing import Dict

from torch import Tensor

from src.model.ae_heads import *
from src.utils.fileName_to_params import file_name_to_params
import re
import os
import matplotlib.pyplot as plt
from src.model.noise import *

def _strip_leading_id_prefix(filename: str) -> str:

    name = Path(filename).name  # ensure just the name + extension
    return re.sub(r"^(?:0?[1-9]|1\d|20)_", "", name)


def _infer_ref_index_from_path(p: Path) -> float:

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

    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    if len(lines) != 2500:
        raise ValueError(f"Expected 2500 lines, got {len(lines)} in: {p}")
    vals = [float(s.replace("D", "E")) for s in lines]
    arr = np.asarray(vals, dtype=np.float32).reshape(50, 50)
    image_tensor = torch.from_numpy(arr).unsqueeze(0)

    return image_tensor, labels


def change_block(size, img:Tensor) -> Tensor:
    if size < 1:
        raise ValueError("size must be >= 1")

    # img is [1, 50, 50] (from load_rbc_txt_image_and_labels)
    # work on a copy to avoid in-place side effects
    out = img.clone()

    _, H, W = out.shape  # expected 50x50
    b = int(size)
    b = min(b, H, W)  # clamp to image size

    # choose top-left corner uniformly
    i = torch.randint(0, H - b + 1, (1,)).item()
    j = torch.randint(0, W - b + 1, (1,)).item()

    # fill with random values sampled uniformly between image min/max
    low = float(out.min())
    high = float(out.max())
    patch = torch.empty((1, b, b), dtype=out.dtype, device=out.device).uniform_(low, high)

    out[:, i:i + b, j:j + b] = patch
    return out


def jitter_block(size: int, img: Tensor, strength: float = 0.1) -> Tensor:
    """
    Randomly select a size×size block in a [1,50,50] image and make small random changes.
    - strength: scales the noise (as a fraction of the block's local std). e.g., 0.1 = 10%.
    - clamp: if True, clamp the modified block to the image's [min, max] range.
    """
    if size < 1:
        raise ValueError("size must be >= 1")

    out = img.clone()
    _, H, W = out.shape
    b = int(size)
    b = min(b, H, W)

    # choose top-left corner
    i = torch.randint(0, H - b + 1, (1,)).item()
    j = torch.randint(0, W - b + 1, (1,)).item()

    # current block and its local stats
    block = out[:, i:i+b, j:j+b]
    local_std = float(block.std())
    global_std = float(out.std())
    sigma = strength * (local_std if local_std > 0.0 else global_std)

    if sigma > 0.0:
        noise = torch.randn_like(block) * sigma
        block = block + noise  # small random change
    # else sigma==0 -> block is constant; leave it as-is


    low = float(out.min())
    high = float(out.max())
    block = block.clamp(min=low, max=high)

    out[:, i:i+b, j:j+b] = block
    return out

def show_img(img: torch.Tensor):
    """
    Display a [1,50,50] (or [50,50]) tensor as a grayscale image.
    """
    arr = img.detach().cpu()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    arr = arr.numpy()

    vmin, vmax = float(arr.min()), float(arr.max())

    fig = plt.figure(figsize=(4.5, 4.5))
    ax = plt.gca()
    im = ax.imshow(
        arr,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,           # keep fixed range to avoid autoscale shifts
        interpolation="nearest"
    )


    fig.tight_layout()


    plt.show()
    plt.close(fig)

def plot_avg_error_prc(iterations,errors: List[float], save_path: str | None = None):
    LABEL_KEYS = ["diameter", "thickness", "ratio", "ref_index"]
    vals = [float(v) for v in errors]
    x = np.arange(len(LABEL_KEYS))
    width = 0.6

    fig = plt.figure(figsize=(7.5, 4.5))
    ax = plt.gca()
    ax.set_ylim(0, max(vals) * 1.15)
    bars = ax.bar(x, vals, width, label="Avg |Error|")

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_KEYS)
    ax.set_ylabel("Average Error Percentage")
    ax.set_title(f"Average Error Percentage across {iterations} samples")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.3g}%", xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)

    plt.show()
    plt.close(fig)