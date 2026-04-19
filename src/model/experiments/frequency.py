import math
from typing import Sequence, Tuple, Dict, List
import torch.nn.functional as F
from tests_helper import *

LABEL_KEYS = ["diameter", "thickness", "ratio", "ref_index"]




def _gaussian_kernel1d(sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if sigma <= 0:
        return torch.tensor([1.0], dtype=dtype, device=device)
    radius = int(math.ceil(3.0 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum()
    return k


def _gaussian_blur_2d(img: torch.Tensor, sigma: float) -> torch.Tensor:

    if sigma <= 0:
        return img.clone()

    c, h, w = img.shape
    assert c == 1, "expected single-channel image [1,H,W]"

    k1d = _gaussian_kernel1d(sigma, img.dtype, img.device)
    ky = k1d.view(1, 1, -1, 1)
    kx = k1d.view(1, 1, 1, -1)


    x = img.unsqueeze(0)


    pad_y = (0, 0, k1d.numel() // 2, k1d.numel() // 2)
    y = F.pad(x, pad_y, mode="reflect")
    y = F.conv2d(y, ky, padding=0, groups=1)


    pad_x = (k1d.numel() // 2, k1d.numel() // 2, 0, 0)
    y = F.pad(y, pad_x, mode="reflect")
    y = F.conv2d(y, kx, padding=0, groups=1)

    return y.squeeze(0)

def _unsharp(img: torch.Tensor, amount: float, sigma: float, clamp: bool = True) -> torch.Tensor:

    if amount <= 0 or sigma <= 0:
        return img.clone()
    blur = _gaussian_blur_2d(img, sigma)
    mask = img - blur
    out = img + amount * mask
    if clamp:
        lo, hi = float(img.min()), float(img.max())
        out = out.clamp(min=lo, max=hi)
    return out


def test_gaussian_blur_sweep(
    model: torch.nn.Module,
    dir_path: str | Path,
    sigmas: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
    pct_eps: float = 1e-8,

) -> Dict[float, List[float]]:

    dir_path = Path(dir_path).resolve()
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".f06"]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No '.f06' files found under: {dir_path}")

    model.eval()
    dev = next(model.parameters()).device

    results: Dict[float, List[float]] = {}
    macro_series: List[Tuple[float, float]] = []

    for sigma in sigmas:

        sum_pct = torch.zeros(4, dtype=torch.float32)
        n = 0

        for f in files:
            img, lbl_true = load_rbc_txt_image_and_labels(f)
            img_blur = _gaussian_blur_2d(img, float(sigma))

            x = img_blur.unsqueeze(0).to(dev)
            with torch.no_grad():
                lbl_pred = model(x).squeeze(0).detach().cpu()

            lbl_true_cpu = lbl_true.cpu()
            abs_err = (lbl_pred - lbl_true_cpu).abs()
            pct_err = abs_err / (lbl_true_cpu.abs() + pct_eps) * 100.0

            sum_pct += pct_err.to(sum_pct.dtype)
            n += 1

        avg_pct = (sum_pct / max(1, n)).tolist()
        results[float(sigma)] = avg_pct
        macro = float(np.mean(avg_pct))
        macro_series.append((float(sigma), macro))


        print(f"[sigma={sigma:.2f}] avg % errors -> "
              + ", ".join(f"{k}={v:.2f}%" for k, v in zip(LABEL_KEYS, avg_pct))
              + f"  | macro={macro:.2f}%")


    sig_arr = np.array([s for s in results.keys()], dtype=float)
    per_label = np.array([results[s] for s in results.keys()], dtype=float)  # [S,4]

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for i, key in enumerate(LABEL_KEYS):
        ax.plot(sig_arr, per_label[:, i], marker="o", linewidth=1.5, label=key)
        for sx, vy in zip(sig_arr, per_label[:, i]):
            ax.annotate(f"{vy:.2f}%", xy=(sx, vy), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Gaussian blur σ (pixels)")
    ax.set_ylabel("Average Absolute Percentage Error (%)")
    ax.set_title(f"Blur sweep (low-pass) over {len(files)} samples")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    ax.set_ylim(0, max(1.0, float(np.nanmax(per_label)) * 1.15))

    fig.tight_layout()

    plt.show()
    plt.close(fig)

    return results

def test_unsharp_sweep(
    model: torch.nn.Module,
    dir_path: str | Path,
    amounts: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    sigma: float = 1.0,
    pct_eps: float = 1e-8,
) -> Dict[float, List[float]]:

    dir_path = Path(dir_path).resolve()
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".f06"]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No '.f06' files found under: {dir_path}")

    model.eval()
    dev = next(model.parameters()).device

    results: Dict[float, List[float]] = {}
    for amt in amounts:
        sum_pct = torch.zeros(4, dtype=torch.float32)
        n = 0

        for f in files:
            img, lbl_true = load_rbc_txt_image_and_labels(f)     # img: [1,50,50], lbl_true: [4]
            img_sharp = _unsharp(img, float(amt), float(sigma), clamp=True)

            x = img_sharp.unsqueeze(0).to(dev)                   # [1,1,50,50]
            with torch.no_grad():
                lbl_pred = model(x).squeeze(0).detach().cpu()    # [4]

            y = lbl_true.cpu()
            abs_err = (lbl_pred - y).abs()
            pct_err = abs_err / (y.abs() + pct_eps) * 100.0      # [4]

            sum_pct += pct_err.to(sum_pct.dtype)
            n += 1

        avg_pct = (sum_pct / max(1, n)).tolist()
        results[float(amt)] = avg_pct

        macro = float(np.mean(avg_pct))
        print(f"[amount={amt:.2f}, sigma={sigma:.2f}] avg % -> "
              + ", ".join(f"{k}={v:.2f}%" for k, v in zip(LABEL_KEYS, avg_pct))
              + f"  | macro={macro:.2f}%")


    amts = np.array(list(results.keys()), dtype=float)
    per_label = np.array([results[a] for a in results.keys()], dtype=float)  # [A,4]

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for i, key in enumerate(LABEL_KEYS):
        ax.plot(amts, per_label[:, i], marker="o", linewidth=1.5, label=key)
        for ax_, ay in zip(amts, per_label[:, i]):
            ax.annotate(f"{ay:.2f}%", xy=(ax_, ay), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel(f"Unsharp amount (mask strength), σ={sigma}")
    ax.set_ylabel("Average Absolute Percentage Error (%)")
    ax.set_title(f"Unsharp sweep over {len(files)} samples")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    ymax = float(np.nanmax(per_label)) if per_label.size else 1.0
    ax.set_ylim(0, max(1.0, ymax * 1.15))

    fig.tight_layout()

    plt.show()
    plt.close(fig)

    return results




