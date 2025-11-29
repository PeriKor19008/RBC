from __future__ import annotations
from tests_helper import *

LABEL_KEYS = ["diameter", "thickness", "ratio", "ref_index"]
def macro_pct_error(model: nn.Module, x: Tensor, y_true: Tensor, dev: torch.device,vec:bool=False, eps: float = 1e-8) -> float:
    with torch.no_grad():
        y_pred = model(x.to(dev)).squeeze(0).detach().cpu()
    y_true = y_true.cpu()
    abs_err = (y_pred - y_true).abs()
    pct = abs_err / (y_true.abs() + eps) * 100.0
    if vec:
        return pct
    return float(pct.mean().item())



def occlusion_map_simple(
    model: nn.Module,
    img: Tensor,
    lbl_true: Tensor,
    k: int = 5,
    stride: int = 2,
    fill: str = "mean",
    eps: float = 1e-8,
):
    dev = next(model.parameters()).device
    model.eval()


    x0 = img.unsqueeze(0)
    baseline = macro_pct_error(model, x0, lbl_true, dev,False, eps)


    _, H, W = img.shape
    k = int(k)
    stride = int(stride)
    Hpos = 1 + (H - k) // stride
    Wpos = 1 + (W - k) // stride

    heat = torch.zeros((Hpos, Wpos), dtype=torch.float32)  # Δ% per position


    if fill == "mean":
        fill_val = float(img.mean())
    elif fill == "zero":
        fill_val = 0.0
    else:
        fill_val = float(img.mean())


    for ri, i in enumerate(range(0, H - k + 1, stride)):
        for cj, j in enumerate(range(0, W - k + 1, stride)):
            x_occ = img.clone()
            x_occ[:, i:i + k, j:j + k] = fill_val
            x_occ = x_occ.unsqueeze(0)  # [1,1,H,W]
            occ_macro = macro_pct_error(model, x_occ, lbl_true, dev, False, eps)
            heat[ri, cj] = occ_macro - baseline  # Δ% (positive = harmful)

    return heat, baseline

def occlusion_map_simple_avg(
    model: nn.Module,
    dir_path: str | Path,
    *,
    k: int = 5,
    stride: int = 2,
    fill: str = "mean",
    eps: float = 1e-8,
):

    dir_path = Path(dir_path).resolve()
    model.eval()

    heat_sum: Tensor | None = None
    base_sum = 0.0
    n = 0
    ref_img: Tensor | None = None

    for f in dir_path.iterdir():
        if not f.is_file() or f.suffix.lower() != ".f06":
            continue

        img, lbl_true = load_rbc_txt_image_and_labels(f)

        heat, baseline = occlusion_map_simple(
            model=model,
            img=img,
            lbl_true=lbl_true,
            k=k,
            stride=stride,
            fill=fill,
            eps=eps,
        )

        if heat_sum is None:
            heat_sum = torch.zeros_like(heat, dtype=torch.float32)
            ref_img = img.clone()

        heat_sum += heat.to(torch.float32)
        base_sum += float(baseline)
        n += 1




    avg_heat = heat_sum / n
    avg_baseline = base_sum / n
    return avg_heat, avg_baseline, n, ref_img

def plot_occlusion_map_simple(
    img: Tensor,
    heat: Tensor,
    k: int,
    stride: int,
    title: str = "Occlusion Δ% (macro)",

):

    arr = img.detach().cpu()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    arr = arr.numpy()
    heat_np = heat.detach().cpu().numpy()

    vmin, vmax = float(arr.min()), float(arr.max())

    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax1.set_title("Input image")
    ax1.set_xticks([]); ax1.set_yticks([])

    ax2 = fig.add_subplot(1, 2, 2)

    im = ax2.imshow(heat_np, cmap="hot", interpolation="nearest", origin="upper")
    ax2.set_title(title + f"\n(k={k}, stride={stride})")
    ax2.set_xlabel("x (top-left of block)"); ax2.set_ylabel("y")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Δ% error")

    fig.tight_layout()


    plt.show()
    plt.close(fig)





def plot_occlusion_maps_per_label(
    img: Tensor,
    heat: Tensor,
    k: int,
    stride: int,
    label_idx: int | None = None,

):

    arr = img.detach().cpu()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    arr = arr.numpy()
    vmin, vmax = float(arr.min()), float(arr.max())

    if label_idx is None:
        fig, axes = plt.subplots(1, 5, figsize=(14, 3.6))

        ax0 = axes[0]
        ax0.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax0.set_title("Input")
        ax0.set_xticks([]); ax0.set_yticks([])

        for idx in range(4):
            ax = axes[idx + 1]
            hm = heat[idx].detach().cpu().numpy()
            im = ax.imshow(hm, cmap="hot", interpolation="nearest", origin="upper")
            ax.set_title(f"{LABEL_KEYS[idx]}\nΔ% (k={k}, s={stride})")
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
    else:
        fig = plt.figure(figsize=(9, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax1.set_title("Input image")
        ax1.set_xticks([]); ax1.set_yticks([])

        ax2 = fig.add_subplot(1, 2, 2)
        hm = heat[label_idx].detach().cpu().numpy()
        im = ax2.imshow(hm, cmap="hot", interpolation="nearest", origin="upper")
        ax2.set_title(f"{LABEL_KEYS[label_idx]} Δ% (k={k}, s={stride})")
        ax2.set_xlabel("x (top-left)"); ax2.set_ylabel("y")
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Δ% error")
        fig.tight_layout()



    plt.show()
    plt.close(fig)

def occlusion_map_per_label(
    model: nn.Module,
    img: Tensor,
    lbl_true: Tensor,
    k: int = 5,
    stride: int = 2,
    fill: str = "mean",
    eps: float = 1e-8,
):
    dev = next(model.parameters()).device
    model.eval()


    x0 = img.unsqueeze(0)
    base_vec = macro_pct_error(model, x0, lbl_true, dev, True, eps)  # [4]

    # positions
    _, H, W = img.shape
    k = int(k)
    stride = int(stride)
    Hpos = 1 + (H - k) // stride
    Wpos = 1 + (W - k) // stride


    heat = torch.zeros((4, Hpos, Wpos), dtype=torch.float32)


    if fill == "mean":
        fill_val = float(img.mean())
    elif fill == "zero":
        fill_val = 0.0
    else:
        fill_val = float(img.mean())


    for ri, i in enumerate(range(0, H - k + 1, stride)):
        for cj, j in enumerate(range(0, W - k + 1, stride)):
            x_occ = img.clone()
            x_occ[:, i:i + k, j:j + k] = fill_val
            x_occ = x_occ.unsqueeze(0)

            occ_vec = macro_pct_error(model, x_occ, lbl_true, dev,True, eps)  # [4]
            heat[:, ri, cj] = occ_vec - base_vec  # per-label Δ%

    return heat, base_vec



def occlusion_map_per_label_avg(
    model: nn.Module,
    dir_path: str | Path,
    *,
    k: int = 5,
    stride: int = 2,
    fill: str = "mean",
    eps: float = 1e-8,
):

    dir_path = Path(dir_path)
    model.eval()

    heat_sum = None
    base_sum = torch.zeros(4, dtype=torch.float32)
    n = 0
    ref_img: Tensor | None = None

    for f in dir_path.iterdir():
        if not f.is_file() or f.suffix.lower() != ".f06":
            continue

        img, lbl_true = load_rbc_txt_image_and_labels(f)
        if ref_img is None:
            ref_img = img.clone()

        heat, base_vec = occlusion_map_per_label(
            model=model,
            img=img,
            lbl_true=lbl_true,
            k=k,
            stride=stride,
            fill=fill,
            eps=eps,
        )

        if heat_sum is None:
            heat_sum = torch.zeros_like(heat, dtype=torch.float32)

        heat_sum += heat.to(torch.float32)
        base_sum += base_vec.to(torch.float32)
        n += 1





    avg_heat = heat_sum / n
    avg_base = base_sum / n
    return avg_heat, avg_base, n, ref_img




