import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt


def find_last_conv(module: nn.Module) -> nn.Conv2d:
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found.")
    return last_conv

class GradCAM:

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model.eval()
        self.model_zero_grad = lambda: self.model.zero_grad(set_to_none=True)

        #find last conv layer
        if target_layer is None:

            if hasattr(model, "conv_layers"):
                target_layer = find_last_conv(model.conv_layers)
            else:
                target_layer = find_last_conv(model)

        self.target_layer = target_layer


        self._activations = None
        self._gradients = None


        def forward_hook(module, inp, out):
            # out: [B, C, H', W']
            self._activations = out.detach()

            def _save_grad(grad):
                self._gradients = grad.detach()
            out.register_hook(_save_grad)

        self._fwd_hook = self.target_layer.register_forward_hook(forward_hook)

    def remove_hooks(self):
        if self._fwd_hook is not None:
            self._fwd_hook.remove()
            self._fwd_hook = None

    @torch.no_grad()
    def _normalize(self, cam: torch.Tensor) -> torch.Tensor:
        # cam: [B, 1, H, W] ή [H, W]
        cam_min = cam.amin(dim=(-2, -1), keepdim=True)
        cam_max = cam.amax(dim=(-2, -1), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam

    def __call__(self, x: torch.Tensor, target_index: int) -> torch.Tensor:

        assert x.ndim == 4 and x.size(0) == 1, "Δώσε batch με 1 εικόνα: [1,C,H,W]"

        # -- Forward --
        self.model_zero_grad()
        y = self.model(x)  # [1, 4]


        self.model.zero_grad(set_to_none=True)
        y[:, target_index].backward(retain_graph=True)


        # self._activations: [1, C', H', W']
        # self._gradients:   [1, C', H', W']
        if self._activations is None or self._gradients is None:
            raise RuntimeError("Δεν καταγράφηκαν activations/gradients. Έλεγξε το target layer και τα hooks.")

        A = self._activations
        dY = self._gradients


        weights = dY.mean(dim=(2, 3), keepdim=True)


        cam = (weights * A).sum(dim=1, keepdim=True)


        cam = F.relu(cam)


        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)  # [1,1,H,W]


        cam = self._normalize(cam)[0, 0]  # [H, W]
        return cam


def overlay_heatmap(img: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:

    import matplotlib.cm as cm
    if img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=-1)
    else:
        img_rgb = img.copy()
        if img_rgb.max() > 1.0:  # ασφάλεια
            img_rgb = img_rgb / 255.0

    heat = cm.jet(cam)[..., :3]  # jet colormap σε RGB
    out = (1 - alpha) * img_rgb + alpha * heat
    out = np.clip(out, 0, 1)
    return out

def plot_grad_cam(cam, img, k):
    img_np = img[0, 0].detach().cpu().numpy()
    cam_np = cam.detach().cpu().numpy()
    overlay = overlay_heatmap(img_np, cam_np, alpha=0.45)
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].imshow(img_np, cmap="gray")
    axs[0].set_title("Input")
    axs[0].axis("off")
    axs[1].imshow(cam_np, cmap="jet")
    axs[1].set_title(f"Grad-CAM (y[{k}])")
    axs[1].axis("off")
    axs[2].imshow(overlay)
    axs[2].set_title("Overlay")
    axs[2].axis("off")
    plt.tight_layout()
    plt.show()
