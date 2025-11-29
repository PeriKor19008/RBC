from __future__ import annotations
import numpy as np

from src.model.ae_heads import *
from src.model.model import FCAutoencoder
from src.utils.fileName_to_params import file_name_to_params
import re
import os
import matplotlib.pyplot as plt
from src.model.noise import *

from pathlib import Path
from skimage.metrics import structural_similarity as ssim_sk
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.feature import canny
from src.model.noise import *
from src.model.experiments.tests_helper import load_rbc_txt_image_and_labels
from src.utils.paths import rel_to_root




def mse_np(x, y):
    return float(torch.mean((x - y) ** 2))

def ssim_np(x, y):
    # win_size=7 for 50x50
    return float(ssim_sk(x, y, data_range=x.max()-x.min(), win_size=7))

def psnr_np(x, y):
    return float(psnr_sk(x, y, data_range=x.max()-x.min()))

def edge_f1(x, y, sigma=1.0):
    ex = canny(x, sigma=sigma)
    ey = canny(y, sigma=sigma)
    tp = np.logical_and(ex, ey).sum()
    px = ex.sum()
    py = ey.sum()
    if px + py == 0:
        return 1.0
    return float((2.0 * tp) / (px + py + 1e-8))

def test_ae(model: nn.Module, dir_path: str | Path, save_path_pct: str | None ):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mse = []
    ssim = []
    psnr = []
    edge = []
    i = 0
    for f in dir_path.iterdir():
        if not f.is_file() or f.suffix.lower() != ".f06":
            continue
        img, lbl_true = load_rbc_txt_image_and_labels(f)
        x = img.unsqueeze(0).to(dev)
        with torch.no_grad():
            y = model(x).squeeze(0).detach().cpu()
        mse.append(mse_np(x,y))
        ssim.append(ssim_np(x.squeeze(0).squeeze(0).detach().cpu().numpy(), y.squeeze(0).detach().cpu().numpy()))
        psnr.append(psnr_np(x.squeeze(0).squeeze(0).detach().cpu().numpy(), y.squeeze(0).detach().cpu().numpy()))
        edge.append(edge_f1(x.squeeze(0).squeeze(0).detach().cpu().numpy(), y.squeeze(0).detach().cpu().numpy()))
        i += 1
    mse_avg = sum(mse) / len(mse)
    ssim_avg = sum(ssim) / len(ssim)
    psnr_avg = sum(psnr) / len(psnr)
    edge_avg = sum(edge) / len(edge)
    mse_norm = 1 / (mse_avg + 1)
    print("averages:")
    print("mse avg:", mse_avg)
    print("mse normalized (big = good):", mse_norm)
    print("ssim avg:", ssim_avg)
    print("psnr avg:", psnr_avg)
    print("edge avg:", edge_avg)



def minmax_norm(values, invert=False):
    values = np.array(values, dtype=np.float32)
    vmin = values.min()
    vmax = values.max()
    if vmax - vmin < 1e-12:
        out = np.zeros_like(values)
    else:
        out = (values - vmin) / (vmax - vmin)
    if invert:
        out = 1.0 - out
    return out.tolist()


