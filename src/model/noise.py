import torch
from torch import nn
from src.model.RBCDataset import *


class WithTransform(Dataset):
    def __init__(self, base, transform=None):
        self.base = base
        self.transform = transform
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return (self.transform(x), y) if self.transform else (x, y)


class AddGaussianNoise(nn.Module):
    def __init__(self, std: float = 0.02, p: float = 0.5):
        super().__init__()
        self.std = float(std)
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1,50,50] or [B,1,50,50], float
        if torch.rand(()) < self.p:
            return x + torch.randn_like(x) * self.std
        return x


class AddSpeckleNoise(nn.Module):
    def __init__(self, std: float = 0.05, p: float = 0.5):
        super().__init__()
        self.std = float(std)
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < self.p:
            return x + x * (torch.randn_like(x) * self.std)
        return x


class AddPoissonNoise(nn.Module):
    """
    Simulate Poisson shot noise. Assumes x ∈ [0,1].
    `peak` is the expected photon count at x=1.
    """
    def __init__(self, peak: float = 20.0, p: float = 0.5):
        super().__init__()
        self.peak = float(peak)
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) >= self.p:
            return x
        x_clamped = x.clamp(0, 1)
        photons = x_clamped * self.peak
        noisy = torch.poisson(photons)
        return (noisy / self.peak).to(x.dtype)


class RandomSaltPepper(nn.Module):
    def __init__(self, amount: float = 0.01, s_vs_p: float = 0.5, p: float = 0.3):
        super().__init__()
        self.amount = float(amount)   # fraction of pixels affected
        self.s_vs_p = float(s_vs_p)   # salt vs pepper balance
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) >= self.p:
            return x
        x = x.clone()
        numel = x.numel()
        num_salt = int(self.amount * self.s_vs_p * numel)
        num_pepper = int(self.amount * (1 - self.s_vs_p) * numel)
        idx = torch.randperm(numel, device=x.device)
        x.view(-1)[idx[:num_salt]] = 1.0
        x.view(-1)[idx[num_salt:num_salt+num_pepper]] = 0.0
        return x
