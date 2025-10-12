# src/model/ae_heads.py
from __future__ import annotations

from typing import Optional, Sequence, Callable, List, Tuple
import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name in {"relu", "relu6"}:
        return nn.ReLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


class AERegressor(nn.Module):
    """
    Wrap a trained AutoEncoder's encoder with a small MLP head to regress targets.

    Typical usage:
        ae = FCAutoencoder(latent_dim=64, hidden_dims=[1024,512,128])
        ae.load_state_dict(torch.load(".../autoencoder_final.pt", map_location="cpu"))
        model = AERegressor.from_autoencoder(ae, latent_dim=64,
                                             head_hidden=(128,64), freeze_encoder=True)

    Notes:
    - Assumes the AE's encoder takes a flattened input [B, 2500] by default.
      If your encoder expects image-shaped input, set `flatten_input=False`
      and adapt the forward accordingly.
    """

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        *,
        num_outputs: int = 4,
        head_hidden: Sequence[int] = (128,),
        activation: str = "relu",
        dropout: float = 0.0,
        freeze_encoder: bool = True,
        flatten_input: bool = True,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.latent_dim = int(latent_dim)
        self.num_outputs = int(num_outputs)
        self.flatten_input = bool(flatten_input)

        if freeze_encoder:
            self.freeze_encoder(True)

        layers: List[nn.Module] = []
        in_dim = self.latent_dim
        act = _activation(activation)

        for h in head_hidden:
            h = int(h)
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act)
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            in_dim = h

        layers.append(nn.Linear(in_dim, self.num_outputs))
        self.head = nn.Sequential(*layers)

    # ---------- convenience constructors ----------

    @classmethod
    def from_autoencoder(
        cls,
        ae: nn.Module,
        latent_dim: int,
        *,
        num_outputs: int = 4,
        head_hidden: Sequence[int] = (128,),
        activation: str = "relu",
        dropout: float = 0.0,
        freeze_encoder: bool = True,
        flatten_input: bool = True,
        use_batchnorm: bool = False,
    ) -> "AERegressor":
        """
        Build an AERegressor from an instantiated & (optionally) preloaded AE.
        Uses `ae.encoder` as the feature extractor.
        """
        if not hasattr(ae, "encoder"):
            raise AttributeError("Autoencoder instance must have an `encoder` attribute.")
        return cls(
            encoder=ae.encoder,
            latent_dim=latent_dim,
            num_outputs=num_outputs,
            head_hidden=head_hidden,
            activation=activation,
            dropout=dropout,
            freeze_encoder=freeze_encoder,
            flatten_input=flatten_input,
            use_batchnorm=use_batchnorm,
        )

    @classmethod
    def from_checkpoint(
        cls,
        ae_builder: Callable[..., nn.Module],
        ckpt_path: str,
        *,
        ae_kwargs: Optional[dict] = None,
        latent_dim: int = 64,
        num_outputs: int = 4,
        head_hidden: Sequence[int] = (128,),
        activation: str = "relu",
        dropout: float = 0.0,
        freeze_encoder: bool = True,
        flatten_input: bool = True,
        use_batchnorm: bool = False,
        device: Optional[torch.device] = None,
        strict: bool = True,
    ) -> "AERegressor":
        """
        Build AERegressor by constructing an AE via `ae_builder(**ae_kwargs)`, loading a checkpoint,
        then taking its `.encoder`.
        """
        ae_kwargs = dict(ae_kwargs or {})
        ae = ae_builder(**ae_kwargs)
        state = torch.load(ckpt_path, map_location=device or "cpu")
        ae.load_state_dict(state, strict=strict)
        if device is not None:
            ae.to(device)
        return cls.from_autoencoder(
            ae,
            latent_dim=latent_dim,
            num_outputs=num_outputs,
            head_hidden=head_hidden,
            activation=activation,
            dropout=dropout,
            freeze_encoder=freeze_encoder,
            flatten_input=flatten_input,
            use_batchnorm=use_batchnorm,
        )

    # ---------- utils ----------

    def freeze_encoder(self, freeze: bool = True) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not freeze

    def unfreeze_last_encoder_layers(self, n_linear: int = 1) -> None:
        """
        Unfreeze the last `n_linear` nn.Linear layers of the encoder (useful for light fine-tuning).
        """
        linear_layers: List[nn.Linear] = [m for m in self.encoder.modules() if isinstance(m, nn.Linear)]
        if not linear_layers:
            return
        for layer in linear_layers[-n_linear:]:
            for p in layer.parameters():
                p.requires_grad = True

    def param_groups(
        self,
        encoder_lr: Optional[float] = None,
        head_lr: Optional[float] = None,
        weight_decay: float = 0.0,
    ) -> List[dict]:
        """
        Build optimizer param groups so you can give the encoder a smaller LR than the head.
        Example:
            opt = torch.optim.Adam(
                model.param_groups(encoder_lr=1e-4, head_lr=1e-3, weight_decay=1e-5)
            )
        """
        groups: List[dict] = []
        enc_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params = [p for p in self.head.parameters() if p.requires_grad]

        if enc_params:
            groups.append({"params": enc_params, "lr": encoder_lr, "weight_decay": weight_decay})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 50, 50] → flatten if requested (FCAutoencoder.encoder expects flat)
        if self.flatten_input:
            x = x.view(x.size(0), -1)
        z = self.encoder(x)       # [B, latent_dim]
        y_hat = self.head(z)      # [B, num_outputs]
        return y_hat


# ---------- small helpers (optional) ----------

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count parameters in a model. trainable_only=True counts only requires_grad=True.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def build_ae_regressor_from_checkpoint(
    *,
    ae_builder: Callable[..., nn.Module],
    ckpt_path: str,
    ae_kwargs: Optional[dict],
    latent_dim: int,
    head_hidden: Sequence[int] = (128,),
    num_outputs: int = 4,
    freeze_encoder: bool = True,
    activation: str = "relu",
    dropout: float = 0.0,
    device: Optional[torch.device] = None,
) -> Tuple[AERegressor, nn.Module]:
    """
    Convenience wrapper that returns (regressor, autoencoder) after loading the AE checkpoint.
    Useful if you also want to inspect AE parts (e.g., to partially unfreeze later).
    """
    ae_kwargs = dict(ae_kwargs or {})
    ae = ae_builder(**ae_kwargs)
    state = torch.load(ckpt_path, map_location=device or "cpu")
    ae.load_state_dict(state, strict=True)
    if device is not None:
        ae.to(device)

    reg = AERegressor.from_autoencoder(
        ae=ae,
        latent_dim=latent_dim,
        head_hidden=head_hidden,
        num_outputs=num_outputs,
        freeze_encoder=freeze_encoder,
        activation=activation,
        dropout=dropout,
        flatten_input=True,
    )
    if device is not None:
        reg.to(device)
    return reg, ae
