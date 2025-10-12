from __future__ import annotations
from typing import Tuple, Optional, Dict, Any
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    OneCycleLR, CosineAnnealingLR, CosineAnnealingWarmRestarts,
    ReduceLROnPlateau, StepLR, MultiStepLR, ExponentialLR
)

def current_lr(optim: Optimizer) -> float:
    return float(optim.param_groups[0]["lr"])



def build_scheduler(
    optimizer: Optimizer,
    name: Optional[str],
    *,
    num_epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    base_lr: Optional[float] = None,
    **kwargs: Any
) -> Tuple[Optional[object], str]:
    """
    Returns (scheduler, mode) where mode in {"batch","epoch","plateau","none"}.
    - "batch": call scheduler.step() every batch
    - "epoch": call scheduler.step() every epoch
    - "plateau": call scheduler.step(val_loss) every epoch
    """
    if not name:
        return None, "none"
    name = name.lower()

    if name in {"onecycle", "one_cycle"}:
        assert num_epochs and steps_per_epoch, "OneCycleLR needs num_epochs & steps_per_epoch"
        max_lr = kwargs.get("max_lr")
        if max_lr is None and base_lr is not None:
            max_lr = base_lr * kwargs.get("max_lr_mult", 5.0)
        if max_lr is None:
            raise ValueError("OneCycleLR: provide max_lr or base_lr")
        div_factor = kwargs.get("div_factor", max_lr / (base_lr or (max_lr / 25.0)))
        sched = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=kwargs.get("pct_start", 0.3),
            div_factor=div_factor,
            final_div_factor=kwargs.get("final_div_factor", 1e4),
            cycle_momentum=kwargs.get("cycle_momentum", False),
            three_phase=kwargs.get("three_phase", False),
            anneal_strategy=kwargs.get("anneal_strategy", "cos")
        )
        return sched, "batch"

    if name in {"cosine", "cosineannealing"}:
        t_max = kwargs.get("t_max", num_epochs)
        eta_min = kwargs.get("eta_min", 0.0)
        sched = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        return sched, "epoch"

    if name in {"cosine_restarts", "cosinewarmrestarts"}:
        t0 = kwargs.get("t0", 10)
        t_mult = kwargs.get("t_mult", 1)
        eta_min = kwargs.get("eta_min", 0.0)
        sched = CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=t_mult, eta_min=eta_min)
        return sched, "epoch"

    if name in {"plateau", "reduce_on_plateau", "reduce_lr_on_plateau"}:
        sched = ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "min"),
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 5),
            threshold=kwargs.get("threshold", 1e-4),
            min_lr=kwargs.get("min_lr", 1e-6),
            verbose=kwargs.get("verbose", True),
        )
        return sched, "plateau"

    if name in {"step", "steplr"}:
        sched = StepLR(optimizer, step_size=kwargs.get("step_size", 10), gamma=kwargs.get("gamma", 0.1))
        return sched, "epoch"

    if name in {"multistep", "multisteplr"}:
        milestones = kwargs.get("milestones", [30, 60, 90])
        gamma = kwargs.get("gamma", 0.1)
        sched = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        return sched, "epoch"

    if name in {"exp", "exponential"}:
        sched = ExponentialLR(optimizer, gamma=kwargs.get("gamma", 0.95))
        return sched, "epoch"

    # default: no scheduler
    return None, "none"


def step_scheduler(scheduler, mode: str, *, val_loss: Optional[float] = None):
    if scheduler is None or mode == "none":
        return
    if mode == "batch":
        scheduler.step()
    elif mode == "epoch":
        scheduler.step()
    elif mode == "plateau":
        if val_loss is None:
            raise ValueError("ReduceLROnPlateau requires val_loss")
        scheduler.step(val_loss)
