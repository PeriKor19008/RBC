import os
from src.utils.run_utils import *
def start_run(model, num_epochs, learning_rate, batch_size, layers,
               base_models_dir="models", extra_info=None):
    """
    Create a per-run folder grouped by architecture name and return paths.
    """
    arch_name = type(model).__name__
    cfg = RunConfig(
        arch=arch_name,
        epochs=num_epochs,
        lr=learning_rate,
        batch_size=batch_size,
        notes=f"layers={layers}"
    )

    run_id = make_run_id(cfg, ds_fingerprint="manual")
    arch_base = os.path.join(base_models_dir, arch_name)
    run_dir = ensure_run_dir(arch_base, run_id)
    figs_dir = os.path.join(run_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # NEW: store layers + any extra info (like conv_config/fc_config) into config.json
    extra = {"layers": layers}
    if extra_info:
        extra.update(extra_info)
    write_config(run_dir, cfg, ds_fingerprint="manual", extra=extra)

    return run_dir, figs_dir, run_id, arch_name