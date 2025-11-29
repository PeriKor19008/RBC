from src.utils.run_utils import *
from pathlib import Path

# project root = parent of "src"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODELS_DIR = str(Path(__file__).resolve().parents[3] / "outputs" / "models")

def start_run(model, num_epochs, learning_rate, batch_size, layers,
               base_models_dir=_DEFAULT_MODELS_DIR, extra_info=None):
    Path(_DEFAULT_MODELS_DIR).mkdir(parents=True, exist_ok=True)
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