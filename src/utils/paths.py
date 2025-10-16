# src/utils/paths.py
from __future__ import annotations
from pathlib import Path
import os

def _find_project_root(markers=("pyproject.toml", ".git", "src")) -> Path:
    """
    Walk up from this file until we find a folder that looks like the repo root.
    You can add/remove markers to fit your repo.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if any((parent / m).exists() for m in markers):
            return parent
    # Fallback: assume <root>/src/utils/paths.py → root is parents[2]
    return Path(__file__).resolve().parents[2]

# Allow an override via env var (handy in PyCharm Run Config)
_ENV = os.getenv("RBC_ROOT")
PROJECT_ROOT: Path = Path(_ENV).resolve() if _ENV else _find_project_root()

def rel_to_root(p: str | os.PathLike) -> Path:
    """Return an absolute path; if `p` is relative, join it to PROJECT_ROOT."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p

def from_root(*parts: str | os.PathLike) -> Path:
    """Join multiple parts onto PROJECT_ROOT."""
    return PROJECT_ROOT.joinpath(*parts)
