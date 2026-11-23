"""Where base models, datasets and outputs live.

Three environment variables, read at call time (so a job script can set them before launching):
  NBM_MODELS   base models (Qwen-Image-2.1, Wan2.2-TI2V-5B-Diffusers), default ./models
  NBM_DATA     datasets, default ./data
  NBM_OUTPUTS  training runs, evaluations, caches, default ./outputs
Paths in the configs are relative to these roots; an absolute path is used as it is.
"""
import os
from pathlib import Path


def _root(var: str, default: str) -> Path:
    return Path(os.environ.get(var, default)).expanduser()


def models_root() -> Path:
    return _root("NBM_MODELS", "models")


def data_root() -> Path:
    return _root("NBM_DATA", "data")


def outputs_root() -> Path:
    return _root("NBM_OUTPUTS", "outputs")


def _under(root: Path, p: str | os.PathLike) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else root / p


def model_path(p: str | os.PathLike) -> Path:
    """A base model directory, e.g. model_path("Qwen-Image-2.1")."""
    return _under(models_root(), p)


def data_path(p: str | os.PathLike) -> Path:
    """A dataset directory, e.g. data_path("chess/puzzles")."""
    return _under(data_root(), p)


def output_path(p: str | os.PathLike) -> Path:
    """A path under the outputs root, e.g. output_path("runs/chess_image")."""
    return _under(outputs_root(), p)
