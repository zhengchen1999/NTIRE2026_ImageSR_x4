from __future__ import annotations

import contextlib
import os
from pathlib import Path


TEAM_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEAM_DIR.parent.parent
DEFAULT_HAT_WEIGHT = "HAT-L_SRx4_ImageNet-pretrain.pth"
DEFAULT_BASE_MODEL_DIR = "flux2-klein-base-9b"


def _resolve_repo_path(path_value) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


@contextlib.contextmanager
def _model_dir_overrides(model_dir):
    updates = {}
    if model_dir:
        model_root = _resolve_repo_path(model_dir)
        if model_root.is_dir():
            hat_weight = model_root / DEFAULT_HAT_WEIGHT
            base_model = model_root / DEFAULT_BASE_MODEL_DIR
            if hat_weight.is_file():
                updates["CIPLAB_HAT_WEIGHT_PATH"] = str(hat_weight)
            if base_model.is_dir():
                updates["CIPLAB_BASE_MODEL_PATH"] = str(base_model)
            updates["CIPLAB_RUN_DIR"] = str(model_root)

    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def main(model_dir, input_path: str, output_path: str, device) -> None:
    from .io import main as run_pipeline

    with _model_dir_overrides(model_dir):
        run_pipeline(input_path, output_path, device=device)
