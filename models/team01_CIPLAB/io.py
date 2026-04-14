from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

try:
    from .step1 import hat_backend
except ImportError:
    from step1 import hat_backend


TEAM_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEAM_DIR.parent.parent
TEAM_MODEL_ZOO_DIR = REPO_ROOT / "model_zoo" / "team01_CIPLAB"
INFERENCE_DIR = TEAM_DIR / "step2"
TMP_ROOT = REPO_ROOT / "tmp"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
LORA_ARTIFACT_NAMES = (
    "pytorch_lora_weights.safetensors",
    "pytorch_lora_weights.bin",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_config.json",
)

BASE_MODEL_DIRNAME = "flux2-klein-base-9b"
DEFAULT_PROMPT_NAME = "test_3"
DEFAULT_TILE_SIZE_PX = 1024
DEFAULT_TILE_OVERLAP_PX = 512
DEFAULT_TILE_BATCH_SIZE = 3
DEFAULT_GUIDANCE_SCALE = 4.0
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_DTYPE = "bf16"
DEFAULT_SEED = 0
DEFAULT_CPU_OFFLOAD = True
PREFERRED_RUN_ROOT_NAMES = ("model2_aesop",)

STAGE_SPECS = (
    {
        "label": "stage1(pixel)",
        "adapter_name": "pix",
        "aliases": ("stage1", "pixel", "pix"),
        "env_vars": ("CIPLAB_STAGE1_PATH", "CIPLAB_PIX_LORA_PATH"),
        "required": True,
    },
    {
        "label": "stage2(sem)",
        "adapter_name": "sem",
        "aliases": ("stage2", "sem"),
        "env_vars": ("CIPLAB_STAGE2_PATH", "CIPLAB_SEM_LORA_PATH"),
        "required": True,
    },
    {
        "label": "stage3(sem2)",
        "adapter_name": "sem2",
        "aliases": ("stage3", "sem2"),
        "env_vars": ("CIPLAB_STAGE3_PATH", "CIPLAB_SEM2_LORA_PATH"),
        "required": False,
    },
)


def _iter_existing_dirs(paths):
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if not resolved.exists() or not resolved.is_dir():
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        yield resolved


def _has_image_files(path: Path) -> bool:
    return any(
        item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        for item in sorted(path.iterdir())
    )


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _resolve_override_path(raw_path: str, label: str) -> Path:
    resolved = _resolve_repo_path(raw_path)
    if resolved.exists():
        return resolved
    raise FileNotFoundError(f"Could not resolve {label}: {raw_path}. Tried: {resolved}")


def _contains_lora_artifacts(path: Path) -> bool:
    return any((path / name).exists() for name in LORA_ARTIFACT_NAMES)


def _checkpoint_sort_key(path: Path):
    suffix = path.name.split("checkpoint-", 1)[-1]
    try:
        return (1, int(suffix))
    except ValueError:
        return (0, path.name)


def _resolve_lora_checkpoint(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} path does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"{label} path must be a directory: {resolved}")

    if resolved.name.startswith("checkpoint-") and _contains_lora_artifacts(resolved):
        return resolved

    checkpoints = sorted(
        (
            child.resolve()
            for child in resolved.iterdir()
            if child.is_dir() and child.name.startswith("checkpoint-")
        ),
        key=_checkpoint_sort_key,
    )
    if checkpoints:
        return checkpoints[-1]

    if _contains_lora_artifacts(resolved):
        return resolved

    raise FileNotFoundError(f"Could not find a LoRA checkpoint for {label} under: {resolved}")


def _is_base_model_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "model_index.json").exists()
        or (path / "transformer").is_dir()
        or (path / "vae").is_dir()
    )


def _discover_base_model_path() -> Path:
    for env_name in ("CIPLAB_BASE_MODEL_PATH", "CIPLAB_PRETRAINED_MODEL_PATH"):
        raw_path = os.environ.get(env_name)
        if not raw_path:
            continue
        candidate = _resolve_override_path(raw_path, "base model")
        if not _is_base_model_dir(candidate):
            raise FileNotFoundError(
                f"Resolved base model path does not look like a diffusers checkpoint: {candidate}"
            )
        return candidate

    direct_candidates = [
        TEAM_MODEL_ZOO_DIR / BASE_MODEL_DIRNAME,
        REPO_ROOT / "model_zoo" / BASE_MODEL_DIRNAME,
    ]
    for candidate in direct_candidates:
        if _is_base_model_dir(candidate):
            return candidate.resolve()

    for search_root in _iter_existing_dirs((TEAM_MODEL_ZOO_DIR, REPO_ROOT / "model_zoo")):
        for candidate in sorted(search_root.rglob(BASE_MODEL_DIRNAME)):
            if _is_base_model_dir(candidate):
                return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find `{BASE_MODEL_DIRNAME}` under `{TEAM_MODEL_ZOO_DIR}`. "
        "Set `CIPLAB_BASE_MODEL_PATH` if your layout is different."
    )


def _stage_candidates_from_root(root: Path, stage_spec: dict) -> list[Path]:
    candidates = []
    for alias in stage_spec["aliases"]:
        candidates.extend(
            (
                root / "step2" / alias,
                root / "step2_weight" / alias,
                root / alias,
                root / "train" / alias,
                root / "loras" / alias,
                root / "lora" / alias,
                root / "checkpoints" / alias,
            )
        )
    return list(_iter_existing_dirs(candidates))


def _resolve_stage_from_root(root: Path, stage_spec: dict) -> Path:
    for candidate in _stage_candidates_from_root(root, stage_spec):
        try:
            return _resolve_lora_checkpoint(candidate, stage_spec["label"])
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Could not find {stage_spec['label']} under: {root}")


def _try_resolve_stage_from_root(root: Path, stage_spec: dict):
    try:
        return _resolve_stage_from_root(root, stage_spec)
    except FileNotFoundError:
        return None


def _candidate_run_roots():
    roots = [(TEAM_MODEL_ZOO_DIR, False)]

    child_roots = []
    if TEAM_MODEL_ZOO_DIR.is_dir():
        for child in TEAM_MODEL_ZOO_DIR.iterdir():
            if not child.is_dir():
                continue
            if child.name == BASE_MODEL_DIRNAME or child.name.startswith("."):
                continue
            child_roots.append(child.resolve())
    preferred_names = {name: index for index, name in enumerate(PREFERRED_RUN_ROOT_NAMES)}
    child_roots.sort(
        key=lambda path: (
            0 if path.name in preferred_names else 1,
            preferred_names.get(path.name, len(preferred_names)),
            -path.stat().st_mtime,
            path.name,
        )
    )
    roots.extend((path, True) for path in child_roots)

    repo_train_dir = REPO_ROOT / "train"
    if repo_train_dir.is_dir():
        roots.append((repo_train_dir.resolve(), False))

    return roots


def _discover_run_root() -> Path:
    for env_name in ("CIPLAB_RUN_DIR", "CIPLAB_MODEL_RUN_DIR"):
        raw_path = os.environ.get(env_name)
        if not raw_path:
            continue
        candidate = _resolve_override_path(raw_path, "run root")
        if not candidate.is_dir():
            raise NotADirectoryError(f"Resolved run root is not a directory: {candidate}")
        missing_required = [
            stage_spec["label"]
            for stage_spec in STAGE_SPECS
            if stage_spec["required"] and _try_resolve_stage_from_root(candidate, stage_spec) is None
        ]
        if missing_required:
            raise FileNotFoundError(
                f"Resolved run root is missing required stages {missing_required}: {candidate}"
            )
        return candidate

    checked_roots = []
    for root, _ in _candidate_run_roots():
        checked_roots.append(str(root))
        missing_required = [
            stage_spec["label"]
            for stage_spec in STAGE_SPECS
            if stage_spec["required"] and _try_resolve_stage_from_root(root, stage_spec) is None
        ]
        if not missing_required:
            return root

    checked = ", ".join(checked_roots) if checked_roots else "<none>"
    raise FileNotFoundError(
        "Could not find a single run directory containing required stage1/stage2 LoRA weights. "
        f"Checked: {checked}. Set `CIPLAB_RUN_DIR` if needed."
    )


def _discover_stage_paths(run_root: Path) -> dict[str, Path]:
    stage_paths = {}
    for stage_spec in STAGE_SPECS:
        stage_path = None
        for env_name in stage_spec["env_vars"]:
            raw_path = os.environ.get(env_name)
            if raw_path:
                stage_path = _resolve_lora_checkpoint(
                    _resolve_override_path(raw_path, stage_spec["label"]),
                    stage_spec["label"],
                )
                break

        if stage_path is None:
            if stage_spec["required"]:
                stage_path = _resolve_stage_from_root(run_root, stage_spec)
            else:
                stage_path = _try_resolve_stage_from_root(run_root, stage_spec)

        if stage_path is not None:
            stage_paths[stage_spec["adapter_name"]] = stage_path

    return stage_paths


def _make_temp_json_path(prefix: str) -> Path:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=".json", dir=str(TMP_ROOT))
    os.close(fd)
    return Path(temp_path)


def _load_prompt_text(prompt_name: str) -> str:
    prompts_path = (INFERENCE_DIR / "prompts.json").resolve()
    if not prompts_path.is_file():
        raise FileNotFoundError(f"Missing prompts.json: {prompts_path}")

    with prompts_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"prompts.json must be a list: {prompts_path}")

    for entry in data:
        if not isinstance(entry, dict):
            continue
        if entry.get("name") != prompt_name:
            continue
        prompt = entry.get("prompt")
        if prompt is None or not str(prompt).strip():
            raise ValueError(f"Prompt `{prompt_name}` has an empty `prompt` field in {prompts_path}")
        return str(prompt).strip()

    raise KeyError(f"Prompt `{prompt_name}` not found in {prompts_path}")


def _resolve_runtime(output_path: str) -> dict:
    output_dir = _resolve_repo_path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root = _discover_run_root()
    stage_paths = _discover_stage_paths(run_root)
    base_model_path = _discover_base_model_path()

    return {
        "output_dir": output_dir,
        "run_root": run_root,
        "base_model_path": base_model_path,
        "stage_paths": stage_paths,
    }

def _print_launch_summary(input_dir: Path, sample_count: int, runtime: dict) -> None:
    stage_paths = runtime["stage_paths"]
    sem2_path = stage_paths.get("sem2")
    print("[team01_CIPLAB] Launch configuration", flush=True)
    print(f"  input_dir            : {input_dir}", flush=True)
    print(f"  output_dir           : {runtime['output_dir']}", flush=True)
    print(f"  num_samples          : {sample_count}", flush=True)
    print(f"  base_model           : {runtime['base_model_path']}", flush=True)
    print(f"  run_root             : {runtime['run_root']}", flush=True)
    print(f"  stage1_pix           : {stage_paths['pix']}", flush=True)
    print(f"  stage2_sem           : {stage_paths['sem']}", flush=True)
    print(f"  stage3_sem2          : {sem2_path if sem2_path else '<disabled>'}", flush=True)
    print(f"  prompt_name          : {DEFAULT_PROMPT_NAME}", flush=True)
    print(
        f"  tile                 : size={DEFAULT_TILE_SIZE_PX}, overlap={DEFAULT_TILE_OVERLAP_PX}, batch={DEFAULT_TILE_BATCH_SIZE}",
        flush=True,
    )
    print(
        f"  guidance/steps/dtype : {DEFAULT_GUIDANCE_SCALE} / {DEFAULT_NUM_INFERENCE_STEPS} / {DEFAULT_DTYPE}",
        flush=True,
    )
    print(
        f"  cpu_offload          : {'on' if DEFAULT_CPU_OFFLOAD else 'off'}",
        flush=True,
    )


def _run_hat_preprocess(input_path: str, device=None) -> Path:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="ciplab_hat_pre_", dir=str(TMP_ROOT))).resolve()
    print("[team01_CIPLAB] Starting HAT preprocessing stage...", flush=True)
    print(f"  hat_output_dir       : {temp_dir}", flush=True)
    hat_backend.run_from_input_dir(input_path, str(temp_dir), device=device)
    print("[team01_CIPLAB] HAT preprocessing finished.", flush=True)
    return temp_dir


def main(input_path: str, output_path: str, device=None):
    if input_path is None or output_path is None:
        raise ValueError("`input_path` and `output_path` are required.")

    try:
        from .step2 import inference as step2_inference
    except ImportError:
        from step2 import inference as step2_inference

    hat_output_dir = _run_hat_preprocess(input_path, device=device)
    stage2_config_path: Path | None = None

    try:
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        runtime = _resolve_runtime(output_path)
        input_dir = hat_output_dir
        sample_count = len(
            [
                path
                for path in input_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".png"
            ]
        )
        _print_launch_summary(input_dir, sample_count, runtime)
        stage2_config = {
            "instance_prompt": _load_prompt_text(DEFAULT_PROMPT_NAME),
            "pretrained_model_name_or_path": str(runtime["base_model_path"]),
            "resolution": DEFAULT_TILE_SIZE_PX,
            "tile_overlap_px": DEFAULT_TILE_OVERLAP_PX,
            "tile_batch_size": DEFAULT_TILE_BATCH_SIZE,
            "num_inference_steps": DEFAULT_NUM_INFERENCE_STEPS,
            "guidance_scale": DEFAULT_GUIDANCE_SCALE,
            "max_sequence_length": 512,
            "seed": DEFAULT_SEED,
            "adapter_scale": 1.0,
            "cpu_offload": DEFAULT_CPU_OFFLOAD,
            "input_dir": str(input_dir),
            "pix_lora_weights_path": str(runtime["stage_paths"]["pix"]),
            "sem_lora_weights_path": str(runtime["stage_paths"]["sem"]),
            "output_dir": str(runtime["output_dir"]),
        }
        stage2_config_path = _make_temp_json_path("ciplab_stage2_")
        stage2_config_path.write_text(
            json.dumps(stage2_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[team01_CIPLAB] Stage2 config: {stage2_config_path}", flush=True)
        step2_inference.main([str(stage2_config_path)])
    finally:
        if stage2_config_path is not None:
            stage2_config_path.unlink(missing_ok=True)
        keep_hat_output = os.environ.get("CIPLAB_KEEP_HAT_OUTPUT", "").strip().lower() in {"1", "true", "yes", "on"}
        if keep_hat_output:
            print(f"[team01_CIPLAB] Keeping intermediate HAT output: {hat_output_dir}", flush=True)
        else:
            shutil.rmtree(hat_output_dir, ignore_errors=True)


def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run team01_CIPLAB inference with auto-discovered base model and stage LoRAs."
    )
    parser.add_argument("input_path", type=str, help="Image directory or dataset root containing an `LQ` folder.")
    parser.add_argument("output_path", type=str, help="Directory where restored images will be written.")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_cli_args()
    main(cli_args.input_path, cli_args.output_path)
