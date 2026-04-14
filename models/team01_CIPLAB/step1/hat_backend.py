from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from PIL.ImageOps import exif_transpose

from .hat_arch import HAT


HAT_DIR = Path(__file__).resolve().parent
TEAM_DIR = HAT_DIR.parent
REPO_ROOT = TEAM_DIR.parent.parent

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
DEFAULT_MODEL_KEY = "hat"
DEFAULT_WEIGHT_KEY = "hat_l_srx4_imagenet_pretrain"
DEFAULT_TEST_NAME = "hat_l_srx4_imagenet_pretrain"
DEFAULT_WEIGHT_FILENAME = "HAT-L_SRx4_ImageNet-pretrain.pth"
DEFAULT_WEIGHT_PATH = REPO_ROOT / "model_zoo" / "team01_CIPLAB" / DEFAULT_WEIGHT_FILENAME


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


RESULTS_DIR = _resolve_repo_path(os.environ.get("CIPLAB_HAT_RESULTS_ROOT", str(REPO_ROOT / "tmp")))
DEFAULT_STAGE_ROOT = _resolve_repo_path(
    os.environ.get("CIPLAB_HAT_STAGE_ROOT", str(REPO_ROOT / "tmp" / "team01_ciplab_hat_stage"))
)


@dataclass(frozen=True)
class SampleRef:
    lr_path: Path
    hr_path: str | None
    staged_name: str
    output_name: str


@dataclass(frozen=True)
class StageBundle:
    stage_dir: Path
    lr_dir: Path
    dataset_name: str
    samples: list[SampleRef]


@dataclass(frozen=True)
class HatRuntime:
    opt_path: Path
    weight_path: Path
    network_kwargs: dict
    scale: int
    tile_size: int | None
    tile_pad: int
    window_size: int
    param_key_g: str
    strict_load_g: bool


def _safe_name(name: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in "._-" else "_" for char in name.strip())
    return sanitized[:80] or "team01_hat"


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"JSON not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {path}\n{error}") from error


def _resolve_override_path(raw_path: str, label: str) -> Path:
    resolved = _resolve_repo_path(raw_path)
    if resolved.exists():
        return resolved
    raise FileNotFoundError(f"Could not resolve {label}: {raw_path}. Tried: {resolved}")


def _resolve_existing_input_path(raw_path: str, label: str, base_dir: Path | None = None) -> Path:
    del base_dir
    resolved = _resolve_repo_path(raw_path)
    if resolved.exists():
        return resolved
    raise FileNotFoundError(f"Could not resolve {label}: {raw_path}. Tried: {resolved}")


def _resolve_output_path(raw_path: str, base_dir: Path | None = None) -> Path:
    del base_dir
    return _resolve_repo_path(raw_path)


def _has_image_files(path: Path) -> bool:
    return any(item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS for item in path.iterdir())


def resolve_input_dir(input_path: str) -> Path:
    input_dir = _resolve_repo_path(input_path)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path must be a directory of images: {input_dir}")

    if _has_image_files(input_dir):
        return input_dir

    for child_name in ("LQ", "lq", "LR", "lr", "input", "inputs"):
        child_dir = input_dir / child_name
        if child_dir.is_dir() and _has_image_files(child_dir):
            return child_dir.resolve()

    raise ValueError(
        f"No input images found under: {input_dir}. "
        "Expected images directly in the folder or under an `LQ` subdirectory."
    )


def _list_image_paths(input_dir: Path) -> list[Path]:
    image_paths = [
        item.resolve()
        for item in sorted(input_dir.iterdir())
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_paths:
        raise ValueError(f"No input images found under: {input_dir}")
    return image_paths


def _resolve_weight_path() -> Path:
    for env_name in ("CIPLAB_HAT_WEIGHT_PATH", "CIPLAB_HAT_WEIGHT"):
        raw_path = os.environ.get(env_name)
        if raw_path:
            resolved = _resolve_override_path(raw_path, "HAT weight")
            if resolved.is_file():
                return resolved
            raise FileNotFoundError(f"HAT weight must be a file: {resolved}")

    if DEFAULT_WEIGHT_PATH.is_file():
        return DEFAULT_WEIGHT_PATH.resolve()

    team_model_zoo_dir = DEFAULT_WEIGHT_PATH.parent
    if team_model_zoo_dir.is_dir():
        matches = sorted(team_model_zoo_dir.rglob(DEFAULT_WEIGHT_FILENAME))
        if matches:
            return matches[0].resolve()

    raise FileNotFoundError(
        f"Could not find `{DEFAULT_WEIGHT_FILENAME}` under `{team_model_zoo_dir}`. "
        "Set `CIPLAB_HAT_WEIGHT_PATH` if your layout is different."
    )


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid HAT option file: {path}")
    return data


def resolve_runtime(weight_key: str = DEFAULT_WEIGHT_KEY) -> HatRuntime:
    if weight_key != DEFAULT_WEIGHT_KEY:
        raise ValueError(
            f"Unsupported weight: {weight_key!r}. "
            f"Only `{DEFAULT_WEIGHT_KEY}` is implemented in team01_CIPLAB."
        )

    opt_path = HAT_DIR / "hat_l_srx4_imagenet_pretrain.yml"
    if not opt_path.is_file():
        raise FileNotFoundError(f"Missing HAT option file: {opt_path}")

    opt = _load_yaml(opt_path)
    network_kwargs = dict(opt.get("network_g", {}))
    network_type = network_kwargs.pop("type", None)
    if network_type != "HAT":
        raise ValueError(f"Unsupported network type in {opt_path}: {network_type!r}")

    tile_opt = opt.get("tile", {}) or {}
    path_opt = opt.get("path", {}) or {}

    return HatRuntime(
        opt_path=opt_path.resolve(),
        weight_path=_resolve_weight_path(),
        network_kwargs=network_kwargs,
        scale=int(opt.get("scale", network_kwargs.get("upscale", 1))),
        tile_size=int(tile_opt["tile_size"]) if tile_opt.get("tile_size") else None,
        tile_pad=int(tile_opt.get("tile_pad", 0)),
        window_size=int(network_kwargs.get("window_size", 1)),
        param_key_g=str(path_opt.get("param_key_g", "params_ema")),
        strict_load_g=bool(path_opt.get("strict_load_g", True)),
    )


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _validate_unique_stems(samples: list[SampleRef]) -> None:
    seen: dict[str, Path] = {}
    for sample in samples:
        stem = Path(sample.staged_name).stem
        previous = seen.get(stem)
        if previous is not None and previous != sample.lr_path:
            raise ValueError(
                "Duplicate LR stem detected. HAT outputs are keyed by stem, so these would overwrite "
                f"each other: {previous} and {sample.lr_path}"
            )
        seen[stem] = sample.lr_path


def _make_stage_dir(name_prefix: str) -> Path:
    DEFAULT_STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{_safe_name(name_prefix)}_", dir=str(DEFAULT_STAGE_ROOT))).resolve()


def _build_stage_bundle(samples: list[SampleRef], name_prefix: str) -> StageBundle:
    if not samples:
        raise ValueError("At least one sample is required for HAT inference.")

    _validate_unique_stems(samples)

    stage_dir = _make_stage_dir(name_prefix)
    lr_dir = (stage_dir / "LR").resolve()
    lr_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        _link_or_copy(sample.lr_path, lr_dir / sample.staged_name)

    return StageBundle(
        stage_dir=stage_dir,
        lr_dir=lr_dir,
        dataset_name=stage_dir.name,
        samples=samples,
    )


def build_stage_bundle_from_input_dir(input_path: str, name_prefix: str = DEFAULT_TEST_NAME) -> StageBundle:
    input_dir = resolve_input_dir(input_path)
    image_paths = _list_image_paths(input_dir)
    samples = [
        SampleRef(
            lr_path=image_path,
            hr_path=None,
            staged_name=image_path.name,
            output_name=f"{image_path.stem}.png",
        )
        for image_path in image_paths
    ]
    return _build_stage_bundle(samples, name_prefix)


def build_stage_bundle_from_manifest_json(manifest_path: Path, name_prefix: str) -> StageBundle:
    data = _load_json(manifest_path)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Manifest must be a non-empty list: {manifest_path}")

    samples: list[SampleRef] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry must be an object at index {index}: {manifest_path}")
        if "lr" not in entry:
            raise KeyError(f"Manifest entry is missing `lr` at index {index}: {manifest_path}")

        lr_path = _resolve_existing_input_path(str(entry["lr"]), f"manifest lr[{index}]", manifest_path.parent)
        hr_path = entry.get("hr")
        resolved_hr = None
        if hr_path is not None and str(hr_path).strip():
            resolved_hr = str(
                _resolve_existing_input_path(str(hr_path), f"manifest hr[{index}]", manifest_path.parent)
            )

        samples.append(
            SampleRef(
                lr_path=lr_path,
                hr_path=resolved_hr,
                staged_name=lr_path.name,
                output_name=f"{lr_path.stem}.png",
            )
        )

    return _build_stage_bundle(samples, name_prefix)


def _normalize_device(device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str) and device.strip():
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor


def _save_image_tensor(tensor: torch.Tensor, path: Path) -> None:
    image = tensor.detach().clamp_(0.0, 1.0).cpu().squeeze(0).permute(1, 2, 0).numpy()
    image_uint8 = np.rint(image * 255.0).clip(0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_uint8).save(path)


def _resolve_state_dict(checkpoint, runtime: HatRuntime) -> dict:
    if isinstance(checkpoint, dict):
        for key in (runtime.param_key_g, "params_ema", "params", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise KeyError(
        f"Could not find a valid state dict in `{runtime.weight_path}` using "
        f"`{runtime.param_key_g}`, `params_ema`, `params`, or `state_dict`."
    )


def _strip_state_dict_prefix(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "net_g.", "generator."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def _build_model(runtime: HatRuntime, device: torch.device) -> torch.nn.Module:
    model = HAT(**runtime.network_kwargs)
    checkpoint = torch.load(runtime.weight_path, map_location="cpu")
    state_dict = _strip_state_dict_prefix(_resolve_state_dict(checkpoint, runtime))
    model.load_state_dict(state_dict, strict=runtime.strict_load_g)
    model = model.to(device)
    model.eval()
    return model


def _pre_process(lq: torch.Tensor, runtime: HatRuntime) -> tuple[torch.Tensor, int, int]:
    mod_pad_h = 0
    mod_pad_w = 0
    _, _, height, width = lq.size()
    if runtime.window_size > 1:
        if height % runtime.window_size != 0:
            mod_pad_h = runtime.window_size - height % runtime.window_size
        if width % runtime.window_size != 0:
            mod_pad_w = runtime.window_size - width % runtime.window_size
    padded = torch.nn.functional.pad(lq, (0, mod_pad_w, 0, mod_pad_h), "reflect")
    return padded, mod_pad_h, mod_pad_w


def _forward_image(model: torch.nn.Module, padded: torch.Tensor, runtime: HatRuntime) -> torch.Tensor:
    if runtime.tile_size is None or runtime.tile_size <= 0:
        return model(padded)

    batch, channel, height, width = padded.shape
    output = padded.new_zeros((batch, channel, height * runtime.scale, width * runtime.scale))
    tiles_x = (width + runtime.tile_size - 1) // runtime.tile_size
    tiles_y = (height + runtime.tile_size - 1) // runtime.tile_size

    for y in range(tiles_y):
        for x in range(tiles_x):
            ofs_x = x * runtime.tile_size
            ofs_y = y * runtime.tile_size

            input_start_x = ofs_x
            input_end_x = min(ofs_x + runtime.tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + runtime.tile_size, height)

            input_start_x_pad = max(input_start_x - runtime.tile_pad, 0)
            input_end_x_pad = min(input_end_x + runtime.tile_pad, width)
            input_start_y_pad = max(input_start_y - runtime.tile_pad, 0)
            input_end_y_pad = min(input_end_y + runtime.tile_pad, height)

            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y
            input_tile = padded[
                :,
                :,
                input_start_y_pad:input_end_y_pad,
                input_start_x_pad:input_end_x_pad,
            ]
            output_tile = model(input_tile)

            output_start_x = input_start_x * runtime.scale
            output_end_x = input_end_x * runtime.scale
            output_start_y = input_start_y * runtime.scale
            output_end_y = input_end_y * runtime.scale

            output_start_x_tile = (input_start_x - input_start_x_pad) * runtime.scale
            output_end_x_tile = output_start_x_tile + input_tile_width * runtime.scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * runtime.scale
            output_end_y_tile = output_start_y_tile + input_tile_height * runtime.scale

            output[
                :,
                :,
                output_start_y:output_end_y,
                output_start_x:output_end_x,
            ] = output_tile[
                :,
                :,
                output_start_y_tile:output_end_y_tile,
                output_start_x_tile:output_end_x_tile,
            ]

    return output


def _post_process(output: torch.Tensor, runtime: HatRuntime, mod_pad_h: int, mod_pad_w: int) -> torch.Tensor:
    _, _, height, width = output.size()
    end_h = height - mod_pad_h * runtime.scale if mod_pad_h else height
    end_w = width - mod_pad_w * runtime.scale if mod_pad_w else width
    return output[:, :, 0:end_h, 0:end_w]


def _run_hat_direct(bundle: StageBundle, runtime: HatRuntime, run_name: str, device) -> None:
    device_obj = _normalize_device(device)
    run_results_dir = RESULTS_DIR / run_name
    visualization_dir = run_results_dir / "visualization" / bundle.dataset_name
    visualization_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        model = _build_model(runtime, device_obj)
        for sample in bundle.samples:
            input_tensor = _load_image_tensor(sample.lr_path).to(device_obj)
            padded, mod_pad_h, mod_pad_w = _pre_process(input_tensor, runtime)
            output = _forward_image(model, padded, runtime)
            output = _post_process(output, runtime, mod_pad_h, mod_pad_w)
            raw_path = visualization_dir / f"{Path(sample.staged_name).stem}_x4.png"
            _save_image_tensor(output, raw_path)

            del input_tensor
            del padded
            del output

    if device_obj.type == "cuda":
        torch.cuda.empty_cache()


def _locate_visualization_dir(bundle: StageBundle, run_name: str) -> Path:
    preferred = RESULTS_DIR / run_name / "visualization" / bundle.dataset_name
    if preferred.is_dir():
        return preferred.resolve()

    visualization_root = RESULTS_DIR / run_name / "visualization"
    if visualization_root.is_dir():
        for candidate in sorted(path for path in visualization_root.iterdir() if path.is_dir()):
            missing = [
                sample.staged_name
                for sample in bundle.samples
                if not (candidate / f"{Path(sample.staged_name).stem}_x4.png").is_file()
            ]
            if not missing:
                return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find HAT visualization outputs for run `{run_name}` under {visualization_root}"
    )


def _copy_outputs(bundle: StageBundle, run_name: str, output_dir: Path) -> None:
    raw_dir = _locate_visualization_dir(bundle, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample in bundle.samples:
        raw_file = raw_dir / f"{Path(sample.staged_name).stem}_x4.png"
        if not raw_file.is_file():
            raise FileNotFoundError(f"Expected HAT output not found: {raw_file}")
        shutil.copy2(raw_file, output_dir / sample.output_name)


def write_result_json(bundle: StageBundle, output_dir: Path) -> Path:
    items = []
    for sample in bundle.samples:
        entry = {"res": str((output_dir / sample.output_name).resolve())}
        if sample.hr_path:
            entry["hr"] = sample.hr_path
        items.append(entry)

    items.sort(key=lambda item: Path(item["res"]).name)
    payload = {"items": items}
    result_path = (output_dir / "result.json").resolve()
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result_path


def _print_summary(bundle: StageBundle, output_dir: Path, runtime: HatRuntime, run_name: str, device) -> None:
    device_obj = _normalize_device(device)
    print("[team01_CIPLAB] HAT-L inference", flush=True)
    print(f"  run_name             : {run_name}", flush=True)
    print(f"  input_dir            : {bundle.lr_dir}", flush=True)
    print(f"  output_dir           : {output_dir}", flush=True)
    print(f"  num_samples          : {len(bundle.samples)}", flush=True)
    print(f"  opt                  : {runtime.opt_path}", flush=True)
    print(f"  weight               : {runtime.weight_path}", flush=True)
    print(f"  dataset_name         : {bundle.dataset_name}", flush=True)
    print(f"  device               : {device_obj}", flush=True)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(
        f"  CUDA_VISIBLE_DEVICES : {cuda_visible_devices if cuda_visible_devices else '<inherit>'}",
        flush=True,
    )


def _run_hat(bundle: StageBundle, output_dir: Path, runtime: HatRuntime, run_name: str, device=None) -> None:
    _print_summary(bundle, output_dir, runtime, run_name, device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("  backend              : raw_HAT_direct", flush=True)
    _run_hat_direct(bundle, runtime, run_name, device)
    _copy_outputs(bundle, run_name, output_dir)


def run_from_input_dir(input_path: str, output_path: str, run_name: str = DEFAULT_TEST_NAME, device=None) -> None:
    runtime = resolve_runtime()
    output_dir = _resolve_repo_path(output_path)
    bundle = build_stage_bundle_from_input_dir(input_path, name_prefix=run_name)
    try:
        _run_hat(bundle, output_dir, runtime, run_name, device=device)
    finally:
        shutil.rmtree(bundle.stage_dir, ignore_errors=True)


def run_experiment(exp_path: str, dry_run: bool = False, device=None) -> None:
    exp_file = _resolve_repo_path(exp_path)
    experiment = _load_json(exp_file)
    if not isinstance(experiment, dict):
        raise ValueError(f"Experiment JSON must be an object: {exp_file}")

    data_input = experiment.get("data_input")
    output_path = experiment.get("output_path")
    settings = experiment.get("setting")

    if not isinstance(data_input, str) or not data_input.strip():
        raise ValueError(f"`data_input` must be a non-empty string: {exp_file}")
    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError(f"`output_path` must be a non-empty string: {exp_file}")
    if not isinstance(settings, list) or not settings:
        raise ValueError(f"`setting` must be a non-empty list: {exp_file}")

    if experiment.get("eval_env") or experiment.get("viz_env"):
        print("[team01_CIPLAB] `eval_env` and `viz_env` are ignored in the minimal HAT-only runner.", flush=True)

    manifest_path = _resolve_existing_input_path(data_input, "experiment.data_input", exp_file.parent)
    output_root = _resolve_output_path(output_path, exp_file.parent)
    runtime = resolve_runtime()
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"Manifest must be a non-empty list: {manifest_path}")

    seen_test_names: set[str] = set()

    for index, setting in enumerate(settings):
        if not isinstance(setting, dict):
            raise ValueError(f"experiment.setting[{index}] must be an object.")

        test_name = setting.get("test_name")
        model_key = setting.get("model")
        weight_key = setting.get("weight")
        if not isinstance(test_name, str) or not test_name.strip():
            raise ValueError(f"experiment.setting[{index}].test_name must be a non-empty string.")
        if test_name in seen_test_names:
            raise ValueError(f"Duplicate test_name: {test_name}")
        seen_test_names.add(test_name)

        if model_key != DEFAULT_MODEL_KEY:
            raise ValueError(
                f"Unsupported model: {model_key!r}. Only `{DEFAULT_MODEL_KEY}` is implemented in team01_CIPLAB."
            )
        if weight_key != DEFAULT_WEIGHT_KEY:
            raise ValueError(
                f"Unsupported weight: {weight_key!r}. "
                f"Only `{DEFAULT_WEIGHT_KEY}` is implemented in team01_CIPLAB."
            )

        output_dir = (output_root / test_name).resolve()

        print(f"[team01_CIPLAB] experiment setting: {test_name}", flush=True)
        print(f"  manifest             : {manifest_path}", flush=True)
        print(f"  output_dir           : {output_dir}", flush=True)
        print(f"  num_samples          : {len(manifest)}", flush=True)

        if dry_run:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        bundle = build_stage_bundle_from_manifest_json(manifest_path, name_prefix=test_name)
        try:
            _run_hat(bundle, output_dir, runtime, test_name, device=device)
            result_json = write_result_json(bundle, output_dir)
            print(f"  result_json          : {result_json}", flush=True)
        finally:
            shutil.rmtree(bundle.stage_dir, ignore_errors=True)
