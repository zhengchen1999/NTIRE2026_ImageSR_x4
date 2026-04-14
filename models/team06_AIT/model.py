import glob
import json
import os
import shutil
import sys
from argparse import Namespace


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _str2bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_config(model_dir):
    candidates = [
        os.path.join(os.path.dirname(__file__), "team06_ait_model.json"),
        os.path.join(model_dir, "team06_ait.json"),
    ]
    for cfg_path in candidates:
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def _resolve_inference_root(model_dir, cfg):
    repo_root = _repo_root()
    cfg_inference_root = cfg.get("inference_root")
    cfg_candidates = []
    if cfg_inference_root:
        cfg_candidates = [
            cfg_inference_root,
            os.path.join(model_dir, cfg_inference_root),
            os.path.join(repo_root, cfg_inference_root),
        ]
    candidates = [
        os.environ.get("SEESR_INFER_ROOT"),
        *cfg_candidates,
        os.path.join(model_dir, "ait_backend"),
        os.path.join(repo_root, "models/team06_AIT/ait_backend"),
        os.path.join(repo_root, "inference_only"),
        os.path.join(model_dir, "inference_only"),
    ]
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, "test_seesr.py")):
            return path
    raise FileNotFoundError(
        "Cannot find inference_only/test_seesr.py. "
        "Please set SEESR_INFER_ROOT or put inference_only under model_dir."
    )


def _env_or_cfg(env_key, cfg, cfg_key, default=None):
    if os.environ.get(env_key) is not None:
        return os.environ[env_key]
    if cfg_key in cfg:
        return cfg[cfg_key]
    return default


def _to_abs(path, base_dir):
    if path is None:
        return None
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base_dir, path))


def _resolve_ram_pretrained_path(model_dir, cfg, inference_root, repo_root):
    cfg_path = _env_or_cfg("SEESR_RAM_PRETRAINED_PATH", cfg, "ram_pretrained_path", None)
    candidates = []

    if cfg_path:
        candidates.extend(
            [
                cfg_path,
                os.path.join(model_dir, cfg_path),
                os.path.join(repo_root, cfg_path),
            ]
        )

    # Backward-compatible defaults used in upstream SeeSR layouts.
    candidates.extend(
        [
            os.path.join(inference_root, "present/models/ram_swin_large_14m.pth"),
            os.path.join(repo_root, "present/models/ram_swin_large_14m.pth"),
            os.path.join(model_dir, "ram_swin_large_14m.pth"),
            os.path.join(repo_root, "model_zoo/team06_AIT/ram_swin_large_14m.pth"),
        ]
    )

    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)

    raise FileNotFoundError(
        "Cannot find RAM pretrained weight 'ram_swin_large_14m.pth'. "
        "Please set SEESR_RAM_PRETRAINED_PATH or add 'ram_pretrained_path' in team06_ait_model.json."
    )


def _cleanup_intermediate_dirs(output_path):
    for pattern in ("sample*", "txt", "_seesr_tmp"):
        for path in glob.glob(os.path.join(output_path, pattern)):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)


def _flatten_png_outputs(output_path):
    src_files = sorted(glob.glob(os.path.join(output_path, "**", "*.png"), recursive=True))
    if not src_files:
        raise RuntimeError("SeeSR produced no PNG outputs.")

    for src in src_files:
        dst = os.path.join(output_path, os.path.basename(src))
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)

    _cleanup_intermediate_dirs(output_path)


def main(model_dir, input_path, output_path, device):
    """
    NTIRE required interface:
      main(model_dir, input_path, output_path, device)
    """
    caller_cwd = os.getcwd()
    model_dir = _to_abs(model_dir, caller_cwd)
    input_path = _to_abs(input_path, caller_cwd)
    output_path = _to_abs(output_path, caller_cwd)

    cfg = _load_config(model_dir)
    repo_root = _repo_root()
    inference_root = _resolve_inference_root(model_dir, cfg)

    if inference_root not in sys.path:
        sys.path.insert(0, inference_root)

    from test_seesr import main as seesr_main

    os.makedirs(output_path, exist_ok=True)

    pretrained_model_path = _env_or_cfg(
        "SEESR_PRETRAINED_MODEL_PATH",
        cfg,
        "pretrained_model_path",
        os.path.join(model_dir, "stable-diffusion-2-base"),
    )
    if pretrained_model_path and not os.path.isabs(pretrained_model_path):
        pretrained_model_path = os.path.join(repo_root, pretrained_model_path)
    seesr_model_path = _env_or_cfg(
        "SEESR_MODEL_PATH",
        cfg,
        "seesr_model_path",
        model_dir,
    )
    if seesr_model_path and not os.path.isabs(seesr_model_path):
        seesr_model_path = os.path.join(repo_root, seesr_model_path)
    ram_ft_path = _env_or_cfg("SEESR_RAM_FT_PATH", cfg, "ram_ft_path", None)
    ram_pretrained_path = _resolve_ram_pretrained_path(model_dir, cfg, inference_root, repo_root)

    adapter_path = _env_or_cfg(
        "SEESR_DEGRADATION_ADAPTER_PATH",
        cfg,
        "degradation_token_adapter_path",
        os.path.join(seesr_model_path, "degradation_token_adapter.bin"),
    )
    use_degradation_token = _str2bool(
        _env_or_cfg(
            "SEESR_USE_DEGRADATION_TOKEN",
            cfg,
            "use_degradation_token",
            os.path.isfile(adapter_path),
        )
    )
    use_dynamic_degradation_token = _str2bool(
        _env_or_cfg(
            "SEESR_USE_DYNAMIC_DEGRADATION_TOKEN",
            cfg,
            "use_dynamic_degradation_token",
            True,
        )
    )

    # Keep final outputs flat under output_path and remove SeeSR temp folders.
    _cleanup_intermediate_dirs(output_path)

    args = Namespace(
        seesr_model_path=seesr_model_path,
        ram_ft_path=ram_ft_path,
        ram_pretrained_path=ram_pretrained_path,
        pretrained_model_path=pretrained_model_path,
        prompt=_env_or_cfg("SEESR_PROMPT", cfg, "prompt", ""),
        added_prompt=_env_or_cfg(
            "SEESR_ADDED_PROMPT", cfg, "added_prompt", "clean, high-resolution, 8k"
        ),
        negative_prompt=_env_or_cfg(
            "SEESR_NEGATIVE_PROMPT",
            cfg,
            "negative_prompt",
            "dotted, noise, blur, lowres, smooth",
        ),
        image_path=input_path,
        output_dir=output_path,
        mixed_precision=_env_or_cfg("SEESR_MIXED_PRECISION", cfg, "mixed_precision", "fp16"),
        guidance_scale=float(_env_or_cfg("SEESR_GUIDANCE_SCALE", cfg, "guidance_scale", 4.0)),
        conditioning_scale=float(
            _env_or_cfg("SEESR_CONDITIONING_SCALE", cfg, "conditioning_scale", 1.0)
        ),
        blending_alpha=float(_env_or_cfg("SEESR_BLENDING_ALPHA", cfg, "blending_alpha", 1.0)),
        num_inference_steps=int(
            _env_or_cfg("SEESR_NUM_INFERENCE_STEPS", cfg, "num_inference_steps", 50)
        ),
        process_size=int(_env_or_cfg("SEESR_PROCESS_SIZE", cfg, "process_size", 512)),
        vae_decoder_tiled_size=int(
            _env_or_cfg("SEESR_VAE_DECODER_TILED_SIZE", cfg, "vae_decoder_tiled_size", 128)
        ),
        vae_encoder_tiled_size=int(
            _env_or_cfg("SEESR_VAE_ENCODER_TILED_SIZE", cfg, "vae_encoder_tiled_size", 1024)
        ),
        latent_tiled_size=int(
            _env_or_cfg("SEESR_LATENT_TILED_SIZE", cfg, "latent_tiled_size", 64)
        ),
        latent_tiled_overlap=int(
            _env_or_cfg("SEESR_LATENT_TILED_OVERLAP", cfg, "latent_tiled_overlap", 4)
        ),
        upscale=int(_env_or_cfg("SEESR_UPSCALE", cfg, "upscale", 4)),
        seed=cfg.get("seed", None),
        sample_times=int(_env_or_cfg("SEESR_SAMPLE_TIMES", cfg, "sample_times", 1)),
        align_method=_env_or_cfg("SEESR_ALIGN_METHOD", cfg, "align_method", "adain"),
        start_steps=int(_env_or_cfg("SEESR_START_STEPS", cfg, "start_steps", 999)),
        start_point=_env_or_cfg("SEESR_START_POINT", cfg, "start_point", "lr"),
        save_prompts=_str2bool(_env_or_cfg("SEESR_SAVE_PROMPTS", cfg, "save_prompts", False)),
        use_degradation_token=use_degradation_token,
        degradation_token_adapter_path=adapter_path if use_degradation_token else None,
        degradation_feat_dim=int(
            _env_or_cfg("SEESR_DEGRADATION_FEAT_DIM", cfg, "degradation_feat_dim", 6)
        ),
        degradation_token_dim=int(
            _env_or_cfg("SEESR_DEGRADATION_TOKEN_DIM", cfg, "degradation_token_dim", 512)
        ),
        degradation_token_dropout=float(
            _env_or_cfg(
                "SEESR_DEGRADATION_TOKEN_DROPOUT",
                cfg,
                "degradation_token_dropout",
                0.1,
            )
        ),
        use_dynamic_degradation_token=use_dynamic_degradation_token,
        degradation_token_timestep_dim=int(
            _env_or_cfg(
                "SEESR_DEGRADATION_TOKEN_TIMESTEP_DIM",
                cfg,
                "degradation_token_timestep_dim",
                128,
            )
        ),
    )

    old_cwd = os.getcwd()
    try:
        # Some SeeSR internal paths are relative (e.g., "present/models/...").
        # Run from inference_root to keep original path assumptions valid.
        os.chdir(inference_root)
        seesr_main(args)
    finally:
        os.chdir(old_cwd)

    _flatten_png_outputs(output_path)

