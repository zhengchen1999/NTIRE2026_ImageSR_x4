#!/usr/bin/env python3
# coding=utf-8
"""
Flux.2 Klein tiled LoRA inference with SUPIR-style fixed Gaussian overlap blending,
using minimal reflect padding and crop-back to preserve the input resolution.

Key points:
- preserve the original output resolution exactly
- only pad to the next latent-compatible multiple when needed
- overlap blending uses SUPIR-style fixed gaussian weights (var=0.01)
- tile_batch_size is supported for throughput
- transformer guidance input is kept as None to match Flux2 Klein __call__ semantics

Usage:
  python3 -m models.team01_CIPLAB.step2.inference /path/to/config.json
"""

from __future__ import annotations

import inspect
import json
import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import Flux2KleinPipeline
from PIL import Image
from PIL.ImageOps import exif_transpose
from tqdm.auto import tqdm
from .colorfix import adain_color_fix


DTYPE = torch.bfloat16


def absolute_path(path_value: str, field_name: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path: {path_value}")
    return path.resolve()


def load_config(config_path: str):
    path = absolute_path(config_path, "config_path")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    config["_config_path"] = str(path)
    return config


def require(config: dict, key: str):
    value = config.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required config key: {key}")
    return value


def get_config_value(config: dict, key: str, default):
    return config[key] if key in config else default


def load_input_images(input_dir: str):
    directory = absolute_path(input_dir, "input_dir")
    if not directory.is_dir():
        raise ValueError(f"input_dir must be a directory: {directory}")

    image_paths = sorted(path for path in directory.glob("*.png") if path.is_file())
    if not image_paths:
        raise ValueError(f"No PNG images found in {directory}")
    return image_paths


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image.copy()


def make_generator(seed: int | None, index: int):
    if seed is None or seed < 0:
        return None
    return torch.Generator(device="cuda").manual_seed(seed + index)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def retrieve_timesteps(scheduler, num_inference_steps: int, device: torch.device, sigmas, mu: float):
    if sigmas is not None:
        accepts_sigmas = "sigmas" in inspect.signature(scheduler.set_timesteps).parameters
        if not accepts_sigmas:
            raise ValueError(f"{scheduler.__class__.__name__} does not support custom sigmas")
        scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
    return scheduler.timesteps


@dataclass
class TileCoord:
    y0: int
    y1: int
    x0: int
    x1: int


@dataclass
class PreparedCondition:
    tensors: list[torch.Tensor]
    output_height: int
    output_width: int


def make_starts(total: int, tile: int, overlap: int):
    if tile <= overlap:
        raise ValueError("tile size must be larger than overlap")
    if total <= tile:
        return [0]

    stride = tile - overlap
    starts = []
    position = 0
    while position + tile < total:
        starts.append(position)
        position += stride
    starts.append(total - tile)
    return sorted(set(starts))


def make_tile_coords(height: int, width: int, tile_h: int, tile_w: int, overlap_h: int, overlap_w: int):
    return [
        TileCoord(y0=y0, y1=min(y0 + tile_h, height), x0=x0, x1=min(x0 + tile_w, width))
        for y0 in make_starts(height, tile_h, overlap_h)
        for x0 in make_starts(width, tile_w, overlap_w)
    ]


def make_supir_gaussian_weight(tile_h: int, tile_w: int, channels: int, device: torch.device):
    """Match SUPIR's fixed gaussian_weights(..., var=0.01) behavior."""
    var = 0.01
    ys = torch.arange(tile_h, device=device, dtype=torch.float32)
    xs = torch.arange(tile_w, device=device, dtype=torch.float32)
    center_y = tile_h / 2.0
    center_x = tile_w / 2.0
    gy = torch.exp(-((ys - center_y) ** 2) / float(tile_h * tile_h) / (2.0 * var))
    gx = torch.exp(-((xs - center_x) ** 2) / float(tile_w * tile_w) / (2.0 * var))
    weight = gy[:, None] * gx[None, :]
    return weight[None, None].repeat(1, channels, 1, 1)


def offset_ids(ids: torch.Tensor, coord: TileCoord):
    shifted = ids.clone()
    shifted[..., 1] += coord.y0
    shifted[..., 2] += coord.x0
    return shifted


def build_condition_ids(pipe: Flux2KleinPipeline, cond_tile_map: torch.Tensor, coord: TileCoord):
    cond_ids = pipe._prepare_image_ids([cond_tile_map[0].unsqueeze(0)]).to(device=cond_tile_map.device)
    cond_ids = cond_ids.view(cond_tile_map.shape[0], -1, cond_ids.shape[-1])
    return offset_ids(cond_ids, coord)


def cache_prompt_embeddings(pipe: Flux2KleinPipeline, prompt: str, max_sequence_length: int, guidance_scale: float):
    device = pipe._execution_device
    prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        prompt_embeds=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )

    negative_prompt_embeds = None
    negative_text_ids = None
    do_cfg = (guidance_scale > 1.0) and (not pipe.config.is_distilled)
    if do_cfg:
        negative_prompt_embeds, negative_text_ids = pipe.encode_prompt(
            prompt="",
            prompt_embeds=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
    return prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids


class TiledInferenceRunner:
    def __init__(self, pipe: Flux2KleinPipeline, resolution: int, tile_overlap_px: int, tile_batch_size: int):
        self.pipe = pipe
        self.resolution = resolution
        self.tile_overlap_px = tile_overlap_px
        self.tile_batch_size = tile_batch_size

    def _tile_config_from_latent_shape(self, latent_h: int, latent_w: int):
        multiple = self.pipe.vae_scale_factor * 2
        if self.resolution % multiple != 0:
            raise ValueError(f"resolution ({self.resolution}) must be divisible by {multiple}")
        if self.tile_overlap_px < 0:
            raise ValueError("tile_overlap_px must be non-negative")
        if self.tile_overlap_px % multiple != 0:
            raise ValueError(f"tile_overlap_px ({self.tile_overlap_px}) must be divisible by {multiple}")

        tile_h = self.resolution // multiple
        tile_w = self.resolution // multiple
        overlap_h = self.tile_overlap_px // multiple
        overlap_w = self.tile_overlap_px // multiple

        if overlap_h >= tile_h or overlap_w >= tile_w:
            raise ValueError("tile_overlap_px is too large for resolution")
        if tile_h <= 0 or tile_w <= 0:
            raise ValueError("resolution is too small for the current VAE packing factor")
        if latent_h <= 0 or latent_w <= 0:
            raise ValueError("latent spatial shape must be positive")

        return {
            "tile_h": tile_h,
            "tile_w": tile_w,
            "overlap_h": overlap_h,
            "overlap_w": overlap_w,
        }

    def _prepare_condition(self, image: Image.Image) -> PreparedCondition:
        self.pipe.image_processor.check_image_input(image)

        output_width, output_height = image.size
        multiple_of = self.pipe.vae_scale_factor * 2

        tensor = self.pipe.image_processor.preprocess(
            image,
            height=output_height,
            width=output_width,
            resize_mode="crop",
        )

        tensor_height = tensor.shape[-2]
        tensor_width = tensor.shape[-1]
        render_width = math.ceil(tensor_width / multiple_of) * multiple_of
        render_height = math.ceil(tensor_height / multiple_of) * multiple_of
        pad_right = render_width - tensor_width
        pad_bottom = render_height - tensor_height
        if pad_right or pad_bottom:
            tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode="reflect")

        return PreparedCondition(
            tensors=[tensor],
            output_height=output_height,
            output_width=output_width,
        )

    def _joint_attention_kwargs(self) -> dict | None:
        if hasattr(self.pipe, "attention_kwargs"):
            return getattr(self.pipe, "attention_kwargs")
        return getattr(self.pipe, "_attention_kwargs", None)

    def _run_transformer(
        self,
        model_input: torch.Tensor,
        timestep_batch: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor,
        joint_attention_kwargs: dict | None,
        cache_context_name: str | None = None,
    ):
        cache_ctx = (
            self.pipe.transformer.cache_context(cache_context_name)
            if cache_context_name is not None and hasattr(self.pipe.transformer, "cache_context")
            else nullcontext()
        )
        with cache_ctx:
            return self.pipe.transformer(
                hidden_states=model_input,
                timestep=timestep_batch / 1000,
                guidance=None,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
            )[0]

    def _predict_noise_tiled(
        self,
        latents_map: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        image_latents_map: torch.Tensor,
        guidance_scale: float,
        negative_prompt_embeds: torch.Tensor | None,
        negative_text_ids: torch.Tensor | None,
        tile_cfg,
    ):
        device = latents_map.device
        channels = latents_map.shape[1]
        coords = make_tile_coords(
            latents_map.shape[-2],
            latents_map.shape[-1],
            tile_cfg["tile_h"],
            tile_cfg["tile_w"],
            tile_cfg["overlap_h"],
            tile_cfg["overlap_w"],
        )

        # Keep overlap accumulation in fp32 so very small corner weights are not distorted.
        noise_accum = torch.zeros_like(latents_map, dtype=torch.float32)
        weight_accum = torch.zeros_like(latents_map, dtype=torch.float32)
        weight_cache = {}
        do_cfg = (guidance_scale > 1.0) and (not self.pipe.config.is_distilled)

        for start in range(0, len(coords), self.tile_batch_size):
            batch_coords = coords[start : start + self.tile_batch_size]
            packed_tiles = []
            tile_ids_local = []
            tile_ids_abs = []
            cond_tiles = []
            cond_ids_abs = []

            for coord in batch_coords:
                tile_map = latents_map[:, :, coord.y0:coord.y1, coord.x0:coord.x1]
                packed_tiles.append(self.pipe._pack_latents(tile_map))
                local_ids = self.pipe._prepare_latent_ids(tile_map).to(tile_map.device)
                tile_ids_local.append(local_ids)
                tile_ids_abs.append(offset_ids(local_ids, coord))

                cond_tile_map = image_latents_map[:, :, coord.y0:coord.y1, coord.x0:coord.x1]
                cond_tiles.append(self.pipe._pack_latents(cond_tile_map))
                cond_ids_abs.append(build_condition_ids(self.pipe, cond_tile_map, coord))

            tile_latents_packed = torch.cat(packed_tiles, dim=0)
            tile_local_ids = torch.cat(tile_ids_local, dim=0)
            tile_abs_ids = torch.cat(tile_ids_abs, dim=0)
            cond_latents_packed = torch.cat(cond_tiles, dim=0)
            cond_abs_ids = torch.cat(cond_ids_abs, dim=0)

            model_input = torch.cat(
                [tile_latents_packed, cond_latents_packed.to(tile_latents_packed.dtype)],
                dim=1,
            ).to(self.pipe.transformer.dtype)
            image_ids = torch.cat([tile_abs_ids, cond_abs_ids], dim=1)
            timestep_batch = timestep.expand(model_input.shape[0]).to(model_input.dtype)

            prompt_rep = prompt_embeds.repeat(len(batch_coords), 1, 1)
            text_ids_rep = text_ids.repeat(len(batch_coords), 1, 1)

            noise_pred = self._run_transformer(
                model_input=model_input,
                timestep_batch=timestep_batch,
                encoder_hidden_states=prompt_rep,
                txt_ids=text_ids_rep,
                img_ids=image_ids,
                joint_attention_kwargs=self._joint_attention_kwargs(),
                cache_context_name="cond",
            )
            noise_pred = noise_pred[:, : tile_latents_packed.shape[1], :]

            if do_cfg:
                neg_prompt_rep = negative_prompt_embeds.repeat(len(batch_coords), 1, 1)
                neg_text_ids_rep = negative_text_ids.repeat(len(batch_coords), 1, 1)
                neg_noise_pred = self._run_transformer(
                    model_input=model_input,
                    timestep_batch=timestep_batch,
                    encoder_hidden_states=neg_prompt_rep,
                    txt_ids=neg_text_ids_rep,
                    img_ids=image_ids,
                    joint_attention_kwargs=self._joint_attention_kwargs(),
                    cache_context_name="uncond",
                )
                neg_noise_pred = neg_noise_pred[:, : tile_latents_packed.shape[1], :]
                noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

            for pred_chunk, local_ids_chunk, coord in zip(
                noise_pred.chunk(len(batch_coords), dim=0),
                tile_local_ids.chunk(len(batch_coords), dim=0),
                batch_coords,
            ):
                local_h = coord.y1 - coord.y0
                local_w = coord.x1 - coord.x0
                weight_key = (local_h, local_w)
                if weight_key not in weight_cache:
                    weight_cache[weight_key] = make_supir_gaussian_weight(
                        local_h,
                        local_w,
                        channels,
                        device,
                    ).to(dtype=torch.float32)
                weight = weight_cache[weight_key]
                noise_tile = self.pipe._unpack_latents_with_ids(pred_chunk, local_ids_chunk)
                noise_accum[:, :, coord.y0:coord.y1, coord.x0:coord.x1] += noise_tile * weight
                weight_accum[:, :, coord.y0:coord.y1, coord.x0:coord.x1] += weight

        if torch.any(weight_accum <= 0):
            raise RuntimeError("Encountered zero tile weight during tiled blending")

        return (noise_accum / weight_accum).to(latents_map.dtype)

    @torch.no_grad()
    def generate(self, image: Image.Image, prompt_cache, num_inference_steps: int, guidance_scale: float, generator):
        prepared_condition = self._prepare_condition(image)
        condition_tensors = prepared_condition.tensors
        prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids = prompt_cache
        render_height = condition_tensors[0].shape[-2]
        render_width = condition_tensors[0].shape[-1]

        latents_packed, latent_ids = self.pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=self.pipe.transformer.config.in_channels // 4,
            height=render_height,
            width=render_width,
            dtype=prompt_embeds.dtype,
            device=self.pipe._execution_device,
            generator=generator,
            latents=None,
        )
        condition_latents, condition_ids = self.pipe.prepare_image_latents(
            images=condition_tensors,
            batch_size=1,
            generator=generator,
            device=self.pipe._execution_device,
            dtype=self.pipe.vae.dtype,
        )

        latents_map = self.pipe._unpack_latents_with_ids(latents_packed, latent_ids)
        image_latents_map = self.pipe._unpack_latents_with_ids(condition_latents, condition_ids)

        if image_latents_map.shape[-2:] != latents_map.shape[-2:]:
            raise RuntimeError(
                "Condition latents and sampled latents have different spatial shapes. "
                f"condition={tuple(image_latents_map.shape[-2:])}, sampled={tuple(latents_map.shape[-2:])}. "
                "Use identical preprocessing and latent preparation paths."
            )

        tile_cfg = self._tile_config_from_latent_shape(latents_map.shape[-2], latents_map.shape[-1])

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if hasattr(self.pipe.scheduler.config, "use_flow_sigmas") and self.pipe.scheduler.config.use_flow_sigmas:
            sigmas = None

        mu = compute_empirical_mu(latents_packed.shape[1], num_inference_steps)
        timesteps = retrieve_timesteps(
            self.pipe.scheduler,
            num_inference_steps,
            self.pipe._execution_device,
            sigmas=sigmas,
            mu=mu,
        )
        self.pipe.scheduler.set_begin_index(0)

        for timestep in timesteps:
            noise_pred_map = self._predict_noise_tiled(
                latents_map=latents_map,
                timestep=timestep,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                image_latents_map=image_latents_map,
                guidance_scale=guidance_scale,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_text_ids=negative_text_ids,
                tile_cfg=tile_cfg,
            )
            noise_pred_packed = self.pipe._pack_latents(noise_pred_map)
            latents_packed = self.pipe.scheduler.step(
                noise_pred_packed,
                timestep,
                latents_packed,
                return_dict=False,
            )[0]
            latents_map = self.pipe._unpack_latents_with_ids(latents_packed, latent_ids)

        final_latents = self.pipe._unpack_latents_with_ids(latents_packed, latent_ids)
        latents_bn_mean = self.pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(final_latents.device, final_latents.dtype)
        latents_bn_std = torch.sqrt(
            self.pipe.vae.bn.running_var.view(1, -1, 1, 1) + self.pipe.vae.config.batch_norm_eps
        ).to(final_latents.device, final_latents.dtype)
        final_latents = final_latents * latents_bn_std + latents_bn_mean
        final_latents = self.pipe._unpatchify_latents(final_latents)
        decoded = self.pipe.vae.decode(final_latents, return_dict=False)[0]
        output_image = self.pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        if (
            output_image.width >= prepared_condition.output_width
            and output_image.height >= prepared_condition.output_height
            and output_image.size != (prepared_condition.output_width, prepared_condition.output_height)
        ):
            output_image = output_image.crop(
                (0, 0, prepared_condition.output_width, prepared_condition.output_height)
            )
        elif output_image.size != (prepared_condition.output_width, prepared_condition.output_height):
            output_image = output_image.resize(
                (prepared_condition.output_width, prepared_condition.output_height),
                resample=Image.Resampling.BICUBIC,
            )
        return output_image


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 1:
        raise SystemExit("usage: python3 -m models.team01_CIPLAB.step2.inference /path/to/config.json")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    config = load_config(argv[0])

    pretrained_model_name_or_path = str(
        absolute_path(require(config, "pretrained_model_name_or_path"), "pretrained_model_name_or_path")
    )
    pix_lora_weights_path = str(absolute_path(require(config, "pix_lora_weights_path"), "pix_lora_weights_path"))
    sem_lora_weights_path = str(absolute_path(require(config, "sem_lora_weights_path"), "sem_lora_weights_path"))
    input_dir = require(config, "input_dir")
    instance_prompt = require(config, "instance_prompt")
    output_dir = absolute_path(require(config, "output_dir"), "output_dir")

    resolution = get_config_value(config, "resolution", 1024)
    tile_overlap_px = get_config_value(config, "tile_overlap_px", resolution //  2)
    tile_batch_size = get_config_value(config, "tile_batch_size", 4)
    guidance_scale = get_config_value(config, "guidance_scale", 1.0)
    num_inference_steps = get_config_value(config, "num_inference_steps", 30)
    max_sequence_length = get_config_value(config, "max_sequence_length", 512)
    seed = get_config_value(config, "seed", 0)
    adapter_scale = get_config_value(config, "adapter_scale", 1.0)
    cpu_offload = get_config_value(config, "cpu_offload", False)

    if tile_batch_size <= 0:
        raise ValueError("tile_batch_size must be >= 1")

    image_paths = load_input_images(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = Flux2KleinPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=DTYPE)
    pipe.load_lora_weights(pix_lora_weights_path, adapter_name="pix")
    pipe.load_lora_weights(sem_lora_weights_path, adapter_name="sem")
    if hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters(["pix", "sem"], adapter_weights=[adapter_scale, adapter_scale])
        except TypeError:
            pipe.set_adapters(["pix", "sem"], [adapter_scale, adapter_scale])

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    prompt_cache = cache_prompt_embeddings(pipe, instance_prompt, max_sequence_length, guidance_scale)
    runner = TiledInferenceRunner(
        pipe=pipe,
        resolution=resolution,
        tile_overlap_px=tile_overlap_px,
        tile_batch_size=tile_batch_size,
    )

    results = []
    for index, condition_path in enumerate(tqdm(image_paths, desc="inference"), start=1):
        condition_image = load_rgb_image(condition_path)
        output_image = runner.generate(
            image=condition_image,
            prompt_cache=prompt_cache,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=make_generator(seed, index),
        )
        
        postfix = adain_color_fix(output_image, condition_image)
        
        output_path = output_dir / f"{condition_path.name}"
        postfix.save(output_path)

        result = {
            "condition_image": str(condition_path),
            "output_image": str(output_path.resolve()),
        }
        results.append(result)

    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump({**config, "output_dir": str(output_dir), "dtype": "bf16"}, handle, indent=2, ensure_ascii=False)
    print(f"saved {len(results)} outputs to {output_dir}")


if __name__ == "__main__":
    main()
