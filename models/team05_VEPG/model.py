import os
import json
import glob
import re
import time
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

from diffusers import (
    AutoencoderKLFlux2,
    Flux2Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
)
from diffusers.training_utils import free_memory
from peft import PeftModel


def adain_color_fix(target: Image.Image, source: Image.Image):
    to_tensor = transforms.ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)
    to_image = transforms.ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))
    return result_image


def wavelet_color_fix(target: Image.Image, source: Image.Image):
    to_tensor = transforms.ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)
    to_image = transforms.ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))
    return result_image


def calc_mean_std(feat: Tensor, eps: float = 1e-5):
    size = feat.size()
    if len(size) != 4:
        raise ValueError("Expected 4D tensor for mean/std calculation.")
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat: Tensor, style_feat: Tensor):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(
        size
    )
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def wavelet_blur(image: Tensor, radius: int):
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None]
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output


def wavelet_decomposition(image: Tensor, levels: int = 5):
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += image - low_freq
        image = low_freq
    return high_freq, low_freq


def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor):
    content_high_freq, _ = wavelet_decomposition(content_feat)
    _, style_low_freq = wavelet_decomposition(style_feat)
    return content_high_freq + style_low_freq


def _is_adapter_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "adapter_config.json")):
        return False
    return (
        os.path.isfile(os.path.join(path, "adapter_model.safetensors"))
        or os.path.isfile(os.path.join(path, "adapter_model.bin"))
    )


def _resolve_adapter_path(base_dir: str, candidates):
    if _is_adapter_dir(base_dir):
        return base_dir
    for c in candidates:
        p = os.path.join(base_dir, c)
        if _is_adapter_dir(p):
            return p
    return None


def retrieve_latents(encoder_output: Tensor, sample_mode: str = "sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample()
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


def _patchify_latents(latents: Tensor):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
    return latents


def _unpatchify_latents(latents: Tensor):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels_latents // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch_size, num_channels_latents // 4, height * 2, width * 2)
    return latents


def _pack_latents(latents: Tensor):
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
    return latents


def _unpack_latents_with_ids(latents: Tensor, latent_ids: Tensor):
    unpacked = []
    for data, pos in zip(latents, latent_ids):
        _, channels = data.shape
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        height = torch.max(h_ids) + 1
        width = torch.max(w_ids) + 1
        flat_ids = h_ids * width + w_ids
        out = torch.zeros((height * width, channels), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, channels), data)
        out = out.view(height, width, channels).permute(2, 0, 1)
        unpacked.append(out)
    return torch.stack(unpacked, dim=0)


def _prepare_text_ids(prompt_embeds: Tensor):
    batch_size, seq_len, _ = prompt_embeds.shape
    text_ids = torch.cartesian_prod(
        torch.arange(1, device=prompt_embeds.device),
        torch.arange(1, device=prompt_embeds.device),
        torch.arange(1, device=prompt_embeds.device),
        torch.arange(seq_len, device=prompt_embeds.device),
    )
    return text_ids.unsqueeze(0).repeat(batch_size, 1, 1)


def _prepare_latent_ids(latents: Tensor):
    batch_size, _, height, width = latents.shape
    latent_ids = torch.cartesian_prod(
        torch.arange(1, device=latents.device),
        torch.arange(height, device=latents.device),
        torch.arange(width, device=latents.device),
        torch.arange(1, device=latents.device),
    )
    return latent_ids.unsqueeze(0).repeat(batch_size, 1, 1)


def _repeat_batch_dim(tensor: Tensor, batch_size: int, tensor_name: str):
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        repeat_shape = [batch_size] + [1] * (tensor.ndim - 1)
        return tensor.repeat(*repeat_shape)
    raise ValueError(
        f"{tensor_name} batch dim {tensor.shape[0]} cannot match inference batch {batch_size}."
    )


def _calculate_dynamic_shift_mu(image_seq_len: int, scheduler: FlowMatchEulerDiscreteScheduler):
    base_seq_len = int(getattr(scheduler.config, "base_image_seq_len", 256))
    max_seq_len = int(getattr(scheduler.config, "max_image_seq_len", 4096))
    base_shift = float(getattr(scheduler.config, "base_shift", 0.5))
    max_shift = float(getattr(scheduler.config, "max_shift", 1.15))
    if max_seq_len == base_seq_len:
        return base_shift
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return float(image_seq_len * m + b)


def encode_images(pixels: Tensor, vae: torch.nn.Module, weight_dtype):
    image_latents = retrieve_latents(vae.encode(pixels.to(vae.dtype)), sample_mode="argmax")
    image_latents = _patchify_latents(image_latents)

    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        image_latents.device, image_latents.dtype
    )
    image_latents = (image_latents - latents_bn_mean) / latents_bn_std
    return image_latents.to(weight_dtype)


def decode_images(latents: Tensor, vae: torch.nn.Module):
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        latents.device, latents.dtype
    )
    latents = latents * latents_bn_std + latents_bn_mean
    latents = _unpatchify_latents(latents)
    return vae.decode(latents.to(vae.dtype), return_dict=False)[0]


def _build_tile_starts(length: int, tile_size: int, tile_overlap: int):
    if tile_size >= length:
        return [0]
    stride = max(1, tile_size - tile_overlap)
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != (length - tile_size):
        starts.append(length - tile_size)
    return starts


class VEPG_Flux2KleinInfer(torch.nn.Module):
    def __init__(
        self,
        flux2_path: str,
        lora_path: str,
        device: str,
        weight_dtype: torch.dtype = torch.bfloat16,
        mid_timestep: int = 244,
        process_size: int = 512,
    ):
        super().__init__()
        self.device = device
        self.weight_dtype = weight_dtype
        self.mid_timestep = mid_timestep

        vae = AutoencoderKLFlux2.from_pretrained(flux2_path, subfolder="vae")
        transformer = Flux2Transformer2DModel.from_pretrained(flux2_path, subfolder="transformer")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(flux2_path, subfolder="scheduler")

        vae_adapter = _resolve_adapter_path(
            lora_path,
            candidates=[
                "vae_encoder_adapter",
                "vae_encoder_lora_adapter",
                "vae_encoder",
                "encoder",
            ],
        )
        if vae_adapter is None:
            raise FileNotFoundError(f"Cannot find VAE encoder adapter under: {lora_path}")

        transformer_adapter = _resolve_adapter_path(
            lora_path,
            candidates=[
                "flux_adapter",
                "transformer_lora_adapter",
                "transformer_adapter",
                "transformer",
            ],
        )
        if transformer_adapter is None:
            raise FileNotFoundError(f"Cannot find Flux2 transformer adapter under: {lora_path}")

        vae.encoder = PeftModel.from_pretrained(vae.encoder, vae_adapter, is_trainable=False)
        vae.encoder = vae.encoder.merge_and_unload()
        transformer = PeftModel.from_pretrained(transformer, transformer_adapter, is_trainable=False)
        transformer = transformer.merge_and_unload()

        vae.requires_grad_(False)
        transformer.requires_grad_(False)
        vae = vae.to(device=device, dtype=weight_dtype).eval()
        transformer = transformer.to(device=device, dtype=weight_dtype).eval()

        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler

        num_inference_steps = int(getattr(self.scheduler.config, "num_train_timesteps", 1000))
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        latent_base = vae_scale_factor * 2
        latent_height = 2 * (int(process_size) // latent_base)
        latent_width = 2 * (int(process_size) // latent_base)
        image_seq_len = (latent_height // 2) * (latent_width // 2)
        if image_seq_len <= 0:
            raise ValueError(
                f"Invalid process_size={process_size} for Flux2 latent packing with factor {latent_base}."
            )

        set_timesteps_kwargs = {}
        if getattr(self.scheduler.config, "use_dynamic_shifting", False):
            mu = _calculate_dynamic_shift_mu(image_seq_len=image_seq_len, scheduler=self.scheduler)
            set_timesteps_kwargs["mu"] = mu

        self.scheduler.set_timesteps(num_inference_steps, device=device, **set_timesteps_kwargs)
        sigmas = self.scheduler.sigmas.to(device=device, dtype=torch.float32)
        timesteps = self.scheduler.timesteps.to(device=device, dtype=torch.float32)
        if mid_timestep < 0 or mid_timestep >= len(sigmas):
            raise ValueError(f"mid_timestep={mid_timestep} is out of range for scheduler of size {len(sigmas)}")

        t_index = min(len(timesteps) - 1, len(sigmas) - 1 - mid_timestep)
        self.sigma_t = sigmas[-(mid_timestep + 1)]
        self.t_mid = timesteps[t_index]

    def _predict_denoised_latent(self, lq_latent, prompt_embeds, text_ids):
        bsz = lq_latent.shape[0]
        prompt_embeds_batch = _repeat_batch_dim(prompt_embeds, bsz, "prompt_embeds")
        text_ids_batch = _repeat_batch_dim(text_ids, bsz, "text_ids")

        packed_lq_latent = _pack_latents(lq_latent)
        latent_image_ids = _prepare_latent_ids(lq_latent)
        timestep = self.t_mid.expand(bsz).to(device=lq_latent.device, dtype=packed_lq_latent.dtype)
        sigma_local = self.sigma_t.to(device=lq_latent.device, dtype=packed_lq_latent.dtype)

        model_pred = self.transformer(
            hidden_states=packed_lq_latent,
            timestep=timestep / 1000.0,
            guidance=None,
            encoder_hidden_states=prompt_embeds_batch.to(self.transformer.dtype),
            txt_ids=text_ids_batch,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        denoised_latent = packed_lq_latent - sigma_local * model_pred
        denoised_latent = _unpack_latents_with_ids(denoised_latent, latent_image_ids)
        return denoised_latent

    def _forward_no_tile(self, lq_latent, prompt_embeds, text_ids):
        denoised_latent = self._predict_denoised_latent(lq_latent, prompt_embeds, text_ids)
        return decode_images(denoised_latent, self.vae)

    def _gaussian_weights(self, tile_width: int, tile_height: int, channels: int, sigma: float = 0.3):
        xs = torch.linspace(-1, 1, tile_width, device=self.device, dtype=torch.float32)
        ys = torch.linspace(-1, 1, tile_height, device=self.device, dtype=torch.float32)
        xx = xs.unsqueeze(0).expand(tile_height, tile_width)
        yy = ys.unsqueeze(1).expand(tile_height, tile_width)
        weights = torch.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
        return weights.unsqueeze(0).unsqueeze(0).repeat(1, channels, 1, 1)

    def _forward_tile(self, lq_latent, prompt_embeds, text_ids, tile_size, tile_overlap):
        _, channels, h, w = lq_latent.shape
        tile_size = min(tile_size, h, w)
        tile_weights = self._gaussian_weights(tile_size, tile_size, channels).to(dtype=lq_latent.dtype)

        x_starts = _build_tile_starts(w, tile_size, tile_overlap)
        y_starts = _build_tile_starts(h, tile_size, tile_overlap)

        denoised_sum = torch.zeros_like(lq_latent)
        contributors = torch.zeros_like(lq_latent)

        for y0 in y_starts:
            for x0 in x_starts:
                y1 = y0 + tile_size
                x1 = x0 + tile_size
                tile = lq_latent[:, :, y0:y1, x0:x1]
                denoised_tile = self._predict_denoised_latent(tile, prompt_embeds, text_ids)
                denoised_sum[:, :, y0:y1, x0:x1] += denoised_tile * tile_weights
                contributors[:, :, y0:y1, x0:x1] += tile_weights

        denoised_latent = denoised_sum / torch.clamp(contributors, min=1e-6)
        return decode_images(denoised_latent, self.vae)

    def forward(self, lq_img, prompt_embeds, text_ids, tile_size, tile_overlap):
        if lq_img.is_cuda:
            torch.cuda.synchronize(lq_img.device)
        start_time = time.time()

        lq_latent = encode_images(lq_img.to(self.vae.dtype), self.vae, self.weight_dtype)
        _, _, h, w = lq_latent.shape
        if h * w <= tile_size * tile_size:
            pred_img = self._forward_no_tile(lq_latent, prompt_embeds, text_ids)
        else:
            pred_img = self._forward_tile(lq_latent, prompt_embeds, text_ids, tile_size, tile_overlap)

        if lq_img.is_cuda:
            torch.cuda.synchronize(lq_img.device)
        t = time.time() - start_time
        return pred_img, t


# def encode_prompt_flux2klein(flux_path: str, prompt: str, device: str, weight_dtype: torch.dtype):
#     pipe = Flux2KleinPipeline.from_pretrained(
#         flux_path, transformer=None, vae=None, torch_dtype=weight_dtype
#     ).to(device)

#     with torch.no_grad():
#         encoded = pipe.encode_prompt(prompt)

#     pipe = pipe.to("cpu")
#     del pipe
#     free_memory()

#     prompt_embeds = None
#     text_ids = None
#     if isinstance(encoded, dict):
#         prompt_embeds = encoded.get("prompt_embeds", None)
#         text_ids = encoded.get("text_ids", None)
#     elif isinstance(encoded, (tuple, list)):
#         if len(encoded) > 0:
#             prompt_embeds = encoded[0]
#         if len(encoded) > 1:
#             text_ids = encoded[1]
#     else:
#         prompt_embeds = encoded

#     if prompt_embeds is None:
#         raise ValueError("Failed to obtain prompt_embeds from Flux2KleinPipeline.encode_prompt.")

#     if prompt_embeds.ndim == 2:
#         prompt_embeds = prompt_embeds.unsqueeze(0)
#     prompt_embeds = prompt_embeds.to(device=device, dtype=weight_dtype)

#     if text_ids is not None and text_ids.ndim == 2:
#         text_ids = text_ids.unsqueeze(0)
#     if text_ids is None or text_ids.shape[-1] != 4:
#         text_ids = _prepare_text_ids(prompt_embeds)
#     text_ids = text_ids.to(device=device, dtype=torch.long)

#     return prompt_embeds, text_ids

# def encode_prompt_flux2klein(
#     flux_path: str,
#     prompt: str,
#     device: str,
#     weight_dtype: torch.dtype,
#     save_path: str = '/data/zfk/code/NTIRE2026_ImageSR_x4/model_zoo/05_VEPG',
# ):
#     pipe = Flux2KleinPipeline.from_pretrained(
#         flux_path,
#         transformer=None,
#         vae=None,
#         torch_dtype=weight_dtype
#     ).to(device)

#     with torch.no_grad():
#         encoded = pipe.encode_prompt(prompt)

#     pipe = pipe.to("cpu")
#     del pipe
#     free_memory()

#     prompt_embeds = None
#     text_ids = None

#     if isinstance(encoded, dict):
#         prompt_embeds = encoded.get("prompt_embeds", None)
#         text_ids = encoded.get("text_ids", None)
#     elif isinstance(encoded, (tuple, list)):
#         if len(encoded) > 0:
#             prompt_embeds = encoded[0]
#         if len(encoded) > 1:
#             text_ids = encoded[1]
#     else:
#         prompt_embeds = encoded

#     if prompt_embeds is None:
#         raise ValueError("Failed to obtain prompt_embeds from Flux2KleinPipeline.encode_prompt.")

#     if prompt_embeds.ndim == 2:
#         prompt_embeds = prompt_embeds.unsqueeze(0)

#     if text_ids is not None and text_ids.ndim == 2:
#         text_ids = text_ids.unsqueeze(0)

#     if text_ids is None or text_ids.shape[-1] != 4:
#         text_ids = _prepare_text_ids(prompt_embeds)

#     prompt_embeds = prompt_embeds.to(device=device, dtype=weight_dtype)
#     text_ids = text_ids.to(device=device, dtype=torch.long)

#     if save_path is not None:
#         # 如果 save_path 看起来像目录，自动补文件名
#         if (
#             os.path.isdir(save_path)
#             or not os.path.splitext(save_path)[1]  # 没有后缀名，通常视为目录
#         ):
#             os.makedirs(save_path, exist_ok=True)
#             save_file = os.path.join(save_path, "prompt_embeds.pt")
#         else:
#             save_dir = os.path.dirname(save_path)
#             if save_dir:
#                 os.makedirs(save_dir, exist_ok=True)
#             save_file = save_path

#         save_obj = {
#             "prompt": prompt,
#             "prompt_embeds": prompt_embeds.detach().cpu(),
#             "text_ids": text_ids.detach().cpu(),
#             "weight_dtype": str(weight_dtype),
#         }
#         torch.save(save_obj, save_file)
#         print(f"[Info] prompt embeds saved to: {save_file}")
#         assert 1==0
#     return prompt_embeds, text_ids

def load_prompt_embeds(
    load_path: str,
    device: str,
    weight_dtype: torch.dtype,
):
    data = torch.load(load_path, map_location="cpu")

    if "prompt_embeds" not in data:
        raise ValueError(f"'prompt_embeds' not found in {load_path}")
    if "text_ids" not in data:
        raise ValueError(f"'text_ids' not found in {load_path}")

    prompt_embeds = data["prompt_embeds"]
    text_ids = data["text_ids"]

    if prompt_embeds.ndim == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)
    if text_ids.ndim == 2:
        text_ids = text_ids.unsqueeze(0)

    prompt_embeds = prompt_embeds.to(device=device, dtype=weight_dtype)
    text_ids = text_ids.to(device=device, dtype=torch.long)

    print(f"[Info] prompt embeds loaded from: {load_path}")
    return prompt_embeds, text_ids


def pad_to_multiple_pil(img: Image.Image, multiple: int = 16, mode: str = "reflect"):
    w, h = img.size
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple
    pad_w = new_w - w
    pad_h = new_h - h

    if pad_w == 0 and pad_h == 0:
        return img, (w, h)

    if mode != "reflect":
        raise ValueError(f"Unsupported pad mode: {mode}")

    padded = Image.new(img.mode, (new_w, new_h))
    padded.paste(img, (0, 0))

    if pad_w > 0:
        right = img.transpose(Image.FLIP_LEFT_RIGHT).crop((0, 0, pad_w, h))
        padded.paste(right, (w, 0))

    if pad_h > 0:
        y0 = max(0, h - pad_h)
        bottom = padded.crop((0, y0, new_w, h)).transpose(Image.FLIP_TOP_BOTTOM)
        padded.paste(bottom, (0, h))

    return padded, (w, h)


def _load_config(model_dir: str):
    for name in ["config.json", "vepg_config.json"]:
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def _is_lora_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    vae_candidates = [
        "vae_encoder_adapter",
        "vae_encoder_lora_adapter",
        "vae_encoder",
        "encoder",
    ]
    transformer_candidates = [
        "flux_adapter",
        "transformer_lora_adapter",
        "transformer_adapter",
        "transformer",
    ]
    has_vae = any(os.path.isdir(os.path.join(path, c)) for c in vae_candidates)
    has_transformer = any(
        os.path.isdir(os.path.join(path, c)) for c in transformer_candidates
    )
    return has_vae and has_transformer


def _weight_step(path: str) -> int:
    m = re.search(r"weight-(\d+)", os.path.basename(path))
    if m:
        return int(m.group(1))
    return -1


def _select_lora_path(model_dir: str) -> str:
    explicit_dir = os.path.join(model_dir, "model_lora_weights")
    if _is_lora_dir(explicit_dir):
        return explicit_dir
    if _is_lora_dir(model_dir):
        return model_dir
    candidates = sorted(glob.glob(os.path.join(model_dir, "weight-*")), key=_weight_step)
    for p in reversed(candidates):
        if _is_lora_dir(p):
            return p
    raise FileNotFoundError(
        "Cannot find LoRA weight folder. Expected a folder containing "
        "flux_adapter and vae_encoder_adapter, or weight-* subfolders."
    )


def _read_first_line(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    return line
    except OSError:
        pass
    return None


def _parse_base_model_from_readme(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("base_model:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _resolve_flux_path(model_dir: str, lora_path: str, config: dict) -> str:
    for key in ["flux_path", "base_model"]:
        if key in config and config[key]:
            flux_val = config[key]
            if not os.path.isabs(flux_val):
                flux_val = os.path.join(model_dir, flux_val)
            return flux_val

    for env_key in ["VEPG_FLUX_PATH", "FLUX_PATH"]:
        env_val = os.getenv(env_key)
        if env_val:
            return env_val

    for fname in ["flux_path.txt", "base_model.txt"]:
        for root in [model_dir, lora_path]:
            p = os.path.join(root, fname)
            val = _read_first_line(p)
            if val:
                if not os.path.isabs(val):
                    val = os.path.join(model_dir, val)
                return val

    default_flux_dir = os.path.join(model_dir, "flux2_klein_base")
    if os.path.isdir(default_flux_dir):
        return default_flux_dir

    base_model = _parse_base_model_from_readme(os.path.join(lora_path, "README.md"))
    if base_model:
        return base_model

    raise FileNotFoundError(
        "Cannot resolve flux_path. Provide it via config.json, "
        "VEPG_FLUX_PATH/FLUX_PATH, or place flux2_klein_base under model_dir."
    )


def _resolve_prompt(config: dict) -> str:
    if "prompt" in config:
        return str(config["prompt"])
    if "VEPG_PROMPT" in os.environ:
        return os.environ["VEPG_PROMPT"]
    return ""


def _resolve_align_method(config: dict) -> str:
    if "align_method" in config:
        return str(config["align_method"]).lower()
    env_val = os.getenv("VEPG_ALIGN_METHOD")
    if env_val:
        return str(env_val).lower()
    return "adain"


def _resolve_weight_dtype_name(config: dict) -> str:
    if "weight_dtype" in config:
        return str(config["weight_dtype"]).lower()
    env_val = os.getenv("VEPG_WEIGHT_DTYPE")
    if env_val:
        return str(env_val).lower()
    return "bf16"


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported weight dtype: {name}")
    dtype = mapping[name]
    if device.type == "cpu" and dtype is not torch.float32:
        return torch.float32
    if device.type == "cuda" and dtype is torch.bfloat16:
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            return torch.float16
    return dtype


def _resolve_int(config: dict, key: str, default: int) -> int:
    if key in config:
        try:
            return int(config[key])
        except (TypeError, ValueError):
            pass
    env_val = os.getenv(f"VEPG_{key.upper()}")
    if env_val:
        try:
            return int(env_val)
        except (TypeError, ValueError):
            pass
    return default


class VEPG(torch.nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        self.model_dir = model_dir
        self.config = _load_config(model_dir)
        self.lora_path = _select_lora_path(model_dir)
        self.flux_path = _resolve_flux_path(model_dir, self.lora_path, self.config)

        self.prompt = _resolve_prompt(self.config)
        self.align_method = _resolve_align_method(self.config)
        if self.align_method not in ["adain", "wavelet", "nofix"]:
            raise ValueError(f"Unsupported align_method: {self.align_method}")

        self.process_size = _resolve_int(self.config, "process_size", 512)
        self.mid_timestep = _resolve_int(self.config, "mid_timestep", 244)
        self.upscale = _resolve_int(self.config, "upscale", 4)
        self.weight_dtype_name = _resolve_weight_dtype_name(self.config)

        self._device = None
        self._weight_dtype = None
        self._prompt_embeds = None
        self._text_ids = None
        self._model = None

    def to(self, *args, **kwargs):
        device = None
        if args:
            if isinstance(args[0], (torch.device, str)):
                device = torch.device(args[0])
        if "device" in kwargs and kwargs["device"] is not None:
            device = torch.device(kwargs["device"])

        if self._model is not None and device is not None and device != self._device:
            raise RuntimeError(
                f"VEPG already initialized on {self._device}, cannot move to {device}."
            )
        if device is not None:
            self._device = device
        return self

    def _ensure_loaded(self, device: torch.device):
        if self._model is not None:
            return
        self._device = device
        self._weight_dtype = _resolve_dtype(self.weight_dtype_name, device)

        # self._prompt_embeds, self._text_ids = encode_prompt_flux2klein(
        #     self.flux_path, self.prompt, str(device), self._weight_dtype
        # )

        self._prompt_embeds, self._text_ids = load_prompt_embeds(
            load_path="./model_zoo/team05_VEPG/prompt_embeds.pt",
            device=str(device),
            weight_dtype=self._weight_dtype,
        )

        self._model = VEPG_Flux2KleinInfer(
            flux2_path=self.flux_path,
            lora_path=self.lora_path,
            device=str(device),
            weight_dtype=self._weight_dtype,
            mid_timestep=self.mid_timestep,
            process_size=self.process_size,
        )
        self._model.eval()

    @torch.no_grad()
    def forward(self, img_lq: Tensor):
        if img_lq.ndim != 4 or img_lq.size(1) != 3:
            raise ValueError("Expected input shape (N, 3, H, W).")
        if img_lq.size(0) != 1:
            outputs = []
            for i in range(img_lq.size(0)):
                outputs.append(self.forward(img_lq[i : i + 1]))
            return torch.cat(outputs, dim=0)

        device = img_lq.device
        self._ensure_loaded(device)

        img_cpu = img_lq[0].detach().clamp(0.0, 1.0).cpu()
        input_image = transforms.ToPILImage()(img_cpu)

        ori_width, ori_height = input_image.size
        rscale = self.upscale
        resize_flag = False

        if ori_width < self.process_size // rscale or ori_height < self.process_size // rscale:
            scale = (self.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize(
                (int(scale * ori_width), int(scale * ori_height))
            )
            resize_flag = True

        input_image = input_image.resize(
            (input_image.size[0] * rscale, input_image.size[1] * rscale)
        )
        input_image_no_pad = input_image
        input_image, (target_w, target_h) = pad_to_multiple_pil(
            input_image, multiple=16, mode="reflect"
        )

        tile_size = max(1, self.process_size // 16)
        tile_overlap = tile_size // 2

        lq_img = (
            TF.to_tensor(input_image)
            .unsqueeze(0)
            .to(device=device, dtype=self._weight_dtype)
            * 2
            - 1
        )

        output_image, _ = self._model(
            lq_img, self._prompt_embeds, self._text_ids, tile_size, tile_overlap
        )
        output_image = output_image * 0.5 + 0.5
        output_image = torch.clamp(output_image, 0, 1).float()
        output_pil = transforms.ToPILImage()(output_image[0].cpu())
        output_pil = output_pil.crop((0, 0, target_w, target_h))

        if self.align_method == "adain":
            output_pil = adain_color_fix(target=output_pil, source=input_image_no_pad)
        elif self.align_method == "wavelet":
            output_pil = wavelet_color_fix(target=output_pil, source=input_image_no_pad)

        if resize_flag:
            output_pil = output_pil.resize(
                (int(rscale * ori_width), int(rscale * ori_height))
            )

        output = transforms.ToTensor()(output_pil).unsqueeze(0).to(device=device)
        return output
