import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.nn import functional as F
from typing import Literal, List, overload
import os
from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig
import logging
from typing import Mapping, Any, Tuple, Callable, Literal
import importlib
import os
from urllib.parse import urlparse
from dataclasses import dataclass
from contextlib import contextmanager
from torch import Tensor
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
import argparse
import os
from pathlib import Path
from time import time
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from . import model_mamba

from timm.models.layers import DropPath, trunc_normal_
from einops.layers.torch import Rearrange
from einops import rearrange

import math
import numpy as np

def _parse_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default

def _resolve_weight_dtype(default: torch.dtype = torch.bfloat16) -> torch.dtype:
    raw = os.environ.get("HYPIR_WEIGHT_DTYPE")
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    return default

def _parse_env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            value = default
    if minimum is not None:
        value = max(minimum, value)
    return value

def _parse_env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError:
            value = default
    if minimum is not None:
        value = max(minimum, value)
    return value

def wavelet_blur(image: Tensor, radius: int):
    """
    Apply wavelet blur to the input tensor.
    """
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    # apply convolution
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output

@dataclass(frozen=True)
class TileIndex:
    hi: int
    hi_end: int
    wi: int
    wi_end: int

def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    levels = _parse_env_int("HYPIR_WAVELET_LEVELS", int(levels), minimum=1)
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq

    return high_freq, low_freq

def wavelet_reconstruction(content_feat:Tensor, style_feat:Tensor):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    if not _parse_env_bool("HYPIR_WAVELET_ENABLE", True):
        return content_feat

    blend = _parse_env_float("HYPIR_WAVELET_BLEND", 1.0, minimum=0.0)
    blend = min(1.0, blend)

    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    # reconstruct the content feature with the style's high frequency
    reconstructed = content_high_freq + style_low_freq
    if blend >= 1.0:
        return reconstructed
    return reconstructed * blend + content_feat * (1.0 - blend)

def sliding_windows(h: int, w: int, tile_size: int, tile_stride: int) -> Tuple[int, int, int, int]:
    hi_list = list(range(0, h - tile_size + 1, tile_stride))
    if (h - tile_size) % tile_stride != 0:
        hi_list.append(h - tile_size)
    
    wi_list = list(range(0, w - tile_size + 1, tile_stride))
    if (w - tile_size) % tile_stride != 0:
        wi_list.append(w - tile_size)
    
    coords = []
    for hi in hi_list:
        for wi in wi_list:
            coords.append((hi, hi + tile_size, wi, wi + tile_size))
    return coords

# https://github.com/csslc/CCSR/blob/main/model/q_sampler.py#L503
def gaussian_weights(tile_width: int, tile_height: int) -> np.ndarray:
    """Generates a gaussian mask of weights for tile contributions"""
    latent_width = tile_width
    latent_height = tile_height
    var = _parse_env_float("HYPIR_TILE_GAUSSIAN_VAR", 0.01, minimum=1e-6)
    midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
    x_probs = [
        np.exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for x in range(latent_width)]
    midpoint = latent_height / 2
    y_probs = [
        np.exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for y in range(latent_height)]
    weights = np.outer(y_probs, x_probs)
    return weights

def make_tiled_fn(
    fn: Callable[[torch.Tensor], torch.Tensor],
    size: int,
    stride: int,
    scale_type: Literal["up", "down"] = "up",
    scale: int = 1,
    channel: int | None = None,
    weight: Literal["uniform", "gaussian"] = "gaussian",
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    progress: bool = True,
    desc: str=None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    # Only split the first input of function.
    def tiled_fn(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if scale_type == "up":
            scale_fn = lambda n: int(n * scale)
        else:
            scale_fn = lambda n: int(n // scale)

        b, c, h, w = x.size()
        out_dtype = dtype or x.dtype
        out_device = device or x.device
        out_channel = channel or c
        out = torch.zeros(
            (b, out_channel, scale_fn(h), scale_fn(w)),
            dtype=out_dtype,
            device=out_device,
        )
        count = torch.zeros_like(out, dtype=torch.float32)
        weight_mode = os.environ.get("HYPIR_TILE_WEIGHT", weight).strip().lower()
        if weight_mode not in {"uniform", "gaussian"}:
            weight_mode = weight
        weight_size = scale_fn(size)
        weights = (
            gaussian_weights(weight_size, weight_size)[None, None]
            if weight_mode == "gaussian"
            else np.ones((1, 1, weight_size, weight_size))
        )
        weights = torch.tensor(
            weights,
            dtype=out_dtype,
            device=out_device,
        )

        indices = sliding_windows(h, w, size, stride)
        pbar_desc = f"[{desc}]: Tiled Processing" if desc else "Tiled Processing"
        pbar = tqdm(
            indices, desc=pbar_desc, disable=not progress
        )
        for hi, hi_end, wi, wi_end in pbar:
            x_tile = x[..., hi:hi_end, wi:wi_end]
            out_hi, out_hi_end, out_wi, out_wi_end = map(
                scale_fn, (hi, hi_end, wi, wi_end)
            )
            if len(args) or len(kwargs):
                kwargs.update(index=TileIndex(hi=hi, hi_end=hi_end, wi=wi, wi_end=wi_end))
            out[..., out_hi:out_hi_end, out_wi:out_wi_end] += (
                fn(x_tile, *args, **kwargs) * weights
            )
            count[..., out_hi:out_hi_end, out_wi:out_wi_end] += weights
        out = out / count
        return out

    return tiled_fn

class BaseEnhancer:

    def __init__(
        self,
        base_model_path,
        weight_path,
        lora_modules,
        lora_rank,
        model_t,
        coeff_t,
        device,
    ):
        self.base_model_path = base_model_path
        self.weight_path = weight_path
        self.lora_modules = lora_modules
        self.lora_rank = lora_rank
        self.model_t = model_t
        self.coeff_t = coeff_t

        self.weight_dtype = _resolve_weight_dtype()
        self.device = device
        self.vae_latent_mode = os.environ.get("HYPIR_VAE_LATENT_MODE", "sample").strip().lower()
        if self.vae_latent_mode not in {"sample", "mode"}:
            self.vae_latent_mode = "sample"
        self.final_resize_antialias = _parse_env_bool("HYPIR_FINAL_RESIZE_ANTIALIAS", True)

    def init_models(self):
        self.init_scheduler()
        self.init_text_models()
        self.init_vae()
        self.init_generator()

    @overload
    def init_scheduler(self):
        ...

    @overload
    def init_text_models(self):
        ...

    def init_vae(self):
        self.vae = AutoencoderKL.from_pretrained(
            self.base_model_path, subfolder="vae", torch_dtype=self.weight_dtype).to(self.device)
        
        # state_dict = torch.load("/root/workspace/hypir-kainan/denoise_enc_state_dict.pth", map_location="cpu")
        # state_dict = {k.replace("encoder.", "", 1): v for k, v in state_dict.items()}
        # self.vae.encoder.load_state_dict(state_dict, strict=True)
        
        self.vae.eval().requires_grad_(False)

    @overload
    def init_generator(self):
        ...

    @overload
    def prepare_inputs(self, batch_size, prompt):
        ...

    @overload
    def forward_generator(self, z_lq: torch.Tensor) -> torch.Tensor:
        ...

    @torch.no_grad()
    def enhance(
        self,
        lq: torch.Tensor,
        prompt: str,
        scale_by: Literal["factor", "longest_side"] = "factor",
        upscale: int = 1,
        target_longest_side: int | None = None,
        patch_size: int = 512,
        stride: int = 256,
        return_type: Literal["pt", "np", "pil"] = "pt",
    ) -> torch.Tensor | np.ndarray | List[Image.Image]:
        if stride <= 0:
            raise ValueError("Stride must be greater than 0.")
        if patch_size <= 0:
            raise ValueError("Patch size must be greater than 0.")
        if patch_size < stride:
            raise ValueError("Patch size must be greater than or equal to stride.")

        # Prepare low-quality inputs
        bs = len(lq)
        if scale_by == "factor":
            lq = F.interpolate(lq, scale_factor=upscale, mode="bicubic")
        elif scale_by == "longest_side":
            if target_longest_side is None:
                raise ValueError("target_longest_side must be specified when scale_by is 'longest_side'.")
            h, w = lq.shape[2:]
            if h >= w:
                new_h = target_longest_side
                new_w = int(w * (target_longest_side / h))
            else:
                new_w = target_longest_side
                new_h = int(h * (target_longest_side / w))
            lq = F.interpolate(lq, size=(new_h, new_w), mode="bicubic")
        else:
            raise ValueError(f"Unsupported scale_by method: {scale_by}")
        ref = lq
        h0, w0 = lq.shape[2:]
        if min(h0, w0) <= patch_size:
            lq = self.resize_at_least(lq, size=patch_size)

        # VAE encoding
        lq = (lq * 2 - 1).to(dtype=self.weight_dtype, device=self.device)
        h1, w1 = lq.shape[2:]
        # Pad vae input size to multiples of vae_scale_factor,
        # otherwise image size will be changed
        vae_scale_factor = 8
        ph = (h1 + vae_scale_factor - 1) // vae_scale_factor * vae_scale_factor - h1
        pw = (w1 + vae_scale_factor - 1) // vae_scale_factor * vae_scale_factor - w1
        lq = F.pad(lq, (0, pw, 0, ph), mode="constant", value=0)
        # Encode
        def _encode_tile(lq_tile: torch.Tensor) -> torch.Tensor:
            latent_dist = self.vae.encode(lq_tile).latent_dist
            if self.vae_latent_mode == "mode":
                return latent_dist.mode()
            return latent_dist.sample()

        z_lq = make_tiled_fn(
            fn=_encode_tile,
            size=patch_size,
            stride=stride,
            scale_type="down",
            scale=vae_scale_factor,
            progress=True,
            channel=self.vae.config.latent_channels,
            desc="VAE encoding",
        )(lq.to(self.weight_dtype))

        # Generator forward
        self.prepare_inputs(batch_size=bs, prompt=prompt)
        z = make_tiled_fn(
            fn=lambda z_lq_tile: self.forward_generator(z_lq_tile),
            size=patch_size // vae_scale_factor,
            stride=stride // vae_scale_factor,
            progress=True,
            desc="Generator Forward",
        )(z_lq.to(self.weight_dtype))
        
        # Decode
        x = make_tiled_fn(
            fn=lambda lq_tile: self.vae.decode(lq_tile).sample.float(),
            size=patch_size//vae_scale_factor,
            stride=stride//vae_scale_factor,
            scale_type="up",
            scale=vae_scale_factor,
            progress=True,
            channel=3,
            desc="VAE decoding",
        )(z.to(self.weight_dtype))
        
        x = x[..., :h1, :w1]
        x = (x + 1) / 2
        x = F.interpolate(
            input=x,
            size=(h0, w0),
            mode="bicubic",
            antialias=self.final_resize_antialias,
        )
        x = wavelet_reconstruction(x, ref.to(device=self.device))

        if return_type == "pt":
            return x.clamp(0, 1).cpu()
        elif return_type == "np":
            return self.tensor2image(x)
        else:
            return [Image.fromarray(img) for img in self.tensor2image(x)]

    @staticmethod
    def tensor2image(img_tensor):
        return (
            (img_tensor * 255.0)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )

    @staticmethod
    def resize_at_least(imgs: torch.Tensor, size: int) -> torch.Tensor:
        _, _, h, w = imgs.size()
        if h == w:
            new_h, new_w = size, size
        elif h < w:
            new_h, new_w = size, int(w * (size / h))
        else:
            new_h, new_w = int(h * (size / w)), size
        return F.interpolate(imgs, size=(new_h, new_w), mode="bicubic", antialias=True)

class SD2Enhancer(BaseEnhancer):

    def init_scheduler(self):
        self.scheduler = DDPMScheduler.from_pretrained(self.base_model_path, subfolder="scheduler")

    def init_text_models(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.base_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.base_model_path, subfolder="text_encoder", torch_dtype=self.weight_dtype).to(self.device)
        self.text_encoder.eval().requires_grad_(False)

    def init_generator(self):
        self.G: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            self.base_model_path, subfolder="unet", torch_dtype=self.weight_dtype).to(self.device)
        target_modules = self.lora_modules
        G_lora_cfg = LoraConfig(r=self.lora_rank, lora_alpha=self.lora_rank,
            init_lora_weights="gaussian", target_modules=target_modules)
        self.G.add_adapter(G_lora_cfg)

        print(f"Load model weights from {self.weight_path}")
        state_dict = torch.load(self.weight_path, map_location="cpu", weights_only=False)
        self.G.load_state_dict(state_dict, strict=False)
        input_keys = set(state_dict.keys())
        required_keys = set([k for k in self.G.state_dict().keys() if "lora" in k])
        missing = required_keys - input_keys
        unexpected = input_keys - required_keys
        assert required_keys == input_keys, f"Missing: {missing}, Unexpected: {unexpected}"

        self.G.eval().requires_grad_(False)

    def prepare_inputs(self, batch_size, prompt):
        bs = batch_size
        txt_ids = self.tokenizer(
            [prompt] * bs,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_embed = self.text_encoder(txt_ids.to(self.device))[0]
        c_txt = {"text_embed": text_embed}
        timesteps = torch.full((bs,), self.model_t, dtype=torch.long, device=self.device)
        self.inputs = dict(
            c_txt=c_txt,
            timesteps=timesteps,
        )

    def forward_generator(self, z_lq):
        z_in = z_lq * self.vae.config.scaling_factor
        eps = self.G(
            z_in, self.inputs["timesteps"],
            encoder_hidden_states=self.inputs["c_txt"]["text_embed"],
        ).sample
        z = self.scheduler.step(eps, self.coeff_t, z_in).pred_original_sample
        z_out = z / self.vae.config.scaling_factor
        return z_out

class Captioner:

    def __init__(self, device: torch.device) -> "Captioner":
        self.device = device

    @overload
    def __call__(self, image: Image.Image) -> str: ...


class EmptyCaptioner(Captioner):

    def __call__(self, image: Image.Image) -> str:
        return ""

def main(model_dir, input_path, output_path, device):
    model_dir = Path(model_dir)
    input_path = Path(input_path)
    output_path = Path(output_path)

    seed = 231
    set_seed(seed)
    lora_modules = [
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "conv",
        "conv1",
        "conv2",
        "conv_shortcut",
        "conv_out",
        "proj_in",
        "proj_out",
        "ff.net.2",
        "ff.net.0.proj",
    ]
    base_model_dir = model_dir / "stable-diffusion-2-1-base"
    weight_path = model_dir / "hypir_sd21.pth"
    txt_dir = model_dir / "text_folder"
    
    model = SD2Enhancer(
        base_model_path=base_model_dir,
        weight_path=weight_path,
        lora_modules=lora_modules,
        lora_rank=256,
        model_t=200,
        coeff_t=200,
        device=device,
    )
    print("Start loading models")
    load_start = time()
    model.init_models()
    print(f"Models loaded in {time() - load_start:.2f} seconds.")
    mamba_device = model_mamba.resolve_device(device)
    if mamba_device.type == "cuda":
        if mamba_device.index is not None:
            torch.cuda.set_device(mamba_device)
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    checkpoint_path = model_mamba.resolve_checkpoint_path(model_dir)
    runtime_opt = model_mamba.get_runtime_options(checkpoint_path, output_path, mamba_device)
    model_mamba_runner = model_mamba.build_model(runtime_opt)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                full_path = Path(root) / file
                images.append(full_path)
    images.sort(key=lambda x: str(x.relative_to(input_path)))
    print(f"Found {len(images)} images in {input_path}.")

    if txt_dir.exists() and txt_dir.is_dir():
        pass
    else:
        captioner = EmptyCaptioner(device)

    to_tensor = transforms.ToTensor()

    result_dir = output_path / "result"
    prompt_dir = output_path / "prompt"
    for file_path in images:
        print(f"Process file: \033[92m{os.path.basename(file_path)}\033[0m")

        relative_path = file_path.relative_to(input_path)
        result_path = result_dir / relative_path.with_suffix(".png")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path = prompt_dir / relative_path.with_suffix(".txt")
        prompt_path.parent.mkdir(parents=True, exist_ok=True)

        lq_pil = Image.open(file_path).convert("RGB")
        lq_tensor = to_tensor(lq_pil).unsqueeze(0)

        if txt_dir.exists() and txt_dir.is_dir():
            with open(txt_dir / relative_path.with_suffix(".txt"), "r") as fp:
                prompt = fp.read().strip()
        else:
            prompt = captioner(lq_pil)
        with open(prompt_path, "w") as fp:
            fp.write(prompt)
        print(f"Prompt: \033[94m{prompt}\033[0m")

        model_mamba_runner.feed_data({"lq": model_mamba.load_image_tensor(file_path)})
        model_mamba_runner.test()
        sr_img = model_mamba.tensor2img(model_mamba_runner.get_current_visuals()["result"])
        sr_tensor = to_tensor(Image.fromarray(sr_img[..., ::-1])).unsqueeze(0)
        del model_mamba_runner.lq
        del model_mamba_runner.output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = model.enhance(
            lq=sr_tensor,
            prompt=prompt,
            scale_by="factor",
            upscale=1,
            target_longest_side=None,
            patch_size=512,
            stride=256,
            return_type="pil",
        )[0]
        result.save(result_path)
    print(f"Done. \033[92mEnjoy your results in {result_dir}.\033[0m")
