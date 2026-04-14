#!/usr/bin/env python3
"""
predict.py - HMANet 4x Super-Resolution Inference Script

Usage:
    python predict.py --model_path /path/to/model.net --input_dir /path/to/input --output_dir /path/to/output
    python predict.py --model_path /path/to/model.net --input_dir /path/to/input --output_dir /path/to/output --tile 256 --tile_overlap 32
"""

import os
import glob
from dataclasses import dataclass, field
from typing import List

import torch
import numpy as np
import cv2
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import get_network


@dataclass
class ModelParams:
    """Model hyperparameters dataclass."""
    embed_dim: int = 180
    depths: List[int] = field(default_factory=lambda: [6, 6, 6, 6, 6, 6])
    num_heads: List[int] = field(default_factory=lambda: [6, 6, 6, 6, 6, 6])
    window_size: int = 16
    interval_size: int = 4
    mlp_ratio: float = 2.0
    drop_path_rate: float = 0.1
    upscale: int = 4
    flag_tune_last_tail: bool = False


@dataclass
class InferenceConfig:
    """Inference configuration dataclass."""
    # Tile processing
    tile: int = 256
    tile_overlap: int = 64
    
    # Data settings
    ext: str = "png"
    max_range: float = 1.0
    
    # Model hyperparameters
    embed_dim: int = 180
    depths: List[int] = field(default_factory=lambda: [6, 6, 6, 6, 6, 6])
    num_heads: List[int] = field(default_factory=lambda: [6, 6, 6, 6, 6, 6])
    window_size: int = 16
    interval_size: int = 4
    mlp_ratio: float = 2.0
    
    # Device settings
    num_gpus: int = 1


def imread_uint(path):
    """Read image as uint8 format."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


def imwrite_uint(img, path):
    """Write uint8 image to file."""
    cv2.imwrite(path, img)


def uint2tensor4(img, max_range=1.0):
    """
    Convert uint8 image to 4D tensor.
    img: HWC, uint8, [0, 255]
    max_range: Output data range, 1 for [0, 1], 255 for [0, 255]
    return: BCHW, float32, [0, max_range]
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float32)
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0 * 255.0
    else:
        img = img.astype(np.float32)
    
    if max_range == 1.0:
        img = img / 255.0
    
    # HWC -> CHW
    if len(img.shape) == 3:
        img = np.transpose(img, (2, 0, 1))
    else:
        img = np.expand_dims(img, axis=0)
    
    # CHW -> BCHW
    img = np.expand_dims(img, axis=0)
    
    return torch.from_numpy(img)


def tensor2uint(tensor, max_range=1.0):
    """
    Convert tensor to uint8 image.
    tensor: BCHW or CHW, float32
    max_range: Input data range, 1 for [0, 1], 255 for [0, 255]
    return: HWC, uint8, [0, 255]
    """
    # Remove batch dimension
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    
    # CHW -> HWC
    tensor = tensor.permute(1, 2, 0)
    
    # Convert to numpy
    img = tensor.cpu().numpy()
    
    # Scale to [0, 255]
    if max_range == 1.0:
        img = img * 255.0
    
    img = np.round(np.clip(img, 0, 255)).astype(np.uint8)
    
    return img


def pad_image_to_multiple(img, multiple=128):
    """
    Pad image to multiple of specified size.
    img: HWC numpy array
    return: padded_img, (orig_h, orig_w, h_pad, w_pad)
    """
    h, w = img.shape[:2]
    h_pad = (multiple - h % multiple) % multiple
    w_pad = (multiple - w % multiple) % multiple
    
    if h_pad == 0 and w_pad == 0:
        return img, (h, w, 0, 0)
    
    if len(img.shape) == 3:
        padded = np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)), mode='reflect')
    else:
        padded = np.pad(img, ((0, h_pad), (0, w_pad)), mode='reflect')
    
    return padded, (h, w, h_pad, w_pad)


def process_image_with_tile(model, img_lq_tensor, tile, tile_overlap=32, scale=4):
    """
    Process image with tile mode and overlap blending.
    
    Args:
        model: Model
        img_lq_tensor: Input image tensor (BCHW)
        tile: Tile size
        tile_overlap: Overlap size between tiles (default: 32)
        scale: Super-resolution scale
    
    Returns:
        output: Super-resolution result tensor
    """
    b, c, h, w = img_lq_tensor.size()
    tile = min(tile, h, w)
    
    # Ensure tile size is valid
    if tile < tile_overlap * 2:
        tile_overlap = tile // 4
    
    # Calculate stride
    stride = tile - tile_overlap
    
    # Calculate tile positions
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    
    # Test one patch to get actual output size
    with torch.no_grad():
        test_patch = img_lq_tensor[..., 0:tile, 0:tile]
        test_out = model(test_patch)
        _, _, out_h_tile, out_w_tile = test_out.size()
    
    # Create output buffer and weight buffer
    E = torch.zeros(b, c, h * scale, w * scale).type_as(img_lq_tensor)
    W = torch.zeros_like(E)
    
    # Process each tile
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            # Extract input tile
            in_patch = img_lq_tensor[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            
            # Inference
            with torch.no_grad():
                out_patch = model(in_patch)
            
            _, _, out_h, out_w = out_patch.size()
            
            # Create blending weight
            out_patch_mask = torch.ones_like(out_patch)
            
            # Accumulate to output buffer
            h_start = h_idx * scale
            w_start = w_idx * scale
            E[..., h_start:h_start + out_h, w_start:w_start + out_w].add_(out_patch)
            W[..., h_start:h_start + out_h, w_start:w_start + out_w].add_(out_patch_mask)
    
    # Weighted average (blend overlap regions)
    output = E.div_(W)
    
    return output


def process_image(model, img_lq, device="cuda", tile=None, tile_overlap=32, 
                  scale=4, pad_multiple=128, max_range=1.0):
    """
    Process single image - 4x Super-Resolution.
    Input image is padded to multiple of pad_multiple, output is cropped accordingly.
    
    Args:
        model: Model (may be wrapped by DataParallel)
        img_lq: Low-resolution image (HWC numpy array)
        device: Device
        tile: Tile size (None for whole image processing)
        tile_overlap: Overlap size between tiles (default: 32, only for tile mode)
        scale: Super-resolution scale
        pad_multiple: Pad to multiple of this value
        max_range: Model input data range, 1 for [0, 1], 255 for [0, 255]
    
    Returns:
        img_sr: Super-resolution result (cropped)
        orig_h, orig_w: Original input dimensions
    """
    orig_h, orig_w = img_lq.shape[:2]
    
    # Pad to multiple of pad_multiple
    img_lq_padded, (_, _, h_pad, w_pad) = pad_image_to_multiple(img_lq, multiple=pad_multiple)
    
    # Convert to tensor
    if isinstance(device, str) and device.startswith('cuda'):
        img_lq_tensor = uint2tensor4(img_lq_padded, max_range=max_range).to('cuda:0')
    else:
        img_lq_tensor = uint2tensor4(img_lq_padded, max_range=max_range).to(device)

    # Inference (4x SR)
    with torch.no_grad():
        if tile is not None:
            # Tile mode
            output = process_image_with_tile(model, img_lq_tensor, tile, tile_overlap, scale)
        else:
            # Whole image processing
            output = model(img_lq_tensor)

    # Convert back to uint8
    img_sr_padded = tensor2uint(output, max_range=max_range)
    
    # Crop padding
    target_h = orig_h * scale
    target_w = orig_w * scale
    img_sr = img_sr_padded[:target_h, :target_w]

    return img_sr, orig_h, orig_w


def load_model(model_path, device='cuda', model_hp=None, num_gpus=1):
    """
    Load model from checkpoint.
    
    Args:
        model_path: Model path
        device: Device
        model_hp: ModelParams object, used if checkpoint doesn't contain model_hp
        num_gpus: Number of GPUs to use (>1 enables DataParallel)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model hyperparameters from checkpoint or use defaults
    model_hp_dict = checkpoint.get("model_hp")
    
    if model_hp_dict is not None:
        model_hp = ModelParams(**model_hp_dict)
        print(f"Model config from checkpoint: embed_dim={model_hp.embed_dim}, "
              f"depths={model_hp.depths}, num_heads={model_hp.num_heads}")
    elif model_hp is not None:
        print(f"Model config from args: embed_dim={model_hp.embed_dim}, "
              f"depths={model_hp.depths}, num_heads={model_hp.num_heads}")
    else:
        # Default configuration
        model_hp = ModelParams()
        print(f"Model config (default): embed_dim={model_hp.embed_dim}, "
              f"depths={model_hp.depths}, num_heads={model_hp.num_heads}")
    
    # Create model
    model = get_network(model_hp)
    
    # Load weights
    if "params" in checkpoint:
        state_dict = checkpoint["params"]
        print("Using standard weights")
    else:
        state_dict = checkpoint
        print("Using checkpoint as state_dict")
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
    
    model = model.to(device)
    
    model.eval()
    
    # Freeze parameters
    for _, v in model.named_parameters():
        v.requires_grad = False
    
    print(f"Model loaded successfully!")
    return model


def main(model_dir: str, input_path: str, output_path: str, device: str = "cuda", 
         config: InferenceConfig = None):
    """
    Main inference function.
    
    Args:
        model_dir: Directory containing the model file
        input_path: Input directory containing LR images
        output_path: Output directory for SR images
        device: Device to use (cuda or cpu)
        config: Inference configuration (uses default if None)
    """
    input_dir = input_path
    output_dir = output_path
    
    # Use default config if not provided
    if config is None:
        config = InferenceConfig()

    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Create ModelParams from config
    model_hp = ModelParams(
        embed_dim=config.embed_dim,
        depths=list(config.depths),
        num_heads=list(config.num_heads),
        window_size=config.window_size,
        interval_size=config.interval_size,
        mlp_ratio=config.mlp_ratio,
        drop_path_rate=0.1,
    )
    
    # Load model
    model_path = os.path.join(model_dir, "team23_AxeraSR.net")
    model_path = "/data2/NTIRE26/NTIRE2026/SR/Team23/NTIRE2026_ImageSR_x4-main-AxeraAI/model_zoo/team23_AxeraSR/AxeraSR.net"
    model = load_model(model_path, device=device, model_hp=model_hp, 
                       num_gpus=config.num_gpus)
    
    # Get input image list
    input_pattern = os.path.join(input_dir, f"*.{config.ext}")
    img_paths = sorted(glob.glob(input_pattern))
    
    if not img_paths:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(img_paths)} images to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Process each image
    processed_files = []
    dimension_errors = []
    
    for img_path in tqdm(img_paths, desc="Processing"):
        img_name = os.path.basename(img_path)
        save_path = os.path.join(output_dir, img_name)
        
        try:
            # Read original image
            img_lq = imread_uint(img_path)
            orig_h, orig_w = img_lq.shape[:2]
            
            # Process image (4x SR)
            img_sr, out_h, out_w = process_image(
                model, img_lq, device=device, tile=config.tile, 
                tile_overlap=config.tile_overlap, scale=4, max_range=config.max_range
            )
            
            # Verify dimensions: output should be 4x of input
            expected_h, expected_w = orig_h * 4, orig_w * 4
            if img_sr.shape[0] != expected_h or img_sr.shape[1] != expected_w:
                dimension_errors.append({
                    'file': img_name,
                    'input_size': (orig_h, orig_w),
                    'expected_output': (expected_h, expected_w),
                    'actual_output': (img_sr.shape[0], img_sr.shape[1])
                })
            
            # Save result
            cv2.imwrite(save_path, img_sr)
            processed_files.append(img_name)
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue
    
    # Validation
    print(f"\n{'='*60}")
    print("Validation Results:")
    print(f"{'='*60}")
    
    # Check file count
    output_files = sorted([f for f in os.listdir(output_dir) if f.endswith(f'.{config.ext}')])
    if len(output_files) == len(img_paths):
        print(f"✓ File count check PASSED: {len(output_files)} files")
    else:
        print(f"✗ File count check FAILED: input={len(img_paths)}, output={len(output_files)}")
    
    # Check dimension errors
    if dimension_errors:
        print(f"✗ Dimension check FAILED: {len(dimension_errors)} files have incorrect dimensions")
        for err in dimension_errors[:5]:
            print(f"  - {err['file']}: input={err['input_size']}, expected={err['expected_output']}, got={err['actual_output']}")
    else:
        print(f"✓ Dimension check PASSED: all outputs are 4x of inputs")
    
    print(f"{'='*60}")
    print(f"Done! Results saved to: {output_dir}")
    print(f"Processed {len(processed_files)} images successfully")


if __name__ == "__main__":
    model_dir = "model_zoo/23_AxeraSR"
    input_dir = "/research_llcv/pengxiaoping/public_testsets/Urban100"
    output_dir = "cache/output"
    device = "cuda"
    
    # Create custom config if needed (optional)
    # config = InferenceConfig(tile=512, tile_overlap=128)
    # main(model_dir, input_dir, output_dir, device, config=config)
    
    main(model_dir, input_dir, output_dir, device)
