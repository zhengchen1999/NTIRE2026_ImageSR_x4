from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def parse_hw(values: list[int] | tuple[int, ...] | None) -> tuple[int, int] | None:
    if values is None:
        return None
    if len(values) == 1:
        return (int(values[0]), int(values[0]))
    if len(values) == 2:
        return (int(values[0]), int(values[1]))
    raise ValueError(f"Expected 1 or 2 integers, got {values}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().clamp(0.0, 1.0).mul(255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def save_image_tensor(tensor: torch.Tensor, path: str | Path) -> None:
    image = tensor_to_image(tensor)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_image_strip(tensors: list[torch.Tensor], path: str | Path, gap: int = 4) -> None:
    images = [tensor_to_image(tensor) for tensor in tensors]
    if not images:
        raise ValueError("save_image_strip requires at least one image")

    total_width = sum(image.width for image in images) + gap * max(0, len(images) - 1)
    max_height = max(image.height for image in images)
    canvas = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

    offset_x = 0
    for image in images:
        canvas.paste(image, (offset_x, 0))
        offset_x += image.width + gap

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def forward_with_tiling(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    input_mask: torch.Tensor,
    tile_size: int = 0,
    tile_overlap: int = 32,
) -> torch.Tensor:
    if tile_size <= 0:
        return model(inputs, input_mask)

    if inputs.shape[0] != 1:
        raise ValueError("Tiled inference currently expects batch_size == 1")

    _, _, _, height, width = inputs.shape
    if max(height, width) <= tile_size:
        return model(inputs, input_mask)

    stride = max(1, tile_size - tile_overlap)
    prediction = torch.zeros((1, 3, height, width), device=inputs.device, dtype=inputs.dtype)
    weight = torch.zeros((1, 1, height, width), device=inputs.device, dtype=inputs.dtype)

    top_positions = list(range(0, max(height - tile_size, 0) + 1, stride))
    left_positions = list(range(0, max(width - tile_size, 0) + 1, stride))
    if top_positions[-1] != max(height - tile_size, 0):
        top_positions.append(max(height - tile_size, 0))
    if left_positions[-1] != max(width - tile_size, 0):
        left_positions.append(max(width - tile_size, 0))

    for top in top_positions:
        for left in left_positions:
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            top = max(0, bottom - tile_size)
            left = max(0, right - tile_size)
            tiled_inputs = inputs[..., top:bottom, left:right]
            tiled_prediction = model(tiled_inputs, input_mask)
            prediction[..., top:bottom, left:right] += tiled_prediction
            weight[..., top:bottom, left:right] += 1.0

    return prediction / weight.clamp_min(1.0)
