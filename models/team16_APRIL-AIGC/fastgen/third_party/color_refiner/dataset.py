from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    input_paths: tuple[str, ...]
    target_path: str | None


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode.upper() == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        return background
    return image.convert("RGB")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _parse_path_list(raw_value: str | None, separator: str) -> list[str]:
    if raw_value is None:
        return []
    return [item.strip() for item in str(raw_value).split(separator) if item.strip()]


def _resolve_path(path_str: str, root_dir: str | None) -> str:
    path = Path(path_str)
    if path.is_absolute() or not root_dir:
        return str(path)
    return str(Path(root_dir) / path)


def _normalize_hw(value: int | Sequence[int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            return None
        return (value, value)
    values = [int(item) for item in value]
    if len(values) == 1:
        if values[0] <= 0:
            return None
        return (int(values[0]), int(values[0]))
    if len(values) == 2:
        if values[0] <= 0 or values[1] <= 0:
            return None
        return (int(values[0]), int(values[1]))
    raise ValueError(f"Expected int or 2-tuple, got {value}")


def _resize_image(image: Image.Image, resize_hw: tuple[int, int] | None) -> Image.Image:
    if resize_hw is None:
        return image
    height, width = resize_hw
    return image.resize((width, height), Image.BICUBIC)


def _pad_tensor_to_minimum(tensor: torch.Tensor, min_h: int, min_w: int) -> torch.Tensor:
    pad_h = max(0, min_h - tensor.shape[-2])
    pad_w = max(0, min_w - tensor.shape[-1])
    if pad_h == 0 and pad_w == 0:
        return tensor
    can_reflect = (
        tensor.shape[-2] > 1
        and tensor.shape[-1] > 1
        and pad_h < tensor.shape[-2]
        and pad_w < tensor.shape[-1]
    )
    pad_mode = "reflect" if can_reflect else "replicate"
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode=pad_mode)


class MultiInputImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        root_dir: str = "",
        input_col: str = "input_image",
        target_col: str = "gt",
        id_col: str = "id",
        input_separator: str = ",",
        max_inputs: int | None = None,
        patch_size: int | Sequence[int] | None = None,
        resize: int | Sequence[int] | None = None,
        training: bool = False,
        augment: bool = False,
    ) -> None:
        self.csv_path = csv_path
        self.root_dir = root_dir
        self.input_col = input_col
        self.target_col = target_col
        self.id_col = id_col
        self.input_separator = input_separator
        self.patch_size = _normalize_hw(patch_size)
        self.resize = _normalize_hw(resize)
        self.training = training
        self.augment = augment and training
        self.records = self._load_records()
        inferred_max_inputs = max(len(record.input_paths) for record in self.records)
        self.max_inputs = max_inputs or inferred_max_inputs
        if self.max_inputs < 1:
            raise ValueError("max_inputs must be >= 1")

    def _load_records(self) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        with open(self.csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader):
                input_paths = _parse_path_list(row.get(self.input_col), self.input_separator)
                if not input_paths:
                    continue
                sample_id = str(row.get(self.id_col) or row_index)
                target_path = row.get(self.target_col)
                if target_path is not None and not str(target_path).strip():
                    target_path = None
                records.append(
                    SampleRecord(
                        sample_id=sample_id,
                        input_paths=tuple(_resolve_path(path, self.root_dir) for path in input_paths),
                        target_path=_resolve_path(str(target_path), self.root_dir) if target_path else None,
                    )
                )
        if not records:
            raise ValueError(f"No valid samples found in {self.csv_path}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: str) -> Image.Image:
        image = Image.open(path)
        return _to_rgb(image)

    def _load_sample_images(self, record: SampleRecord) -> tuple[list[Image.Image], Image.Image | None]:
        input_images = [self._load_image(path) for path in record.input_paths]
        target_image = self._load_image(record.target_path) if record.target_path else None
        return input_images, target_image

    def _match_sizes(
        self,
        input_images: list[Image.Image],
        target_image: Image.Image | None,
    ) -> tuple[list[Image.Image], Image.Image | None]:
        if self.resize is not None:
            resized_inputs = [_resize_image(image, self.resize) for image in input_images]
            resized_target = _resize_image(target_image, self.resize) if target_image is not None else None
            return resized_inputs, resized_target

        if target_image is not None:
            target_hw = (target_image.height, target_image.width)
        else:
            target_hw = (input_images[0].height, input_images[0].width)

        resize_hw = target_hw
        resized_inputs = [_resize_image(image, resize_hw) for image in input_images]
        resized_target = _resize_image(target_image, resize_hw) if target_image is not None else None
        return resized_inputs, resized_target

    def _crop_and_augment(
        self,
        input_tensors: list[torch.Tensor],
        target_tensor: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        tensors = list(input_tensors) + [target_tensor]
        if self.patch_size is not None:
            patch_h, patch_w = self.patch_size
            tensors = [_pad_tensor_to_minimum(tensor, patch_h, patch_w) for tensor in tensors]
            height, width = tensors[0].shape[-2:]
            top = random.randint(0, height - patch_h)
            left = random.randint(0, width - patch_w)
            tensors = [tensor[:, top : top + patch_h, left : left + patch_w] for tensor in tensors]

        if self.augment:
            if random.random() < 0.5:
                tensors = [tensor.flip(-1) for tensor in tensors]
            if random.random() < 0.5:
                tensors = [tensor.flip(-2) for tensor in tensors]
            if tensors[0].shape[-2] == tensors[0].shape[-1] and random.random() < 0.5:
                tensors = [tensor.transpose(-1, -2).contiguous() for tensor in tensors]

        return tensors[:-1], tensors[-1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        input_images, target_image = self._load_sample_images(record)
        input_images, target_image = self._match_sizes(input_images, target_image)

        input_tensors = [_pil_to_tensor(image) for image in input_images]
        if target_image is not None:
            target_tensor = _pil_to_tensor(target_image)
        else:
            target_tensor = torch.zeros_like(input_tensors[0])

        input_tensors, target_tensor = self._crop_and_augment(input_tensors, target_tensor)

        stacked_inputs = torch.zeros(
            self.max_inputs,
            3,
            input_tensors[0].shape[-2],
            input_tensors[0].shape[-1],
            dtype=input_tensors[0].dtype,
        )
        input_mask = torch.zeros(self.max_inputs, dtype=torch.float32)

        for input_index, input_tensor in enumerate(input_tensors[: self.max_inputs]):
            stacked_inputs[input_index] = input_tensor
            input_mask[input_index] = 1.0

        return {
            "inputs": stacked_inputs,
            "input_mask": input_mask,
            "target": target_tensor,
            "has_target": torch.tensor(target_image is not None, dtype=torch.bool),
            "sample_id": record.sample_id,
            "input_paths": "|".join(record.input_paths[: self.max_inputs]),
            "target_path": record.target_path or "",
        }
