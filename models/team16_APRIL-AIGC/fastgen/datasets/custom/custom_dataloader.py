# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from torch.utils.data import DataLoader

from fastgen.datasets.custom.restoration_dataset import RestorationMultiCondDataset
from fastgen.datasets.custom.restoration_dataset_face import FaceRestorationDegradeDataset
from fastgen.datasets.custom.restoration_dataset_denoising import ImageDenoisingDataset
from fastgen.datasets.custom.restoration_dataset_super_resolution import ImageSuperResolutionX4Dataset
from fastgen.datasets.custom.MultiResolutionSampler import ImageMultiResolutionSampler
from fastgen.datasets.custom.bucket_image import ImageBucket 


def _collect_dataset_kwargs(kwargs: dict, allowed_keys: tuple[str, ...]) -> dict:
    return {key: kwargs[key] for key in allowed_keys if key in kwargs}


def _build_loader(
    dataset,
    dataset_path: str,
    resolutions: list,
    batch_size_config: dict,
    shuffle: bool,
    sampler_start_idx: int,
    exact_resolutions: list,
):
    bucket = ImageBucket(
        resolutions=resolutions,
        batch_size_config=batch_size_config,
        exact_resolutions=exact_resolutions,
    )

    dataset_sampler = ImageMultiResolutionSampler(
        dataset_path=dataset_path,
        bucket=bucket,
        shuffle=shuffle,
        start_idx=sampler_start_idx,
    )
    data_loader_kwargs = dict(
        num_workers=4,  # don't change this, otherwise it will cause BadZipFile error
        pin_memory=True,
        prefetch_factor=2,
    )
    return DataLoader(dataset=dataset, batch_sampler=dataset_sampler, **data_loader_kwargs)


class CustomImageLoader:
    def __init__(
        self,
        dataset_path: str,
        resolutions: list,
        batch_size_config: dict,
        task : str, #default prompt
        use_labels: bool = True,
        cache: bool = True,
        shuffle: bool = True,
        sampler_start_idx: int = 0,
        exact_resolutions = [],
        **kwargs,
    ):

        self.dataset = RestorationMultiCondDataset(
            dataset_path=dataset_path, task=task,
        )
        self.loader = _build_loader(
            dataset=self.dataset,
            dataset_path=dataset_path,
            resolutions=resolutions,
            batch_size_config=batch_size_config,
            shuffle=shuffle,
            sampler_start_idx=sampler_start_idx,
            exact_resolutions=exact_resolutions,
        )

    def __iter__(self):
        return iter(self.loader)


class CustomFaceImageLoader:
    def __init__(
        self,
        dataset_path: str,
        resolutions: list,
        batch_size_config: dict,
        use_labels: bool = True,
        cache: bool = True,
        shuffle: bool = True,
        sampler_start_idx: int = 0,
        exact_resolutions = [],
        **kwargs,
    ):


        self.dataset = FaceRestorationDegradeDataset(
            dataset_path=dataset_path,
        )
        self.loader = _build_loader(
            dataset=self.dataset,
            dataset_path=dataset_path,
            resolutions=resolutions,
            batch_size_config=batch_size_config,
            shuffle=shuffle,
            sampler_start_idx=sampler_start_idx,
            exact_resolutions=exact_resolutions,
        )

    def __iter__(self):
        return iter(self.loader)


class CustomDenoisingImageLoader:
    def __init__(
        self,
        dataset_path: str,
        resolutions: list,
        batch_size_config: dict,
        use_labels: bool = True,
        cache: bool = True,
        shuffle: bool = True,
        sampler_start_idx: int = 0,
        exact_resolutions = [],
        **kwargs,
    ):
        dataset_kwargs = _collect_dataset_kwargs(
            kwargs,
            (
                "root_dir",
                "cfg_prob",
                "sigma",
                "deterministic_noise",
                "noise_seed",
                "transform",
                "realesrgan_prob",
                "degrade_resize_bak",
                "degradation_device",
            ),
        )
        self.dataset = ImageDenoisingDataset(dataset_path=dataset_path, **dataset_kwargs)
        self.loader = _build_loader(
            dataset=self.dataset,
            dataset_path=dataset_path,
            resolutions=resolutions,
            batch_size_config=batch_size_config,
            shuffle=shuffle,
            sampler_start_idx=sampler_start_idx,
            exact_resolutions=exact_resolutions,
        )

    def __iter__(self):
        return iter(self.loader)


class CustomSuperResolutionImageLoader:
    def __init__(
        self,
        dataset_path: str,
        resolutions: list,
        batch_size_config: dict,
        use_labels: bool = True,
        cache: bool = True,
        shuffle: bool = True,
        sampler_start_idx: int = 0,
        exact_resolutions = [],
        **kwargs,
    ):
        dataset_kwargs = _collect_dataset_kwargs(
            kwargs,
            (
                "root_dir",
                "cfg_prob",
                "scale",
                "transform",
                "realesrgan_prob",
                "degrade_resize_bak",
                "degradation_device",
            ),
        )
        self.dataset = ImageSuperResolutionX4Dataset(dataset_path=dataset_path, **dataset_kwargs)
        self.loader = _build_loader(
            dataset=self.dataset,
            dataset_path=dataset_path,
            resolutions=resolutions,
            batch_size_config=batch_size_config,
            shuffle=shuffle,
            sampler_start_idx=sampler_start_idx,
            exact_resolutions=exact_resolutions,
        )

    def __iter__(self):
        return iter(self.loader)
