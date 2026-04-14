"""
Fusion-based Ensemble Super-Resolution

A lightweight learned fusion approach combining multiple frozen SR backbones
to achieve better performance with minimal training data and time.

Architecture:
    - Frozen SR backbones (BBox MSHAT + XiaomiMM HATIQCMix)
    - Lightweight trainable FusionNet (~30K parameters)
    - 8-way Test-Time Augmentation

Usage:
    # Training
    python -m models.fusion.train \
        --lr_dir /path/to/DIV2K_train_LR_bicubic/X4 \
        --hr_dir /path/to/DIV2K_train_HR \
        --output_dir ./experiments/fusion \
        --epochs 100

    # Inference
    python -m models.fusion.inference \
        --input_dir /path/to/test_LR \
        --output_dir /path/to/output \
        --fusion_weights ./experiments/fusion/best.pth
"""

from .fusion_net import (
    FusionNet,
    FusionNetLarge,
    CombinedLoss,
    CharbonnierLoss,
    calculate_psnr,
    calculate_psnr_y
)

__all__ = [
    'FusionNet',
    'FusionNetLarge', 
    'CombinedLoss',
    'CharbonnierLoss',
    'calculate_psnr',
    'calculate_psnr_y',
]
