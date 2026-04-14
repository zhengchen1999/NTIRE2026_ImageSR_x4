"""
Fusion Networks for Multi-Model SR Ensemble

FusionNet   — v1 original: ~30K params, softmax attention, 2-layer encoder
FusionNetV2 — v2 improved: ~120K params, dilated multi-scale encoder,
              sigmoid attention, variance-map input, larger receptive field.
              Based on NTIRE 2024 findings: Charbonnier + FFT + Sobel loss,
              per-pixel disagreement map, cosine LR schedule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention for adaptive feature weighting."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FusionNet(nn.Module):
    """
    Lightweight fusion network for combining multiple SR model outputs.
    
    Args:
        num_models: Number of SR models to fuse (default: 2 for BBox + XiaomiMM)
        base_channels: Base channel count for encoder (default: 32)
        use_residual: Add residual refinement layer (default: True)
    
    Input: Concatenated SR outputs [B, num_models*3, H, W]
    Output: Fused SR image [B, 3, H, W]
    """
    
    def __init__(self, num_models=2, base_channels=32, use_residual=True):
        super().__init__()
        self.num_models = num_models
        self.use_residual = use_residual
        in_channels = num_models * 3
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels * 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Channel attention
        self.ca = ChannelAttention(base_channels)
        
        # Attention weight predictor (outputs num_models weights per pixel)
        self.attention = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 2, num_models, 1),
        )
        
        # Optional residual refinement
        if use_residual:
            self.refine = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(16, 3, 3, padding=1),
            )
            self.residual_scale = nn.Parameter(torch.tensor(0.1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_weights=False):
        """
        Args:
            x: Concatenated SR outputs [B, num_models*3, H, W]
            return_weights: If True, also return attention weights
        
        Returns:
            fused: Fused SR image [B, 3, H, W]
            weights: (optional) Attention weights [B, num_models, H, W]
        """
        # Split input into individual model outputs
        outputs = x.chunk(self.num_models, dim=1)  # List of [B, 3, H, W]
        
        # Encode features
        feat = self.encoder(x)
        feat = self.ca(feat)
        
        # Predict attention weights
        weights = self.attention(feat)  # [B, num_models, H, W]
        weights = F.softmax(weights, dim=1)  # Normalize to sum to 1
        
        # Weighted fusion
        fused = torch.zeros_like(outputs[0])
        for i, out in enumerate(outputs):
            w = weights[:, i:i+1, :, :]  # [B, 1, H, W]
            fused = fused + w * out
        
        # Optional residual refinement
        if self.use_residual:
            residual = self.refine(fused)
            fused = fused + residual * self.residual_scale
        
        fused = torch.clamp(fused, 0, 1)
        
        if return_weights:
            return fused, weights
        return fused
    
    def get_param_count(self):
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FusionNetLarge(nn.Module):
    """
    Larger fusion network with multi-scale processing.
    Parameters: ~100K
    """
    
    def __init__(self, num_models=2, base_channels=64):
        super().__init__()
        self.num_models = num_models
        in_channels = num_models * 3
        
        # Multi-scale encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Fusion with multi-scale features
        self.fuse_feat = nn.Conv2d(base_channels * 2 + base_channels // 2, base_channels, 1)
        
        # Attention
        self.attention = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 2, num_models, 1),
        )
        
        # Refinement
        self.refine = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
        )
    
    def forward(self, x, return_weights=False):
        outputs = x.chunk(self.num_models, dim=1)
        
        # Multi-scale encoding
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        
        # Fuse multi-scale features
        feat = torch.cat([f1, f2, f3], dim=1)
        feat = self.fuse_feat(feat)
        
        # Attention weights
        weights = F.softmax(self.attention(feat), dim=1)
        
        # Weighted fusion
        fused = sum(weights[:, i:i+1] * out for i, out in enumerate(outputs))
        
        # Refine
        fused = fused + self.refine(fused) * 0.1
        fused = torch.clamp(fused, 0, 1)
        
        if return_weights:
            return fused, weights
        return fused


# ============================================================================
# FusionNetV2 — Improved fusion with variance map + dilated encoder
# ============================================================================

class FusionNetV2(nn.Module):
    """
    Improved fusion network (v2) for combining multiple SR model outputs.

    Key improvements over FusionNet v1:
      - Accepts per-pixel variance map as extra input channel (+1 ch)
        giving the network an explicit disagreement signal between models
      - 4-layer dilated encoder [d=1,2,4,1] → receptive field ~29×29 px
        vs. 5×5 in v1; enough context to route by region type
      - Multi-scale feature aggregation: attention uses all encoder levels
      - Sigmoid per-model attention (not softmax): allows both models
        to have weight >0.5 in high-consensus regions (amplification)
      - Deeper refinement block for sharper details

    Args:
        num_models  : number of SR models (default 2)
        base_ch     : base channel width (default 64)
        use_var_map : whether variance map is included in input (default True)

    Input : cat([model_1, ..., model_N, var_map]) → [B, N*3+1, H, W]
            (if use_var_map=False:            → [B, N*3,   H, W])
    Output: fused SR image [B, 3, H, W]
    """

    def __init__(self, num_models=2, base_ch=64, use_var_map=True):
        super().__init__()
        self.num_models  = num_models
        self.use_var_map = use_var_map
        in_ch = num_models * 3 + (1 if use_var_map else 0)

        # Dilated encoder: receptive field grows 3→7→15→17 px per level
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch,    base_ch, 3, padding=1, dilation=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch,  base_ch, 3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_ch,  base_ch, 3, padding=4, dilation=4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_ch,  base_ch // 2, 3, padding=1, dilation=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 1×1 projection of concatenated multi-scale features
        ms_ch = base_ch * 3 + base_ch // 2  # enc1+enc2+enc3+enc4
        self.fuse_feat = nn.Sequential(
            nn.Conv2d(ms_ch, base_ch, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Channel attention on fused features
        self.ca = ChannelAttention(base_ch, reduction=4)

        # Per-pixel sigmoid attention (one weight per model, independent)
        self.attention = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch // 2, num_models, 1),
            nn.Sigmoid(),           # independent sigmoid — allows >0.5 on both
        )

        # Normalize sigmoid weights to sum=1 per pixel (done in forward)

        # Residual refinement
        self.refine = nn.Sequential(
            nn.Conv2d(3, base_ch // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch // 2, base_ch // 4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch // 4, 3, 3, padding=1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_weights=False):
        """
        x : [B, num_models*3 (+ 1 if use_var_map), H, W]
            First num_models*3 channels are SR outputs (3 ch each).
            Last channel (if use_var_map) is the cross-model variance map.
        """
        # Split off per-model SR outputs (first num_models*3 channels)
        sr_concat = x[:, :self.num_models * 3]
        outputs   = sr_concat.chunk(self.num_models, dim=1)  # List[B,3,H,W]

        # Multi-scale dilated encoding
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)

        feat = torch.cat([f1, f2, f3, f4], dim=1)
        feat = self.fuse_feat(feat)
        feat = self.ca(feat)

        # Per-pixel attention weights (sigmoid, then L1-normalize)
        raw_w = self.attention(feat)                        # [B, N, H, W]
        weights = raw_w / (raw_w.sum(dim=1, keepdim=True) + 1e-6)  # sum-to-1

        # Weighted fusion
        fused = sum(weights[:, i:i+1] * out
                    for i, out in enumerate(outputs))

        # Residual refinement
        fused = fused + self.refine(fused) * self.residual_scale
        fused = torch.clamp(fused, 0, 1)

        if return_weights:
            return fused, weights
        return fused

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Loss Functions
# ============================================================================

class CombinedLoss(nn.Module):
    """Combined L1 + MSE loss optimized for PSNR."""
    
    def __init__(self, l1_weight=0.8, mse_weight=0.2):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.l1_weight * self.l1(pred, target) + self.mse_weight * self.mse(pred, target)


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (smooth L1)."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps))


class FFTLoss(nn.Module):
    """
    Focal Frequency Loss (Jiang et al., ICCV 2021).
    Penalises L1 difference of DFT magnitude spectra, recovering
    high-frequency details lost by pixel-space L1/MSE training.
    """

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target):
        # rfft2 is faster than full fft2 for real inputs
        pred_f   = torch.fft.rfft2(pred,   norm='ortho')
        target_f = torch.fft.rfft2(target, norm='ortho')
        # Compare magnitude spectra
        loss = F.l1_loss(pred_f.abs(), target_f.abs())
        return self.weight * loss


class SobelLoss(nn.Module):
    """
    Gradient (Sobel) loss: penalises errors in horizontal + vertical
    image gradients, directly encouraging sharp edges/textures.
    Used by NTIRE 2024 3rd-place (UCAS-SCST, HFT model, 31.28 dB).
    """

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        # Sobel kernels as fixed buffers (not parameters)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def _grad(self, img):
        """Compute gradient magnitude for a [B,3,H,W] image."""
        # Process each channel independently
        B, C, H, W = img.shape
        img_flat = img.view(B * C, 1, H, W)
        gx = F.conv2d(img_flat, self.kx, padding=1)
        gy = F.conv2d(img_flat, self.ky, padding=1)
        grad = (gx ** 2 + gy ** 2 + 1e-6).sqrt()
        return grad.view(B, C, H, W)

    def forward(self, pred, target):
        return self.weight * F.l1_loss(self._grad(pred), self._grad(target))


class CombinedLossV2(nn.Module):
    """
    NTIRE 2024-inspired composite loss for fusion training:
        L = CharbonnierLoss + fft_w * FFTLoss + sobel_w * SobelLoss

    Default weights (fft_w=0.05, sobel_w=0.1) follow the confirmed ratio
    used by UCAS-SCST (31.28 dB, NTIRE 2024 3rd place).
    """

    def __init__(self, fft_w=0.05, sobel_w=0.1, eps=1e-6):
        super().__init__()
        self.charb  = CharbonnierLoss(eps=eps)
        self.fft    = FFTLoss(weight=fft_w)
        self.sobel  = SobelLoss(weight=sobel_w)

    def forward(self, pred, target):
        return self.charb(pred, target) + \
               self.fft  (pred, target) + \
               self.sobel(pred, target)


# ============================================================================
# Utility Functions
# ============================================================================

def calculate_psnr(pred, target, max_val=1.0):
    """Calculate PSNR between two images."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)


def calculate_psnr_y(pred, target, shave=4):
    """Calculate PSNR on Y channel (luminance) with border shaving."""
    # RGB to Y
    def rgb_to_y(img):
        return 16/255 + (65.481/255 * img[:, 0] + 128.553/255 * img[:, 1] + 24.966/255 * img[:, 2])
    
    pred_y = rgb_to_y(pred)
    target_y = rgb_to_y(target)
    
    # Shave borders
    if shave > 0:
        pred_y = pred_y[:, shave:-shave, shave:-shave]
        target_y = target_y[:, shave:-shave, shave:-shave]
    
    mse = torch.mean((pred_y - target_y) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)


if __name__ == '__main__':
    # Test the fusion network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = FusionNet(num_models=2, base_channels=32, use_residual=True).to(device)
    print(f"FusionNet parameters: {model.get_param_count():,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 6, 256, 256).to(device)  # 2 models × 3 channels
    output, weights = model(dummy_input, return_weights=True)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights sum: {weights.sum(dim=1).mean().item():.4f} (should be 1.0)")
