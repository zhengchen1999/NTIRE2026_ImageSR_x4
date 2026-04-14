"""
Multi-Model Fusion Architecture for Image Super-Resolution
Combines HAT, DAT, and NAFNet with 3x3 convolution fusion layers
"""
import torch
import torch.nn as nn
from .hat_arch import HAT
from .dat_arch import DAT
from .nafnet_arch import NAFNetSR


class _Registry:
    def register(self, cls=None):
        def wrapper(cls):
            return cls
        return wrapper if cls is None else cls
ARCH_REGISTRY = _Registry()


class ChannelAttention(nn.Module):
    """Channel Attention for fusion"""
    def __init__(self, num_feat, reduction=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // reduction, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // reduction, num_feat, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)


class SpatialFusion(nn.Module):
    """Spatial Fusion Module with 3x3 convolutions"""
    def __init__(self, in_channels, out_channels):
        super(SpatialFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        x = self.ca(x)
        return x


@ARCH_REGISTRY.register()
class HAT_DAT_NAFNet_Fusion_3x3(nn.Module):
    """
    Multi-Model Fusion Architecture with 3x3 Convolution Fusion
    Combines features from HAT, DAT, and NAFNet for improved super-resolution

    Difference from original: Uses 3x3 convolutions in fusion layers instead of 1x1
    """
    def __init__(
        self,
        upscale=4,
        in_chans=3,
        img_range=1.,
        freeze_backbone=True,
        freeze_strategy='all',
        # HAT parameters
        hat_embed_dim=180,
        hat_depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        hat_num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        hat_window_size=16,
        hat_compress_ratio=3,
        hat_squeeze_factor=30,
        hat_conv_scale=0.01,
        hat_overlap_ratio=0.5,
        hat_resi_connection='1conv',
        # DAT parameters
        dat_embed_dim=180,
        dat_depth=[6, 6, 6, 6, 6, 6],
        dat_num_heads=[6, 6, 6, 6, 6, 6],
        dat_expansion_factor=2,
        dat_split_size=[8, 32],
        dat_resi_connection='1conv',
        # NAFNet parameters
        nafnet_width=64,
        nafnet_middle_blk_num=12,
        nafnet_enc_blk_nums=[2, 2, 4, 8],
        nafnet_dec_blk_nums=[2, 2, 2, 2],
    ):
        super(HAT_DAT_NAFNet_Fusion_3x3, self).__init__()

        self.upscale = upscale
        self.img_range = img_range
        self.in_chans = in_chans
        self.freeze_backbone = freeze_backbone

        # Mean for normalization
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        # ========== Build HAT ==========
        self.hat = HAT(
            img_size=64,
            upscale=upscale,
            in_chans=in_chans,
            img_range=img_range,
            depths=hat_depths,
            embed_dim=hat_embed_dim,
            num_heads=hat_num_heads,
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection=hat_resi_connection,
            window_size=hat_window_size,
            compress_ratio=hat_compress_ratio,
            squeeze_factor=hat_squeeze_factor,
            conv_scale=hat_conv_scale,
            overlap_ratio=hat_overlap_ratio
        )

        # ========== Build DAT ==========
        self.dat = DAT(
            img_size=64,
            upscale=upscale,
            in_chans=in_chans,
            img_range=img_range,
            depth=dat_depth,
            embed_dim=dat_embed_dim,
            num_heads=dat_num_heads,
            expansion_factor=dat_expansion_factor,
            resi_connection=dat_resi_connection,
            split_size=dat_split_size,
            upsampler='pixelshuffle'
        )

        # ========== Build NAFNet ==========
        self.nafnet = NAFNetSR(
            img_channel=in_chans,
            width=nafnet_width,
            middle_blk_num=nafnet_middle_blk_num,
            enc_blk_nums=nafnet_enc_blk_nums,
            dec_blk_nums=nafnet_dec_blk_nums,
            upscale=upscale
        )

        # ========== Feature dimension matching ==========
        # HAT feature dim: hat_embed_dim (180)
        # DAT feature dim: dat_embed_dim (180)
        # NAFNet feature dim: nafnet_width (64)

        # Project NAFNet features to match HAT/DAT dimension
        self.nafnet_proj = nn.Conv2d(nafnet_width, hat_embed_dim, 1, 1, 0)

        # ========== Fusion Module with 3x3 Convolutions ==========
        # Input: hat_embed_dim + dat_embed_dim + hat_embed_dim = 180 + 180 + 180 = 540
        total_dim = hat_embed_dim + dat_embed_dim + hat_embed_dim

        # 使用3x3卷积进行融合，增加空间信息交互
        self.fusion = nn.Sequential(
            nn.Conv2d(total_dim, total_dim // 2, kernel_size=3, padding=1),  # 540 -> 270
            nn.GELU(),
            nn.Conv2d(total_dim // 2, hat_embed_dim, kernel_size=3, padding=1),  # 270 -> 180
        )

        # ========== Reconstruction Head ==========
        self.conv_before_upsample = nn.Conv2d(hat_embed_dim, 64 * (upscale ** 2), 3, 1, 1)
        self.upsample = nn.PixelShuffle(upscale)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # Apply initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for trainable parts"""
        for m in [self.nafnet_proj, self.fusion, self.conv_before_upsample, self.conv_last]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_pretrain(self, hat_path=None, dat_path=None, nafnet_path=None):
        """Load pretrained weights for backbone models"""
        if hat_path is not None:
            print(f"Loading HAT pretrained from: {hat_path}")
            checkpoint = torch.load(hat_path, map_location='cpu')
            if 'params_ema' in checkpoint:
                state_dict = checkpoint['params_ema']
            elif 'params' in checkpoint:
                state_dict = checkpoint['params']
            else:
                state_dict = checkpoint
            self.hat.load_state_dict(state_dict, strict=True)

        if dat_path is not None:
            print(f"Loading DAT pretrained from: {dat_path}")
            checkpoint = torch.load(dat_path, map_location='cpu')
            if 'params_ema' in checkpoint:
                state_dict = checkpoint['params_ema']
            elif 'params' in checkpoint:
                state_dict = checkpoint['params']
            else:
                state_dict = checkpoint
            self.dat.load_state_dict(state_dict, strict=True)

        if nafnet_path is not None:
            print(f"Loading NAFNet pretrained from: {nafnet_path}")
            checkpoint = torch.load(nafnet_path, map_location='cpu')
            if 'params_ema' in checkpoint:
                state_dict = checkpoint['params_ema']
            elif 'params' in checkpoint:
                state_dict = checkpoint['params']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            self.nafnet.load_state_dict(state_dict, strict=True)

    def freeze_backbones(self):
        """Freeze backbone model parameters"""
        if self.freeze_backbone:
            print("Freezing backbone models (HAT, DAT, NAFNet)")
            for param in self.hat.parameters():
                param.requires_grad = False
            for param in self.dat.parameters():
                param.requires_grad = False
            for param in self.nafnet.parameters():
                param.requires_grad = False

    def unfreeze_backbones(self):
        """Unfreeze backbone model parameters"""
        print("Unfreezing backbone models")
        for param in self.hat.parameters():
            param.requires_grad = True
        for param in self.dat.parameters():
            param.requires_grad = True
        for param in self.nafnet.parameters():
            param.requires_grad = True

    def freeze_hat_dat_only(self):
        """Freeze only HAT and DAT, keep NAFNet trainable"""
        print("Freezing HAT and DAT only (NAFNet + fusion + projection remain trainable)")
        for param in self.hat.parameters():
            param.requires_grad = False
        for param in self.dat.parameters():
            param.requires_grad = False
        # Unfreeze NAFNet to ensure it's trainable
        for param in self.nafnet.parameters():
            param.requires_grad = True
        # NAFNet, nafnet_proj, fusion, and reconstruction heads remain trainable
        print(f"  Frozen parameters: HAT, DAT")
        print(f"  Trainable: NAFNet, nafnet_proj, fusion (3x3), reconstruction heads")

    def freeze_all_backbones(self):
        """Freeze all backbone models (HAT, DAT, NAFNet)"""
        print("Freezing all backbones (HAT, DAT, NAFNet)")
        for param in self.hat.parameters():
            param.requires_grad = False
        for param in self.dat.parameters():
            param.requires_grad = False
        for param in self.nafnet.parameters():
            param.requires_grad = False
        print(f"  Frozen: HAT, DAT, NAFNet")
        print(f"  Trainable: nafnet_proj, fusion (3x3), reconstruction heads")

    def extract_features(self, x):
        """Extract features from all backbone models"""
        # Get HAT features (after deep feature extraction)
        self.hat.mean = self.hat.mean.type_as(x)
        hat_input = (x - self.hat.mean) * self.hat.img_range
        hat_shallow_feat = self.hat.conv_first(hat_input)  # Extract shallow features
        hat_feat = self.hat.conv_after_body(self.hat.forward_features(hat_shallow_feat)) + hat_shallow_feat

        # Get DAT features (after deep feature extraction)
        self.dat.mean = self.dat.mean.type_as(x)
        dat_input = (x - self.dat.mean) * self.dat.img_range
        dat_shallow_feat = self.dat.conv_first(dat_input)  # Extract shallow features
        dat_feat = self.dat.conv_after_body(self.dat.forward_features(dat_shallow_feat)) + dat_shallow_feat

        # Get NAFNet features (processed features)
        nafnet_feat = self.nafnet.intro(x)

        # Encode
        encs = []
        for encoder, down in zip(self.nafnet.encoders, self.nafnet.downs):
            nafnet_feat = encoder(nafnet_feat)
            encs.append(nafnet_feat)
            nafnet_feat = down(nafnet_feat)

        # Middle
        nafnet_feat = self.nafnet.middle_blks(nafnet_feat)

        # Decode
        for decoder, up, enc_skip in zip(self.nafnet.decoders, self.nafnet.ups, encs[::-1]):
            nafnet_feat = up(nafnet_feat)
            nafnet_feat = nafnet_feat + enc_skip
            nafnet_feat = decoder(nafnet_feat)

        nafnet_feat = self.nafnet.ending(nafnet_feat) + self.nafnet.intro(x)

        # Project NAFNet features to match dimension
        nafnet_feat = self.nafnet_proj(nafnet_feat)

        return hat_feat, dat_feat, nafnet_feat

    def forward(self, x):
        """
        Forward pass with 3x3 convolution fusion
        Args:
            x: Input LR image (B, 3, H, W)
        Returns:
            output: SR image (B, 3, H*upscale, W*upscale)
        """
        self.mean = self.mean.type_as(x)

        # Extract features from frozen backbones
        # 使用torch.no_grad()时需要判断是否在训练模式
        # 如果self.training=False（验证模式），强制使用no_grad节省显存
        use_no_grad = self.freeze_backbone or not self.training
        with torch.no_grad() if use_no_grad else torch.enable_grad():
            hat_feat, dat_feat, nafnet_feat = self.extract_features(x)

        # Concatenate features
        fused_feat = torch.cat([hat_feat, dat_feat, nafnet_feat], dim=1)

        # Fusion module with 3x3 convolutions (trainable)
        fused_feat = self.fusion(fused_feat)

        # Reconstruction
        out = self.conv_before_upsample(fused_feat)
        out = self.upsample(out)
        out = self.conv_last(out)

        # Denormalize
        out = out / self.img_range + self.mean

        return out


if __name__ == '__main__':
    # Test the model
    model = HAT_DAT_NAFNet_Fusion_3x3(
        upscale=4,
        freeze_backbone=False,
    ).cuda().eval()

    # Test forward pass
    x = torch.randn(1, 3, 64, 64).cuda()
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Count fusion layer parameters
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    print(f"Fusion layer (3x3) parameters: {fusion_params:,}")
    print(f"Expected: ~1.7M (vs ~194K for 1x1 version)")
