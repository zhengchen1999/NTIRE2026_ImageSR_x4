import torch.nn.functional as F
import torch 
import torch.nn as nn
import torch.nn.functional as F 
####################################################################################################################

# =============================================================================
# Path A: Spectral-Spatial Processing
# =============================================================================


class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization for (B, C, H, W) tensors (NAFNet convention)."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SpectralPositionalEncoding(nn.Module):
    """Learnable positional encoding in the frequency domain."""

    def __init__(self, dim, max_freq_h=64, max_freq_w=64):
        super().__init__()
        self.freq_embed_h = nn.Parameter(torch.randn(1, dim // 2, max_freq_h, 1) * 0.02)
        self.freq_embed_w = nn.Parameter(torch.randn(1, dim // 2, 1, max_freq_w) * 0.02)
        self.proj = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        embed_h = F.interpolate(self.freq_embed_h, size=(H, 1), mode='bilinear', align_corners=False)
        embed_w = F.interpolate(self.freq_embed_w, size=(1, W), mode='bilinear', align_corners=False)
        return x + self.proj(embed_h + embed_w)


class AdaptiveSpectralModulation(nn.Module):
    """SE-style channel reweighting applied to frequency-domain magnitude."""

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )
        self.spectral_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.importance = nn.Parameter(torch.ones(1, 1, 1, 1))

    def forward(self, x):
        weights = self.weight_generator(x.abs())
        return self.spectral_conv(x) * weights * self.importance


class SpectralFeatureExtractor(nn.Module):
    """Global feature extraction via FFT processing."""

    def __init__(self, dim):
        super().__init__()
        self.spectral_pe = SpectralPositionalEncoding(dim)
        self.adaptive_mod = AdaptiveSpectralModulation(dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.gate = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        B, C, H, W = x.shape
        x_freq = torch.fft.fft2(x, norm='ortho')
        x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
        magnitude = x_freq.abs()
        phase = x_freq.angle()
        magnitude = self.spectral_pe(magnitude)
        magnitude = self.adaptive_mod(magnitude)
        mag_flat = magnitude.permute(0, 2, 3, 1)
        mag_flat = self.norm(mag_flat)
        mag_flat = self.mlp(mag_flat)
        magnitude = mag_flat.permute(0, 3, 1, 2)
        x_freq = magnitude * torch.exp(1j * phase)
        x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
        x_global = torch.fft.ifft2(x_freq, norm='ortho').real
        return x_global * torch.sigmoid(self.gate)


class SpatialFeatureExtractor(nn.Module):
    """Local feature extraction via multi-scale depthwise convolutions."""

    def __init__(self, dim, expansion=2):
        super().__init__()
        hidden = dim * expansion
        self.proj_in = nn.Conv2d(dim, hidden, 1)
        self.dw_conv_3x3 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.dw_conv_5x5 = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden)
        self.gate_proj = nn.Conv2d(hidden, hidden, 1)
        self.proj_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        x = self.proj_in(x)
        x_local = self.dw_conv_3x3(x) + self.dw_conv_5x5(x)
        return self.proj_out(x_local * torch.sigmoid(self.gate_proj(x)))


class SpectralSpatialMixer(nn.Module):
    """Parallel spectral-spatial feature fusion."""

    def __init__(self, dim):
        super().__init__()
        self.spatial_branch = SpatialFeatureExtractor(dim)
        self.spectral_branch = SpectralFeatureExtractor(dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )
        self.w_spatial = nn.Parameter(torch.ones(1) * 0.5)
        self.w_spectral = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        x_spatial = self.spatial_branch(x) * torch.sigmoid(self.w_spatial)
        x_spectral = self.spectral_branch(x) * torch.sigmoid(self.w_spectral)
        return self.fusion(torch.cat([x_spatial, x_spectral], dim=1))


class ChannelSplitGate(nn.Module):
    """Channel-split gating: bisect channels and element-wise multiply (NAFNet, Chen et al. ECCV 2022)."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SpectralSpatialBlock(nn.Module):
    """
    Spectral-Spatial Processing Block (FFT global + DWConv local).
    Norm -> SpectralSpatialMixer -> Residual -> Norm -> GatedFFN -> Residual
    Input/Output: (B, H, W, C)
    """

    def __init__(self, dim, d_state=64, ffn_expansion=2.66):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)
        self.mixer = SpectralSpatialMixer(dim)
        ffn_hidden = int(dim * ffn_expansion)
        self.ffn_proj = nn.Conv2d(dim, ffn_hidden * 2, 1)
        self.ffn_gate = ChannelSplitGate()
        self.ffn_out = nn.Conv2d(ffn_hidden, dim, 1)
        self.residual_scale_1 = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.residual_scale_2 = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        residual = x
        x = self.mixer(self.norm1(x))
        x = residual + x * self.residual_scale_1
        residual = x
        x = self.ffn_out(self.ffn_gate(self.ffn_proj(self.norm2(x))))
        x = residual + x * self.residual_scale_2
        return x.permute(0, 2, 3, 1)


# =============================================================================
# Path B: Kolmogorov-Arnold Adaptive Processing
# =============================================================================


class KolmogorovArnoldLinear(nn.Module):
    """Linear layer with learnable polynomial activation functions on edges (KAN)."""

    def __init__(self, in_features, out_features, num_basis=5, groups=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_basis = num_basis
        self.groups = groups
        self.basis_coeffs = nn.Parameter(
            torch.randn(in_features, out_features // groups, num_basis) * 0.1
        )
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.pathway_balance = nn.Parameter(torch.ones(2) * 0.5)

    def forward(self, x):
        shape = x.shape[:-1]
        x_flat = x.view(-1, self.in_features)
        x_norm = torch.tanh(x_flat)
        powers = torch.arange(self.num_basis, device=x.device, dtype=x.dtype)
        basis = x_norm.unsqueeze(-1) ** powers
        nonlinear_out = torch.einsum('bin,ion->bo', basis, self.basis_coeffs)
        if self.groups > 1:
            nonlinear_out = nonlinear_out.repeat(1, self.groups)
        linear_out = self.linear(x_flat)
        y = self.pathway_balance[0] * nonlinear_out + self.pathway_balance[1] * linear_out
        return y.view(*shape, self.out_features)


class SelectiveGatedMixer(nn.Module):
    """KAN-projected selective gating with depthwise local context."""

    def __init__(self, dim, d_conv=3, expansion=2.0):
        super().__init__()
        self.d_inner = int(expansion * dim)
        self.in_proj = KolmogorovArnoldLinear(dim, self.d_inner * 2, num_basis=5)
        self.dw_conv = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=(d_conv - 1) // 2,
            groups=self.d_inner
        )
        self.skip_scale = nn.Parameter(torch.ones(self.d_inner))
        self.gate_proj = nn.Linear(self.d_inner, self.d_inner)
        self.out_proj = KolmogorovArnoldLinear(self.d_inner, dim, num_basis=5)
        self.norm = nn.LayerNorm(self.d_inner)

    def forward(self, x):
        B, L, D = x.shape
        x_proj = self.in_proj(x)
        x_main, z = x_proj.chunk(2, dim=-1)
        x_main = x_main.transpose(1, 2)
        x_main = self.dw_conv(x_main)
        x_main = x_main.transpose(1, 2)
        x_main = F.silu(x_main)
        gate = torch.sigmoid(F.softplus(self.gate_proj(x_main)))
        y = x_main * gate + x_main * self.skip_scale
        y = self.norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        return y


class KolmogorovArnoldBlock(nn.Module):
    """
    Kolmogorov-Arnold Adaptive Processing Block.
    Norm -> SelectiveGatedMixer -> Residual -> Norm -> KAN-GatedFFN -> Residual
    Input/Output: (B, H, W, C)
    """

    def __init__(self, dim, d_state=64, ffn_expansion=2.66):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gated_mixer = SelectiveGatedMixer(dim)
        hidden = int(dim * ffn_expansion)
        self.ffn_in = KolmogorovArnoldLinear(dim, hidden * 2, num_basis=5)
        self.ffn_dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.ffn_out = KolmogorovArnoldLinear(hidden, dim, num_basis=5)
        self.residual_scale_1 = nn.Parameter(torch.ones(1) * 0.1)
        self.residual_scale_2 = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        B, H, W, C = x.shape
        residual = x
        x = self.norm1(x)
        x = self.gated_mixer(x.view(B, H * W, C)).view(B, H, W, C)
        x = residual + x * self.residual_scale_1
        residual = x
        x = self.norm2(x)
        x1, x2 = self.ffn_in(x).chunk(2, dim=-1)
        x1 = self.ffn_dw(x1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.ffn_out(x1 * F.gelu(x2))
        return residual + x * self.residual_scale_2


# =============================================================================
# DUSKANBlock: Dual Spectral KAN
# =============================================================================


class DUSKANBlock(nn.Module):
    """
    Complementary dual-path block combining:
      - Path A: SpectralSpatialBlock (global frequency + local spatial)
      - Path B: KolmogorovArnoldBlock (selective gating + polynomial activations)
    Learned blending via sigmoid-gated scalar.
    Input/Output: (B, H, W, C)
    """

    def __init__(self, dim, d_state=64):
        super().__init__()
        self.spectral_path = SpectralSpatialBlock(dim, d_state)
        self.adaptive_path = KolmogorovArnoldBlock(dim, d_state)
        self.blend_logit = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y_spectral = self.spectral_path(x)
        y_adaptive = self.adaptive_path(x)
        alpha = torch.sigmoid(self.blend_logit)
        return alpha * y_spectral + (1.0 - alpha) * y_adaptive


# =============================================================================
# Network Architecture: DUSKAN
# =============================================================================


class DUSKAN(nn.Module):
    """
    Dual Spectral Kolmogorov-Arnold Network.
    U-Net encoder-decoder with DUSKANBlock and global residual learning.
    """

    def __init__(self, in_channels=3, base_width=48, middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1], d_state=64):
        super().__init__()
        self.intro = nn.Conv2d(in_channels, base_width, 3, padding=1)
        self.ending = nn.Conv2d(base_width, in_channels, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        dim = base_width
        for n in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[DUSKANBlock(dim, d_state) for _ in range(n)])
            )
            self.downsamples.append(nn.Conv2d(dim, dim * 2, 2, 2))
            dim *= 2

        self.bottleneck = nn.Sequential(
            *[DUSKANBlock(dim, d_state) for _ in range(middle_blk_num)]
        )

        for n in dec_blk_nums:
            self.upsamples.append(nn.Sequential(
                nn.Conv2d(dim, dim * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            dim //= 2
            self.decoders.append(
                nn.Sequential(*[DUSKANBlock(dim, d_state) for _ in range(n)])
            )

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self._pad(inp)
        x = self.intro(inp)
        x = x.permute(0, 2, 3, 1)

        skip_connections = []
        for encoder, down in zip(self.encoders, self.downsamples):
            x = encoder(x)
            skip_connections.append(x)
            x = down(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        x = self.bottleneck(x)

        for decoder, up, skip in zip(self.decoders, self.upsamples, reversed(skip_connections)):
            x = up(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = x + skip
            x = decoder(x)

        x = self.ending(x.permute(0, 3, 1, 2))
        x = x + inp
        return x[:, :, :H, :W]

    def _pad(self, x):
        _, _, h, w = x.size()
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h))

