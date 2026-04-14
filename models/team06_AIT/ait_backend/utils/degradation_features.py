import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    if timesteps.ndim == 0:
        timesteps = timesteps[None]
    timesteps = timesteps.float()
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(float(max_period), device=timesteps.device))
        * torch.arange(0, half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = timesteps[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _to_grayscale(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] == 1:
        return image
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def compute_degradation_features(image: torch.Tensor) -> torch.Tensor:
    """Compute lightweight degradation descriptors from [0,1] images.

    Args:
        image: Tensor in shape (B, C, H, W), range [0, 1].

    Returns:
        Tensor in shape (B, 6): [blur, noise, jpeg_blocking, edge_density, brightness, contrast]
    """
    if image.ndim != 4:
        raise ValueError(f"Expected image shape (B,C,H,W), got {tuple(image.shape)}")

    image = image.clamp(0.0, 1.0)
    gray = _to_grayscale(image)

    lap_kernel = image.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    lap = F.conv2d(gray, lap_kernel, padding=1)
    lap_var = lap.var(dim=(2, 3), unbiased=False)
    blur = 1.0 / (lap_var + 1e-6)

    smooth = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
    noise = (gray - smooth).abs().mean(dim=(2, 3))

    if gray.shape[-1] > 8:
        v_diff = (gray[:, :, :, 8:] - gray[:, :, :, :-8]).abs()
        h_diff = (gray[:, :, 8:, :] - gray[:, :, :-8, :]).abs()
        jpeg_blocking = 0.5 * (
            v_diff[:, :, :, ::8].mean(dim=(2, 3)) + h_diff[:, :, ::8, :].mean(dim=(2, 3))
        )
    else:
        jpeg_blocking = torch.zeros_like(noise)

    sobel_x = image.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    sobel_y = image.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    grad = (F.conv2d(gray, sobel_x, padding=1).pow(2) + F.conv2d(gray, sobel_y, padding=1).pow(2)).sqrt()
    edge_density = (grad > 0.08).float().mean(dim=(2, 3))

    brightness = gray.mean(dim=(2, 3))
    contrast = gray.std(dim=(2, 3), unbiased=False)

    features = torch.cat([blur, noise, jpeg_blocking, edge_density, brightness, contrast], dim=1)
    features = torch.log1p(features)
    return features


class DegradationTokenAdapter(nn.Module):
    def __init__(
        self,
        in_dim: int = 6,
        out_dim: int = 512,
        dropout: float = 0.1,
        use_timestep_condition: bool = False,
        timestep_dim: int = 128,
        timestep_max_period: int = 10000,
    ):
        super().__init__()
        self.use_timestep_condition = use_timestep_condition
        self.timestep_dim = timestep_dim
        self.timestep_max_period = timestep_max_period
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )
        if self.use_timestep_condition:
            self.time_mlp = nn.Sequential(
                nn.Linear(timestep_dim, 128),
                nn.SiLU(),
                nn.Linear(128, out_dim * 2),
            )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, degradation_features: torch.Tensor, timesteps: torch.Tensor = None) -> torch.Tensor:
        token = self.mlp(degradation_features)
        if self.use_timestep_condition:
            if timesteps is None:
                raise ValueError("timesteps must be provided when use_timestep_condition=True.")
            if timesteps.ndim == 0:
                timesteps = timesteps.view(1)
            if timesteps.shape[0] != token.shape[0]:
                if timesteps.shape[0] == 1:
                    timesteps = timesteps.expand(token.shape[0])
                else:
                    raise ValueError(
                        f"timesteps batch ({timesteps.shape[0]}) != token batch ({token.shape[0]})."
                    )
            t_emb = timestep_embedding(
                timesteps.to(device=token.device),
                dim=self.timestep_dim,
                max_period=self.timestep_max_period,
            ).to(dtype=token.dtype)
            t_scale_shift = self.time_mlp(t_emb)
            scale, shift = torch.chunk(t_scale_shift, 2, dim=-1)
            token = token * (1.0 + scale) + shift
        return self.norm(token)
