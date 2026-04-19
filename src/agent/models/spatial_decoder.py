import torch
import torch.nn as nn


def tactical_center_spatial_bias(
    spatial_resolution: int,
    ref: torch.Tensor,
    *,
    amplitude: float = 1e-3,
    sigma: float = 3.0,
    center_gx: float = 16.0,
    center_gy: float = 15.0,
) -> torch.Tensor:
    """
    Tiny Gaussian bump over the tactical grid, centered where the ObservationTransformer
    places the active unit (gx=16, gy=15 on 32×32). Breaks greedy argmax ties that
    otherwise default to spatial index 0 (corner → map-edge clamp / one stuck point).
    """
    device, dtype = ref.device, ref.dtype
    idx = torch.arange(spatial_resolution * spatial_resolution, device=device, dtype=dtype)
    gx = idx % spatial_resolution
    gy = idx // spatial_resolution
    return amplitude * torch.exp(
        -((gx - center_gx) ** 2 + (gy - center_gy) ** 2) / (2.0 * sigma**2)
    )


class SpatialDecoder(nn.Module):
    """
    Maps fused 16x16 spatial features + policy latent to flat 32x32 spatial logits.
    """

    def __init__(
        self,
        latent_dim: int,
        fused_channels: int,
        spatial_resolution: int = 32,
        latent_inject_channels: int = 32,
    ):
        super().__init__()
        self.spatial_resolution = spatial_resolution
        self.grid_area = spatial_resolution * spatial_resolution
        self.latent_inject_channels = latent_inject_channels

        self.latent_proj = nn.Linear(latent_dim, latent_inject_channels * 16 * 16)
        in_ch = fused_channels + latent_inject_channels
        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, latent_pi: torch.Tensor, spatial_fused: torch.Tensor) -> torch.Tensor:
        b = latent_pi.shape[0]
        z = self.latent_proj(latent_pi).view(b, self.latent_inject_channels, 16, 16)
        x = torch.cat([spatial_fused, z], dim=1)
        out = self.decoder(x)
        return out.view(b, self.grid_area)
