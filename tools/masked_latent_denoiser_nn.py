"""Shared small UNet-style denoiser used by train / infer masked latent tools."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return F.silu(x + self.net(x))


class MaskedLatentDenoiser(nn.Module):
    def __init__(self, latent_channels, base_channels):
        super().__init__()
        self.in_proj = nn.Conv2d(latent_channels + 2, base_channels, 3, padding=1)
        self.down = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)
        self.mid = nn.Sequential(ResidualBlock(base_channels), ResidualBlock(base_channels))
        self.up = nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1)
        self.out = nn.Sequential(ResidualBlock(base_channels), nn.Conv2d(base_channels, latent_channels, 3, padding=1))

    def forward(self, latent, mask, sigma):
        sigma_map = sigma[:, None, None, None].expand(mask.shape[0], 1, mask.shape[-2], mask.shape[-1])
        h = self.in_proj(torch.cat([latent, mask, sigma_map], dim=1))
        skip = h
        h = self.down(h)
        h = self.mid(h)
        h = self.up(h)
        if h.shape[-2:] != skip.shape[-2:]:
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(h + skip)
