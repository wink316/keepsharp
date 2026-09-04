from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(t))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class LiteCondUNet(nn.Module):
    """LQ-conditioned UNet that predicts DDPM noise. Architecture is Diffusion-native."""

    def __init__(self, base: int = 32, time_dim: int = 128) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.in_conv = nn.Conv2d(6, base, 3, padding=1)
        self.down1 = ResBlock(base, base, time_dim)
        self.down2 = ResBlock(base, base * 2, time_dim)
        self.down3 = ResBlock(base * 2, base * 4, time_dim)
        self.pool = nn.AvgPool2d(2)
        self.mid = ResBlock(base * 4, base * 4, time_dim)
        self.up3 = ResBlock(base * 8, base * 2, time_dim)
        self.up2 = ResBlock(base * 4, base, time_dim)
        self.up1 = ResBlock(base * 2, base, time_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, 3, 3, padding=1),
        )

    def forward(self, xt: torch.Tensor, lq: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(t)
        x = self.in_conv(torch.cat([xt, lq], dim=1))
        h1 = self.down1(x, temb)
        h2 = self.down2(self.pool(h1), temb)
        h3 = self.down3(self.pool(h2), temb)
        mid = self.mid(self.pool(h3), temb)
        u3 = self.up3(torch.cat([F.interpolate(mid, size=h3.shape[-2:], mode="nearest"), h3], dim=1), temb)
        u2 = self.up2(torch.cat([F.interpolate(u3, size=h2.shape[-2:], mode="nearest"), h2], dim=1), temb)
        u1 = self.up1(torch.cat([F.interpolate(u2, size=h1.shape[-2:], mode="nearest"), h1], dim=1), temb)
        return self.out(u1)
