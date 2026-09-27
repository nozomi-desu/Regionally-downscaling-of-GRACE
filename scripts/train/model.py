"""Baseline neural network for GRACE adaptive downscaling."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from utils.mass_closure import masked_pool2d_mean, upsample_coarse_residual


class ConvBlock(nn.Module):
    """Two-layer convolutional block with GroupNorm and GELU."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = max(1, min(8, out_channels // 8))
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Downsampling stage."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsampling stage with skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class BaselineUNet(nn.Module):
    """A compact U-Net baseline with optional residual correction to coarse context."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        dropout: float = 0.0,
        residual_to_coarse: bool = True,
    ) -> None:
        super().__init__()
        self.residual_to_coarse = residual_to_coarse
        self.stem = ConvBlock(in_channels, base_channels, dropout=dropout)
        self.down1 = DownBlock(base_channels, base_channels * 2, dropout=dropout)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, dropout=dropout)
        self.bottleneck = DownBlock(base_channels * 4, base_channels * 8, dropout=dropout)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4, dropout=dropout)
        self.up1 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2, dropout=dropout)
        self.up0 = UpBlock(base_channels * 2, base_channels, base_channels, dropout=dropout)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def decode_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode and decode inputs, returning the final full-resolution features."""
        x0 = self.stem(inputs)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        xb = self.bottleneck(x2)
        x = self.up2(xb, x2)
        x = self.up1(x, x1)
        return self.up0(x, x0)

    def forward(
        self,
        inputs: torch.Tensor,
        coarse_context: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        area_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del valid_mask, area_weight
        x = self.decode_features(inputs)
        pred = self.head(x)
        if self.residual_to_coarse and coarse_context is not None:
            pred = pred + coarse_context
        return pred


class VariationalUNet(BaselineUNet):
    """Uz-style variational U-Net approximation with a stochastic bottleneck."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        dropout: float = 0.0,
        residual_to_coarse: bool = True,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            dropout=dropout,
            residual_to_coarse=residual_to_coarse,
        )
        latent_channels = base_channels * 8
        self.latent_mu = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.latent_logvar = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.last_kl_loss: torch.Tensor | None = None

    def decode_features(self, inputs: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(inputs)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        xb = self.bottleneck(x2)
        mu = self.latent_mu(xb)
        logvar = self.latent_logvar(xb).clamp(min=-12.0, max=12.0)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        self.last_kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
        x = self.up2(z, x2)
        x = self.up1(x, x1)
        return self.up0(x, x0)


class ScaleSeparationUNet(BaselineUNet):
    """Learn only within-coarse-cell redistribution, optionally gated by reliability."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        dropout: float = 0.0,
        residual_to_coarse: bool = True,
        coarse_factor: int = 6,
        gate_mode: str = "none",
        gate_smoothing_kernel: int = 7,
        gate_application: str = "output",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            dropout=dropout,
            residual_to_coarse=residual_to_coarse,
        )
        if gate_mode not in {"none", "learned", "oracle"}:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")
        if gate_smoothing_kernel < 1 or gate_smoothing_kernel % 2 == 0:
            raise ValueError("gate_smoothing_kernel must be a positive odd integer.")
        if gate_application not in {"output", "loss_only"}:
            raise ValueError(f"Unsupported gate_application: {gate_application}")
        self.coarse_factor = int(coarse_factor)
        self.gate_mode = gate_mode
        self.gate_smoothing_kernel = int(gate_smoothing_kernel)
        self.gate_application = gate_application
        self.gate_head = nn.Conv2d(base_channels, 1, kernel_size=1) if gate_mode == "learned" else None
        self.last_gate: torch.Tensor | None = None
        self.last_high_frequency: torch.Tensor | None = None

    def forward(
        self,
        inputs: torch.Tensor,
        coarse_context: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        area_weight: torch.Tensor | None = None,
        oracle_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if coarse_context is None:
            raise ValueError("ScaleSeparationUNet requires coarse_context.")
        features = self.decode_features(inputs)
        raw_high = self.head(features)
        if valid_mask is None:
            valid_mask = torch.ones_like(raw_high)
        weights = valid_mask.to(raw_high.dtype)
        if area_weight is not None:
            weights = weights * area_weight.to(raw_high.dtype)
        coarse_high, _ = masked_pool2d_mean(
            raw_high,
            weights,
            kernel_size=self.coarse_factor,
            eps=1e-12,
        )
        high = raw_high - upsample_coarse_residual(
            coarse_high,
            self.coarse_factor,
            output_shape=raw_high.shape[-2:],
        )
        if self.gate_mode == "oracle":
            if oracle_gate is None:
                raise ValueError("gate_mode='oracle' requires closed-loop reliability truth.")
            gate = oracle_gate.to(device=high.device, dtype=high.dtype).clamp(0.0, 1.0)
        elif self.gate_head is None:
            gate = torch.ones_like(high)
        else:
            gate_logits = self.gate_head(features)
            padding = self.gate_smoothing_kernel // 2
            gate_logits = F.avg_pool2d(
                gate_logits,
                kernel_size=self.gate_smoothing_kernel,
                stride=1,
                padding=padding,
            )
            gate = torch.sigmoid(gate_logits)
        self.last_gate = gate
        self.last_high_frequency = high
        if self.gate_application == "loss_only":
            return coarse_context + high
        return coarse_context + gate * high


def build_model_from_config(in_channels: int, model_cfg: dict) -> nn.Module:
    """Instantiate one registered architecture from a shared configuration schema."""
    architecture = str(model_cfg.get("architecture", "baseline_unet")).lower()
    common = {
        "in_channels": in_channels,
        "base_channels": int(model_cfg["base_channels"]),
        "dropout": float(model_cfg["dropout"]),
        "residual_to_coarse": bool(model_cfg.get("residual_to_coarse", True)),
    }
    if architecture == "baseline_unet":
        return BaselineUNet(**common)
    if architecture == "variational_unet":
        return VariationalUNet(**common)
    if architecture == "scale_separation_unet":
        return ScaleSeparationUNet(
            **common,
            coarse_factor=int(model_cfg.get("coarse_factor", 6)),
            gate_mode=str(model_cfg.get("gate_mode", "none")),
            gate_smoothing_kernel=int(model_cfg.get("gate_smoothing_kernel", 7)),
            gate_application=str(model_cfg.get("gate_application", "output")),
        )
    raise ValueError(f"Unsupported model architecture: {architecture}")
