"""Loss functions for GRACE adaptive soft-constraint training."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.mass_closure import masked_pool2d_mean


def ensure_nchw(field: torch.Tensor) -> torch.Tensor:
    """Promote CHW tensors to NCHW for consistent loss computation."""
    if field.ndim == 3:
        return field.unsqueeze(0)
    return field


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute the weighted mean over valid entries."""
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(eps)


def aggregate_coarse_mean(
    field: torch.Tensor,
    mask: torch.Tensor,
    factor: int,
    area_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate using the shared masked cos(latitude)-weighted operator."""
    weights = mask if area_weight is None else mask * area_weight
    return masked_pool2d_mean(field, weights, kernel_size=factor)


def gradient_xy(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute first-order finite differences in x and y."""
    grad_y = field[:, :, 1:, :] - field[:, :, :-1, :]
    grad_x = field[:, :, :, 1:] - field[:, :, :, :-1]
    return grad_x, grad_y


def batch_spatial_pearson(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return the mean per-sample 2D Pearson correlation used as Uz-style lambda."""
    pred = ensure_nchw(prediction)
    ref = ensure_nchw(reference)
    weights = ensure_nchw(valid_mask).to(pred.dtype)
    reduce_dims = (-2, -1)
    counts = weights.sum(dim=reduce_dims).clamp_min(eps)
    pred_mean = (pred * weights).sum(dim=reduce_dims) / counts
    ref_mean = (ref * weights).sum(dim=reduce_dims) / counts
    pred_centered = (pred - pred_mean[..., None, None]) * weights
    ref_centered = (ref - ref_mean[..., None, None]) * weights
    covariance = (pred_centered * ref_centered).sum(dim=reduce_dims)
    pred_scale = torch.sqrt(pred_centered.square().sum(dim=reduce_dims).clamp_min(eps))
    ref_scale = torch.sqrt(ref_centered.square().sum(dim=reduce_dims).clamp_min(eps))
    correlation = covariance / (pred_scale * ref_scale).clamp_min(eps)
    return correlation.mean().clamp(0.0, 1.0)


def sobel_gradient_mse(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Uz-style squared Sobel-gradient discrepancy on valid fine cells."""
    dtype = prediction.dtype
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=prediction.device,
        dtype=dtype,
    ).reshape(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    pred_x = F.conv2d(prediction, kernel_x, padding=1)
    pred_y = F.conv2d(prediction, kernel_y, padding=1)
    ref_x = F.conv2d(reference, kernel_x, padding=1)
    ref_y = F.conv2d(reference, kernel_y, padding=1)
    diff2 = 0.5 * ((pred_x - ref_x).square() + (pred_y - ref_y).square())
    return masked_mean(diff2, valid_mask)


def build_uz_dynamic_loss(
    pred_fine: torch.Tensor,
    batch: dict[str, torch.Tensor],
    loss_cfg: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Approximate Uz et al. inverse dynamic soft-constraint objective."""
    pred = ensure_nchw(pred_fine)
    target = ensure_nchw(batch["target_fine"])
    wghm = ensure_nchw(batch["wghm_fine"])
    valid = ensure_nchw(batch["valid_mask"])
    factor = int(loss_cfg["coarse_factor"])

    pred_coarse, coarse_valid = masked_pool2d_mean(pred, valid, kernel_size=factor)
    target_coarse, target_valid = masked_pool2d_mean(target, valid, kernel_size=factor)
    reconstruction_lr = masked_mean(
        (pred_coarse - target_coarse).square(),
        coarse_valid * target_valid,
    )
    kl = batch.get("model_kl_loss")
    if kl is None:
        kl = torch.zeros((), device=pred.device, dtype=pred.dtype)
    kl = kl.to(dtype=pred.dtype)
    reconstruction = reconstruction_lr + float(loss_cfg.get("kl_weight", 1.0)) * kl

    high_resolution = masked_mean((pred - wghm).square(), valid)
    gradient = sobel_gradient_mse(pred, wghm, valid)
    constraint = high_resolution + float(loss_cfg.get("uz_gradient_weight", 1.0)) * gradient

    dynamic_lambda = batch_spatial_pearson(pred, wghm, valid).detach()
    total = dynamic_lambda * reconstruction + (1.0 - dynamic_lambda) * constraint
    components = {
        "loss_total": total.detach(),
        "loss_mass": reconstruction_lr.detach(),
        "loss_value_adaptive": high_resolution.detach(),
        "loss_gradient_adaptive": gradient.detach(),
        "loss_fine": torch.zeros_like(total.detach()),
        "loss_spectral": torch.zeros_like(total.detach()),
        "loss_low_grace": reconstruction.detach(),
        "loss_high_wghm": constraint.detach(),
        "loss_basin": torch.zeros_like(total.detach()),
        "loss_kl": kl.detach(),
        "dynamic_lambda": dynamic_lambda.detach(),
    }
    return total, components


def coarse_supervision_loss(
    pred_fine: torch.Tensor,
    target_coarse: torch.Tensor,
    valid_mask: torch.Tensor,
    factor: int,
    area_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize mismatch between aggregated prediction and coarse JPL supervision."""
    pred_coarse, coarse_mask = aggregate_coarse_mean(
        pred_fine,
        valid_mask,
        factor,
        area_weight=area_weight,
    )
    diff2 = (pred_coarse - target_coarse) ** 2
    return masked_mean(diff2, coarse_mask)


def fine_target_loss(
    pred_fine: torch.Tensor,
    target_fine: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Optional fine-grid reconstruction loss against JPL on valid cells."""
    diff2 = (pred_fine - target_fine) ** 2
    return masked_mean(diff2, valid_mask)


def value_constraint_loss(
    pred_fine: torch.Tensor,
    wghm_fine: torch.Tensor,
    alpha: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Adaptive soft value constraint toward WGHM."""
    diff = F.smooth_l1_loss(pred_fine, wghm_fine, reduction="none")
    return masked_mean(diff, alpha * valid_mask)


def gradient_constraint_loss(
    pred_fine: torch.Tensor,
    wghm_fine: torch.Tensor,
    beta: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Adaptive soft gradient constraint toward WGHM."""
    pred_dx, pred_dy = gradient_xy(pred_fine)
    wghm_dx, wghm_dy = gradient_xy(wghm_fine)
    mask_dx = valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]
    mask_dy = valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]
    beta_dx = 0.5 * (beta[:, :, :, 1:] + beta[:, :, :, :-1])
    beta_dy = 0.5 * (beta[:, :, 1:, :] + beta[:, :, :-1, :])
    loss_dx = masked_mean(torch.abs(pred_dx - wghm_dx), beta_dx * mask_dx)
    loss_dy = masked_mean(torch.abs(pred_dy - wghm_dy), beta_dy * mask_dy)
    return 0.5 * (loss_dx + loss_dy)


def basin_integral_constraint_loss(
    pred_fine: torch.Tensor,
    reference_fine: torch.Tensor,
    basin_id: torch.Tensor,
    valid_mask: torch.Tensor,
    basin_cfg: dict[str, float | int | bool],
) -> torch.Tensor:
    """Encourage basin-mean agreement inside available HydroBASINS regions.

    This prototype loss operates on rasterized basin IDs currently available for the
    Asia HydroBASINS domain. Outside those regions the IDs are zero and the term
    naturally vanishes.
    """
    if not basin_cfg.get("enabled", False):
        return pred_fine.new_tensor(0.0)

    min_cells = int(basin_cfg.get("min_cells", 12))
    basin_id = ensure_nchw(basin_id).to(torch.int64)
    valid_mask = ensure_nchw(valid_mask)
    losses: list[torch.Tensor] = []

    for batch_idx in range(pred_fine.shape[0]):
        basin_map = basin_id[batch_idx, 0]
        valid_map = valid_mask[batch_idx, 0] > 0
        if not torch.any(valid_map):
            continue
        basin_values = torch.unique(basin_map[valid_map])
        basin_values = basin_values[basin_values > 0]
        for basin_value in basin_values.tolist():
            basin_mask = (basin_map == basin_value) & valid_map
            if int(basin_mask.sum().item()) < min_cells:
                continue
            pred_mean = pred_fine[batch_idx, 0][basin_mask].mean()
            ref_mean = reference_fine[batch_idx, 0][basin_mask].mean()
            losses.append((pred_mean - ref_mean) ** 2)

    if not losses:
        return pred_fine.new_tensor(0.0)
    return torch.stack(losses).mean()


def build_gaussian_kernel(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build a normalized 2D Gaussian kernel with shape ``[1, 1, K, K]``."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd for centered Gaussian blur.")
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-6)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum().clamp_min(1e-6)
    return kernel_2d.view(1, 1, kernel_size, kernel_size)


def masked_gaussian_blur(
    field: torch.Tensor,
    mask: torch.Tensor,
    kernel_size: int,
    sigma: float,
) -> torch.Tensor:
    """Apply a NaN-safe, mask-aware Gaussian blur to ``[N, C, H, W]`` tensors."""
    kernel = build_gaussian_kernel(kernel_size, sigma, field.device, field.dtype)
    padding = kernel_size // 2

    valid = torch.isfinite(field) & torch.isfinite(mask) & (mask > 0)
    weights = torch.where(valid, mask.to(field.dtype), torch.zeros_like(mask, dtype=field.dtype))
    values = torch.where(valid, field, torch.zeros_like(field))

    padded_values = F.pad(values * weights, (padding, padding, padding, padding), mode="reflect")
    padded_weights = F.pad(weights, (padding, padding, padding, padding), mode="reflect")
    groups = field.shape[1]
    kernel_grouped = kernel.expand(groups, 1, kernel_size, kernel_size)

    blurred_sum = F.conv2d(padded_values, kernel_grouped, groups=groups)
    blurred_weight = F.conv2d(padded_weights, kernel_grouped, groups=groups)
    return blurred_sum / blurred_weight.clamp_min(1e-6)


def spectral_constraint_loss(
    pred_fine: torch.Tensor,
    jpl_fine: torch.Tensor,
    wghm_fine: torch.Tensor,
    beta: torch.Tensor,
    valid_mask: torch.Tensor,
    spectral_cfg: dict[str, float | bool],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine low-frequency GRACE matching and high-frequency WGHM matching."""
    if not spectral_cfg.get("enabled", False):
        zero = pred_fine.new_tensor(0.0)
        return zero, zero, zero

    kernel_size = int(spectral_cfg.get("kernel_size", 13))
    sigma = float(spectral_cfg.get("sigma", 3.0))
    w_low = float(spectral_cfg.get("w_low", 0.2))
    w_high = float(spectral_cfg.get("w_high", 0.1))

    low_pred = masked_gaussian_blur(pred_fine, valid_mask, kernel_size=kernel_size, sigma=sigma)
    low_jpl = masked_gaussian_blur(jpl_fine, valid_mask, kernel_size=kernel_size, sigma=sigma)
    high_pred = pred_fine - low_pred
    high_wghm = wghm_fine - masked_gaussian_blur(wghm_fine, valid_mask, kernel_size=kernel_size, sigma=sigma)

    loss_low = masked_mean((low_pred - low_jpl) ** 2, valid_mask)
    loss_high = masked_mean(((high_pred - high_wghm) ** 2), beta * valid_mask)
    total = w_low * loss_low + w_high * loss_high
    return total, loss_low, loss_high


def _resolve_loss_weight(loss_cfg: dict, canonical_key: str, legacy_key: str, default: float = 0.0) -> float:
    """Support both round1 ``lambda_*`` keys and round2 ``*_weight`` keys."""
    if canonical_key in loss_cfg:
        return float(loss_cfg[canonical_key])
    if legacy_key in loss_cfg:
        return float(loss_cfg[legacy_key])
    return float(default)


def gate_identifiability_loss(
    gate: torch.Tensor | None,
    reliability_truth: torch.Tensor | None,
    valid_mask: torch.Tensor,
    gate_cfg: dict[str, float | str | bool],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise a synthetic reliability gate and penalize constant collapse."""
    zero = valid_mask.new_tensor(0.0)
    if not gate_cfg.get("enabled", False):
        return zero, {
            "loss_gate_supervision": zero,
            "loss_gate_variance": zero,
            "gate_mae": zero,
            "gate_std": zero,
            "gate_saturation_fraction": zero,
        }
    if gate is None or reliability_truth is None:
        raise ValueError(
            "Enabled gate_identifiability_loss requires model_gate and reliability_truth."
        )

    gate = ensure_nchw(gate)
    target = ensure_nchw(reliability_truth).to(device=gate.device, dtype=gate.dtype).clamp(0.0, 1.0)
    valid = ensure_nchw(valid_mask).to(device=gate.device, dtype=gate.dtype)
    mode = str(gate_cfg.get("supervision", "binary_cross_entropy")).lower()
    if mode == "binary_cross_entropy":
        elementwise = F.binary_cross_entropy(
            gate.clamp(1.0e-6, 1.0 - 1.0e-6),
            target,
            reduction="none",
        )
    elif mode == "smooth_l1":
        elementwise = F.smooth_l1_loss(gate, target, reduction="none")
    else:
        raise ValueError(f"Unsupported gate supervision mode: {mode}")
    supervision = masked_mean(elementwise, valid)

    gate_mean = masked_mean(gate, valid)
    gate_variance = masked_mean((gate - gate_mean).square(), valid)
    gate_std = torch.sqrt(gate_variance.clamp_min(1.0e-12))
    min_std = float(gate_cfg.get("min_std", 0.10))
    variance_penalty = F.relu(gate.new_tensor(min_std) - gate_std).square()
    variance_weight = float(gate_cfg.get("variance_weight", 0.0))
    total = supervision + variance_weight * variance_penalty

    saturation_threshold = float(gate_cfg.get("saturation_threshold", 0.02))
    saturated = ((gate <= saturation_threshold) | (gate >= 1.0 - saturation_threshold)).to(gate.dtype)
    diagnostics = {
        "loss_gate_supervision": supervision.detach(),
        "loss_gate_variance": variance_penalty.detach(),
        "gate_mae": masked_mean(torch.abs(gate - target), valid).detach(),
        "gate_std": gate_std.detach(),
        "gate_saturation_fraction": masked_mean(saturated, valid).detach(),
    }
    return total, diagnostics


def build_total_loss(
    pred_fine: torch.Tensor,
    batch: dict[str, torch.Tensor],
    loss_cfg: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the weighted sum of all configured loss terms."""
    if str(loss_cfg.get("mode", "rwsc_static")).lower() == "uz_dynamic":
        return build_uz_dynamic_loss(pred_fine, batch, loss_cfg)
    pred_fine = ensure_nchw(pred_fine)
    target_coarse = ensure_nchw(batch["target_coarse"])
    target_fine = ensure_nchw(batch["target_fine"])
    wghm_fine = ensure_nchw(batch["wghm_fine"])
    alpha = ensure_nchw(batch["alpha"])
    beta = ensure_nchw(batch["beta"])
    valid_mask = ensure_nchw(batch["valid_mask"])
    area_weight = ensure_nchw(batch["area_weight"]) if "area_weight" in batch else None
    coarse_factor = int(loss_cfg["coarse_factor"])
    lambda_mass = _resolve_loss_weight(loss_cfg, "mass_weight", "lambda_coarse", default=1.0)
    lambda_value = _resolve_loss_weight(loss_cfg, "value_weight", "lambda_value", default=0.0)
    lambda_gradient = _resolve_loss_weight(loss_cfg, "gradient_weight", "lambda_gradient", default=0.0)
    lambda_fine = _resolve_loss_weight(loss_cfg, "fine_weight", "lambda_fine", default=0.0)
    lambda_spectral = _resolve_loss_weight(loss_cfg, "spectral_weight", "lambda_spectral", default=0.0)
    lambda_basin = _resolve_loss_weight(loss_cfg, "basin_weight", "lambda_basin", default=0.0)
    lambda_gate = float(loss_cfg.get("gate_weight", 0.0))
    if bool(loss_cfg.get("gate_modulates_constraints", False)):
        model_gate = batch.get("model_gate")
        if model_gate is None:
            raise ValueError("gate_modulates_constraints requires model_gate.")
        # The gate is identified by its explicit objective. Stop-gradient here
        # prevents the value/gradient losses from winning by collapsing it to 0.
        constraint_gate = ensure_nchw(model_gate).detach().to(device=alpha.device, dtype=alpha.dtype)
        alpha = alpha * constraint_gate
        beta = beta * constraint_gate

    coarse = coarse_supervision_loss(
        pred_fine,
        target_coarse,
        valid_mask,
        factor=coarse_factor,
        area_weight=area_weight,
    )
    value = value_constraint_loss(
        pred_fine,
        wghm_fine,
        alpha,
        valid_mask,
    )
    gradient = gradient_constraint_loss(
        pred_fine,
        wghm_fine,
        beta,
        valid_mask,
    )
    fine = fine_target_loss(
        pred_fine,
        target_fine,
        valid_mask,
    )
    spectral, low_grace, high_wghm = spectral_constraint_loss(
        pred_fine,
        target_fine,
        wghm_fine,
        beta,
        valid_mask,
        spectral_cfg=loss_cfg.get("spectral_loss", {}),
    )
    basin_reference = target_fine
    basin_cfg = loss_cfg.get("basin_integral_loss", {})
    if str(basin_cfg.get("reference", "jpl")).lower() == "wghm":
        basin_reference = wghm_fine
    basin = basin_integral_constraint_loss(
        pred_fine,
        basin_reference,
        batch["basin_id"],
        valid_mask,
        basin_cfg=basin_cfg,
    )
    gate, gate_diagnostics = gate_identifiability_loss(
        batch.get("model_gate"),
        batch.get("reliability_truth"),
        valid_mask,
        gate_cfg=loss_cfg.get("gate_identifiability_loss", {}),
    )
    total = (
        lambda_mass * coarse
        + lambda_value * value
        + lambda_gradient * gradient
        + lambda_fine * fine
        + lambda_spectral * spectral
        + lambda_basin * basin
        + lambda_gate * gate
    )
    components = {
        "loss_total": total.detach(),
        "loss_mass": coarse.detach(),
        "loss_value_adaptive": value.detach(),
        "loss_gradient_adaptive": gradient.detach(),
        "loss_fine": fine.detach(),
        "loss_spectral": spectral.detach(),
        "loss_low_grace": low_grace.detach(),
        "loss_high_wghm": high_wghm.detach(),
        "loss_basin": basin.detach(),
        **gate_diagnostics,
    }
    return total, components
