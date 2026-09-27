"""Mass-closure correction helpers for fine-to-coarse GRACE consistency."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _ensure_nchw(tensor: torch.Tensor) -> torch.Tensor:
    """Promote 2D/3D tensors to NCHW for pooled operations."""
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got shape {tuple(tensor.shape)}.")
    return tensor


def _ensure_hw(mask: np.ndarray) -> np.ndarray:
    """Promote arrays to HxW or NxHxW-compatible form for NumPy utilities."""
    if mask.ndim not in {2, 3, 4}:
        raise ValueError(f"Expected 2D, 3D, or 4D array, got shape {mask.shape}.")
    return mask


def masked_pool2d_mean(
    field: torch.Tensor,
    mask: torch.Tensor,
    kernel_size: int,
    stride: int | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute masked mean pooling on NCHW tensors.

    Args:
        field: Fine-resolution field with shape ``[N, C, H, W]`` or broadcastable lower-rank shape.
        mask: Non-negative validity weights with the same shape semantics as ``field``.
        kernel_size: Pooling window size along both spatial axes.
        stride: Pooling stride. Defaults to ``kernel_size``.
        eps: Small constant guarding division by zero.

    Returns:
        Tuple ``(pooled_mean, pooled_mask)`` where both have shape ``[N, C, Hc, Wc]``.
    """
    stride = stride or kernel_size
    field_nchw = _ensure_nchw(field)
    mask_nchw = _ensure_nchw(mask).to(field_nchw.dtype)
    if field_nchw.shape != mask_nchw.shape:
        mask_nchw = torch.broadcast_to(mask_nchw, field_nchw.shape)

    valid = torch.isfinite(field_nchw) & torch.isfinite(mask_nchw) & (mask_nchw > 0)
    weights = torch.where(valid, mask_nchw, torch.zeros_like(mask_nchw))
    values = torch.where(valid, field_nchw, torch.zeros_like(field_nchw))

    pooled_sum = F.avg_pool2d(values * weights, kernel_size=kernel_size, stride=stride) * (kernel_size * kernel_size)
    pooled_weight = F.avg_pool2d(weights, kernel_size=kernel_size, stride=stride) * (kernel_size * kernel_size)
    pooled_mean = pooled_sum / pooled_weight.clamp_min(eps)
    pooled_mask = (pooled_weight > eps).to(field_nchw.dtype)
    return pooled_mean, pooled_mask


def masked_pool2d_mean_numpy(
    field: np.ndarray,
    mask: np.ndarray,
    factor: int,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute masked block means on arrays with trailing ``H, W`` dimensions."""
    field_dtype = np.result_type(np.asarray(field).dtype, np.float32)
    field_arr = np.asarray(field, dtype=field_dtype)
    mask_arr = np.asarray(mask, dtype=field_dtype)
    _ensure_hw(field_arr)
    _ensure_hw(mask_arr)

    if field_arr.shape[-2:] != mask_arr.shape[-2:]:
        raise ValueError("field and mask must share the same trailing spatial shape.")
    if field_arr.shape[-2] % factor != 0 or field_arr.shape[-1] % factor != 0:
        raise ValueError("Spatial shape must be divisible by factor.")

    leading_shape = field_arr.shape[:-2]
    coarse_h = field_arr.shape[-2] // factor
    coarse_w = field_arr.shape[-1] // factor
    reshaped_field = field_arr.reshape(*leading_shape, coarse_h, factor, coarse_w, factor)
    reshaped_mask = mask_arr.reshape(*leading_shape, coarse_h, factor, coarse_w, factor)

    valid = np.isfinite(reshaped_field) & np.isfinite(reshaped_mask) & (reshaped_mask > 0)
    weights = np.where(valid, reshaped_mask, 0.0)
    values = np.where(valid, reshaped_field, 0.0)

    summed = (values * weights).sum(axis=(-3, -1))
    counts = weights.sum(axis=(-3, -1))
    pooled = summed / np.clip(counts, a_min=eps, a_max=None)
    pooled_mask = counts > eps
    return pooled.astype(field_dtype), pooled_mask.astype(bool)


def upsample_coarse_residual(
    coarse_residual: torch.Tensor | np.ndarray,
    factor: int,
    output_shape: tuple[int, int] | None = None,
) -> torch.Tensor | np.ndarray:
    """Repeat a coarse residual back to fine resolution.

    Each coarse residual is broadcast to a ``factor x factor`` spatial block.
    """
    if isinstance(coarse_residual, torch.Tensor):
        upsampled = coarse_residual.repeat_interleave(factor, dim=-2).repeat_interleave(factor, dim=-1)
        if output_shape is not None:
            upsampled = upsampled[..., : output_shape[0], : output_shape[1]]
        return upsampled

    coarse_dtype = np.result_type(np.asarray(coarse_residual).dtype, np.float32)
    coarse_arr = np.asarray(coarse_residual, dtype=coarse_dtype)
    upsampled_arr = np.repeat(np.repeat(coarse_arr, factor, axis=-2), factor, axis=-1)
    if output_shape is not None:
        upsampled_arr = upsampled_arr[..., : output_shape[0], : output_shape[1]]
    return upsampled_arr.astype(coarse_dtype)


def latitude_area_weights(
    latitudes: torch.Tensor | np.ndarray,
    width: int | None = None,
    eps: float = 1e-12,
) -> torch.Tensor | np.ndarray:
    """Return normalized cos(latitude) cell-area weights.

    The result has shape ``[H, 1]`` or ``[H, W]`` and can be broadcast over
    leading batch/channel dimensions. Latitude is interpreted in degrees.
    """
    if isinstance(latitudes, torch.Tensor):
        lat = latitudes.to(dtype=torch.float64)
        weights = torch.cos(torch.deg2rad(lat)).clamp_min(eps)
        weights = weights / weights.max().clamp_min(eps)
        weights = weights.reshape(-1, 1)
        if width is not None:
            weights = weights.expand(-1, width)
        return weights
    lat_arr = np.asarray(latitudes, dtype=np.float64)
    weights_arr = np.clip(np.cos(np.deg2rad(lat_arr)), a_min=eps, a_max=None)
    weights_arr = weights_arr / np.max(weights_arr)
    weights_arr = weights_arr.reshape(-1, 1)
    if width is not None:
        weights_arr = np.broadcast_to(weights_arr, (lat_arr.size, width)).copy()
    return weights_arr


def _combine_projection_weights_torch(
    pred_nchw: torch.Tensor,
    mask: torch.Tensor,
    area_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Combine validity and area weights for a differentiable projection."""
    mask_nchw = _ensure_nchw(mask).to(pred_nchw.dtype)
    if mask_nchw.shape != pred_nchw.shape:
        mask_nchw = torch.broadcast_to(mask_nchw, pred_nchw.shape)
    if area_weights is None:
        return mask_nchw
    area = area_weights.to(device=pred_nchw.device, dtype=pred_nchw.dtype)
    if area.ndim == 1:
        area = area.reshape(1, 1, -1, 1)
    elif area.ndim == 2:
        area = area.reshape(1, 1, *area.shape)
    elif area.ndim == 3:
        area = area.unsqueeze(1)
    if area.shape != pred_nchw.shape:
        area = torch.broadcast_to(area, pred_nchw.shape)
    return mask_nchw * area.clamp_min(0)


def project_to_coarse_constraint_torch(
    pred_fine: torch.Tensor,
    target_coarse: torch.Tensor,
    mask: torch.Tensor,
    factor: int,
    area_weights: torch.Tensor | None = None,
    refinement_steps: int = 2,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Differentiably project a fine field onto exact coarse block means.

    This is the strict block-support projection used by rescue-v1. It adds the
    area-weighted coarse residual uniformly to valid fine cells in each block.
    Repeating the residual update removes float32 round-off without clipping or
    partial blending. Invalid blocks are left unchanged.
    """
    if factor <= 0:
        raise ValueError("factor must be positive.")
    if refinement_steps < 1:
        raise ValueError("refinement_steps must be at least one.")
    pred_nchw = _ensure_nchw(pred_fine)
    if pred_nchw.dtype in {torch.float16, torch.bfloat16, torch.float32}:
        pred_nchw = pred_nchw.to(torch.float64)
    target_nchw = _ensure_nchw(target_coarse).to(device=pred_nchw.device, dtype=pred_nchw.dtype)
    weights = _combine_projection_weights_torch(pred_nchw, mask, area_weights)
    coarse_before, coarse_valid = masked_pool2d_mean(
        pred_nchw,
        weights,
        kernel_size=factor,
        eps=eps,
    )
    if target_nchw.shape != coarse_before.shape:
        raise ValueError(
            f"target_coarse shape {tuple(target_nchw.shape)} does not match "
            f"aggregated prediction shape {tuple(coarse_before.shape)}."
        )

    projected = pred_nchw
    residual = torch.zeros_like(coarse_before)
    for _ in range(refinement_steps):
        current_coarse, current_valid = masked_pool2d_mean(
            projected,
            weights,
            kernel_size=factor,
            eps=eps,
        )
        residual = torch.where(
            current_valid > 0,
            target_nchw - current_coarse,
            torch.zeros_like(current_coarse),
        )
        residual_fine = upsample_coarse_residual(
            residual,
            factor,
            output_shape=pred_nchw.shape[-2:],
        )
        projected = torch.where(weights > 0, projected + residual_fine, projected)

    coarse_after, _ = masked_pool2d_mean(projected, weights, kernel_size=factor, eps=eps)
    valid_bool = coarse_valid > 0
    abs_before = torch.where(valid_bool, torch.abs(target_nchw - coarse_before), torch.zeros_like(coarse_before))
    abs_after = torch.where(valid_bool, torch.abs(target_nchw - coarse_after), torch.zeros_like(coarse_after))
    diagnostics = {
        "coarse_before": coarse_before,
        "coarse_after": coarse_after,
        "coarse_target": target_nchw,
        "coarse_valid": coarse_valid,
        "last_coarse_residual": residual,
        "refinement_steps": int(refinement_steps),
        "mass_closure_error_before": abs_before.sum() / coarse_valid.sum().clamp_min(eps),
        "mass_closure_error_after": abs_after.sum() / coarse_valid.sum().clamp_min(eps),
        "mass_closure_max_abs_after": abs_after.max(),
        "operator": "coslat_area_weighted_block_mean" if area_weights is not None else "masked_block_mean",
    }
    return projected, diagnostics


def project_to_coarse_constraint_numpy(
    pred_fine: np.ndarray,
    target_coarse: np.ndarray,
    mask: np.ndarray,
    factor: int,
    area_weights: np.ndarray | None = None,
    refinement_steps: int = 2,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """NumPy counterpart of :func:`project_to_coarse_constraint_torch`."""
    if factor <= 0:
        raise ValueError("factor must be positive.")
    if refinement_steps < 1:
        raise ValueError("refinement_steps must be at least one.")
    pred_arr = np.asarray(pred_fine, dtype=np.float64)
    target_arr = np.asarray(target_coarse, dtype=np.float64)
    mask_arr = np.asarray(mask, dtype=np.float64)
    if mask_arr.shape != pred_arr.shape:
        mask_arr = np.broadcast_to(mask_arr, pred_arr.shape)
    if area_weights is not None:
        area_arr = np.asarray(area_weights, dtype=np.float64)
        if area_arr.ndim == 1:
            area_arr = area_arr.reshape(-1, 1)
        area_arr = np.broadcast_to(area_arr, pred_arr.shape)
        weights = mask_arr * np.clip(area_arr, a_min=0.0, a_max=None)
        operator = "coslat_area_weighted_block_mean"
    else:
        weights = mask_arr
        operator = "masked_block_mean"

    coarse_before, coarse_valid = masked_pool2d_mean_numpy(pred_arr, weights, factor=factor, eps=eps)
    coarse_before = coarse_before.astype(np.float64)
    if target_arr.shape != coarse_before.shape:
        raise ValueError(
            f"target_coarse shape {target_arr.shape} does not match aggregated prediction shape {coarse_before.shape}."
        )
    projected = pred_arr.copy()
    residual = np.zeros_like(coarse_before)
    for _ in range(refinement_steps):
        current_coarse, current_valid = masked_pool2d_mean_numpy(projected, weights, factor=factor, eps=eps)
        residual = np.where(current_valid, target_arr - current_coarse.astype(np.float64), 0.0)
        residual_fine = upsample_coarse_residual(residual, factor, output_shape=pred_arr.shape[-2:])
        projected = np.where(weights > 0, projected + residual_fine, projected)

    coarse_after, _ = masked_pool2d_mean_numpy(projected, weights, factor=factor, eps=eps)
    valid = coarse_valid.astype(bool)
    before = np.where(valid, np.abs(target_arr - coarse_before), np.nan)
    after = np.where(valid, np.abs(target_arr - coarse_after.astype(np.float64)), np.nan)
    diagnostics = {
        "coarse_before": coarse_before,
        "coarse_after": coarse_after.astype(np.float64),
        "coarse_target": target_arr,
        "coarse_valid": valid,
        "last_coarse_residual": residual,
        "refinement_steps": int(refinement_steps),
        "mass_closure_error_before": float(np.nanmean(before)),
        "mass_closure_error_after": float(np.nanmean(after)),
        "mass_closure_max_abs_after": float(np.nanmax(after)),
        "operator": operator,
    }
    return projected, diagnostics


def apply_mass_closure_correction(
    pred_fine: torch.Tensor | np.ndarray,
    observed_fine: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    factor: int,
    blend_factor: float = 1.0,
    max_residual_abs: float | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor | np.ndarray, dict[str, Any]]:
    """Apply coarse residual redistribution so pooled prediction matches pooled observation.

    Args:
        pred_fine: Fine-grid prediction with shape ``[..., H, W]``.
        observed_fine: Fine-grid observed field on the same grid.
        mask: Land/valid mask with the same spatial shape as ``pred_fine``.
        factor: Coarse-to-fine aggregation factor.
        blend_factor: Fraction of the coarse residual redistributed back to fine cells.
        max_residual_abs: Optional absolute cap applied before the residual is redistributed.
        eps: Small denominator guard.

    Returns:
        Tuple ``(corrected_prediction, diagnostics)``.
    """
    if isinstance(pred_fine, torch.Tensor):
        pred_nchw = _ensure_nchw(pred_fine)
        obs_nchw = _ensure_nchw(observed_fine)
        mask_nchw = _ensure_nchw(mask).to(pred_nchw.dtype)
        if pred_nchw.shape != obs_nchw.shape:
            raise ValueError("pred_fine and observed_fine must have the same shape.")
        if mask_nchw.shape != pred_nchw.shape:
            mask_nchw = torch.broadcast_to(mask_nchw, pred_nchw.shape)

        coarse_pred, coarse_pred_mask = masked_pool2d_mean(pred_nchw, mask_nchw, kernel_size=factor, eps=eps)
        coarse_obs, coarse_obs_mask = masked_pool2d_mean(obs_nchw, mask_nchw, kernel_size=factor, eps=eps)
        coarse_valid = coarse_pred_mask * coarse_obs_mask
        coarse_residual = torch.where(coarse_valid > 0, coarse_obs - coarse_pred, torch.zeros_like(coarse_pred))
        if max_residual_abs is not None:
            coarse_residual = torch.clamp(coarse_residual, min=-max_residual_abs, max=max_residual_abs)
        residual_fine = upsample_coarse_residual(
            coarse_residual * float(blend_factor),
            factor,
            output_shape=pred_nchw.shape[-2:],
        )
        corrected = torch.where(mask_nchw > 0, pred_nchw + residual_fine, pred_nchw)

        mass_before = torch.abs(coarse_obs - coarse_pred)
        corrected_coarse, _ = masked_pool2d_mean(corrected, mask_nchw, kernel_size=factor, eps=eps)
        mass_after = torch.abs(coarse_obs - corrected_coarse)
        diagnostics = {
            "coarse_pred": coarse_pred,
            "coarse_obs": coarse_obs,
            "coarse_residual": coarse_residual,
            "residual_fine": residual_fine,
            "blend_factor": float(blend_factor),
            "mass_closure_error_before": (mass_before * coarse_valid).sum() / coarse_valid.sum().clamp_min(eps),
            "mass_closure_error_after": (mass_after * coarse_valid).sum() / coarse_valid.sum().clamp_min(eps),
        }
        return corrected, diagnostics

    pred_arr = np.asarray(pred_fine, dtype=np.float32)
    obs_arr = np.asarray(observed_fine, dtype=np.float32)
    mask_arr = np.asarray(mask, dtype=np.float32)
    if pred_arr.shape != obs_arr.shape:
        raise ValueError("pred_fine and observed_fine must have the same shape.")
    if mask_arr.shape != pred_arr.shape:
        mask_arr = np.broadcast_to(mask_arr, pred_arr.shape).astype(np.float32, copy=False)

    coarse_pred, coarse_pred_mask = masked_pool2d_mean_numpy(pred_arr, mask_arr, factor=factor, eps=eps)
    coarse_obs, coarse_obs_mask = masked_pool2d_mean_numpy(obs_arr, mask_arr, factor=factor, eps=eps)
    coarse_valid = coarse_pred_mask & coarse_obs_mask
    coarse_residual = np.where(coarse_valid, coarse_obs - coarse_pred, 0.0).astype(np.float32)
    if max_residual_abs is not None:
        coarse_residual = np.clip(coarse_residual, a_min=-max_residual_abs, a_max=max_residual_abs).astype(np.float32)
    residual_fine = upsample_coarse_residual(
        coarse_residual * float(blend_factor),
        factor,
        output_shape=pred_arr.shape[-2:],
    )
    corrected = np.where(mask_arr > 0, pred_arr + residual_fine, pred_arr).astype(np.float32)

    corrected_coarse, _ = masked_pool2d_mean_numpy(corrected, mask_arr, factor=factor, eps=eps)
    before = np.where(coarse_valid, np.abs(coarse_obs - coarse_pred), np.nan)
    after = np.where(coarse_valid, np.abs(coarse_obs - corrected_coarse), np.nan)
    diagnostics = {
        "coarse_pred": coarse_pred.astype(np.float32),
        "coarse_obs": coarse_obs.astype(np.float32),
        "coarse_residual": coarse_residual.astype(np.float32),
        "residual_fine": residual_fine.astype(np.float32),
        "blend_factor": float(blend_factor),
        "mass_closure_error_before": float(np.nanmean(before)),
        "mass_closure_error_after": float(np.nanmean(after)),
    }
    return corrected, diagnostics
