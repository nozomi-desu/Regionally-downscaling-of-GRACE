"""Native JPL mascon aggregation, projection, and null-space utilities.

The released JPL fields are sampled on a 0.5-degree grid but are estimated on
native equal-area mascons.  This module uses the released ``mascon_ID`` raster
as the observation support instead of inventing regular 6x6 blocks.

For area weights ``w_i`` and mascon ``m``, the observation operator is

    A_m(x) = sum_{i in m} w_i x_i / sum_{i in m} w_i.

Under the matching area-weighted quadratic metric, the minimum-change
projection adds one constant residual to every valid cell in a mascon.  The
same operator yields an explicit null-space component by removing each
mascon's weighted mean.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _validate_spatial_shape(field_shape: tuple[int, ...], spatial_shape: tuple[int, int]) -> None:
    if len(field_shape) < 2 or tuple(field_shape[-2:]) != tuple(spatial_shape):
        raise ValueError(
            f"Expected trailing spatial shape {spatial_shape}, got {field_shape}."
        )


def _mascon_layout_numpy(
    mascon_ids: np.ndarray,
    valid_mask: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids_raw = np.asarray(mascon_ids)
    if ids_raw.ndim != 2:
        raise ValueError(f"mascon_ids must be 2D, got {ids_raw.shape}.")
    finite = np.isfinite(ids_raw)
    rounded = np.where(finite, np.rint(ids_raw), 0).astype(np.int64)
    if np.any(finite & (np.abs(ids_raw - rounded) > 1e-8)):
        raise ValueError("mascon_ids must contain integer-valued identifiers.")
    base_valid = finite & (rounded > 0)
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.shape != ids_raw.shape:
            raise ValueError("valid_mask and mascon_ids must have identical shapes.")
        base_valid &= np.isfinite(mask) & (mask > 0)

    if labels is None:
        label_arr = np.unique(rounded[base_valid])
    else:
        label_arr = np.asarray(labels, dtype=np.int64)
        if label_arr.ndim != 1 or np.any(np.diff(label_arr) <= 0):
            raise ValueError("labels must be a strictly increasing 1D array.")
    if label_arr.size == 0:
        raise ValueError("No positive valid mascon identifiers were found.")

    index_map = np.full(ids_raw.shape, -1, dtype=np.int64)
    positions = np.searchsorted(label_arr, rounded[base_valid])
    matched = (positions < label_arr.size) & (label_arr[np.minimum(positions, label_arr.size - 1)] == rounded[base_valid])
    valid_positions = np.flatnonzero(base_valid)
    flat_index = index_map.reshape(-1)
    flat_index[valid_positions[matched]] = positions[matched]
    layout_valid = index_map >= 0
    return label_arr, index_map, layout_valid


def _mascon_layout_torch(
    mascon_ids: torch.Tensor | np.ndarray,
    valid_mask: torch.Tensor | np.ndarray | None = None,
    labels: torch.Tensor | np.ndarray | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids_raw = torch.as_tensor(mascon_ids, device=device)
    if ids_raw.ndim != 2:
        raise ValueError(f"mascon_ids must be 2D, got {tuple(ids_raw.shape)}.")
    finite = torch.isfinite(ids_raw) if ids_raw.is_floating_point() else torch.ones_like(ids_raw, dtype=torch.bool)
    rounded = torch.where(finite, torch.round(ids_raw.to(torch.float64)), torch.zeros_like(ids_raw, dtype=torch.float64)).to(torch.int64)
    if ids_raw.is_floating_point():
        error = torch.where(finite, torch.abs(ids_raw.to(torch.float64) - rounded.to(torch.float64)), torch.zeros_like(ids_raw, dtype=torch.float64))
        if bool(torch.any(error > 1e-8)):
            raise ValueError("mascon_ids must contain integer-valued identifiers.")
    base_valid = finite & (rounded > 0)
    if valid_mask is not None:
        mask = torch.as_tensor(valid_mask, device=ids_raw.device)
        if tuple(mask.shape) != tuple(ids_raw.shape):
            raise ValueError("valid_mask and mascon_ids must have identical shapes.")
        mask_finite = torch.isfinite(mask) if mask.is_floating_point() else torch.ones_like(mask, dtype=torch.bool)
        base_valid &= mask_finite & (mask > 0)

    if labels is None:
        label_tensor = torch.unique(rounded[base_valid], sorted=True)
    else:
        label_tensor = torch.as_tensor(labels, dtype=torch.int64, device=ids_raw.device)
        if label_tensor.ndim != 1 or bool(torch.any(label_tensor[1:] <= label_tensor[:-1])):
            raise ValueError("labels must be a strictly increasing 1D tensor.")
    if label_tensor.numel() == 0:
        raise ValueError("No positive valid mascon identifiers were found.")

    index_map = torch.full_like(rounded, -1)
    positions = torch.searchsorted(label_tensor, rounded[base_valid])
    safe_positions = positions.clamp(max=label_tensor.numel() - 1)
    matched = (positions < label_tensor.numel()) & (label_tensor[safe_positions] == rounded[base_valid])
    flat_index = index_map.reshape(-1)
    valid_positions = torch.nonzero(base_valid.reshape(-1), as_tuple=False).squeeze(1)
    flat_index[valid_positions[matched]] = positions[matched]
    layout_valid = index_map >= 0
    return label_tensor, index_map, layout_valid


def mascon_area_weighted_mean_numpy(
    field: np.ndarray,
    mascon_ids: np.ndarray,
    area_weights: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate arrays with trailing ``H,W`` dimensions to native mascons."""
    arr = np.asarray(field, dtype=np.float64)
    ids = np.asarray(mascon_ids)
    _validate_spatial_shape(arr.shape, ids.shape)
    label_arr, index_map, layout_valid = _mascon_layout_numpy(ids, valid_mask, labels)
    spatial_weights = np.ones(ids.shape, dtype=np.float64) if area_weights is None else np.asarray(area_weights, dtype=np.float64)
    if spatial_weights.shape != ids.shape:
        spatial_weights = np.broadcast_to(spatial_weights, ids.shape)
    spatial_weights = np.where(layout_valid & np.isfinite(spatial_weights) & (spatial_weights > 0), spatial_weights, 0.0)

    leading = arr.shape[:-2]
    flat = arr.reshape((-1, ids.size))
    idx = index_map.reshape(-1)
    base_weight = spatial_weights.reshape(-1)
    means = np.full((flat.shape[0], label_arr.size), np.nan, dtype=np.float64)
    valid_out = np.zeros_like(means, dtype=bool)
    for row_index, row in enumerate(flat):
        valid = (idx >= 0) & (base_weight > 0) & np.isfinite(row)
        if not np.any(valid):
            continue
        numerator = np.bincount(idx[valid], weights=row[valid] * base_weight[valid], minlength=label_arr.size)
        denominator = np.bincount(idx[valid], weights=base_weight[valid], minlength=label_arr.size)
        row_valid = denominator > eps
        means[row_index, row_valid] = numerator[row_valid] / denominator[row_valid]
        valid_out[row_index] = row_valid
    return means.reshape((*leading, label_arr.size)), valid_out.reshape((*leading, label_arr.size)), label_arr


def mascon_area_weighted_mean_torch(
    field: torch.Tensor,
    mascon_ids: torch.Tensor | np.ndarray,
    area_weights: torch.Tensor | np.ndarray | None = None,
    valid_mask: torch.Tensor | np.ndarray | None = None,
    labels: torch.Tensor | np.ndarray | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiably aggregate tensors with trailing ``H,W`` dimensions."""
    if not field.is_floating_point():
        raise TypeError("field must be a floating-point tensor.")
    ids = torch.as_tensor(mascon_ids, device=field.device)
    _validate_spatial_shape(tuple(field.shape), tuple(ids.shape))
    label_tensor, index_map, layout_valid = _mascon_layout_torch(ids, valid_mask, labels, field.device)
    if area_weights is None:
        spatial_weights = torch.ones(ids.shape, dtype=field.dtype, device=field.device)
    else:
        spatial_weights = torch.as_tensor(area_weights, dtype=field.dtype, device=field.device)
        if tuple(spatial_weights.shape) != tuple(ids.shape):
            spatial_weights = torch.broadcast_to(spatial_weights, ids.shape)
    spatial_weights = torch.where(
        layout_valid & torch.isfinite(spatial_weights) & (spatial_weights > 0),
        spatial_weights,
        torch.zeros_like(spatial_weights),
    )

    leading = tuple(field.shape[:-2])
    flat = field.reshape((-1, ids.numel()))
    idx = index_map.reshape(-1)
    safe_idx = idx.clamp_min(0)
    base_weight = spatial_weights.reshape(-1)
    finite = torch.isfinite(flat)
    weight = torch.where(finite & (idx.unsqueeze(0) >= 0), base_weight.unsqueeze(0), torch.zeros_like(flat))
    values = torch.where(finite, flat, torch.zeros_like(flat))
    expanded_idx = safe_idx.unsqueeze(0).expand(flat.shape[0], -1)
    numerator = torch.zeros((flat.shape[0], label_tensor.numel()), dtype=field.dtype, device=field.device)
    denominator = torch.zeros_like(numerator)
    numerator.scatter_add_(1, expanded_idx, values * weight)
    denominator.scatter_add_(1, expanded_idx, weight)
    valid_out = denominator > eps
    means = numerator / denominator.clamp_min(eps)
    means = torch.where(valid_out, means, torch.full_like(means, torch.nan))
    return means.reshape((*leading, label_tensor.numel())), valid_out.reshape((*leading, label_tensor.numel())), label_tensor


def broadcast_mascon_values_numpy(
    values: np.ndarray,
    mascon_ids: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray | None = None,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Broadcast trailing mascon values back to their native grid supports."""
    value_arr = np.asarray(values, dtype=np.float64)
    label_arr, index_map, layout_valid = _mascon_layout_numpy(mascon_ids, valid_mask, labels)
    if value_arr.shape[-1] != label_arr.size:
        raise ValueError("The trailing values dimension must equal the number of labels.")
    leading = value_arr.shape[:-1]
    flat_values = value_arr.reshape((-1, label_arr.size))
    safe_idx = np.clip(index_map.reshape(-1), 0, None)
    gathered = flat_values[:, safe_idx]
    gathered[:, ~layout_valid.reshape(-1)] = fill_value
    return gathered.reshape((*leading, *index_map.shape))


def broadcast_mascon_values_torch(
    values: torch.Tensor,
    mascon_ids: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    valid_mask: torch.Tensor | np.ndarray | None = None,
    fill_value: float = float("nan"),
) -> torch.Tensor:
    """Differentiably broadcast trailing mascon values to a native grid."""
    label_tensor, index_map, layout_valid = _mascon_layout_torch(
        mascon_ids, valid_mask, labels, values.device
    )
    if values.shape[-1] != label_tensor.numel():
        raise ValueError("The trailing values dimension must equal the number of labels.")
    leading = tuple(values.shape[:-1])
    flat_values = values.reshape((-1, label_tensor.numel()))
    safe_idx = index_map.reshape(-1).clamp_min(0)
    gathered = flat_values[:, safe_idx]
    gathered = torch.where(
        layout_valid.reshape(1, -1),
        gathered,
        torch.full_like(gathered, fill_value),
    )
    return gathered.reshape((*leading, *index_map.shape))


def project_to_native_mascons_numpy(
    pred_fine: np.ndarray,
    target_mascon: np.ndarray,
    mascon_ids: np.ndarray,
    area_weights: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    refinement_steps: int = 2,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project a field onto exact native-mascon area-weighted means."""
    if refinement_steps < 1:
        raise ValueError("refinement_steps must be at least one.")
    pred = np.asarray(pred_fine, dtype=np.float64)
    before, coarse_valid, label_arr = mascon_area_weighted_mean_numpy(
        pred, mascon_ids, area_weights, valid_mask, labels, eps
    )
    target = np.asarray(target_mascon, dtype=np.float64)
    if target.shape != before.shape:
        raise ValueError(f"target_mascon shape {target.shape} does not match {before.shape}.")
    projected = pred.copy()
    residual = np.zeros_like(before)
    for _ in range(refinement_steps):
        current, current_valid, _ = mascon_area_weighted_mean_numpy(
            projected, mascon_ids, area_weights, valid_mask, label_arr, eps
        )
        update_valid = current_valid & np.isfinite(target)
        residual = np.where(update_valid, target - current, 0.0)
        fine_residual = broadcast_mascon_values_numpy(
            residual, mascon_ids, label_arr, valid_mask, fill_value=0.0
        )
        projected = np.where(np.isfinite(fine_residual) & (fine_residual != 0), projected + fine_residual, projected)
    after, _, _ = mascon_area_weighted_mean_numpy(
        projected, mascon_ids, area_weights, valid_mask, label_arr, eps
    )
    valid = coarse_valid & np.isfinite(target)
    abs_before = np.where(valid, np.abs(target - before), np.nan)
    abs_after = np.where(valid, np.abs(target - after), np.nan)
    diagnostics = {
        "mascon_ids": label_arr,
        "mascon_before": before,
        "mascon_after": after,
        "mascon_target": target,
        "mascon_valid": valid,
        "last_mascon_residual": residual,
        "mass_closure_error_before": float(np.nanmean(abs_before)),
        "mass_closure_error_after": float(np.nanmean(abs_after)),
        "mass_closure_max_abs_after": float(np.nanmax(abs_after)),
        "operator": "jpl_native_mascon_area_weighted_mean",
    }
    return projected, diagnostics


def project_to_native_mascons_torch(
    pred_fine: torch.Tensor,
    target_mascon: torch.Tensor,
    mascon_ids: torch.Tensor | np.ndarray,
    area_weights: torch.Tensor | np.ndarray | None = None,
    valid_mask: torch.Tensor | np.ndarray | None = None,
    labels: torch.Tensor | np.ndarray | None = None,
    refinement_steps: int = 2,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Differentiably project a field onto exact native-mascon means."""
    if refinement_steps < 1:
        raise ValueError("refinement_steps must be at least one.")
    working = pred_fine.to(torch.float64) if pred_fine.dtype in {torch.float16, torch.bfloat16, torch.float32} else pred_fine
    before, coarse_valid, label_tensor = mascon_area_weighted_mean_torch(
        working, mascon_ids, area_weights, valid_mask, labels, eps
    )
    target = target_mascon.to(device=working.device, dtype=working.dtype)
    if tuple(target.shape) != tuple(before.shape):
        raise ValueError(f"target_mascon shape {tuple(target.shape)} does not match {tuple(before.shape)}.")
    projected = working
    residual = torch.zeros_like(before)
    for _ in range(refinement_steps):
        current, current_valid, _ = mascon_area_weighted_mean_torch(
            projected, mascon_ids, area_weights, valid_mask, label_tensor, eps
        )
        update_valid = current_valid & torch.isfinite(target)
        residual = torch.where(update_valid, target - current, torch.zeros_like(current))
        fine_residual = broadcast_mascon_values_torch(
            residual, mascon_ids, label_tensor, valid_mask, fill_value=0.0
        )
        projected = projected + fine_residual
    after, _, _ = mascon_area_weighted_mean_torch(
        projected, mascon_ids, area_weights, valid_mask, label_tensor, eps
    )
    valid = coarse_valid & torch.isfinite(target)
    abs_before = torch.where(valid, torch.abs(target - before), torch.zeros_like(before))
    abs_after = torch.where(valid, torch.abs(target - after), torch.zeros_like(after))
    diagnostics = {
        "mascon_ids": label_tensor,
        "mascon_before": before,
        "mascon_after": after,
        "mascon_target": target,
        "mascon_valid": valid,
        "last_mascon_residual": residual,
        "mass_closure_error_before": abs_before.sum() / valid.sum().clamp_min(1),
        "mass_closure_error_after": abs_after.sum() / valid.sum().clamp_min(1),
        "mass_closure_max_abs_after": abs_after.max(),
        "operator": "jpl_native_mascon_area_weighted_mean",
    }
    return projected, diagnostics


def native_mascon_nullspace_numpy(
    component: np.ndarray,
    mascon_ids: np.ndarray,
    area_weights: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove native-mascon means so the returned component lies in null(A)."""
    means, valid, label_arr = mascon_area_weighted_mean_numpy(
        component, mascon_ids, area_weights, valid_mask, labels, eps
    )
    mean_field = broadcast_mascon_values_numpy(means, mascon_ids, label_arr, valid_mask, fill_value=0.0)
    null_component = np.asarray(component, dtype=np.float64) - mean_field
    check, check_valid, _ = mascon_area_weighted_mean_numpy(
        null_component, mascon_ids, area_weights, valid_mask, label_arr, eps
    )
    diagnostics = {
        "mascon_ids": label_arr,
        "removed_mascon_mean": means,
        "nullspace_mascon_mean": check,
        "nullspace_max_abs_mean": float(np.nanmax(np.where(check_valid, np.abs(check), np.nan))),
    }
    return null_component, diagnostics


def native_mascon_nullspace_torch(
    component: torch.Tensor,
    mascon_ids: torch.Tensor | np.ndarray,
    area_weights: torch.Tensor | np.ndarray | None = None,
    valid_mask: torch.Tensor | np.ndarray | None = None,
    labels: torch.Tensor | np.ndarray | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Differentiably remove native-mascon means from a fine component."""
    working = component.to(torch.float64) if component.dtype in {torch.float16, torch.bfloat16, torch.float32} else component
    means, valid, label_tensor = mascon_area_weighted_mean_torch(
        working, mascon_ids, area_weights, valid_mask, labels, eps
    )
    mean_field = broadcast_mascon_values_torch(means, mascon_ids, label_tensor, valid_mask, fill_value=0.0)
    null_component = working - mean_field
    check, check_valid, _ = mascon_area_weighted_mean_torch(
        null_component, mascon_ids, area_weights, valid_mask, label_tensor, eps
    )
    check_abs = torch.where(check_valid, torch.abs(check), torch.zeros_like(check))
    diagnostics = {
        "mascon_ids": label_tensor,
        "removed_mascon_mean": means,
        "nullspace_mascon_mean": check,
        "nullspace_max_abs_mean": check_abs.max(),
    }
    return null_component, diagnostics


def decompose_observation_and_nullspace_numpy(
    prior_fine: np.ndarray,
    target_mascon: np.ndarray,
    mascon_ids: np.ndarray,
    area_weights: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return observation component, prior null-space component, and their sum."""
    null_component, null_diag = native_mascon_nullspace_numpy(
        prior_fine, mascon_ids, area_weights, valid_mask, labels, eps
    )
    label_arr = null_diag["mascon_ids"]
    observation_component = broadcast_mascon_values_numpy(
        target_mascon, mascon_ids, label_arr, valid_mask, fill_value=np.nan
    )
    reconstruction = observation_component + null_component
    reconstructed_means, valid, _ = mascon_area_weighted_mean_numpy(
        reconstruction, mascon_ids, area_weights, valid_mask, label_arr, eps
    )
    target = np.asarray(target_mascon, dtype=np.float64)
    closure = np.where(valid & np.isfinite(target), np.abs(reconstructed_means - target), np.nan)
    diagnostics = {
        **null_diag,
        "reconstruction_mascon_mean": reconstructed_means,
        "mass_closure_max_abs": float(np.nanmax(closure)),
        "operator": "jpl_native_mascon_observation_plus_nullspace",
    }
    return observation_component, null_component, reconstruction, diagnostics


def decompose_observation_and_nullspace_with_support_numpy(
    prior_fine: np.ndarray,
    target_mascon: np.ndarray,
    mascon_ids: np.ndarray,
    area_weights: np.ndarray | None = None,
    detail_valid_mask: np.ndarray | None = None,
    output_valid_mask: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Decompose partial fine detail on a complete native-mascon support.

    The observation component is defined over ``output_valid_mask``.  The
    prior contributes detail only on ``detail_valid_mask``; its mascon mean is
    removed on that support and its null-space component is zero elsewhere.
    Mascons without a usable prior are therefore observation-only, while the
    complete output still closes exactly under the native operator.
    """
    prior = np.asarray(prior_fine, dtype=np.float64)
    ids = np.asarray(mascon_ids)
    if prior.ndim != 2:
        raise ValueError("prior_fine must be a single 2D field for supported decomposition.")
    _validate_spatial_shape(prior.shape, ids.shape)
    output_mask = (
        np.isfinite(ids) & (ids > 0)
        if output_valid_mask is None
        else np.asarray(output_valid_mask, dtype=bool)
    )
    if output_mask.shape != ids.shape:
        raise ValueError("output_valid_mask and mascon_ids must have identical shapes.")
    if detail_valid_mask is None:
        detail_mask = output_mask & np.isfinite(prior)
    else:
        detail_mask = np.asarray(detail_valid_mask, dtype=bool)
        if detail_mask.shape != ids.shape:
            raise ValueError("detail_valid_mask and mascon_ids must have identical shapes.")
        detail_mask &= output_mask & np.isfinite(prior)

    label_arr, _, _ = _mascon_layout_numpy(ids, output_mask, labels)
    null_raw, null_diag = native_mascon_nullspace_numpy(
        prior,
        ids,
        area_weights,
        detail_mask,
        label_arr,
        eps,
    )
    null_component = np.where(detail_mask, null_raw, 0.0)
    observation_component = broadcast_mascon_values_numpy(
        target_mascon,
        ids,
        label_arr,
        output_mask,
        fill_value=np.nan,
    )
    reconstruction = np.where(
        output_mask,
        observation_component + null_component,
        np.nan,
    )
    reconstructed_means, reconstructed_valid, _ = mascon_area_weighted_mean_numpy(
        reconstruction,
        ids,
        area_weights,
        output_mask,
        label_arr,
        eps,
    )
    target = np.asarray(target_mascon, dtype=np.float64)
    if target.shape != reconstructed_means.shape:
        raise ValueError(
            f"target_mascon shape {target.shape} does not match {reconstructed_means.shape}."
        )
    closure_valid = reconstructed_valid & np.isfinite(target)
    closure = np.where(
        closure_valid,
        np.abs(reconstructed_means - target),
        np.nan,
    )

    spatial_weights = (
        np.ones(ids.shape, dtype=np.float64)
        if area_weights is None
        else np.broadcast_to(np.asarray(area_weights, dtype=np.float64), ids.shape)
    )
    unit_weights = np.ones(ids.shape, dtype=np.float64)
    full_area_mean, full_valid, _ = mascon_area_weighted_mean_numpy(
        spatial_weights,
        ids,
        unit_weights,
        output_mask,
        label_arr,
        eps,
    )
    detail_area_mean, detail_valid, _ = mascon_area_weighted_mean_numpy(
        np.where(detail_mask, spatial_weights, 0.0),
        ids,
        unit_weights,
        output_mask,
        label_arr,
        eps,
    )
    coverage = np.divide(
        detail_area_mean,
        full_area_mean,
        out=np.zeros_like(detail_area_mean),
        where=full_valid & detail_valid & (full_area_mean > eps),
    )
    diagnostics = {
        **null_diag,
        "reconstruction_mascon_mean": reconstructed_means,
        "mascon_output_valid": reconstructed_valid,
        "detail_area_fraction": coverage,
        "detail_supported_mascon_count": int(np.sum(coverage > 0)),
        "observation_only_mascon_count": int(np.sum(reconstructed_valid & (coverage <= 0))),
        "mass_closure_max_abs": float(np.nanmax(closure)),
        "operator": "jpl_native_mascon_observation_plus_supported_nullspace",
    }
    return observation_component, null_component, reconstruction, diagnostics


def decompose_observation_and_nullspace_torch(
    prior_fine: torch.Tensor,
    target_mascon: torch.Tensor,
    mascon_ids: torch.Tensor | np.ndarray,
    area_weights: torch.Tensor | np.ndarray | None = None,
    valid_mask: torch.Tensor | np.ndarray | None = None,
    labels: torch.Tensor | np.ndarray | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Differentiable observation/null-space decomposition."""
    null_component, null_diag = native_mascon_nullspace_torch(
        prior_fine, mascon_ids, area_weights, valid_mask, labels, eps
    )
    label_tensor = null_diag["mascon_ids"]
    target = target_mascon.to(device=null_component.device, dtype=null_component.dtype)
    observation_component = broadcast_mascon_values_torch(
        target, mascon_ids, label_tensor, valid_mask, fill_value=float("nan")
    )
    reconstruction = observation_component + null_component
    reconstructed_means, valid, _ = mascon_area_weighted_mean_torch(
        reconstruction, mascon_ids, area_weights, valid_mask, label_tensor, eps
    )
    closure = torch.where(
        valid & torch.isfinite(target),
        torch.abs(reconstructed_means - target),
        torch.zeros_like(reconstructed_means),
    )
    diagnostics = {
        **null_diag,
        "reconstruction_mascon_mean": reconstructed_means,
        "mass_closure_max_abs": closure.max(),
        "operator": "jpl_native_mascon_observation_plus_nullspace",
    }
    return observation_component, null_component, reconstruction, diagnostics
