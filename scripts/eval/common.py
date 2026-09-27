"""Shared helpers for inference, evaluation, and experiment comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from utils.mass_closure import masked_pool2d_mean_numpy


def load_yaml(path: Path | str) -> dict:
    """Load a YAML file into a Python dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path | str, payload: dict) -> Path:
    """Write a JSON file with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def build_tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Enumerate tile starts and include the terminal tile."""
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def repeat_coarse_to_fine(coarse_field: np.ndarray, factor: int) -> np.ndarray:
    """Upsample a coarse grid to fine resolution via block repeat."""
    return np.repeat(np.repeat(coarse_field, factor, axis=0), factor, axis=1)


def aggregate_coarse_mean_numpy(field: np.ndarray, mask: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate a fine-grid field to coarse resolution with valid-aware averaging."""
    return masked_pool2d_mean_numpy(field, np.asarray(mask), factor=factor)


def spatial_correlation(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Compute correlation across one spatial field with a boolean mask."""
    valid = np.asarray(mask).astype(bool)
    if valid.sum() < 3:
        return float("nan")
    pred = np.asarray(prediction)[valid].astype(np.float64)
    targ = np.asarray(target)[valid].astype(np.float64)
    pred = pred - pred.mean()
    targ = targ - targ.mean()
    pred_std = pred.std()
    targ_std = targ.std()
    if pred_std == 0 or targ_std == 0:
        return float("nan")
    return float((pred * targ).mean() / (pred_std * targ_std))


def gradient_xy_numpy(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute finite differences in x and y for one 2D field."""
    grad_y = field[1:, :] - field[:-1, :]
    grad_x = field[:, 1:] - field[:, :-1]
    return grad_x, grad_y


def open_prediction_dataset(path: Path | str) -> xr.Dataset:
    """Open a prediction dataset from NetCDF or Zarr."""
    path = Path(path)
    if path.is_dir() and path.suffix.lower() == ".zarr":
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def resolve_prediction_var_name(ds: xr.Dataset, variant: str = "auto") -> str:
    """Resolve which prediction variable should be used for evaluation."""
    if variant == "raw":
        return "predicted_twsa_raw" if "predicted_twsa_raw" in ds.data_vars else "predicted_twsa"
    if variant == "corrected":
        return "predicted_twsa_corrected" if "predicted_twsa_corrected" in ds.data_vars else "predicted_twsa"
    if variant == "projected":
        return "predicted_twsa_projected" if "predicted_twsa_projected" in ds.data_vars else "predicted_twsa"
    default_variant = str(ds.attrs.get("default_prediction_variant", "raw")).lower()
    if default_variant == "corrected" and "predicted_twsa_corrected" in ds.data_vars:
        return "predicted_twsa_corrected"
    if default_variant == "projected" and "predicted_twsa_projected" in ds.data_vars:
        return "predicted_twsa_projected"
    return "predicted_twsa"


def resolve_prediction_coarse_var_name(ds: xr.Dataset, variant: str = "auto") -> str:
    """Resolve the coarse prediction variable name aligned with the selected variant."""
    if variant == "raw":
        return "predicted_twsa_coarse_raw" if "predicted_twsa_coarse_raw" in ds.data_vars else "predicted_twsa_coarse"
    if variant == "corrected":
        return "predicted_twsa_coarse_corrected" if "predicted_twsa_coarse_corrected" in ds.data_vars else "predicted_twsa_coarse"
    if variant == "projected":
        return "predicted_twsa_coarse_projected" if "predicted_twsa_coarse_projected" in ds.data_vars else "predicted_twsa_coarse"
    default_variant = str(ds.attrs.get("default_prediction_variant", "raw")).lower()
    if default_variant == "corrected" and "predicted_twsa_coarse_corrected" in ds.data_vars:
        return "predicted_twsa_coarse_corrected"
    if default_variant == "projected" and "predicted_twsa_coarse_projected" in ds.data_vars:
        return "predicted_twsa_coarse_projected"
    return "predicted_twsa_coarse"
