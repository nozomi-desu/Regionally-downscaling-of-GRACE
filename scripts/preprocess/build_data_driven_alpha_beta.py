"""Generate train-only, coefficient-free data-driven RWSC alpha and beta maps.

Alpha is derived from centred JPL--WGHM disagreement at the existing coarse
resolution. Beta is derived from equal-vote median disagreement between the
WGHM forward-difference gradient and independent parent-product gradients.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMULA_ALPHA = "centered_symmetric_absolute_disagreement_v1"
FORMULA_BETA = "normalized_forward_gradient_multisource_median_v1"
BANNED_BETA_TOKENS = ("smap", "groundwater", "well", "wgms", "glacier", "grace", "jpl", "csr", "gsfc")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    parent_product: str
    path: Path | None
    variable: str
    source_url_or_doi: str
    version: str
    original_grid: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-zarr",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "training_rescue_v2" / "grace_downscaling_training_dataset_mm_area.zarr",
    )
    parser.add_argument(
        "--gldas",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "independent_priors" / "gldas_noah_twsa_05deg_200204_202212.nc",
    )
    parser.add_argument(
        "--clm5",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "independent_priors" / "clm5_twsa_05deg_200204_201412.nc",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data_driven_alpha_beta" / "weights",
    )
    parser.add_argument("--train-end", default="2016-12-01")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    metadata = path / ".zmetadata"
    if metadata.exists():
        return sha256_file(metadata)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(str(child.stat().st_size).encode("ascii"))
    return digest.hexdigest()


def month_start(values: Iterable[object]) -> np.ndarray:
    return pd.to_datetime(np.asarray(list(values))).to_period("M").to_timestamp().values.astype("datetime64[ns]")


def assert_train_only(times: np.ndarray, train_end: str) -> None:
    timestamps = pd.to_datetime(times)
    if len(timestamps) == 0:
        raise ValueError("Training timestamp set is empty.")
    cutoff = pd.Timestamp(train_end)
    leaked = timestamps[timestamps > cutoff]
    if len(leaked):
        raise ValueError(f"Weight-generation leakage: {len(leaked)} timestamps exceed {cutoff.date()}.")
    if len(timestamps) != len(np.unique(timestamps.values.astype("datetime64[M]"))):
        raise ValueError("Weight-generation timestamps contain duplicate months.")


def _finite_bounds(values: np.ndarray, label: str, tolerance: float = 2e-6) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size and (finite.min() < -tolerance or finite.max() > 1.0 + tolerance):
        raise ValueError(f"{label} is outside theoretical [0,1]: min={finite.min()}, max={finite.max()}")
    # This is solely a floating-point safety correction, not an empirical range.
    return np.where(np.isfinite(values), np.minimum(1.0, np.maximum(0.0, values)), values)


def compute_alpha(
    coarse_wghm: np.ndarray,
    coarse_jpl: np.ndarray,
    coarse_support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return alpha, D_alpha, centred WGHM and centred JPL on coarse cells."""
    w = np.asarray(coarse_wghm, dtype=np.float64)
    j = np.asarray(coarse_jpl, dtype=np.float64)
    if w.shape != j.shape or w.ndim != 3:
        raise ValueError("coarse_wghm and coarse_jpl must share shape (time, coarse_lat, coarse_lon).")
    support = np.asarray(coarse_support, dtype=bool)
    valid = np.isfinite(w) & np.isfinite(j) & support[None, :, :]
    w_masked = np.where(valid, w, np.nan)
    j_masked = np.where(valid, j, np.nan)
    w_center = w_masked - np.nanmedian(w_masked, axis=0)[None, :, :]
    j_center = j_masked - np.nanmedian(j_masked, axis=0)[None, :, :]
    numerator = np.nansum(np.abs(w_center - j_center), axis=0)
    denominator = np.nansum(np.abs(w_center) + np.abs(j_center), axis=0)
    d_alpha = np.full(denominator.shape, np.nan, dtype=np.float64)
    informative = support & (denominator > 0)
    d_alpha[informative] = numerator[informative] / denominator[informative]
    d_alpha[support & ~informative] = 1.0  # prescribed conservative alpha=0 rule
    d_alpha = _finite_bounds(d_alpha, "D_alpha")
    alpha = np.where(support, 1.0 - d_alpha, np.nan)
    alpha = _finite_bounds(alpha, "alpha")
    return alpha, d_alpha, w_center, j_center


def forward_gradient(field: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the exact first-order forward differences used by gradient_xy."""
    values = np.asarray(field, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if values.ndim != 3 or mask.shape != values.shape:
        raise ValueError("field and valid must share shape (time, lat, lon).")
    gx = np.full(values.shape, np.nan, dtype=np.float64)
    gy = np.full(values.shape, np.nan, dtype=np.float64)
    valid_x = mask[:, :, :-1] & mask[:, :, 1:]
    valid_y = mask[:, :-1, :] & mask[:, 1:, :]
    dx = values[:, :, 1:] - values[:, :, :-1]
    dy = values[:, 1:, :] - values[:, :-1, :]
    gx[:, :, :-1] = np.where(valid_x, dx, np.nan)
    gy[:, :-1, :] = np.where(valid_y, dy, np.nan)
    vector_valid = np.isfinite(gx) & np.isfinite(gy)
    return gx, gy, vector_valid


def gradient_scale(gx: np.ndarray, gy: np.ndarray, valid: np.ndarray, source_name: str) -> float:
    norms = np.sqrt(gx * gx + gy * gy)
    positive = norms[valid & np.isfinite(norms) & (norms > 0)]
    if positive.size == 0:
        raise ValueError(f"Source {source_name!r} has no positive finite training gradient norm.")
    return float(np.median(positive))


def gradient_disagreement(
    wghm_gx: np.ndarray,
    wghm_gy: np.ndarray,
    wghm_valid: np.ndarray,
    wghm_scale: float,
    source_gx: np.ndarray,
    source_gy: np.ndarray,
    source_valid: np.ndarray,
    source_scale: float,
) -> np.ndarray:
    valid = wghm_valid & source_valid
    wx = wghm_gx / wghm_scale
    wy = wghm_gy / wghm_scale
    sx = source_gx / source_scale
    sy = source_gy / source_scale
    numerator = np.sqrt((wx - sx) ** 2 + (wy - sy) ** 2)
    denominator = np.sqrt(wx * wx + wy * wy) + np.sqrt(sx * sx + sy * sy)
    result = np.full(denominator.shape, np.nan, dtype=np.float64)
    both_zero = valid & (denominator == 0)
    informative = valid & (denominator > 0)
    result[both_zero] = 0.0
    result[informative] = numerator[informative] / denominator[informative]
    return _finite_bounds(result, "gradient disagreement")


def nanmedian_no_warning(values: np.ndarray, axis: int) -> np.ndarray:
    valid_count = np.sum(np.isfinite(values), axis=axis)
    safe = np.where(np.isfinite(values), values, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        result = np.nanmedian(safe, axis=axis)
    return np.where(valid_count > 0, result, np.nan)


def combine_beta(disagreements: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine array (source,time,lat,lon) by equal-vote source and time medians."""
    if disagreements.ndim != 4:
        raise ValueError("disagreements must have shape (source,time,lat,lon).")
    d_time = nanmedian_no_warning(disagreements, axis=0)
    valid_time_count = np.sum(np.isfinite(d_time), axis=0).astype(np.int16)
    d_beta = nanmedian_no_warning(d_time, axis=0)
    no_evidence = valid_time_count == 0
    d_beta[no_evidence] = 1.0  # prescribed conservative beta=0 rule
    d_beta = _finite_bounds(d_beta, "D_beta")
    beta = _finite_bounds(1.0 - d_beta, "beta")
    beta[no_evidence] = 0.0
    return beta, d_beta, valid_time_count


def normalize_source_time(da: xr.DataArray) -> xr.DataArray:
    normalized = da.assign_coords(time=("time", month_start(da["time"].values)))
    if len(np.unique(normalized["time"].values)) != normalized.sizes["time"]:
        normalized = normalized.groupby("time").mean()
    return normalized


def load_source_on_training_grid(spec: SourceSpec, training: xr.Dataset, train_times: np.ndarray) -> np.ndarray:
    if any(token in f"{spec.name} {spec.parent_product}".lower() for token in BANNED_BETA_TOKENS):
        raise ValueError(f"External-validation or GRACE-derived source is forbidden for beta: {spec.name}")
    if spec.path is None:
        da = training[spec.variable]
    else:
        with xr.open_dataset(spec.path) as source_ds:
            da = source_ds[spec.variable].load()
    da = normalize_source_time(da)
    if da["lat"].values[0] > da["lat"].values[-1]:
        da = da.sortby("lat")
    if da["lon"].values[0] > da["lon"].values[-1]:
        da = da.sortby("lon")
    if not np.array_equal(da["lat"].values, training["lat"].values) or not np.array_equal(
        da["lon"].values, training["lon"].values
    ):
        raise ValueError(f"Preprocessed beta source {spec.name} does not match the frozen 0.5-degree grid.")
    da = da.reindex(time=train_times)
    return np.asarray(da.values, dtype=np.float64)


def diagnostic_fields(w: np.ndarray, j: np.ndarray, times: np.ndarray) -> dict[str, np.ndarray]:
    shape = w.shape[1:]
    corr = np.full(shape, np.nan)
    trend = np.full(shape, np.nan)
    amplitude = np.full(shape, np.nan)
    nrmse = np.full(shape, np.nan)
    x = np.arange(w.shape[0], dtype=np.float64)
    months = pd.to_datetime(times).month.to_numpy()
    for row in range(shape[0]):
        for col in range(shape[1]):
            a, b = w[:, row, col], j[:, row, col]
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() < 3:
                continue
            av, bv = a[valid], b[valid]
            if np.std(av) > 0 and np.std(bv) > 0:
                corr[row, col] = np.corrcoef(av, bv)[0, 1]
            trend[row, col] = np.polyfit(x[valid], av, 1)[0] - np.polyfit(x[valid], bv, 1)[0]
            rmse = np.sqrt(np.mean((av - bv) ** 2))
            if np.std(bv) > 0:
                nrmse[row, col] = rmse / np.std(bv)
            clim_a = [np.nanmean(a[months == m]) for m in range(1, 13)]
            clim_b = [np.nanmean(b[months == m]) for m in range(1, 13)]
            amplitude[row, col] = 0.5 * (np.nanmax(clim_a) - np.nanmin(clim_a)) - 0.5 * (
                np.nanmax(clim_b) - np.nanmin(clim_b)
            )
    return {"pcc": corr, "trend_difference": trend, "annual_amplitude_difference": amplitude, "nrmse": nrmse}


def spearman_rows(alpha: np.ndarray, diagnostics: dict[str, np.ndarray]) -> list[dict[str, object]]:
    from scipy.stats import spearmanr

    rows = []
    for name, field in diagnostics.items():
        values = np.abs(field) if name in {"trend_difference", "annual_amplitude_difference"} else field
        valid = np.isfinite(alpha) & np.isfinite(values)
        statistic, pvalue = spearmanr(alpha[valid], values[valid]) if valid.sum() >= 3 else (np.nan, np.nan)
        rows.append({"diagnostic": name, "spearman_rho": statistic, "p_value": pvalue, "n": int(valid.sum())})
    return rows


def area_weighted_mean(field: np.ndarray, lat: np.ndarray, land: np.ndarray) -> float:
    weights = np.cos(np.deg2rad(lat))[:, None] * np.asarray(land, dtype=np.float64)
    valid = np.isfinite(field) & np.isfinite(weights) & (weights > 0)
    return float(np.sum(field[valid] * weights[valid]) / np.sum(weights[valid]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    outputs = [
        output_dir / "alpha_data_driven.nc",
        output_dir / "beta_data_driven.nc",
        output_dir / "beta_source_count.nc",
        output_dir / "weight_generation_metadata.json",
    ]
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise FileExistsError("Weight outputs already exist; use --overwrite only for the isolated new experiment.")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_path = args.training_zarr.resolve()
    training = xr.open_zarr(training_path, consolidated=False)
    if str(training.attrs.get("canonical_water_storage_units")) != "mm EWH":
        raise ValueError("Formal weight generation requires the unit-corrected mm EWH training store.")
    split = np.asarray(training["split_index"].values)
    train_index = np.flatnonzero(split == 0)
    train_times = month_start(training["time"].values[train_index])
    assert_train_only(train_times, args.train_end)
    if train_index.size != 156:
        raise ValueError(f"Frozen original training split must contain 156 months, found {train_index.size}.")

    land = np.asarray(training["land_mask"].values > 0, dtype=bool)
    valid_fine = np.asarray(training["valid_mask"].isel(time=train_index).values > 0, dtype=bool) & land[None, :, :]
    wghm_fine = np.asarray(training["input_wghm_twsa"].isel(time=train_index).values, dtype=np.float64)
    jpl_coarse = np.asarray(training["target_jplm_twsa_coarse"].isel(time=train_index).values, dtype=np.float64)
    wghm_coarse = np.asarray(training["input_wghm_twsa_coarse"].isel(time=train_index).values, dtype=np.float64)

    factor = int(training.attrs.get("coarse_factor", 6))
    if factor != 6 or land.shape[0] % factor or land.shape[1] % factor:
        raise ValueError("Frozen experiment expects a regular 6x6 coarse mapping.")
    area = np.cos(np.deg2rad(training["lat"].values))[:, None] * land
    coarse_support = area.reshape(land.shape[0] // factor, factor, land.shape[1] // factor, factor).sum((1, 3)) > 0
    alpha_coarse, d_alpha_coarse, w_center, j_center = compute_alpha(wghm_coarse, jpl_coarse, coarse_support)
    alpha = np.repeat(np.repeat(alpha_coarse, factor, axis=0), factor, axis=1)
    d_alpha = np.repeat(np.repeat(d_alpha_coarse, factor, axis=0), factor, axis=1)
    alpha = np.where(land, alpha, np.nan)
    d_alpha = np.where(land, d_alpha, np.nan)
    coarse_mapping = np.arange(alpha_coarse.size, dtype=np.int32).reshape(alpha_coarse.shape)
    coarse_mapping = np.repeat(np.repeat(coarse_mapping, factor, axis=0), factor, axis=1)

    specs = [
        SourceSpec("ERA5_TWSA", "ERA5", None, "input_era5_twsa", "https://cds.climate.copernicus.eu/", "project-derived", "0.5 degree processed", "One ERA5 parent vote; ERA5 CWSC excluded as duplicate parent."),
        SourceSpec("GLDAS_Noah_TWSA", "GLDAS-Noah v3.3", args.gldas.resolve(), "gldas_noah_twsa", "10.5067/GGDAS-3NH33", "3.3", "1 degree", "Soil water + snow + canopy; incomplete TWSA structural prior."),
        SourceSpec("CLM5_TWSA", "CLM5.0 GSWP3", args.clm5.resolve(), "clm5_twsa", "project-cached source; see source metadata", "5.0", "approximately 1 degree", "Independent land-model structural prior through 2014-12."),
    ]
    wgx, wgy, wg_valid = forward_gradient(wghm_fine, valid_fine)
    q_values: dict[str, float] = {"WGHM": gradient_scale(wgx, wgy, wg_valid, "WGHM")}
    disagreements: list[np.ndarray] = []
    per_source_valid: list[np.ndarray] = []
    source_rows: list[dict[str, object]] = []
    for spec in specs:
        source = load_source_on_training_grid(spec, training, train_times)
        source_valid = np.isfinite(source) & land[None, :, :]
        sgx, sgy, sg_valid = forward_gradient(source, source_valid)
        q_values[spec.name] = gradient_scale(sgx, sgy, sg_valid, spec.name)
        disagreement = gradient_disagreement(
            wgx, wgy, wg_valid, q_values["WGHM"], sgx, sgy, sg_valid, q_values[spec.name]
        )
        disagreements.append(disagreement.astype(np.float32))
        per_source_valid.append(np.isfinite(disagreement))
        source_rows.append(
            {
                "source_name": spec.name,
                "parent_product": spec.parent_product,
                "q_s": q_values[spec.name],
                "valid_gradient_comparisons": int(np.isfinite(disagreement).sum()),
                "time_start": str(pd.Timestamp(train_times[0]).date()),
                "time_end_available": str(pd.Timestamp(train_times[np.flatnonzero(np.any(np.isfinite(source), axis=(1, 2)))[-1]]).date()),
            }
        )
    disagreement_stack = np.stack(disagreements, axis=0)
    beta, d_beta, valid_time_count = combine_beta(disagreement_stack)
    beta = np.where(land, beta, np.nan)
    d_beta = np.where(land, d_beta, np.nan)
    valid_by_source = np.stack(per_source_valid, axis=0)
    per_source_month_count = valid_by_source.sum(axis=1).astype(np.int16)
    source_count = (per_source_month_count > 0).sum(axis=0).astype(np.int8)

    loo_fields: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
    loo_rows: list[dict[str, object]] = []
    from scipy.stats import spearmanr

    for idx, spec in enumerate(specs):
        keep = [i for i in range(len(specs)) if i != idx]
        beta_loo, _, _ = combine_beta(disagreement_stack[keep])
        beta_loo = np.where(land, beta_loo, np.nan)
        valid = np.isfinite(beta) & np.isfinite(beta_loo)
        rho = spearmanr(beta[valid], beta_loo[valid]).statistic if valid.sum() >= 3 else np.nan
        loo_rows.append(
            {
                "left_out_source": spec.name,
                "spearman_rho": rho,
                "mae": float(np.mean(np.abs(beta[valid] - beta_loo[valid]))),
                "n": int(valid.sum()),
            }
        )
        loo_fields[f"beta_without_{spec.name.lower()}"] = (("lat", "lon"), beta_loo.astype(np.float32))

    diagnostics = diagnostic_fields(w_center, j_center, train_times)
    diagnostic_rows = spearman_rows(alpha_coarse, diagnostics)
    alpha_zero_count = int(np.sum(coarse_support & (alpha_coarse == 0)))
    beta_no_evidence_count = int(np.sum(land & (valid_time_count == 0)))
    alpha_mean = area_weighted_mean(alpha, training["lat"].values, land)
    beta_mean = area_weighted_mean(beta, training["lat"].values, land)

    common_attrs = {
        "train_start": str(pd.Timestamp(train_times[0]).date()),
        "train_end": str(pd.Timestamp(train_times[-1]).date()),
        "train_n_months": int(len(train_times)),
        "train_timestamps": json.dumps([str(pd.Timestamp(t).date()) for t in train_times]),
        "canonical_units": "mm EWH",
        "coarse_factor": factor,
        "coarse_operator": str(training.attrs.get("coarse_operator")),
        "gradient_operator": "first_order_forward_difference_matching scripts.train.losses.gradient_xy",
        "generation_script": str(Path(__file__).resolve()),
        "generation_script_sha256": sha256_file(Path(__file__).resolve()),
        "training_zarr": str(training_path),
        "training_zmetadata_sha256": sha256_path(training_path),
    }
    alpha_ds = xr.Dataset(
        {
            "alpha": (("lat", "lon"), alpha.astype(np.float32)),
            "D_alpha": (("lat", "lon"), d_alpha.astype(np.float32)),
            "valid_mask": (("lat", "lon"), land.astype(np.uint8)),
            "coarse_cell_mapping": (("lat", "lon"), coarse_mapping),
            "alpha_coarse": (("coarse_lat", "coarse_lon"), alpha_coarse.astype(np.float32)),
            "D_alpha_coarse": (("coarse_lat", "coarse_lon"), d_alpha_coarse.astype(np.float32)),
            **{name: (("coarse_lat", "coarse_lon"), values.astype(np.float32)) for name, values in diagnostics.items()},
        },
        coords={
            "lat": training["lat"].values,
            "lon": training["lon"].values,
            "coarse_lat": training["coarse_lat"].values,
            "coarse_lon": training["coarse_lon"].values,
        },
        attrs={**common_attrs, "formula_version": FORMULA_ALPHA, "denominator_zero_rule": "alpha=0"},
    )
    beta_ds = xr.Dataset(
        {
            "beta": (("lat", "lon"), beta.astype(np.float32)),
            "D_beta": (("lat", "lon"), d_beta.astype(np.float32)),
            "source_count": (("lat", "lon"), source_count),
            "valid_time_count": (("lat", "lon"), valid_time_count),
            **{
                f"D_{spec.name.lower()}": (("lat", "lon"), nanmedian_no_warning(disagreements[idx], axis=0).astype(np.float32))
                for idx, spec in enumerate(specs)
            },
            **loo_fields,
        },
        coords={"lat": training["lat"].values, "lon": training["lon"].values},
        attrs={
            **common_attrs,
            "formula_version": FORMULA_BETA,
            "source_names": json.dumps([spec.name for spec in specs]),
            "parent_products": json.dumps([spec.parent_product for spec in specs]),
            "q_s": json.dumps(q_values, sort_keys=True),
            "no_evidence_rule": "beta=0",
        },
    )
    coverage_ds = xr.Dataset(
        {
            "source_count": (("lat", "lon"), source_count),
            "valid_time_count": (("lat", "lon"), valid_time_count),
            **{
                f"valid_months_{spec.name.lower()}": (("lat", "lon"), per_source_month_count[idx])
                for idx, spec in enumerate(specs)
            },
        },
        coords={"lat": training["lat"].values, "lon": training["lon"].values},
        attrs={**common_attrs, "source_names": json.dumps([spec.name for spec in specs])},
    )
    alpha_ds.to_netcdf(outputs[0])
    beta_ds.to_netcdf(outputs[1])
    coverage_ds.to_netcdf(outputs[2])
    write_csv(output_dir / "alpha_diagnostics.csv", diagnostic_rows)
    write_csv(output_dir / "beta_source_diagnostics.csv", source_rows)
    write_csv(output_dir / "beta_leave_one_source_out.csv", loo_rows)

    metadata = {
        "train_timestamps": [str(pd.Timestamp(t).date()) for t in train_times],
        "alpha_formula_version": FORMULA_ALPHA,
        "beta_formula_version": FORMULA_BETA,
        "coarse_operator": common_attrs["coarse_operator"],
        "gradient_operator": common_attrs["gradient_operator"],
        "beta_parent_sources": [spec.__dict__ | {"path": str(spec.path) if spec.path else str(training_path)} for spec in specs],
        "q_s": q_values,
        "missing_data_rules": {
            "alpha_zero_denominator": "alpha=0",
            "beta_both_gradients_zero": "d=0",
            "beta_one_gradient_zero": "formula gives d=1",
            "beta_missing_source_time": "exclude from source median",
            "beta_no_source_evidence": "beta=0",
        },
        "alpha_denominator_zero_cells": alpha_zero_count,
        "beta_no_evidence_land_cells": beta_no_evidence_count,
        "source_coverage_statistics": source_rows,
        "fixed_area_weighted_alpha": alpha_mean,
        "fixed_area_weighted_beta": beta_mean,
        "git_commit": "UNAVAILABLE_NO_GIT_REPOSITORY",
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {"numpy": np.__version__, "pandas": pd.__version__, "xarray": xr.__version__},
        "remote_host_alias": "codex-linux",
        "remote_project_path": "/home/user02/grace_remote_train",
        "gpu_information": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition; queried during audit",
        "job_ids": [],
        "output_files": {path.name: str(path) for path in outputs[:3]},
    }
    outputs[3].write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"alpha_mean": alpha_mean, "beta_mean": beta_mean, "outputs": [str(p) for p in outputs]}, indent=2))


if __name__ == "__main__":
    main()
