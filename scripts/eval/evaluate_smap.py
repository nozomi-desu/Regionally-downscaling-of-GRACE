"""Evaluate predicted TWSA fields against overlapping-period SMAP soil-moisture patterns."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import DEFAULT_LAT, DEFAULT_LON, configure_logging, ensure_dir  # noqa: E402
from scripts.eval.common import open_prediction_dataset, resolve_prediction_var_name, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--smap-dir", type=Path, default=PROJECT_ROOT / "data_raw" / "smap")
    parser.add_argument("--monthly-smap", type=Path, help="Optional pre-aggregated monthly SMAP NetCDF with variable smap_surface_sm.")
    parser.add_argument(
        "--training-zarr",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "training" / "grace_downscaling_training_dataset.zarr",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-start", default="2015-03")
    parser.add_argument("--time-end", default="2016-08")
    parser.add_argument("--max-human-activity", type=float, default=0.5)
    parser.add_argument("--max-glacier-fraction", type=float, default=0.05)
    parser.add_argument("--min-valid-months", type=int, default=6)
    parser.add_argument("--prediction-variant", choices=["auto", "raw", "corrected"], default="auto")
    return parser.parse_args()


def month_token_from_name(path: Path) -> str:
    """Extract YYYY-MM token from an SMAP daily filename."""
    match = re.search(r"(\d{4})(\d{2})(\d{2})", path.name)
    if not match:
        raise ValueError(f"Could not parse date from {path.name}")
    return f"{match.group(1)}-{match.group(2)}"


def load_one_smap_field(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one daily SMAP surface-soil-moisture field, averaging AM/PM retrievals."""
    with h5py.File(path, "r") as handle:
        am_group = handle["Soil_Moisture_Retrieval_Data_AM"]
        pm_group = handle["Soil_Moisture_Retrieval_Data_PM"]

        lat = np.asarray(am_group["latitude"], dtype=np.float32)
        lon = np.asarray(am_group["longitude"], dtype=np.float32)

        am_sm = np.asarray(am_group["soil_moisture"], dtype=np.float32)
        am_qf = np.asarray(am_group["retrieval_qual_flag"], dtype=np.uint16)
        pm_sm = np.asarray(pm_group["soil_moisture_dca_pm"], dtype=np.float32)
        pm_qf = np.asarray(pm_group["retrieval_qual_flag_dca_pm"], dtype=np.uint16)

        am_valid = np.isfinite(am_sm) & (am_sm >= 0.0) & (am_sm <= 1.0) & (am_qf == 0)
        pm_valid = np.isfinite(pm_sm) & (pm_sm >= 0.0) & (pm_sm <= 1.0) & (pm_qf == 0)

        stacked = np.stack(
            [
                np.where(am_valid, am_sm, np.nan),
                np.where(pm_valid, pm_sm, np.nan),
            ],
            axis=0,
        )
        field = np.nanmean(stacked, axis=0).astype(np.float32)
    return lat, lon, field


def aggregate_to_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    field: np.ndarray,
    lat_centers: np.ndarray,
    lon_centers: np.ndarray,
) -> np.ndarray:
    """Aggregate one swath-like field to the project 0.5-degree grid by bin averaging."""
    flat_lat = lat.ravel()
    flat_lon = lon.ravel()
    flat_field = field.ravel()
    valid = np.isfinite(flat_lat) & np.isfinite(flat_lon) & np.isfinite(flat_field)
    if not np.any(valid):
        return np.full((lat_centers.size, lon_centers.size), np.nan, dtype=np.float32)

    flat_lat = flat_lat[valid]
    flat_lon = flat_lon[valid]
    flat_field = flat_field[valid]

    lat_idx = np.floor((flat_lat - (lat_centers[0] - 0.25)) / 0.5).astype(int)
    lon_idx = np.floor((flat_lon - (lon_centers[0] - 0.25)) / 0.5).astype(int)
    in_bounds = (
        (lat_idx >= 0)
        & (lat_idx < lat_centers.size)
        & (lon_idx >= 0)
        & (lon_idx < lon_centers.size)
    )
    lat_idx = lat_idx[in_bounds]
    lon_idx = lon_idx[in_bounds]
    flat_field = flat_field[in_bounds]

    sums = np.zeros((lat_centers.size, lon_centers.size), dtype=np.float64)
    counts = np.zeros((lat_centers.size, lon_centers.size), dtype=np.float64)
    np.add.at(sums, (lat_idx, lon_idx), flat_field)
    np.add.at(counts, (lat_idx, lon_idx), 1.0)
    out = sums / np.where(counts > 0, counts, np.nan)
    return out.astype(np.float32)


def build_monthly_smap_dataset(
    smap_dir: Path,
    time_start: str,
    time_end: str,
    lat_centers: np.ndarray,
    lon_centers: np.ndarray,
    logger,
) -> tuple[xr.Dataset, list[str]]:
    """Read daily SMAP files and aggregate them to monthly 0.5-degree means."""
    monthly_sum: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros((lat_centers.size, lon_centers.size), dtype=np.float64)
    )
    monthly_count: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros((lat_centers.size, lon_centers.size), dtype=np.float64)
    )
    skipped_files: list[str] = []

    daily_files = sorted(smap_dir.glob("SMAP_L3_SM_P_*.h5"))
    if not daily_files:
        raise FileNotFoundError(f"No SMAP HDF5 files found in {smap_dir}")

    for daily_path in daily_files:
        month_token = month_token_from_name(daily_path)
        if month_token < time_start[:7] or month_token > time_end[:7]:
            continue
        try:
            lat, lon, field = load_one_smap_field(daily_path)
        except OSError as exc:
            skipped_files.append(f"{daily_path.name}: {exc}")
            logger.warning("Skipping unreadable SMAP file %s: %s", daily_path.name, exc)
            continue
        gridded = aggregate_to_grid(lat, lon, field, lat_centers, lon_centers)
        valid = np.isfinite(gridded)
        monthly_sum[month_token][valid] += gridded[valid]
        monthly_count[month_token][valid] += 1.0

    available_months = sorted(monthly_sum)
    if not available_months:
        raise ValueError("No SMAP files matched the requested time window.")

    monthly_fields = []
    for month_token in available_months:
        field = monthly_sum[month_token] / np.where(monthly_count[month_token] > 0, monthly_count[month_token], np.nan)
        monthly_fields.append(field.astype(np.float32))
    times = np.array([np.datetime64(f"{month}-01") for month in available_months])
    dataset = xr.Dataset(
        data_vars={"smap_surface_sm": (("time", "lat", "lon"), np.stack(monthly_fields, axis=0))},
        coords={"time": times, "lat": lat_centers, "lon": lon_centers},
    )
    return dataset, skipped_files


def spatial_correlation(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> float:
    """Compute spatial correlation over a masked field."""
    valid = mask & np.isfinite(pred) & np.isfinite(obs)
    if valid.sum() < 3:
        return float("nan")
    pred_v = pred[valid].astype(np.float64)
    obs_v = obs[valid].astype(np.float64)
    pred_v = pred_v - pred_v.mean()
    obs_v = obs_v - obs_v.mean()
    pred_std = pred_v.std()
    obs_std = obs_v.std()
    if pred_std == 0 or obs_std == 0:
        return float("nan")
    return float((pred_v * obs_v).mean() / (pred_std * obs_std))


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger = configure_logging()
    ensure_dir(args.output_dir)

    training_ds = xr.open_zarr(args.training_zarr)
    lat = np.asarray(training_ds["lat"].values, dtype=np.float32)
    lon = np.asarray(training_ds["lon"].values, dtype=np.float32)

    pred_ds = open_prediction_dataset(args.predictions)
    pred_ds = pred_ds.sel(time=slice(args.time_start, args.time_end))
    if pred_ds.sizes.get("time", 0) == 0:
        raise ValueError("No prediction months remain after applying the requested time window.")
    prediction_var = resolve_prediction_var_name(pred_ds, args.prediction_variant)

    skipped_files: list[str] = []
    if args.monthly_smap is not None:
        smap_ds = xr.open_dataset(args.monthly_smap)
        if "smap_surface_sm" not in smap_ds.data_vars:
            raise KeyError(f"{args.monthly_smap} does not contain smap_surface_sm.")
        smap_ds = smap_ds.sel(time=slice(args.time_start, args.time_end))
    else:
        smap_ds, skipped_files = build_monthly_smap_dataset(
            smap_dir=args.smap_dir,
            time_start=args.time_start,
            time_end=args.time_end,
            lat_centers=lat,
            lon_centers=lon,
            logger=logger,
        )
    pred_ds, smap_ds = xr.align(pred_ds, smap_ds, join="inner")
    if pred_ds.sizes.get("time", 0) == 0:
        raise ValueError("No overlapping months between predictions and monthly SMAP aggregates.")

    static = training_ds.sel(lat=lat, lon=lon)
    eval_mask = (
        (np.asarray(static["land_mask"].values) > 0.5)
        & (np.asarray(static["glacier_fraction"].values) <= args.max_glacier_fraction)
        & (np.asarray(static["human_activity_index"].values) <= args.max_human_activity)
    )

    pred = np.asarray(pred_ds[prediction_var].values, dtype=np.float32)
    smap = np.asarray(smap_ds["smap_surface_sm"].values, dtype=np.float32)
    valid_cube = np.isfinite(pred) & np.isfinite(smap) & eval_mask[None, :, :]

    pred_anom = pred - np.nanmean(np.where(valid_cube, pred, np.nan), axis=0, keepdims=True)
    smap_anom = smap - np.nanmean(np.where(valid_cube, smap, np.nan), axis=0, keepdims=True)

    pred_std = np.nanstd(np.where(valid_cube, pred_anom, np.nan), axis=0, keepdims=True)
    smap_std = np.nanstd(np.where(valid_cube, smap_anom, np.nan), axis=0, keepdims=True)
    pred_z = pred_anom / np.where(pred_std > 0, pred_std, np.nan)
    smap_z = smap_anom / np.where(smap_std > 0, smap_std, np.nan)

    monthly_rows: list[dict[str, object]] = []
    spatial_corrs = []
    spatial_rmse_z = []
    for idx, time_value in enumerate(pred_ds["time"].values):
        month_mask = valid_cube[idx]
        corr = spatial_correlation(pred_anom[idx], smap_anom[idx], month_mask)
        valid = month_mask & np.isfinite(pred_z[idx]) & np.isfinite(smap_z[idx])
        rmse_z = float(np.sqrt(np.mean((pred_z[idx][valid] - smap_z[idx][valid]) ** 2))) if valid.any() else float("nan")
        monthly_rows.append(
            {
                "time": str(time_value)[:10],
                "spatial_corr": corr,
                "spatial_rmse_zscore": rmse_z,
                "valid_cell_fraction": float(month_mask.mean()),
            }
        )
        spatial_corrs.append(corr)
        spatial_rmse_z.append(rmse_z)

    temporal_corrs = []
    valid_counts = valid_cube.sum(axis=0)
    for lat_idx in range(pred.shape[1]):
        for lon_idx in range(pred.shape[2]):
            if not eval_mask[lat_idx, lon_idx]:
                continue
            if valid_counts[lat_idx, lon_idx] < args.min_valid_months:
                continue
            pred_series = pred_anom[:, lat_idx, lon_idx]
            smap_series = smap_anom[:, lat_idx, lon_idx]
            mask = np.isfinite(pred_series) & np.isfinite(smap_series)
            if mask.sum() < args.min_valid_months:
                continue
            pred_valid = pred_series[mask].astype(np.float64)
            smap_valid = smap_series[mask].astype(np.float64)
            pred_valid = pred_valid - pred_valid.mean()
            smap_valid = smap_valid - smap_valid.mean()
            pred_var = pred_valid.std()
            smap_var = smap_valid.std()
            if pred_var == 0 or smap_var == 0:
                continue
            temporal_corrs.append(float((pred_valid * smap_valid).mean() / (pred_var * smap_var)))

    smap_monthly_path = args.output_dir / f"{args.label}_smap_monthly_overlap.nc"
    smap_ds.to_netcdf(smap_monthly_path)

    summary = {
        "label": args.label,
        "overlap_start": str(pred_ds["time"].values[0])[:10],
        "overlap_end": str(pred_ds["time"].values[-1])[:10],
        "n_months": int(pred_ds.sizes["time"]),
        "evaluation_mask_fraction": float(eval_mask.mean()),
        "monthly_spatial_corr_mean": float(np.nanmean(spatial_corrs)),
        "monthly_spatial_corr_median": float(np.nanmedian(spatial_corrs)),
        "monthly_spatial_rmse_zscore_mean": float(np.nanmean(spatial_rmse_z)),
        "pixel_temporal_corr_mean": float(np.nanmean(temporal_corrs)) if temporal_corrs else float("nan"),
        "pixel_temporal_corr_median": float(np.nanmedian(temporal_corrs)) if temporal_corrs else float("nan"),
        "n_temporal_corr_cells": int(len(temporal_corrs)),
        "prediction_var_used": prediction_var,
        "predictions_path": str(args.predictions),
        "smap_monthly_path": str(smap_monthly_path),
        "n_skipped_smap_files": len(skipped_files),
        "notes": "SMAP overlap-period external consistency check over non-glacier, low-human-activity land cells.",
    }

    monthly_csv = args.output_dir / f"{args.label}_smap_monthly_metrics.csv"
    import csv

    with monthly_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(monthly_rows[0].keys()))
        writer.writeheader()
        writer.writerows(monthly_rows)

    summary_json = args.output_dir / f"{args.label}_smap_summary.json"
    summary_md = args.output_dir / f"{args.label}_smap_summary.md"
    write_json(summary_json, summary)
    summary_md.write_text(
        "\n".join(
            [
                f"# {args.label} SMAP overlap evaluation",
                "",
                f"- `overlap_start`: {summary['overlap_start']}",
                f"- `overlap_end`: {summary['overlap_end']}",
                f"- `n_months`: {summary['n_months']}",
                f"- `evaluation_mask_fraction`: {summary['evaluation_mask_fraction']:.6f}",
                f"- `monthly_spatial_corr_mean`: {summary['monthly_spatial_corr_mean']:.6f}",
                f"- `monthly_spatial_corr_median`: {summary['monthly_spatial_corr_median']:.6f}",
                f"- `monthly_spatial_rmse_zscore_mean`: {summary['monthly_spatial_rmse_zscore_mean']:.6f}",
                f"- `pixel_temporal_corr_mean`: {summary['pixel_temporal_corr_mean']:.6f}",
                f"- `pixel_temporal_corr_median`: {summary['pixel_temporal_corr_median']:.6f}",
                f"- `n_temporal_corr_cells`: {summary['n_temporal_corr_cells']}",
                f"- `prediction_var_used`: {summary['prediction_var_used']}",
                f"- `n_skipped_smap_files`: {summary['n_skipped_smap_files']}",
                "",
                f"- Notes: {summary['notes']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if skipped_files:
        (args.output_dir / f"{args.label}_smap_skipped_files.txt").write_text("\n".join(skipped_files) + "\n", encoding="utf-8")
    logger.info("Wrote SMAP evaluation summary to %s", summary_json)


if __name__ == "__main__":
    main()
