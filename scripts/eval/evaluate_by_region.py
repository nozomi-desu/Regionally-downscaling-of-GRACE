"""Evaluate model predictions across static heterogeneity strata."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import configure_logging, ensure_dir  # noqa: E402
from scripts.eval.common import gradient_xy_numpy, open_prediction_dataset, resolve_prediction_var_name, spatial_correlation, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "training" / "grace_downscaling_training_dataset.zarr",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-variant", choices=["auto", "raw", "corrected"], default="auto")
    return parser.parse_args()


def mean_ignore_nan(values: list[float]) -> float:
    """Compute a NaN-safe mean."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def build_region_masks(ds: xr.Dataset) -> dict[str, np.ndarray]:
    """Build boolean masks for the main heterogeneity strata."""
    land = np.asarray(ds["land_mask"].values, dtype=np.float32) > 0.5
    aridity = np.asarray(ds["aridity_mask"].values, dtype=np.float32)
    glacier = np.asarray(ds["glacier_fraction"].values, dtype=np.float32)
    human = np.asarray(ds["human_activity_index"].values, dtype=np.float32)

    masks = {
        "global_land": land,
        "arid": land & (aridity >= 0.5),
        "humid": land & (aridity <= 0.1),
        "glacier_influenced": land & (glacier >= 0.01),
        "non_glacier": land & (glacier < 0.001),
        "high_human_activity": land & (human >= 0.05),
        "low_human_activity": land & (human <= 0.001),
        "arid_glacier": land & (aridity >= 0.5) & (glacier >= 0.01),
        "arid_human": land & (aridity >= 0.5) & (human >= 0.05),
    }
    return masks


def summarize_one_region(
    pred_ds: xr.Dataset,
    train_ds: xr.Dataset,
    region_name: str,
    region_mask: np.ndarray,
    prediction_var: str,
) -> dict[str, float | int | str]:
    """Compute region-specific monthly metrics and return a summary row."""
    fine_rmse_values: list[float] = []
    fine_mae_values: list[float] = []
    fine_bias_values: list[float] = []
    fine_corr_values: list[float] = []
    wghm_rmse_values: list[float] = []
    wghm_corr_values: list[float] = []
    gradient_mae_values: list[float] = []
    valid_fraction_values: list[float] = []

    cell_count = int(region_mask.sum())

    for time_idx in range(pred_ds.sizes["time"]):
        prediction = np.asarray(pred_ds[prediction_var].isel(time=time_idx).values, dtype=np.float32)
        target = np.asarray(pred_ds["target_jplm_twsa"].isel(time=time_idx).values, dtype=np.float32)
        valid_mask = np.asarray(train_ds["valid_mask"].isel(time=time_idx).values).astype(bool)
        wghm = np.asarray(pred_ds["input_wghm_twsa"].isel(time=time_idx).values, dtype=np.float32)

        region_valid = region_mask & valid_mask & np.isfinite(prediction) & np.isfinite(target)
        if region_valid.sum() < 9:
            continue

        pred_fill = np.where(np.isfinite(prediction), prediction, 0.0).astype(np.float32)
        fine_diff = pred_fill - target
        fine_rmse_values.append(float(np.sqrt(np.mean((fine_diff[region_valid]) ** 2))))
        fine_mae_values.append(float(np.mean(np.abs(fine_diff[region_valid]))))
        fine_bias_values.append(float(np.mean(fine_diff[region_valid])))
        fine_corr_values.append(spatial_correlation(pred_fill, target, region_valid))

        wghm_diff = pred_fill - wghm
        wghm_rmse_values.append(float(np.sqrt(np.mean((wghm_diff[region_valid]) ** 2))))
        wghm_corr_values.append(spatial_correlation(pred_fill, wghm, region_valid))

        pred_dx, pred_dy = gradient_xy_numpy(pred_fill)
        wghm_dx, wghm_dy = gradient_xy_numpy(wghm)
        grad_mask_x = region_valid[:, 1:] & region_valid[:, :-1]
        grad_mask_y = region_valid[1:, :] & region_valid[:-1, :]
        grad_mae_x = np.mean(np.abs(pred_dx[grad_mask_x] - wghm_dx[grad_mask_x])) if grad_mask_x.any() else np.nan
        grad_mae_y = np.mean(np.abs(pred_dy[grad_mask_y] - wghm_dy[grad_mask_y])) if grad_mask_y.any() else np.nan
        gradient_mae_values.append(float(np.nanmean([grad_mae_x, grad_mae_y])))
        valid_fraction_values.append(float(region_valid.sum() / max(cell_count, 1)))

    return {
        "region": region_name,
        "cell_count": cell_count,
        "n_months": len(fine_rmse_values),
        "fine_rmse_mean": mean_ignore_nan(fine_rmse_values),
        "fine_mae_mean": mean_ignore_nan(fine_mae_values),
        "fine_bias_mean": mean_ignore_nan(fine_bias_values),
        "fine_spatial_corr_mean": mean_ignore_nan(fine_corr_values),
        "wghm_rmse_mean": mean_ignore_nan(wghm_rmse_values),
        "wghm_spatial_corr_mean": mean_ignore_nan(wghm_corr_values),
        "gradient_mae_vs_wghm_mean": mean_ignore_nan(gradient_mae_values),
        "valid_fraction_mean": mean_ignore_nan(valid_fraction_values),
    }


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger = configure_logging()
    pred_ds = open_prediction_dataset(args.predictions)
    prediction_var = resolve_prediction_var_name(pred_ds, args.prediction_variant)
    train_ds = xr.open_zarr(args.zarr_path)
    pred_ds, train_ds = xr.align(pred_ds, train_ds, join="inner")
    ensure_dir(args.output_dir)

    masks = build_region_masks(train_ds)
    rows = [summarize_one_region(pred_ds, train_ds, name, mask, prediction_var=prediction_var) for name, mask in masks.items()]
    summary = {
        "label": args.label,
        "prediction_var_used": prediction_var,
        "predictions_path": str(args.predictions),
        "zarr_path": str(args.zarr_path),
        "regions": rows,
    }

    csv_path = args.output_dir / f"{args.label}_region_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.output_dir / f"{args.label}_region_summary.json"
    md_path = args.output_dir / f"{args.label}_region_summary.md"
    write_json(json_path, summary)

    lines = [f"# {args.label} regional heterogeneity summary", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['region']}",
                "",
                f"- `cell_count`: {row['cell_count']}",
                f"- `n_months`: {row['n_months']}",
                f"- `fine_rmse_mean`: {row['fine_rmse_mean']:.6f}",
                f"- `fine_spatial_corr_mean`: {row['fine_spatial_corr_mean']:.6f}",
                f"- `wghm_rmse_mean`: {row['wghm_rmse_mean']:.6f}",
                f"- `wghm_spatial_corr_mean`: {row['wghm_spatial_corr_mean']:.6f}",
                f"- `gradient_mae_vs_wghm_mean`: {row['gradient_mae_vs_wghm_mean']:.6f}",
                f"- `valid_fraction_mean`: {row['valid_fraction_mean']:.6f}",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote regional heterogeneity summary to %s", json_path)


if __name__ == "__main__":
    main()
