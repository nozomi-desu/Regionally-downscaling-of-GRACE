"""Evaluate model or baseline predictions against internal GRACE/WGHM targets."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import configure_logging, ensure_dir  # noqa: E402
from scripts.eval.common import (  # noqa: E402
    aggregate_coarse_mean_numpy,
    gradient_xy_numpy,
    open_prediction_dataset,
    resolve_prediction_var_name,
    spatial_correlation,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data_processed"
            / "training_rescue_v1"
            / "grace_downscaling_training_dataset_mm.zarr"
        ),
    )
    parser.add_argument("--label", default="experiment")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prediction-variant", choices=["auto", "raw", "corrected", "projected"], default="auto")
    return parser.parse_args()


def mean_ignore_nan(values: list[float]) -> float:
    """Compute mean while ignoring NaNs."""
    arr = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def slope_1d(values: np.ndarray) -> float:
    """Return a linear slope over the valid part of a 1D series."""
    mask = np.isfinite(values)
    if mask.sum() < 3:
        return float("nan")
    x = np.arange(mask.sum(), dtype=np.float64)
    return float(np.polyfit(x, values[mask].astype(np.float64), 1)[0])


def seasonal_amplitude(cube: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Estimate per-cell seasonal amplitude from monthly climatology."""
    climatology = []
    for month in range(1, 13):
        month_mask = months == month
        if not np.any(month_mask):
            continue
        month_values = cube[month_mask]
        if not np.isfinite(month_values).any():
            continue
        finite = np.isfinite(month_values)
        counts = finite.sum(axis=0)
        summed = np.where(finite, month_values, 0.0).sum(axis=0)
        climatology.append(np.where(counts > 0, summed / counts, np.nan))
    if not climatology:
        return np.full(cube.shape[1:], np.nan, dtype=np.float32)
    stacked = np.stack(climatology, axis=0)
    valid = np.isfinite(stacked)
    max_values = np.where(valid, stacked, -np.inf).max(axis=0)
    min_values = np.where(valid, stacked, np.inf).min(axis=0)
    amplitude = 0.5 * (max_values - min_values)
    amplitude = np.where(valid.any(axis=0), amplitude, np.nan)
    return amplitude.astype(np.float32)


def safe_nanmean(values: np.ndarray) -> float:
    """Compute a NaN-safe mean that returns NaN for fully missing arrays."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmean(arr))


def summarize_variant(
    pred_ds: xr.Dataset,
    train_ds: xr.Dataset,
    prediction_var: str,
    coarse_factor: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Compute monthly and aggregate metrics for one prediction variant."""
    monthly_rows: list[dict[str, object]] = []
    fine_rmse_values: list[float] = []
    fine_mae_values: list[float] = []
    fine_bias_values: list[float] = []
    fine_corr_values: list[float] = []
    coarse_rmse_values: list[float] = []
    coarse_mae_values: list[float] = []
    coarse_bias_values: list[float] = []
    coarse_corr_values: list[float] = []
    wghm_rmse_values: list[float] = []
    wghm_corr_values: list[float] = []
    gradient_mae_values: list[float] = []
    mass_closure_values: list[float] = []

    prediction_cube = np.asarray(pred_ds[prediction_var].values, dtype=np.float64)
    target_cube = np.asarray(pred_ds["target_jplm_twsa"].values, dtype=np.float64)
    wghm_cube = np.asarray(pred_ds["input_wghm_twsa"].values, dtype=np.float64)
    valid_cube = np.asarray(pred_ds["valid_mask"].values).astype(bool)
    coarse_target_cube = np.asarray(train_ds["target_jplm_twsa_coarse"].sel(time=pred_ds["time"]).values, dtype=np.float64)
    if "cell_area_weight" in train_ds.data_vars:
        area_weight = np.asarray(train_ds["cell_area_weight"].values, dtype=np.float64)
    else:
        lat_weight = np.cos(np.deg2rad(np.asarray(train_ds["lat"].values, dtype=np.float64)))
        area_weight = np.broadcast_to(lat_weight[:, None], target_cube.shape[-2:]).copy()

    coarse_pred_series = []
    coarse_target_series = []
    coarse_valid_series = []

    for time_idx, time_value in enumerate(pred_ds["time"].values):
        prediction = prediction_cube[time_idx]
        target = target_cube[time_idx]
        valid_mask = valid_cube[time_idx]
        wghm = wghm_cube[time_idx]

        pred_fill = np.where(np.isfinite(prediction), prediction, 0.0).astype(np.float64)
        fine_diff = pred_fill - target
        valid_pred = valid_mask & np.isfinite(prediction) & np.isfinite(target)
        if valid_pred.sum() == 0:
            continue

        fine_rmse = float(np.sqrt(np.mean((fine_diff[valid_pred]) ** 2)))
        fine_mae = float(np.mean(np.abs(fine_diff[valid_pred])))
        fine_bias = float(np.mean(fine_diff[valid_pred]))
        fine_corr = spatial_correlation(pred_fill, target, valid_pred)

        coarse_pred, coarse_mask = aggregate_coarse_mean_numpy(
            pred_fill,
            valid_mask.astype(np.float32) * area_weight,
            coarse_factor,
        )
        coarse_target = coarse_target_cube[time_idx]
        coarse_valid = coarse_mask & np.isfinite(coarse_target)
        coarse_diff = coarse_pred - coarse_target
        coarse_rmse = float(np.sqrt(np.mean((coarse_diff[coarse_valid]) ** 2)))
        coarse_mae = float(np.mean(np.abs(coarse_diff[coarse_valid])))
        coarse_bias = float(np.mean(coarse_diff[coarse_valid]))
        coarse_corr = spatial_correlation(coarse_pred, coarse_target, coarse_valid)
        mass_closure_error = float(np.mean(np.abs(coarse_diff[coarse_valid])))

        wghm_diff = pred_fill - wghm
        wghm_rmse = float(np.sqrt(np.mean((wghm_diff[valid_pred]) ** 2)))
        wghm_corr = spatial_correlation(pred_fill, wghm, valid_pred)

        pred_dx, pred_dy = gradient_xy_numpy(pred_fill)
        wghm_dx, wghm_dy = gradient_xy_numpy(wghm)
        grad_mask_x = valid_pred[:, 1:] & valid_pred[:, :-1]
        grad_mask_y = valid_pred[1:, :] & valid_pred[:-1, :]
        grad_mae_x = np.mean(np.abs(pred_dx[grad_mask_x] - wghm_dx[grad_mask_x])) if grad_mask_x.any() else np.nan
        grad_mae_y = np.mean(np.abs(pred_dy[grad_mask_y] - wghm_dy[grad_mask_y])) if grad_mask_y.any() else np.nan
        gradient_mae = float(np.nanmean([grad_mae_x, grad_mae_y]))

        monthly_rows.append(
            {
                "time": str(time_value)[:10],
                "jpl_grid_consistency_rmse": fine_rmse,
                "jpl_grid_consistency_mae": fine_mae,
                "jpl_grid_consistency_bias": fine_bias,
                "jpl_grid_consistency_spatial_corr": fine_corr,
                "coarse_rmse": coarse_rmse,
                "coarse_mae": coarse_mae,
                "coarse_bias": coarse_bias,
                "coarse_spatial_corr": coarse_corr,
                "mass_closure_error": mass_closure_error,
                "wghm_rmse": wghm_rmse,
                "wghm_spatial_corr": wghm_corr,
                "gradient_mae_vs_wghm": gradient_mae,
                "valid_cell_fraction": float(valid_pred.mean()),
            }
        )
        fine_rmse_values.append(fine_rmse)
        fine_mae_values.append(fine_mae)
        fine_bias_values.append(fine_bias)
        fine_corr_values.append(fine_corr)
        coarse_rmse_values.append(coarse_rmse)
        coarse_mae_values.append(coarse_mae)
        coarse_bias_values.append(coarse_bias)
        coarse_corr_values.append(coarse_corr)
        wghm_rmse_values.append(wghm_rmse)
        wghm_corr_values.append(wghm_corr)
        gradient_mae_values.append(gradient_mae)
        mass_closure_values.append(mass_closure_error)
        coarse_pred_series.append(np.where(coarse_valid, coarse_pred, np.nan))
        coarse_target_series.append(np.where(coarse_valid, coarse_target, np.nan))
        coarse_valid_series.append(coarse_valid)

    coarse_pred_cube = np.stack(coarse_pred_series, axis=0).astype(np.float32)
    coarse_target_cube_eval = np.stack(coarse_target_series, axis=0).astype(np.float32)
    coarse_valid_cube = np.stack(coarse_valid_series, axis=0).astype(bool)
    months = pd.to_datetime(pred_ds["time"].values).month.to_numpy()

    trend_pred = np.apply_along_axis(slope_1d, 0, coarse_pred_cube)
    trend_target = np.apply_along_axis(slope_1d, 0, coarse_target_cube_eval)
    trend_valid = np.isfinite(trend_pred) & np.isfinite(trend_target) & coarse_valid_cube.any(axis=0)
    seasonal_pred = seasonal_amplitude(coarse_pred_cube, months)
    seasonal_target = seasonal_amplitude(coarse_target_cube_eval, months)
    seasonal_valid = np.isfinite(seasonal_pred) & np.isfinite(seasonal_target) & coarse_valid_cube.any(axis=0)

    summary = {
        "n_months": len(monthly_rows),
        "jpl_grid_consistency_rmse_mean": mean_ignore_nan(fine_rmse_values),
        "jpl_grid_consistency_mae_mean": mean_ignore_nan(fine_mae_values),
        "jpl_grid_consistency_bias_mean": mean_ignore_nan(fine_bias_values),
        "jpl_grid_consistency_spatial_corr_mean": mean_ignore_nan(fine_corr_values),
        "coarse_rmse_mean": mean_ignore_nan(coarse_rmse_values),
        "coarse_mae_mean": mean_ignore_nan(coarse_mae_values),
        "coarse_bias_mean": mean_ignore_nan(coarse_bias_values),
        "coarse_spatial_corr_mean": mean_ignore_nan(coarse_corr_values),
        "mass_closure_error_mean": mean_ignore_nan(mass_closure_values),
        "wghm_rmse_mean": mean_ignore_nan(wghm_rmse_values),
        "wghm_spatial_corr_mean": mean_ignore_nan(wghm_corr_values),
        "gradient_mae_vs_wghm_mean": mean_ignore_nan(gradient_mae_values),
        "trend_diff_mean": float(np.nanmean(np.abs(trend_pred[trend_valid] - trend_target[trend_valid]))) if np.any(trend_valid) else float("nan"),
        "seasonal_amp_diff_mean": float(np.nanmean(np.abs(seasonal_pred[seasonal_valid] - seasonal_target[seasonal_valid]))) if np.any(seasonal_valid) else float("nan"),
    }
    return summary, monthly_rows


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger = configure_logging()
    pred_ds = open_prediction_dataset(args.predictions)
    train_ds = xr.open_zarr(args.zarr_path)
    output_dir = args.output_dir or args.predictions.with_suffix("")
    output_dir = output_dir if output_dir.is_dir() else Path(str(output_dir))
    ensure_dir(output_dir)

    pred_ds, train_ds = xr.align(pred_ds, train_ds, join="inner")
    coarse_factor = int(train_ds.attrs.get("coarse_factor", 6))

    default_prediction_var = resolve_prediction_var_name(pred_ds, args.prediction_variant)
    summary, monthly_rows = summarize_variant(pred_ds, train_ds, default_prediction_var, coarse_factor)
    summary.update(
        {
            "label": args.label,
            "prediction_var_used": default_prediction_var,
            "predictions_path": str(args.predictions),
            "jpl_grid_consistency_interpretation": (
                "Consistency with gridded JPL only; not independent 0.5-degree accuracy."
            ),
        }
    )

    if "predicted_twsa_raw" in pred_ds.data_vars:
        raw_summary, _ = summarize_variant(pred_ds, train_ds, "predicted_twsa_raw", coarse_factor)
        summary["jpl_grid_consistency_rmse_raw"] = raw_summary["jpl_grid_consistency_rmse_mean"]
        summary["coarse_rmse_raw"] = raw_summary["coarse_rmse_mean"]
        summary["mass_closure_error_before"] = raw_summary["mass_closure_error_mean"]
    if "predicted_twsa_corrected" in pred_ds.data_vars:
        corrected_summary, _ = summarize_variant(pred_ds, train_ds, "predicted_twsa_corrected", coarse_factor)
        summary["jpl_grid_consistency_rmse_corrected"] = corrected_summary["jpl_grid_consistency_rmse_mean"]
        summary["coarse_rmse_corrected"] = corrected_summary["coarse_rmse_mean"]
        summary["mass_closure_error_after"] = corrected_summary["mass_closure_error_mean"]

    if "mass_closure_error_before" in pred_ds.data_vars:
        summary["mass_closure_error_before_dataset"] = safe_nanmean(pred_ds["mass_closure_error_before"].values)
    if "mass_closure_error_after" in pred_ds.data_vars:
        summary["mass_closure_error_after_dataset"] = safe_nanmean(pred_ds["mass_closure_error_after"].values)

    monthly_csv = output_dir / f"{args.label}_monthly_metrics.csv"
    with monthly_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(monthly_rows[0].keys()))
        writer.writeheader()
        writer.writerows(monthly_rows)

    summary_json = output_dir / f"{args.label}_summary.json"
    summary_md = output_dir / f"{args.label}_summary.md"
    write_json(summary_json, summary)
    summary_lines = [
        f"# {args.label} internal evaluation summary",
        "",
        f"- `n_months`: {summary['n_months']}",
        f"- `prediction_var_used`: {summary['prediction_var_used']}",
        "- Interpretation: gridded-JPL consistency only; not independent 0.5-degree accuracy.",
        f"- `jpl_grid_consistency_rmse_mean`: {summary['jpl_grid_consistency_rmse_mean']:.6f}",
        f"- `jpl_grid_consistency_spatial_corr_mean`: {summary['jpl_grid_consistency_spatial_corr_mean']:.6f}",
        f"- `coarse_rmse_mean`: {summary['coarse_rmse_mean']:.6f}",
        f"- `coarse_spatial_corr_mean`: {summary['coarse_spatial_corr_mean']:.6f}",
        f"- `mass_closure_error_mean`: {summary['mass_closure_error_mean']:.6f}",
        f"- `trend_diff_mean`: {summary['trend_diff_mean']:.6f}",
        f"- `seasonal_amp_diff_mean`: {summary['seasonal_amp_diff_mean']:.6f}",
        f"- `wghm_rmse_mean`: {summary['wghm_rmse_mean']:.6f}",
        f"- `wghm_spatial_corr_mean`: {summary['wghm_spatial_corr_mean']:.6f}",
        f"- `gradient_mae_vs_wghm_mean`: {summary['gradient_mae_vs_wghm_mean']:.6f}",
    ]
    if "jpl_grid_consistency_rmse_raw" in summary:
        summary_lines.append(
            f"- `jpl_grid_consistency_rmse_raw`: {summary['jpl_grid_consistency_rmse_raw']:.6f}"
        )
    if "jpl_grid_consistency_rmse_corrected" in summary:
        summary_lines.append(
            f"- `jpl_grid_consistency_rmse_corrected`: {summary['jpl_grid_consistency_rmse_corrected']:.6f}"
        )
    if "coarse_rmse_raw" in summary:
        summary_lines.append(f"- `coarse_rmse_raw`: {summary['coarse_rmse_raw']:.6f}")
    if "coarse_rmse_corrected" in summary:
        summary_lines.append(f"- `coarse_rmse_corrected`: {summary['coarse_rmse_corrected']:.6f}")
    if "mass_closure_error_before" in summary:
        summary_lines.append(f"- `mass_closure_error_before`: {summary['mass_closure_error_before']:.6f}")
    if "mass_closure_error_after" in summary:
        summary_lines.append(f"- `mass_closure_error_after`: {summary['mass_closure_error_after']:.6f}")
    summary_lines.extend(
        [
            "",
            f"- Monthly metrics CSV: `{monthly_csv}`",
            f"- Summary JSON: `{summary_json}`",
        ]
    )
    summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    logger.info("Wrote evaluation summary to %s", summary_json)


if __name__ == "__main__":
    main()
