"""Evaluate basin-scale integrated TWSA dynamics over available HydroBASINS regions."""

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
from scripts.eval.common import open_prediction_dataset, resolve_prediction_var_name, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--zarr-path", type=Path, default=PROJECT_ROOT / "data_processed" / "training" / "grace_downscaling_training_dataset.zarr")
    parser.add_argument("--label", default="experiment")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-variant", choices=["auto", "raw", "corrected"], default="auto")
    parser.add_argument("--min-cells", type=int, default=20)
    return parser.parse_args()


def corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute a NaN-safe temporal correlation."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xv = x[mask].astype(np.float64)
    yv = y[mask].astype(np.float64)
    xv -= xv.mean()
    yv -= yv.mean()
    xs = xv.std()
    ys = yv.std()
    if xs == 0 or ys == 0:
        return float("nan")
    return float((xv * yv).mean() / (xs * ys))


def basin_series(cube: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute one basin-mean time series from a masked spatiotemporal cube."""
    valid = np.isfinite(cube) & mask
    counts = valid.sum(axis=(1, 2))
    summed = np.where(valid, cube, 0.0).sum(axis=(1, 2))
    return np.where(counts > 0, summed / np.clip(counts, 1, None), np.nan).astype(np.float32)


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger = configure_logging()
    ensure_dir(args.output_dir)

    pred_ds = open_prediction_dataset(args.predictions)
    train_ds = xr.open_zarr(args.zarr_path)
    pred_ds, train_ds = xr.align(pred_ds, train_ds, join="inner")

    if "hydrobasins_basin_id" not in train_ds:
        raise KeyError(
            "hydrobasins_basin_id is missing from the training Zarr dataset. "
            "Run scripts/preprocess/build_hydrobasins_basin_ids.py and rebuild the training dataset first."
        )

    prediction_var = resolve_prediction_var_name(pred_ds, args.prediction_variant)
    pred_cube = np.asarray(pred_ds[prediction_var].values, dtype=np.float32)
    jpl_cube = np.asarray(pred_ds["target_jplm_twsa"].values, dtype=np.float32)
    wghm_cube = np.asarray(pred_ds["input_wghm_twsa"].values, dtype=np.float32)
    era5_cube = np.asarray(train_ds["input_era5_twsa"].sel(time=pred_ds["time"]).values, dtype=np.float32)
    valid_cube = np.asarray(pred_ds["valid_mask"].values).astype(bool)
    basin_id = np.asarray(train_ds["hydrobasins_basin_id"].values, dtype=np.int32)

    basin_values = np.unique(basin_id[(basin_id > 0) & np.isfinite(basin_id)])
    rows: list[dict[str, object]] = []
    pred_vs_jpl = []
    pred_vs_wghm = []
    pred_vs_era5 = []

    for basin_value in basin_values.tolist():
        basin_mask = (basin_id == basin_value)
        valid_any = valid_cube & basin_mask[None, :, :]
        if int(valid_any.sum(axis=(1, 2)).max()) < args.min_cells:
            continue

        pred_series = basin_series(pred_cube, valid_any)
        jpl_series = basin_series(jpl_cube, valid_any)
        wghm_series = basin_series(wghm_cube, valid_any)
        era5_series = basin_series(era5_cube, valid_any)

        corr_jpl = corr_1d(pred_series, jpl_series)
        corr_wghm = corr_1d(pred_series, wghm_series)
        corr_era5 = corr_1d(pred_series, era5_series)
        rmse_jpl = float(np.sqrt(np.nanmean((pred_series - jpl_series) ** 2)))
        rmse_wghm = float(np.sqrt(np.nanmean((pred_series - wghm_series) ** 2)))
        rmse_era5 = float(np.sqrt(np.nanmean((pred_series - era5_series) ** 2)))
        basin_cells = int(basin_mask.sum())

        rows.append(
            {
                "basin_id": int(basin_value),
                "n_cells": basin_cells,
                "corr_pred_vs_jpl": corr_jpl,
                "corr_pred_vs_wghm": corr_wghm,
                "corr_pred_vs_era5_twsa": corr_era5,
                "rmse_pred_vs_jpl": rmse_jpl,
                "rmse_pred_vs_wghm": rmse_wghm,
                "rmse_pred_vs_era5_twsa": rmse_era5,
            }
        )
        pred_vs_jpl.append(corr_jpl)
        pred_vs_wghm.append(corr_wghm)
        pred_vs_era5.append(corr_era5)

    if not rows:
        raise ValueError("No HydroBASINS regions met the minimum cell threshold.")

    rows.sort(key=lambda item: item["n_cells"], reverse=True)
    csv_path = args.output_dir / f"{args.label}_basin_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "label": args.label,
        "prediction_var_used": prediction_var,
        "n_basins": len(rows),
        "min_cells_threshold": args.min_cells,
        "mean_corr_pred_vs_jpl": float(np.nanmean(pred_vs_jpl)),
        "median_corr_pred_vs_jpl": float(np.nanmedian(pred_vs_jpl)),
        "mean_corr_pred_vs_wghm": float(np.nanmean(pred_vs_wghm)),
        "median_corr_pred_vs_wghm": float(np.nanmedian(pred_vs_wghm)),
        "mean_corr_pred_vs_era5_twsa": float(np.nanmean(pred_vs_era5)),
        "median_corr_pred_vs_era5_twsa": float(np.nanmedian(pred_vs_era5)),
        "notes": "HydroBASINS basin-scale diagnostic over the currently available Asia basin-ID coverage.",
    }
    json_path = args.output_dir / f"{args.label}_basin_summary.json"
    md_path = args.output_dir / f"{args.label}_basin_summary.md"
    write_json(json_path, summary)
    md_path.write_text(
        "\n".join(
            [
                f"# {args.label} basin-integral diagnostic",
                "",
                f"- `n_basins`: {summary['n_basins']}",
                f"- `mean_corr_pred_vs_jpl`: {summary['mean_corr_pred_vs_jpl']:.4f}",
                f"- `mean_corr_pred_vs_wghm`: {summary['mean_corr_pred_vs_wghm']:.4f}",
                f"- `mean_corr_pred_vs_era5_twsa`: {summary['mean_corr_pred_vs_era5_twsa']:.4f}",
                "",
                f"- Basin metrics CSV: `{csv_path}`",
                f"- Summary JSON: `{json_path}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote basin-integral summary to %s", json_path)


if __name__ == "__main__":
    main()
