"""Variable-matched validation of model-derived GWSA against independent USGS wells.

The evaluated groundwater-storage anomaly is

    GWSA_candidate = TWSA_candidate - (TWSA_WGHM - GWSA_WGHM)

so every TWSA candidate uses the same WGHM non-groundwater subtraction.  The
well observations remain independent, but the component separation is model
dependent and must be described as such in the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import configure_logging, ensure_dir, normalize_monthly_time_values  # noqa: E402
from scripts.eval.common import open_prediction_dataset, resolve_prediction_var_name, write_json  # noqa: E402


BASELINE_LABELS = ("candidate", "jpl", "wghm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--wghm-groundwater",
        type=Path,
        default=PROJECT_ROOT
        / "data_raw"
        / "wghm"
        / "watergap22e_gswp3-era5_groundwstor_histsoc_monthly_1901_2022.nc",
    )
    parser.add_argument(
        "--regions-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "groundwater_validation_regions_usgs.json",
    )
    parser.add_argument(
        "--groundwater-root",
        type=Path,
        default=PROJECT_ROOT / "data_raw" / "groundwater" / "usgs_ogc",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-variant", choices=["auto", "raw", "corrected"], default="auto")
    parser.add_argument("--baseline-start", default="2004-01-01")
    parser.add_argument("--baseline-end", default="2009-12-31")
    parser.add_argument("--min-overlap-months", type=int, default=12)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def zscore(values: pd.Series) -> pd.Series:
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (values - float(values.mean())) / std


def pearson(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 3:
        return float("nan")
    xv = x.loc[valid].to_numpy(dtype=np.float64)
    yv = y.loc[valid].to_numpy(dtype=np.float64)
    if np.nanstd(xv) == 0.0 or np.nanstd(yv) == 0.0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def first_difference_corr(x: pd.Series, y: pd.Series) -> float:
    merged = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna().sort_index()
    if len(merged) < 4:
        return float("nan")
    monthly_gap = merged.index.to_period("M").astype(int).to_series(index=merged.index).diff()
    diff = merged.diff()
    valid = monthly_gap.eq(1) & diff["x"].notna() & diff["y"].notna()
    return pearson(diff.loc[valid, "x"], diff.loc[valid, "y"])


def linear_slope(values: pd.Series) -> float:
    values = values.dropna().sort_index()
    if len(values) < 3:
        return float("nan")
    month = (values.index.year - values.index.year.min()) * 12 + values.index.month
    return float(np.polyfit(month.to_numpy(dtype=np.float64), values.to_numpy(dtype=np.float64), 1)[0] * 12.0)


def bootstrap_cell_median(
    cell_values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(cell_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(np.nanmedian(values))
    if values.size == 1 or replicates <= 0:
        return estimate, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(replicates, values.size), replace=True)
    medians = np.nanmedian(sampled, axis=1)
    lower, upper = np.nanpercentile(medians, [2.5, 97.5])
    return estimate, float(lower), float(upper)


def groundwater_file_path(root: Path, region_id: str) -> Path | None:
    candidates = sorted(root.glob(f"{region_id}_usgs_field_measurements_*.geojson"))
    return candidates[0] if candidates else None


def load_well_monthly(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates", [np.nan, np.nan])
        try:
            raw_value = float(props.get("value"))
            lon = float(coords[0])
            lat = float(coords[1])
        except (TypeError, ValueError, IndexError):
            continue
        timestamp = pd.to_datetime(props.get("time"), utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue
        unit = str(props.get("unit_of_measure", "")).strip().lower()
        value_m = raw_value * 0.3048 if unit == "ft" else raw_value
        rows.append(
            {
                "site_id": str(props.get("monitoring_location_id", "")),
                "time": timestamp.tz_convert(None).to_period("M").to_timestamp(),
                "well_storage_proxy_m": -value_m,
                "lon": lon,
                "lat": lat,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["site_id", "time", "well_storage_proxy_m", "lon", "lat"])
    frame = pd.DataFrame(rows)
    frame = (
        frame.groupby(["site_id", "time"], as_index=False)
        .agg(well_storage_proxy_m=("well_storage_proxy_m", "mean"), lon=("lon", "median"), lat=("lat", "median"))
        .sort_values(["site_id", "time"])
    )
    return frame


def open_wghm_groundwater(path: Path, baseline_start: str, baseline_end: str) -> xr.DataArray:
    # Decode the WaterGAP 360-day calendar first; converting its raw numeric
    # month offsets through pandas would collapse them near the Unix epoch.
    ds = xr.open_dataset(path, decode_times=True)
    if "groundwstor" not in ds:
        raise KeyError(f"groundwstor not found in {path}")
    da = ds["groundwstor"].assign_coords(time=("time", normalize_monthly_time_values(ds["time"].values)))
    if len(np.unique(da["time"].values)) != da.sizes["time"]:
        da = da.groupby("time").mean()
    baseline_slice = da.sel(time=slice(baseline_start, baseline_end))
    if baseline_slice.sizes.get("time", 0) == 0:
        raise ValueError("WGHM groundwater baseline is empty.")
    baseline = baseline_slice.mean("time")
    return (da - baseline).sortby("lat").rename("wghm_gwsa")


def extract_candidate_series(
    pred_ds: xr.Dataset,
    prediction_var: str,
    wghm_gwsa: xr.DataArray,
    sites: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    site_ids = sites["site_id"].astype(str).tolist()
    site_lat = xr.DataArray(sites["lat"].to_numpy(dtype=np.float64), dims="site", coords={"site": site_ids})
    site_lon = xr.DataArray(sites["lon"].to_numpy(dtype=np.float64), dims="site", coords={"site": site_ids})

    pred_points = pred_ds[[prediction_var, "target_jplm_twsa", "input_wghm_twsa"]].sel(
        lat=site_lat,
        lon=site_lon,
        method="nearest",
    )
    pred_times = pd.to_datetime(pred_points["time"].values).to_period("M").to_timestamp()
    gw_points = wghm_gwsa.sel(time=pred_points["time"], lat=site_lat, lon=site_lon, method="nearest")

    wghm_twsa = np.asarray(pred_points["input_wghm_twsa"].values, dtype=np.float64)
    gwsa_wghm = np.asarray(gw_points.values, dtype=np.float64)
    non_groundwater = wghm_twsa - gwsa_wghm
    arrays = {
        "candidate": np.asarray(pred_points[prediction_var].values, dtype=np.float64) - non_groundwater,
        "jpl": np.asarray(pred_points["target_jplm_twsa"].values, dtype=np.float64) - non_groundwater,
        "wghm": gwsa_wghm,
    }
    return {
        label: pd.DataFrame(values, index=pred_times, columns=site_ids)
        for label, values in arrays.items()
    }


def evaluate_region(
    region_id: str,
    display_name: str,
    wells: pd.DataFrame,
    model_series: dict[str, pd.DataFrame],
    *,
    min_overlap_months: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    site_meta = wells.groupby("site_id", as_index=False).agg(lat=("lat", "median"), lon=("lon", "median"))
    for site in site_meta.itertuples(index=False):
        observed = wells.loc[wells["site_id"] == site.site_id].set_index("time")["well_storage_proxy_m"].sort_index()
        observed = observed.groupby(level=0).mean()
        cell_lat = float(np.round((float(site.lat) + 89.75) / 0.5) * 0.5 - 89.75)
        cell_lon = float(np.round((float(site.lon) + 179.75) / 0.5) * 0.5 - 179.75)
        for source in BASELINE_LABELS:
            if str(site.site_id) not in model_series[source]:
                continue
            modeled = model_series[source][str(site.site_id)].dropna()
            merged = pd.concat([modeled.rename("model"), observed.rename("well")], axis=1, join="inner").dropna()
            if len(merged) < min_overlap_months:
                continue
            model_z = zscore(merged["model"])
            well_z = zscore(merged["well"])
            corr = pearson(model_z, well_z)
            diff_corr = first_difference_corr(model_z, well_z)
            rows.append(
                {
                    "region_id": region_id,
                    "display_name": display_name,
                    "site_id": str(site.site_id),
                    "cell_id": f"{cell_lat:.2f}_{cell_lon:.2f}",
                    "source": source,
                    "n_overlap_months": int(len(merged)),
                    "pearson_r": corr,
                    "first_difference_r": diff_corr,
                    "model_z_trend_per_year": linear_slope(model_z),
                    "well_z_trend_per_year": linear_slope(well_z),
                    "trend_sign_match": int(np.sign(linear_slope(model_z)) == np.sign(linear_slope(well_z))),
                }
            )
    return rows


def summarize_metrics(rows: pd.DataFrame, replicates: int, seed: int) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    grouping = [("ALL", rows)] + list(rows.groupby("region_id", sort=True))
    for group_id, group in grouping:
        for source, source_df in group.groupby("source", sort=True):
            cell_df = source_df.groupby("cell_id", as_index=False).agg(
                pearson_r=("pearson_r", "median"),
                first_difference_r=("first_difference_r", "median"),
                trend_sign_match=("trend_sign_match", "mean"),
            )
            r_med, r_lo, r_hi = bootstrap_cell_median(
                cell_df["pearson_r"].to_numpy(), replicates=replicates, seed=seed
            )
            d_med, d_lo, d_hi = bootstrap_cell_median(
                cell_df["first_difference_r"].to_numpy(), replicates=replicates, seed=seed + 1
            )
            summaries.append(
                {
                    "region_id": group_id,
                    "source": source,
                    "n_wells": int(source_df["site_id"].nunique()),
                    "n_independent_cells": int(cell_df["cell_id"].nunique()),
                    "median_pearson_r": r_med,
                    "pearson_ci95_low": r_lo,
                    "pearson_ci95_high": r_hi,
                    "median_first_difference_r": d_med,
                    "first_difference_ci95_low": d_lo,
                    "first_difference_ci95_high": d_hi,
                    "mean_trend_sign_match": float(cell_df["trend_sign_match"].mean()),
                }
            )
    return summaries


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    ensure_dir(args.output_dir)

    pred_ds = open_prediction_dataset(args.predictions).sortby("lat")
    prediction_var = resolve_prediction_var_name(pred_ds, args.prediction_variant)
    required = {prediction_var, "target_jplm_twsa", "input_wghm_twsa"}
    missing = required.difference(pred_ds.data_vars)
    if missing:
        raise KeyError(f"Prediction dataset is missing required variables: {sorted(missing)}")
    wghm_gwsa = open_wghm_groundwater(args.wghm_groundwater, args.baseline_start, args.baseline_end)
    regions = json.loads(args.regions_config.read_text(encoding="utf-8"))["regions"]

    all_rows: list[dict[str, object]] = []
    for region in regions:
        path = groundwater_file_path(args.groundwater_root, str(region["region_id"]))
        if path is None:
            logger.warning("Missing USGS file for %s", region["region_id"])
            continue
        wells = load_well_monthly(path)
        if wells.empty:
            continue
        sites = wells.groupby("site_id", as_index=False).agg(lat=("lat", "median"), lon=("lon", "median"))
        model_series = extract_candidate_series(pred_ds, prediction_var, wghm_gwsa, sites)
        all_rows.extend(
            evaluate_region(
                str(region["region_id"]),
                str(region["display_name"]),
                wells,
                model_series,
                min_overlap_months=args.min_overlap_months,
            )
        )

    if not all_rows:
        raise ValueError("No wells passed the overlap requirements.")
    rows_df = pd.DataFrame(all_rows)
    summary_rows = summarize_metrics(rows_df, args.bootstrap_replicates, args.seed)

    detail_path = args.output_dir / f"{args.label}_well_metrics.csv"
    summary_path = args.output_dir / f"{args.label}_summary.csv"
    rows_df.to_csv(detail_path, index=False, encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8")

    all_summary = [row for row in summary_rows if row["region_id"] == "ALL"]
    payload = {
        "label": args.label,
        "predictions": str(args.predictions),
        "prediction_var_used": prediction_var,
        "wghm_groundwater": str(args.wghm_groundwater),
        "n_regions": int(rows_df["region_id"].nunique()),
        "n_wells": int(rows_df["site_id"].nunique()),
        "n_independent_cells": int(rows_df["cell_id"].nunique()),
        "all_region_summary": all_summary,
        "independence_note": "USGS wells are independent; GWSA component separation uses WGHM non-groundwater storage for every TWSA candidate.",
        "interpretation_limit": "Unknown specific yield prevents amplitude validation; report standardized temporal association and trend direction only.",
    }
    write_json(args.output_dir / f"{args.label}_summary.json", payload)
    logger.info("Wrote %s and %s", detail_path, summary_path)


if __name__ == "__main__":
    main()
