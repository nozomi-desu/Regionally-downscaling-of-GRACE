"""Assemble a training-ready dataset for GRACE adaptive downscaling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (  # noqa: E402
    configure_logging,
    guess_data_var,
    infer_grid_resolution,
    normalize_monthly_time_values,
    open_dataset_any,
    write_dataset,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jpl",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "anomalies_2004_2009" / "jplm_twsa.nc",
    )
    parser.add_argument(
        "--wghm",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "anomalies_2004_2009" / "wghm_twsa.nc",
    )
    parser.add_argument(
        "--era5-twsa",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "anomalies_2004_2009" / "era5_twsa.nc",
    )
    parser.add_argument(
        "--era5-cwsc",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "anomalies_2004_2009" / "era5_cwsc.nc",
    )
    parser.add_argument(
        "--alpha",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "adaptive_weights_train_only" / "alpha_value_weight.nc",
    )
    parser.add_argument(
        "--beta",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "adaptive_weights_train_only" / "beta_gradient_weight.nc",
    )
    parser.add_argument(
        "--aridity",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "adaptive_weights" / "aridity_mask.nc",
    )
    parser.add_argument(
        "--human",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "adaptive_weights" / "human_activity_index.nc",
    )
    parser.add_argument(
        "--glacier",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "adaptive_weights" / "glacier_fraction.nc",
    )
    parser.add_argument(
        "--hydrobasins",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "grid_05deg" / "hydrobasins_mask.nc",
    )
    parser.add_argument(
        "--hydrobasins-id",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "grid_05deg" / "hydrobasins_level06_pfaf_id.nc",
    )
    parser.add_argument(
        "--land-mask",
        type=Path,
        default=PROJECT_ROOT / "data_raw" / "jpl_mascon" / "GRCTellus.JPL.200204_202603.GLO.RL06.3M.MSCNv04CRI.nc",
    )
    parser.add_argument("--start", default="2002-04-01")
    parser.add_argument("--end", default="2022-12-01")
    parser.add_argument("--train-end", default="2016-12-01")
    parser.add_argument("--val-end", default="2019-12-01")
    parser.add_argument("--coarse-factor", type=int, default=6)
    parser.add_argument(
        "--wghm-abs-max-mm",
        type=float,
        default=10000.0,
        help="Predeclared physical sanity cap; WGHM cells beyond this absolute anomaly are masked.",
    )
    parser.add_argument(
        "--allow-legacy-weights",
        action="store_true",
        help="Permit alpha/beta without train-only provenance. Never use for formal experiments.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "training_rescue_v1" / "grace_downscaling_training_dataset_mm.zarr",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "training_rescue_v1" / "grace_downscaling_training_dataset_summary.md",
    )
    parser.add_argument(
        "--missing-report",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "training_rescue_v1" / "grace_downscaling_missing_supervision_months.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def open_monthly_dataarray(path: Path, aliases: list[str]) -> xr.DataArray:
    """Open one monthly field and normalize time stamps to month starts."""
    ds = open_dataset_any(path)
    variable = guess_data_var(ds, aliases=aliases)
    da = ds[variable]
    if "time" not in da.dims:
        raise ValueError(f"{path} does not contain a time dimension.")
    monthly_time = normalize_monthly_time_values(da["time"].values)
    da = da.assign_coords(time=("time", monthly_time))
    if len(np.unique(monthly_time)) != len(monthly_time):
        da = da.groupby("time").mean()
    return da.astype("float32")


def open_static_dataarray(path: Path, aliases: list[str], template: xr.DataArray | None = None) -> xr.DataArray:
    """Open one static 2D field and align it to the template grid if needed."""
    ds = open_dataset_any(path)
    variable = guess_data_var(ds, aliases=aliases)
    da = ds[variable]
    if "time" in da.dims:
        da = da.isel(time=0, drop=True)
    if template is not None and ("lat" in da.dims and "lon" in da.dims):
        da = da.interp(lat=template["lat"], lon=template["lon"], method="nearest")
    return da.astype("float32")


def canonical_unit_name(units: str | None) -> str:
    """Normalize supported water-storage unit spellings."""
    compact = (units or "").strip().lower().replace("²", "2")
    compact = compact.replace("^", "").replace(" ", "").replace("_", "")
    aliases = {
        "mm": "mm",
        "millimeter": "mm",
        "millimeters": "mm",
        "mmequivalentwaterheight": "mm",
        "cm": "cm",
        "centimeter": "cm",
        "centimeters": "cm",
        "cmequivalentwaterheight": "cm",
        "kgm-2": "kg_m-2",
        "kg/m2": "kg_m-2",
        "kgm−2": "kg_m-2",
    }
    if compact not in aliases:
        raise ValueError(
            f"Unsupported or missing water-storage units {units!r}; expected cm, mm, or kg m-2."
        )
    return aliases[compact]


def convert_to_mm(da: xr.DataArray, variable_label: str) -> xr.DataArray:
    """Convert one dynamic water-storage field to canonical mm EWH."""
    source_units = str(da.attrs.get("units", ""))
    unit_name = canonical_unit_name(source_units)
    factor = 10.0 if unit_name == "cm" else 1.0
    converted = (da.astype("float32") * np.float32(factor)).astype("float32")
    converted.attrs = dict(da.attrs)
    converted.attrs.update(
        {
            "units": "mm EWH",
            "source_units": source_units,
            "unit_conversion_to_mm": factor,
            "variable_label": variable_label,
        }
    )
    return converted


def validate_train_only_weight_files(
    alpha_path: Path,
    beta_path: Path,
    train_end: str,
    allow_legacy: bool = False,
) -> dict[str, object]:
    """Validate that alpha/beta were fitted no later than the training cutoff."""
    alpha_ds = open_dataset_any(alpha_path)
    beta_ds = open_dataset_any(beta_path)
    required = {
        "fit_start_requested",
        "fit_end_requested",
        "fit_start_effective",
        "fit_end_effective",
        "fit_n_months",
        "consistency_source_sha256",
    }
    for label, ds in (("alpha", alpha_ds), ("beta", beta_ds)):
        missing = sorted(required.difference(ds.attrs))
        if missing and not allow_legacy:
            raise ValueError(f"{label} weight file lacks train-only provenance: {missing}")
    if allow_legacy and (required.difference(alpha_ds.attrs) or required.difference(beta_ds.attrs)):
        return {"status": "legacy_allowed", "fit_end_effective": "UNKNOWN"}

    alpha_end = pd.Timestamp(str(alpha_ds.attrs["fit_end_effective"]))
    beta_end = pd.Timestamp(str(beta_ds.attrs["fit_end_effective"]))
    cutoff = pd.Timestamp(train_end)
    if alpha_end > cutoff or beta_end > cutoff:
        raise ValueError(
            f"Adaptive weights leak beyond train_end={cutoff.date()}: "
            f"alpha={alpha_end.date()}, beta={beta_end.date()}."
        )
    if alpha_ds.attrs["consistency_source_sha256"] != beta_ds.attrs["consistency_source_sha256"]:
        raise ValueError("Alpha and beta do not share the same consistency source hash.")
    return {
        "status": "train_only_verified",
        "fit_start_effective": str(alpha_ds.attrs["fit_start_effective"]),
        "fit_end_effective": str(alpha_ds.attrs["fit_end_effective"]),
        "fit_n_months": int(alpha_ds.attrs["fit_n_months"]),
        "consistency_source_sha256": str(alpha_ds.attrs["consistency_source_sha256"]),
    }


def build_split_index(time: xr.DataArray, train_end: str, val_end: str) -> xr.DataArray:
    """Create integer time-split labels: 0 train, 1 val, 2 test."""
    timestamps = pd.to_datetime(time.values)
    split_index = np.full(timestamps.shape, 2, dtype=np.int8)
    split_index[timestamps <= pd.Timestamp(train_end)] = 0
    split_index[(timestamps > pd.Timestamp(train_end)) & (timestamps <= pd.Timestamp(val_end))] = 1
    return xr.DataArray(
        split_index,
        dims=("time",),
        coords={"time": time.values},
        name="split_index",
        attrs={"mapping": "0=train, 1=val, 2=test"},
    )


def aggregate_coarse(
    da: xr.DataArray,
    factor: int,
    valid_mask: xr.DataArray,
    area_weights: xr.DataArray,
    eps: float = 1e-12,
) -> xr.DataArray:
    """Aggregate with the masked cos(latitude) operator used by projection."""
    valid = (valid_mask > 0) & np.isfinite(da) & np.isfinite(area_weights) & (area_weights > 0)
    weights = xr.where(valid, area_weights, 0.0)
    values = xr.where(valid, da, 0.0)
    numerator = (values * weights).coarsen(lat=factor, lon=factor, boundary="trim").sum()
    denominator = weights.coarsen(lat=factor, lon=factor, boundary="trim").sum()
    coarse = (numerator / denominator.where(denominator > eps)).fillna(0.0)
    coarse = coarse.rename({"lat": "coarse_lat", "lon": "coarse_lon"})
    return coarse.astype("float32")


def write_missing_months(path: Path, missing_months: list[str]) -> None:
    """Write missing supervised months to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"missing_month": missing_months})
    frame.to_csv(path, index=False)


def write_summary(path: Path, dataset: xr.Dataset, args: argparse.Namespace, missing_months: list[str]) -> None:
    """Write a Markdown summary for downstream training work."""
    time_values = pd.to_datetime(dataset["time"].values)
    split_index = dataset["split_index"].values
    train_months = int((split_index == 0).sum())
    val_months = int((split_index == 1).sum())
    test_months = int((split_index == 2).sum())
    valid_fraction = float(dataset["valid_mask"].mean().item())
    hydrobasins_fraction = float((dataset["hydrobasins_mask"] > 0).mean().item())
    hydrobasins_source_count = dataset["hydrobasins_mask"].attrs.get("source_count")
    lines = [
        "# Training Dataset Summary",
        "",
        f"- Output: `{args.output}`",
        f"- Time range: `{time_values.min():%Y-%m}` to `{time_values.max():%Y-%m}`",
        f"- Number of months: `{len(time_values)}`",
        f"- Train/val/test months: `{train_months} / {val_months} / {test_months}`",
        f"- Fine-grid shape: `time={dataset.sizes['time']}, lat={dataset.sizes['lat']}, lon={dataset.sizes['lon']}`",
        f"- Coarse-grid shape: `time={dataset.sizes['time']}, coarse_lat={dataset.sizes['coarse_lat']}, coarse_lon={dataset.sizes['coarse_lon']}`",
        f"- Grid resolution: `{infer_grid_resolution(dataset)}`",
        f"- Mean valid-cell fraction: `{valid_fraction:.4f}`",
        f"- Missing supervised months inside requested window: `{len(missing_months)}`",
        "",
        "## Dynamic variables",
        "",
        "- `target_jplm_twsa`",
        "- `input_wghm_twsa`",
        "- `input_era5_twsa`",
        "- `input_era5_cwsc`",
        "",
        "## Static variables",
        "",
        "- `alpha_value_weight`",
        "- `beta_gradient_weight`",
        "- `aridity_mask`",
        "- `human_activity_index`",
        "- `glacier_fraction`",
        "- `hydrobasins_mask`",
        "- `land_mask`",
        "- `cell_area_weight` (relative area proportional to cos(latitude))",
        "",
        "## Auxiliary variables",
        "",
        "- `valid_mask`",
        "- `split_index`",
        "- `month_of_year`",
        "- `target_jplm_twsa_coarse`",
        "- `input_wghm_twsa_coarse`",
        "- `hydrobasins_basin_id` (when available)",
        "",
        "## Notes",
        "",
        "- All dynamic water-storage variables are stored in canonical `mm EWH` units.",
        f"- WGHM values with absolute anomaly above `{args.wghm_abs_max_mm:g} mm` are excluded by a predeclared physical sanity mask.",
        f"- Adaptive-weight provenance: `{dataset.attrs.get('weight_provenance', '')}`.",
        "- `target_jplm_twsa` is a gridded-JPL consistency target, not independent 0.5-degree truth.",
        "- Dynamic variables are filled with `0` outside the monthly `valid_mask`.",
        "- Static variables are filled with `0` outside `land_mask`.",
        f"- `hydrobasins_mask` positive-cell fraction on the target grid: `{hydrobasins_fraction:.4f}`.",
        f"- `hydrobasins_basin_id` is included only when `{args.hydrobasins_id}` exists.",
        "- Coarse supervision uses the masked cos(latitude)-area-weighted operator shared with strict projection.",
        f"- Missing supervised months are listed in `{args.missing_report}`.",
    ]
    if hydrobasins_source_count is not None:
        lines.insert(
            lines.index(f"- `hydrobasins_basin_id` is included only when `{args.hydrobasins_id}` exists."),
            f"- HydroBASINS source shapefiles merged into the mask: `{hydrobasins_source_count}`.",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger = configure_logging()
    if args.output.exists() and not args.overwrite:
        logger.info("Output already exists: %s", args.output)
        return

    logger.info("Opening dynamic fields.")
    jpl = open_monthly_dataarray(args.jpl, ["jplm_twsa", "lwe_thickness_anomaly", "lwe_thickness"])
    wghm = open_monthly_dataarray(args.wghm, ["wghm_twsa", "tws_anomaly", "tws"])
    era5_twsa = open_monthly_dataarray(args.era5_twsa, ["era5_twsa"])
    era5_cwsc = open_monthly_dataarray(args.era5_cwsc, ["cwsc"])

    jpl = convert_to_mm(jpl, "JPL TWSA")
    wghm = convert_to_mm(wghm, "WGHM TWSA")
    era5_twsa = convert_to_mm(era5_twsa, "ERA5 TWSA")
    era5_cwsc = convert_to_mm(era5_cwsc, "ERA5 CWSC")

    jpl = jpl.sel(time=slice(args.start, args.end))
    wghm = wghm.sel(time=slice(args.start, args.end))
    era5_twsa = era5_twsa.sel(time=slice(args.start, args.end))
    era5_cwsc = era5_cwsc.sel(time=slice(args.start, args.end))

    jpl, wghm, era5_twsa, era5_cwsc = xr.align(jpl, wghm, era5_twsa, era5_cwsc, join="inner")
    if jpl.sizes.get("time", 0) == 0:
        raise ValueError("No overlapping monthly time range remains after alignment.")
    requested_months = pd.period_range(args.start[:7], args.end[:7], freq="M").to_timestamp()
    available_months = set(pd.to_datetime(jpl["time"].values))
    missing_months = [ts.strftime("%Y-%m") for ts in requested_months if ts not in available_months]
    logger.info(
        "Aligned common training window: %s to %s (%d months).",
        pd.to_datetime(jpl["time"].values[0]).strftime("%Y-%m"),
        pd.to_datetime(jpl["time"].values[-1]).strftime("%Y-%m"),
        jpl.sizes["time"],
    )

    logger.info("Opening static fields.")
    weight_provenance = validate_train_only_weight_files(
        args.alpha,
        args.beta,
        args.train_end,
        allow_legacy=args.allow_legacy_weights,
    )
    land_mask = open_static_dataarray(args.land_mask, ["land_mask", "mask", "land"], template=jpl)
    land_mask = xr.where(land_mask >= 0.5, 1.0, 0.0).astype("float32")
    latitude_weights = xr.DataArray(
        np.cos(np.deg2rad(jpl["lat"].values)).astype("float32"),
        dims=("lat",),
        coords={"lat": jpl["lat"].values},
    )
    cell_area_weight = (
        latitude_weights.broadcast_like(land_mask)
        .clip(min=0.0)
        .astype("float32")
        .rename("cell_area_weight")
    )
    cell_area_weight.attrs.update(
        {
            "units": "1",
            "description": "Relative 0.5-degree cell area weight proportional to cos(latitude).",
        }
    )
    alpha = open_static_dataarray(args.alpha, ["alpha_value_weight"], template=jpl).where(land_mask > 0, 0.0).fillna(0.0)
    beta = open_static_dataarray(args.beta, ["beta_gradient_weight"], template=jpl).where(land_mask > 0, 0.0).fillna(0.0)
    aridity = open_static_dataarray(args.aridity, ["aridity_mask"], template=jpl).where(land_mask > 0, 0.0).fillna(0.0)
    human = open_static_dataarray(args.human, ["human_activity_index"], template=jpl).where(land_mask > 0, 0.0).fillna(0.0)
    glacier = open_static_dataarray(args.glacier, ["glacier_fraction"], template=jpl).where(land_mask > 0, 0.0).fillna(0.0)
    hydrobasins = open_static_dataarray(args.hydrobasins, ["hydrobasins_mask"], template=jpl).where(land_mask > 0, 0.0).fillna(0.0)
    hydrobasins_id = None
    if args.hydrobasins_id.exists():
        hydrobasins_id = (
            open_static_dataarray(args.hydrobasins_id, ["hydrobasins_basin_id", "pfaf_id"], template=jpl)
            .where(land_mask > 0, 0.0)
            .fillna(0.0)
            .round()
            .astype("int32")
        )
    else:
        logger.warning("HydroBASINS basin-id file not found, basin-integral prototype will be unavailable: %s", args.hydrobasins_id)

    logger.info("Building valid masks and coarse supervision.")
    valid_mask = (
        np.isfinite(jpl)
        & np.isfinite(wghm)
        & np.isfinite(era5_twsa)
        & np.isfinite(era5_cwsc)
        & (np.abs(wghm) <= float(args.wghm_abs_max_mm))
        & (land_mask > 0)
    ).astype("uint8")

    target = jpl.where(valid_mask > 0, 0.0).fillna(0.0).astype("float32").rename("target_jplm_twsa")
    input_wghm = wghm.where(valid_mask > 0, 0.0).fillna(0.0).astype("float32").rename("input_wghm_twsa")
    input_era5_twsa = era5_twsa.where(valid_mask > 0, 0.0).fillna(0.0).astype("float32").rename("input_era5_twsa")
    input_era5_cwsc = era5_cwsc.where(valid_mask > 0, 0.0).fillna(0.0).astype("float32").rename("input_era5_cwsc")
    for field in (target, input_wghm, input_era5_twsa, input_era5_cwsc):
        field.attrs["units"] = "mm EWH"

    coarse_target = aggregate_coarse(
        target,
        args.coarse_factor,
        valid_mask,
        cell_area_weight,
    ).rename("target_jplm_twsa_coarse")
    coarse_wghm = aggregate_coarse(
        input_wghm,
        args.coarse_factor,
        valid_mask,
        cell_area_weight,
    ).rename("input_wghm_twsa_coarse")

    split_index = build_split_index(target["time"], args.train_end, args.val_end)
    month_of_year = xr.DataArray(
        pd.to_datetime(target["time"].values).month.astype(np.int8),
        dims=("time",),
        coords={"time": target["time"].values},
        name="month_of_year",
    )

    dataset = xr.Dataset(
        {
            "target_jplm_twsa": target,
            "input_wghm_twsa": input_wghm,
            "input_era5_twsa": input_era5_twsa,
            "input_era5_cwsc": input_era5_cwsc,
            "alpha_value_weight": alpha,
            "beta_gradient_weight": beta,
            "aridity_mask": aridity,
            "human_activity_index": human,
            "glacier_fraction": glacier,
            "hydrobasins_mask": hydrobasins,
            "land_mask": land_mask,
            "cell_area_weight": cell_area_weight,
            "valid_mask": valid_mask.rename("valid_mask"),
            "split_index": split_index,
            "month_of_year": month_of_year,
            "target_jplm_twsa_coarse": coarse_target,
            "input_wghm_twsa_coarse": coarse_wghm,
        }
    )
    if hydrobasins_id is not None:
        dataset["hydrobasins_basin_id"] = hydrobasins_id
    dataset.attrs.update(
        {
            "description": "Training-ready dataset for GRACE adaptive physical soft-constraint downscaling.",
            "time_range": f"{pd.to_datetime(target['time'].values[0]):%Y-%m} to {pd.to_datetime(target['time'].values[-1]):%Y-%m}",
            "coarse_factor": args.coarse_factor,
            "split_mapping": "0=train, 1=val, 2=test",
            "canonical_water_storage_units": "mm EWH",
            "weight_provenance": json.dumps(weight_provenance, sort_keys=True),
            "wghm_abs_max_mm": float(args.wghm_abs_max_mm),
            "coarse_operator": "masked_coslat_area_weighted_block_mean",
            "fine_jpl_interpretation": "gridded-JPL consistency target; not independent 0.5-degree truth",
            "notes": "Dynamic inputs are converted to mm EWH and aligned on the common 0.5 degree grid and overlap window.",
        }
    )

    logger.info("Writing %s", args.output)
    write_dataset(dataset, args.output)
    write_missing_months(args.missing_report, missing_months)
    write_summary(args.summary, dataset, args, missing_months)
    logger.info("Wrote training summary to %s", args.summary)


if __name__ == "__main__":
    main()
