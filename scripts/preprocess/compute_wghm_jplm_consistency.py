"""Compute coarse-scale WGHM vs JPL Mascon consistency metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (  # noqa: E402
    configure_logging,
    guess_data_var,
    normalize_monthly_time_values,
    open_dataset_any,
    write_dataset,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jpl-input",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "anomalies_2004_2009" / "jplm_twsa.nc",
    )
    parser.add_argument(
        "--wghm-input",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "anomalies_2004_2009" / "wghm_twsa.nc",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data_processed"
            / "adaptive_weights_train_only"
            / "wghm_jplm_consistency_train_only.nc"
        ),
    )
    parser.add_argument(
        "--fit-start",
        default="2002-04-01",
        help="Inclusive first month used to fit consistency statistics.",
    )
    parser.add_argument(
        "--fit-end",
        default="2016-12-01",
        help="Inclusive last month used to fit consistency statistics. Must not exceed the training split.",
    )
    parser.add_argument("--coarse-factor", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def slope_1d(values: np.ndarray) -> float:
    """Return the linear slope across the series index."""
    mask = np.isfinite(values)
    if mask.sum() < 3:
        return np.nan
    x = np.arange(mask.sum(), dtype=float)
    return np.polyfit(x, values[mask], 1)[0]


def season_amp(values: xr.DataArray) -> xr.DataArray:
    """Estimate seasonal amplitude from monthly climatology."""
    climatology = values.groupby("time.month").mean("time")
    return 0.5 * (climatology.max("month") - climatology.min("month"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for one source file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_unit_name(units: str | None) -> str:
    """Normalize common terrestrial-water-storage unit spellings."""
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
            f"Unsupported or missing TWSA units {units!r}. "
            "Declare cm, mm, or kg m-2 before computing consistency."
        )
    return aliases[compact]


def convert_to_mm(da: xr.DataArray) -> tuple[xr.DataArray, str, float]:
    """Convert a TWSA field to millimetres equivalent water height."""
    original_units = str(da.attrs.get("units", ""))
    canonical = canonical_unit_name(original_units)
    factor = 10.0 if canonical == "cm" else 1.0
    converted = da.astype(np.float64) * factor
    converted.attrs = dict(da.attrs)
    converted.attrs["source_units"] = original_units
    converted.attrs["units"] = "mm EWH"
    converted.attrs["unit_conversion_to_mm"] = factor
    return converted, original_units, factor


def align_and_select_fit_window(
    jpl: xr.DataArray,
    wghm: xr.DataArray,
    fit_start: str,
    fit_end: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Normalize monthly timestamps, align sources, and retain only the fit window."""
    jpl = jpl.assign_coords(time=("time", normalize_monthly_time_values(jpl["time"].values)))
    wghm = wghm.assign_coords(time=("time", normalize_monthly_time_values(wghm["time"].values)))
    if len(np.unique(jpl["time"].values)) != jpl.sizes["time"]:
        jpl = jpl.groupby("time").mean()
    if len(np.unique(wghm["time"].values)) != wghm.sizes["time"]:
        wghm = wghm.groupby("time").mean()
    jpl, wghm = xr.align(jpl, wghm, join="inner")
    if "time" not in jpl.dims or "time" not in wghm.dims:
        raise ValueError("Both JPL and WGHM inputs must have time dimensions.")

    fit_start_value = np.datetime64(fit_start, "ns")
    fit_end_value = np.datetime64(fit_end, "ns")
    if fit_start_value > fit_end_value:
        raise ValueError(f"fit-start {fit_start} must not be later than fit-end {fit_end}.")
    fit_mask = (jpl["time"] >= fit_start_value) & (jpl["time"] <= fit_end_value)
    jpl_fit = jpl.where(fit_mask, drop=True)
    wghm_fit = wghm.where(fit_mask, drop=True)
    if jpl_fit.sizes.get("time", 0) < 3:
        raise ValueError(
            "Fewer than three overlapping months remain inside the requested fit window "
            f"[{fit_start}, {fit_end}]."
        )
    return jpl_fit, wghm_fit


def compute_consistency_dataset(
    jpl: xr.DataArray,
    wghm: xr.DataArray,
    *,
    coarse_factor: int,
    fit_start: str,
    fit_end: str,
) -> xr.Dataset:
    """Compute train-window-only, unit-consistent JPL-WGHM consistency fields."""
    if coarse_factor <= 0:
        raise ValueError("coarse_factor must be a positive integer.")
    jpl_mm, jpl_units, jpl_factor = convert_to_mm(jpl)
    wghm_mm, wghm_units, wghm_factor = convert_to_mm(wghm)
    jpl_fit, wghm_fit = align_and_select_fit_window(jpl_mm, wghm_mm, fit_start, fit_end)

    coarse_jpl = jpl_fit.coarsen(lat=coarse_factor, lon=coarse_factor, boundary="trim").mean()
    coarse_wghm = wghm_fit.coarsen(lat=coarse_factor, lon=coarse_factor, boundary="trim").mean()

    corr = xr.corr(coarse_wghm, coarse_jpl, dim="time").rename("corr")
    jpl_slope = xr.apply_ufunc(
        slope_1d,
        coarse_jpl,
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    wghm_slope = xr.apply_ufunc(
        slope_1d,
        coarse_wghm,
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    trend_diff = (wghm_slope - jpl_slope).rename("trend_diff")
    season_amp_diff = (season_amp(coarse_wghm) - season_amp(coarse_jpl)).rename("season_amp_diff")
    nrmse = (
        np.sqrt(((coarse_wghm - coarse_jpl) ** 2).mean("time"))
        / coarse_jpl.std("time")
    ).rename("nrmse")

    coarse_ds = xr.Dataset(
        {
            "corr": corr,
            "trend_diff": trend_diff,
            "season_amp_diff": season_amp_diff,
            "nrmse": nrmse,
        }
    )
    full_res = coarse_ds.interp(lat=jpl_fit["lat"], lon=jpl_fit["lon"], method="nearest")
    effective_start = str(np.datetime_as_string(jpl_fit["time"].values.min(), unit="D"))
    effective_end = str(np.datetime_as_string(jpl_fit["time"].values.max(), unit="D"))
    full_res.attrs.update(
        {
            "coarse_factor": coarse_factor,
            "description": "WGHM versus JPL coarse-scale consistency metrics fitted on training months only.",
            "fit_start_requested": fit_start,
            "fit_end_requested": fit_end,
            "fit_start_effective": effective_start,
            "fit_end_effective": effective_end,
            "fit_n_months": int(jpl_fit.sizes["time"]),
            "canonical_units": "mm EWH",
            "jpl_source_units": jpl_units,
            "wghm_source_units": wghm_units,
            "jpl_conversion_to_mm": jpl_factor,
            "wghm_conversion_to_mm": wghm_factor,
        }
    )
    return full_res


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger = configure_logging()
    if args.output.exists() and not args.overwrite:
        logger.info("Output already exists: %s", args.output)
        return

    jpl_ds = open_dataset_any(args.jpl_input)
    wghm_ds = open_dataset_any(args.wghm_input)
    jpl_var = guess_data_var(jpl_ds, aliases=["jplm_twsa", "lwe_thickness_anomaly", "lwe_thickness"])
    wghm_var = guess_data_var(wghm_ds, aliases=["wghm_twsa", "tws_anomaly", "tws"])

    full_res = compute_consistency_dataset(
        jpl_ds[jpl_var],
        wghm_ds[wghm_var],
        coarse_factor=args.coarse_factor,
        fit_start=args.fit_start,
        fit_end=args.fit_end,
    )
    full_res.attrs.update(
        {
            "jpl_source_path": str(args.jpl_input.resolve()),
            "wghm_source_path": str(args.wghm_input.resolve()),
            "jpl_source_sha256": sha256_file(args.jpl_input),
            "wghm_source_sha256": sha256_file(args.wghm_input),
            "jpl_source_variable": jpl_var,
            "wghm_source_variable": wghm_var,
            "generation_script": str(Path(__file__).resolve()),
            "generation_script_sha256": sha256_file(Path(__file__).resolve()),
            "generation_parameters": json.dumps(
                {
                    "fit_start": args.fit_start,
                    "fit_end": args.fit_end,
                    "coarse_factor": args.coarse_factor,
                },
                sort_keys=True,
            ),
        }
    )
    write_dataset(full_res, args.output)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
