"""Compute ERA5-derived TWSA and CWSC on the common grid."""

from __future__ import annotations

import argparse
import glob
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
    make_target_grid,
    standardize_lat_lon,
    standardize_time,
    write_dataset,
)

SOIL_LAYER_ALIASES = [
    ["swvl1", "volumetric_soil_water_layer_1"],
    ["swvl2", "volumetric_soil_water_layer_2"],
    ["swvl3", "volumetric_soil_water_layer_3"],
    ["swvl4", "volumetric_soil_water_layer_4"],
]
SOIL_LAYER_THICKNESS_M = [0.07, 0.21, 0.72, 1.89]
PRECIP_ALIASES = ["tp", "total_precipitation"]
EVAP_ALIASES = ["e", "evaporation"]
RUNOFF_ALIASES = ["ro", "runoff"]
SWE_ALIASES = ["sd", "snow_depth_water_equivalent", "sdwe"]
CANOPY_ALIASES = ["src", "skin_reservoir_content"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        default=str(PROJECT_ROOT / "data_raw" / "era5" / "*.nc"),
        help="Glob for ERA5 NetCDF files under the canonical raw-data directory.",
    )
    parser.add_argument(
        "--output-twsa",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "anomalies_2004_2009" / "era5_twsa.nc",
    )
    parser.add_argument(
        "--output-cwsc",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "anomalies_2004_2009" / "era5_cwsc.nc",
    )
    parser.add_argument("--baseline-start", default="2004-01-01")
    parser.add_argument("--baseline-end", default="2009-12-31")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def first_available(ds: xr.Dataset, aliases: list[str]) -> xr.DataArray:
    """Return the first available variable from a list of aliases."""
    lowered = {name.lower(): name for name in ds.data_vars}
    for alias in aliases:
        if alias in ds.data_vars:
            return ds[alias]
        if alias.lower() in lowered:
            return ds[lowered[alias.lower()]]
    raise KeyError(f"None of these variables were found: {aliases}")


def to_mm_water(da: xr.DataArray) -> xr.DataArray:
    """Convert a water-thickness-like variable to mm where reasonable."""
    units = str(da.attrs.get("units", "")).lower()
    if "kg m-2" in units or "kg m**-2" in units:
        return da
    if "mm" in units:
        return da
    if "m" in units or units == "":
        converted = da * 1000.0
        converted.attrs.update(da.attrs)
        converted.attrs["units"] = "mm"
        return converted
    return da


def soil_storage_mm(ds: xr.Dataset) -> xr.DataArray:
    """Convert volumetric soil-water layers to integrated mm storage."""
    components = []
    for aliases, thickness in zip(SOIL_LAYER_ALIASES, SOIL_LAYER_THICKNESS_M):
        layer = first_available(ds, aliases)
        components.append(layer * thickness * 1000.0)
    out = sum(components)
    out.name = "soil_moisture_storage_mm"
    out.attrs["units"] = "mm"
    return out


def anomaly(da: xr.DataArray, start: str, end: str) -> xr.DataArray:
    """Compute anomalies relative to a baseline time range."""
    baseline = da.sel(time=slice(start, end)).mean("time")
    return da - baseline


def open_and_harmonize_era5(path: str) -> xr.Dataset:
    """Open one ERA5 file and harmonize coordinates for safe multi-file merging."""
    ds = xr.open_dataset(path)
    ds = standardize_lat_lon(standardize_time(ds))
    if "time" not in ds.coords:
        raise KeyError(f"Could not find a time-like coordinate in {path}")
    monthly_time = pd.to_datetime(ds["time"].values).to_period("M").to_timestamp()
    ds = ds.assign_coords(time=("time", monthly_time))
    if pd.Index(monthly_time).duplicated().any():
        ds = ds.groupby("time").first()
    drop_vars = [name for name in ["number", "expver"] if name in ds.coords or name in ds.variables]
    if drop_vars:
        ds = ds.drop_vars(drop_vars, errors="ignore")
    return ds


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger = configure_logging()
    if args.output_twsa.exists() and args.output_cwsc.exists() and not args.overwrite:
        logger.info("ERA5 processed outputs already exist.")
        return

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No ERA5 files matched {args.input_glob}")

    logger.info("Opening %d ERA5 file(s).", len(files))
    datasets = [open_and_harmonize_era5(path) for path in files]
    ds = xr.merge(datasets, compat="override", join="exact")
    target = make_target_grid()
    ds = ds.interp(lat=target["lat"], lon=target["lon"], method="linear")

    precip = to_mm_water(first_available(ds, PRECIP_ALIASES))
    evap = to_mm_water(first_available(ds, EVAP_ALIASES))
    runoff = to_mm_water(first_available(ds, RUNOFF_ALIASES))
    swe = to_mm_water(first_available(ds, SWE_ALIASES))
    canopy = to_mm_water(first_available(ds, CANOPY_ALIASES))
    soil = soil_storage_mm(ds)

    evap_sign = float(evap.mean(skipna=True))
    evap_loss = -evap if evap_sign < 0 else evap
    if evap_sign < 0:
        logger.info("ERA5 evaporation appears negative by convention; converting to positive loss for CWSC.")

    soil_anom = anomaly(soil, args.baseline_start, args.baseline_end)
    swe_anom = anomaly(swe, args.baseline_start, args.baseline_end)
    canopy_anom = anomaly(canopy, args.baseline_start, args.baseline_end)
    era5_twsa = soil_anom + swe_anom + canopy_anom
    era5_twsa.name = "era5_twsa"
    era5_twsa.attrs["units"] = "mm"
    era5_twsa.attrs["description"] = "ERA5-derived TWSA proxy from soil moisture, snow water equivalent, and canopy water anomalies."

    cwsc_flux = precip - evap_loss - runoff
    cwsc = cwsc_flux.cumsum("time")
    cwsc_anom = anomaly(cwsc, args.baseline_start, args.baseline_end)
    cwsc_anom.name = "cwsc"
    cwsc_anom.attrs["units"] = "mm"
    cwsc_anom.attrs["description"] = "Cumulative water storage change from P - E - R."

    write_dataset(xr.Dataset({"era5_twsa": era5_twsa}), args.output_twsa)
    write_dataset(xr.Dataset({"cwsc": cwsc_anom}), args.output_cwsc)
    logger.info("Wrote %s and %s", args.output_twsa, args.output_cwsc)


if __name__ == "__main__":
    main()
