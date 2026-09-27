"""Standardize raster-like data to the common 0.5 degree lat/lon grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import (  # noqa: E402
    configure_logging,
    guess_data_var,
    make_target_grid,
    open_dataset_any,
    write_dataset,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variable", help="Optional variable name override.")
    parser.add_argument(
        "--method",
        default="linear",
        choices=["linear", "nearest", "cubic"],
        help="Use nearest for categorical rasters and linear for continuous fields.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger = configure_logging()
    if args.output.exists() and not args.overwrite:
        logger.info("Output already exists: %s", args.output)
        return

    ds = open_dataset_any(args.input)
    variable = guess_data_var(
        ds,
        preferred=args.variable,
        aliases=["lwe_thickness", "tws", "TWS", "band_data", Path(args.input).stem],
    )
    da = ds[variable]
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"{args.input} does not expose lat/lon dimensions after standardization.")

    target = make_target_grid()
    logger.info("Interpolating %s to the 0.5 degree grid with method=%s", variable, args.method)
    regridded = da.interp(lat=target["lat"], lon=target["lon"], method=args.method)
    regridded.attrs.update(da.attrs)
    regridded.attrs["grid_standardized_to"] = "0.5 degree global lat/lon grid"
    out_ds = xr.Dataset({variable: regridded})
    write_dataset(out_ds, args.output)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()

