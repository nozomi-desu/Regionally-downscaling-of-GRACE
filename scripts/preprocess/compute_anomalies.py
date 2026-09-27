"""Compute anomalies relative to the 2004-01 to 2009-12 baseline."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variable", help="Optional variable name override.")
    parser.add_argument("--output-var", help="Optional anomaly variable name.")
    parser.add_argument("--baseline-start", default="2004-01-01")
    parser.add_argument("--baseline-end", default="2009-12-31")
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
        aliases=["lwe_thickness", "tws", "twsa", "TWS", "TWSA"],
    )
    da = ds[variable]
    if "time" not in da.dims:
        raise ValueError(f"{args.input} does not have a time dimension.")
    monthly_time = normalize_monthly_time_values(da["time"].values)
    da = da.assign_coords(time=("time", monthly_time))
    if len(np.unique(monthly_time)) != len(monthly_time):
        da = da.groupby("time").mean()

    baseline_slice = da.sel(time=slice(args.baseline_start, args.baseline_end))
    if baseline_slice.sizes.get("time", 0) == 0:
        raise ValueError("Baseline subset is empty. Check the input time coverage.")
    baseline = baseline_slice.mean("time")

    anomaly_name = args.output_var or f"{variable}_anomaly"
    anomaly = da - baseline
    anomaly.attrs.update(da.attrs)
    anomaly.attrs["anomaly_baseline"] = f"{args.baseline_start} to {args.baseline_end}"
    out_ds = xr.Dataset(
        {
            anomaly_name: anomaly,
            f"{variable}_baseline_mean": baseline,
        }
    )
    write_dataset(out_ds, args.output)
    logger.info("Wrote anomalies to %s", args.output)


if __name__ == "__main__":
    main()
