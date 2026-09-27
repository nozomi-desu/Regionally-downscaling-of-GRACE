"""Build the independent CLM5.0 monthly TWS anomaly prior on the 0.5-degree grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import configure_logging, write_dataset  # noqa: E402
from scripts.eval.apply_native_mascon_decomposition import normalize_longitude  # noqa: E402


DEFAULT_TEMPLATE = (
    PROJECT_ROOT
    / "data_processed"
    / "training_rescue_v2"
    / "grace_downscaling_training_dataset_mm_area.zarr"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-start", default="2004-01")
    parser.add_argument("--baseline-end", default="2009-12")
    parser.add_argument("--start", default="2002-04")
    parser.add_argument("--end", default="2014-12")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    source = normalize_longitude(xr.open_dataset(args.input.resolve()))
    if "TWS" not in source:
        raise KeyError("CLM file does not contain TWS.")
    tws = source["TWS"]
    if not {"time", "lat", "lon"}.issubset(tws.dims):
        raise ValueError(f"Unexpected CLM TWS dimensions: {tws.dims}")
    units = str(tws.attrs.get("units", "")).strip().lower()
    if units not in {"mm", "kg/m2", "kg m-2", "kg m^-2"}:
        raise ValueError(f"Unverified CLM TWS units: {tws.attrs.get('units')!r}")

    if "time_bounds" in source:
        # CLM monthly timestamps denote the right edge (e.g. 2002-05-01 for
        # the April mean); assign the lower bound so months align physically.
        month_values = np.asarray(source["time_bounds"].isel(hist_interval=0).values).astype(
            "datetime64[M]"
        )
    else:
        month_values = np.asarray(tws.time.values).astype("datetime64[M]") - np.timedelta64(1, "M")
    tws = tws.assign_coords(time=("time", month_values))
    tws = tws.sel(time=slice(args.start, args.end)).astype(np.float64)
    template = xr.open_zarr(args.template.resolve())
    tws = tws.interp(lat=template.lat, lon=template.lon, method="linear")
    baseline = tws.sel(time=slice(args.baseline_start, args.baseline_end))
    if baseline.sizes.get("time", 0) != 72:
        raise ValueError(f"Expected 72 baseline months; found {baseline.sizes.get('time', 0)}.")
    baseline_mean = baseline.mean("time")
    anomaly = (tws - baseline_mean).astype(np.float32)
    anomaly.name = "clm5_twsa"
    anomaly.attrs.update(
        units="mm EWH",
        long_name="CLM5.0 terrestrial-water-storage anomaly",
        anomaly_baseline=f"{args.baseline_start} to {args.baseline_end}",
        interpolation="bilinear interpolation from native approximately 1-degree grid to 0.5 degree",
    )
    output = xr.Dataset(
        {
            "clm5_twsa": anomaly,
            "clm5_baseline_mean": baseline_mean.astype(np.float32),
        },
        attrs={
            "protocol_id": "clm5_independent_prior_v1",
            "source_product": "CLM5.0 land-only GSWP3 historical simulation",
            "source_file": args.input.name,
            "claim_boundary": "Independent model-family prior; never treated as observed fine-grid truth.",
        },
    )
    write_dataset(output, args.output.resolve())
    logger.info("Wrote CLM5 independent prior to %s", args.output.resolve())


if __name__ == "__main__":
    main()
