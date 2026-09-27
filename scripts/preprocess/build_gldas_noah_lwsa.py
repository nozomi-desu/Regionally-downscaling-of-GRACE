"""Build a 0.5-degree GLDAS Noah land-water-storage anomaly proxy.

The proxy sums four soil-moisture layers, snow water equivalent, and canopy
interception.  All GLDAS components use kg m-2, numerically equivalent to mm EWH.
It excludes groundwater and surface-water storage and must not be labelled total
terrestrial water storage.
"""

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


COMPONENTS = (
    "SoilMoi0_10cm_inst",
    "SoilMoi10_40cm_inst",
    "SoilMoi40_100cm_inst",
    "SoilMoi100_200cm_inst",
    "SWE_inst",
    "CanopInt_inst",
)
DEFAULT_TEMPLATE = (
    PROJECT_ROOT
    / "data_processed"
    / "training_rescue_v2"
    / "grace_downscaling_training_dataset_mm_area.zarr"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-start", default="2004-01-01")
    parser.add_argument("--baseline-end", default="2009-12-31")
    parser.add_argument("--max-files", type=int, help="Structural smoke only.")
    return parser.parse_args()


def area_weighted_half_degree(field: xr.DataArray) -> xr.DataArray:
    """Aggregate the 0.25-degree GLDAS grid to 0.5 degrees by cell area."""
    weights = xr.DataArray(
        np.cos(np.deg2rad(field.lat.values)),
        coords={"lat": field.lat},
        dims=("lat",),
    ).broadcast_like(field.isel(time=0, drop=True))
    valid = field.notnull()
    numerator = (field.fillna(0.0) * weights).coarsen(lat=2, lon=2, boundary="trim").sum()
    denominator = weights.where(valid).fillna(0.0).coarsen(lat=2, lon=2, boundary="trim").sum()
    return numerator / denominator.where(denominator > 0)


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    files = sorted(args.input_dir.resolve().glob("GLDAS_NOAH025_M.A*.021.nc4"))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No GLDAS monthly files found in {args.input_dir}")
    logger.info("Opening %d GLDAS monthly files", len(files))
    dataset = xr.open_mfdataset(files, combine="by_coords", data_vars="minimal", coords="minimal", compat="override")
    missing = [name for name in COMPONENTS if name not in dataset]
    if missing:
        raise KeyError(f"GLDAS files lack required components: {missing}")
    storage = sum((dataset[name].astype(np.float64) for name in COMPONENTS[1:]), dataset[COMPONENTS[0]].astype(np.float64))
    storage.name = "gldas_noah_land_water_storage"
    storage = normalize_longitude(storage.to_dataset()).to_array().isel(variable=0, drop=True)
    storage = area_weighted_half_degree(storage)

    template = xr.open_zarr(args.template.resolve())
    storage = storage.interp(lat=template.lat, lon=template.lon, method="linear")
    baseline = storage.sel(time=slice(args.baseline_start, args.baseline_end))
    if baseline.sizes.get("time", 0) < 12 and args.max_files is None:
        raise ValueError("The requested 2004--2009 anomaly baseline is incomplete.")
    baseline_mean = baseline.mean("time")
    anomaly = (storage - baseline_mean).astype(np.float32)
    anomaly.name = "gldas_noah_lwsa"
    anomaly.attrs.update(
        units="mm EWH",
        long_name="GLDAS Noah land-water-storage anomaly proxy",
        components=", ".join(COMPONENTS),
        anomaly_baseline=f"{args.baseline_start} to {args.baseline_end}",
        scope_warning="Excludes groundwater and surface-water storage; not complete TWSA.",
    )
    output = xr.Dataset(
        {
            "gldas_noah_lwsa": anomaly,
            "gldas_noah_baseline_mean": baseline_mean.astype(np.float32),
        },
        attrs={
            "protocol_id": "gldas_noah_lwsa_proxy_v1",
            "source_product": "GLDAS_NOAH025_M.2.1",
            "aggregation": "cosine-area-weighted 0.25-to-0.5-degree mean",
            "claim_boundary": "Independent land-model proxy, not fine-grid truth or complete TWSA.",
        },
    )
    write_dataset(output, args.output.resolve())
    logger.info("Wrote GLDAS Noah anomaly proxy to %s", args.output.resolve())


if __name__ == "__main__":
    main()

