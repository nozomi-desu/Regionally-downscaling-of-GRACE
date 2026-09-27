"""Build a 0.5-degree GLDAS-Noah v3.3 TWSA prior from PO.DAAC granules.

The source product is a 1-degree monthly terrestrial-water-storage anomaly
aligned to nominal GRACE/GRACE-FO integration months.  It sums Noah soil water,
snow water equivalent, and canopy water; it excludes groundwater and explicit
surface-water storage.  Bilinear interpolation to the 0.5-degree display grid
does not create independent sub-degree information and is used only to place
this independent model-family pattern on the common native-mascon operator.
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


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    files = sorted(args.input_dir.resolve().glob("GLDAS-NOAH_1deg_tws_anomaly_monthly_*.nc"))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No PO.DAAC GLDAS-Noah granules found in {args.input_dir}")
    logger.info("Opening %d PO.DAAC GLDAS-Noah TWS granules", len(files))
    # Avoid making dask a hidden runtime requirement: each granule is only
    # about 0.45 MB, so eager open-and-concat is both deterministic and small.
    granules = [xr.open_dataset(path).load() for path in files]
    dataset = xr.concat(
        granules,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    ).sortby("time")
    # Two official granule pairs share a calendar month but represent distinct
    # nominal integration windows.  Match the established JPL lookup rule:
    # retain the later nominal-window centre for each calendar month.
    retained: dict[np.datetime64, int] = {}
    for index, value in enumerate(dataset.time.values):
        retained[np.datetime64(value, "M")] = index
    selected_indices = np.asarray(sorted(retained.values()), dtype=np.int64)
    duplicate_records_removed = int(dataset.sizes["time"] - selected_indices.size)
    dataset = dataset.isel(time=selected_indices)
    if "TWS_monthly" not in dataset:
        raise KeyError("PO.DAAC granules lack TWS_monthly.")
    storage = dataset["TWS_monthly"].astype(np.float64)
    units = str(storage.attrs.get("units", "")).strip().lower()
    if units not in {"mm", "mm ewh", "mmewh"}:
        raise ValueError(f"Expected PO.DAAC TWS_monthly in mm, found {units!r}.")
    template = xr.open_zarr(args.template.resolve())
    storage = storage.interp(lat=template.lat, lon=template.lon, method="linear")
    baseline = storage.sel(time=slice(args.baseline_start, args.baseline_end))
    if baseline.sizes.get("time", 0) < 36 and args.max_files is None:
        raise ValueError("The 2004--2009 common anomaly baseline is incomplete.")
    baseline_mean = baseline.mean("time")
    anomaly = (storage - baseline_mean).astype(np.float32)
    anomaly.name = "gldas_noah_twsa"
    anomaly.attrs.update(
        units="mm EWH",
        long_name="GLDAS-Noah v3.3 land-water-storage anomaly prior",
        anomaly_baseline=f"{args.baseline_start} to {args.baseline_end}",
        source_resolution="1 degree",
        target_resolution="0.5 degree display grid",
        interpolation="bilinear; no new independent sub-degree information",
        scope_warning=(
            "Noah storage sums soil water, snow and canopy water; groundwater "
            "and explicit surface-water storage are excluded."
        ),
    )
    output = xr.Dataset(
        {
            "gldas_noah_twsa": anomaly,
            "gldas_noah_baseline_mean": baseline_mean.astype(np.float32),
        },
        attrs={
            "protocol_id": "gldas_noah_podaac_twsa_prior_v1",
            "source_product": "TELLUS_GLDAS-NOAH-3.3_TWS-ANOMALY_MONTHLY",
            "source_doi": "10.5067/GGDAS-3NH33",
            "source_components": "soil moisture + snow water equivalent + canopy water",
            "forcing_family": "GLDAS-Noah; independent of ERA5",
            "duplicate_month_policy": "retain later nominal integration-window centre",
            "duplicate_records_removed": duplicate_records_removed,
            "claim_boundary": (
                "Independent land-model family prior, not complete TWSA truth "
                "and not independent 0.5-degree information."
            ),
        },
    )
    write_dataset(output, args.output.resolve())
    logger.info("Wrote %s", args.output.resolve())


if __name__ == "__main__":
    main()
