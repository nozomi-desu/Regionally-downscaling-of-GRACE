"""Build a 0.5 degree aridity penalty mask from the monthly AI archive."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import configure_logging, write_dataset  # noqa: E402

TARGET_TRANSFORM = from_origin(-180.0, 90.0, 0.5, 0.5)
TARGET_SHAPE = (360, 720)
NODATA_SENTINEL = 65535.0
AI_SCALE_FACTOR = 1.0 / 10000.0
DRYLAND_THRESHOLD = 0.65


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ai-zip",
        type=Path,
        default=PROJECT_ROOT / "data_raw" / "aridity_pet" / "Global_AI__monthly_v3_1.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "adaptive_weights" / "aridity_mask.nc",
    )
    parser.add_argument(
        "--output-annual-ai",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "adaptive_weights" / "annual_mean_aridity_index.nc",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def target_coords() -> dict[str, np.ndarray]:
    """Return canonical 0.5 degree coordinates."""
    return {
        "lat": np.arange(-89.75, 90.0, 0.5),
        "lon": np.arange(-179.75, 180.0, 0.5),
    }


def monthly_tif_members(ai_zip: Path) -> list[str]:
    """List the monthly AI GeoTIFF members inside the archive."""
    with zipfile.ZipFile(ai_zip) as zf:
        tif_names = sorted(
            name
            for name in zf.namelist()
            if name.lower().endswith(".tif") and "/ai_v31_" in name.replace("\\", "/").lower()
        )
    if len(tif_names) != 12:
        raise ValueError(f"Expected 12 monthly AI TIFF files in {ai_zip}, found {len(tif_names)}.")
    return tif_names


def reproject_member(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    """Read one TIFF member from zip and reproject it to the 0.5 degree target grid."""
    data = zf.read(member)
    with MemoryFile(data) as memfile:
        with memfile.open() as src:
            source = src.read(1).astype("float32")
            source[source >= NODATA_SENTINEL] = np.nan
            source *= AI_SCALE_FACTOR

            destination = np.full(TARGET_SHAPE, np.nan, dtype="float32")
            reproject(
                source=source,
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=np.nan,
                dst_transform=TARGET_TRANSFORM,
                dst_crs=src.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    return np.flipud(destination)


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger = configure_logging()
    if (
        args.output.exists()
        and args.output_annual_ai.exists()
        and not args.overwrite
    ):
        logger.info("Aridity outputs already exist.")
        return
    if not args.ai_zip.exists():
        raise FileNotFoundError(args.ai_zip)

    members = monthly_tif_members(args.ai_zip)
    logger.info("Processing %d monthly AI rasters from %s", len(members), args.ai_zip)

    monthly_arrays = []
    with zipfile.ZipFile(args.ai_zip) as zf:
        for member in members:
            logger.info("Reprojecting %s", Path(member).name)
            monthly_arrays.append(reproject_member(zf, member))

    annual_ai = np.nanmean(np.stack(monthly_arrays, axis=0), axis=0).astype("float32")
    annual_ai = np.where(np.isfinite(annual_ai), annual_ai, np.nan)
    valid_land = np.isfinite(annual_ai) & (annual_ai > 0.0)
    aridity_penalty = np.zeros_like(annual_ai, dtype="float32")
    aridity_penalty[valid_land] = np.clip(
        (DRYLAND_THRESHOLD - annual_ai[valid_land]) / DRYLAND_THRESHOLD,
        0.0,
        1.0,
    ).astype("float32")

    coords = target_coords()
    ai_da = xr.DataArray(
        annual_ai,
        dims=("lat", "lon"),
        coords=coords,
        name="annual_mean_aridity_index",
        attrs={
            "description": "Annual mean aridity index from the monthly AI v3.1 archive, resampled to 0.5 degree.",
            "source_archive": str(args.ai_zip),
            "units": "dimensionless",
            "scale_factor_applied": AI_SCALE_FACTOR,
        },
    )
    mask_da = xr.DataArray(
        aridity_penalty,
        dims=("lat", "lon"),
        coords=coords,
        name="aridity_mask",
        attrs={
            "description": "Dryland penalty mask derived from annual mean AI using threshold AI < 0.65.",
            "formula": "clip((0.65 - AI) / 0.65, 0, 1)",
            "valid_domain": "Applied only where annual mean AI > 0; oceans and invalid cells are set to 0.",
            "units": "0-1 penalty",
        },
    )

    write_dataset(ai_da.to_dataset(name=ai_da.name), args.output_annual_ai)
    write_dataset(mask_da.to_dataset(name=mask_da.name), args.output)
    logger.info("Wrote %s and %s", args.output_annual_ai, args.output)


if __name__ == "__main__":
    main()
