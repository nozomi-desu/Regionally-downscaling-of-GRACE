"""Rasterize HydroBASINS basin identifiers onto the 0.5 degree grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.features
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import configure_logging, write_dataset  # noqa: E402
from scripts.preprocess.rasterize_masks import TRANSFORM_05, prepare_hydrobasins_paths, read_vector_inputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional HydroBASINS shapefile path override.")
    parser.add_argument(
        "--attribute",
        default=None,
        help="Integer basin attribute to burn. Defaults to HYBAS_ID when multiple domains are available, otherwise PFAF_ID.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "grid_05deg" / "hydrobasins_level06_pfaf_id.nc",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def latlon_from_array(array: np.ndarray, name: str) -> xr.DataArray:
    """Wrap a 2D raster into a 0.5 degree DataArray."""
    lat_desc = np.arange(89.75, -90.0, -0.5)
    lon = np.arange(-179.75, 180.0, 0.5)
    da = xr.DataArray(array, dims=("lat", "lon"), coords={"lat": lat_desc, "lon": lon}, name=name)
    return da.sortby("lat")


def resolve_attribute(requested: str | None, gdf: gpd.GeoDataFrame, n_sources: int) -> str:
    """Choose a basin identifier attribute that stays unique across merged domains."""
    if requested:
        return requested
    if n_sources > 1 and "HYBAS_ID" in gdf.columns:
        return "HYBAS_ID"
    if "PFAF_ID" in gdf.columns:
        return "PFAF_ID"
    if "HYBAS_ID" in gdf.columns:
        return "HYBAS_ID"
    raise KeyError(f"No suitable HydroBASINS basin-ID field found. Available: {list(gdf.columns)}")


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger = configure_logging()
    if args.output.exists() and not args.overwrite:
        logger.info("Output already exists: %s", args.output)
        return

    input_paths = [args.input] if args.input is not None else prepare_hydrobasins_paths()
    logger.info("Reading %d HydroBASINS shapefile(s).", len(input_paths))
    gdf = read_vector_inputs(input_paths)
    attribute = resolve_attribute(args.attribute, gdf, len(input_paths))
    if attribute not in gdf.columns:
        first_name = input_paths[0].name if input_paths else "<none>"
        raise KeyError(f"{attribute!r} is not present in {first_name}. Available: {list(gdf.columns)}")
    logger.info("Using HydroBASINS attribute %s for basin IDs.", attribute)

    shapes = []
    for row in gdf.itertuples(index=False):
        geom = getattr(row, "geometry")
        value = int(getattr(row, attribute))
        if geom is None or geom.is_empty or value <= 0:
            continue
        shapes.append((geom, value))

    raster = rasterio.features.rasterize(
        shapes,
        out_shape=(360, 720),
        transform=TRANSFORM_05,
        fill=0,
        all_touched=True,
        dtype="int32",
    )
    basin_name = "hydrobasins_basin_id"
    da = latlon_from_array(raster.astype(np.int32), basin_name)
    da.attrs["source_attribute"] = attribute
    da.attrs["source_count"] = len(input_paths)
    da.attrs["notes"] = "Rasterized HydroBASINS basin identifier layer on the 0.5 degree training grid."
    write_dataset(da.to_dataset(name=basin_name), args.output)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
