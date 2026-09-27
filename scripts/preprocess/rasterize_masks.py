"""Rasterize vector or raster mask inputs onto the 0.5 degree target grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
from rasterio.transform import from_origin
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import configure_logging, make_target_grid, open_dataset_any, write_dataset  # noqa: E402

TRANSFORM_05 = from_origin(-180.0, 90.0, 0.5, 0.5)
DATASET_CHOICES = [
    "rgi",
    "hydrobasins",
    "human_composite",
    "gmia_aei",
    "gmia_aeigw",
    "gmia_aeisw",
    "gdw_reservoirs",
    "gdw_barriers",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["glacier", "hydrobasins", "human"], required=True)
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        help="Optional source selector. Defaults are rgi for glacier, hydrobasins for hydrobasins, and human_composite for human.",
    )
    parser.add_argument("--input", type=Path, help="Optional vector or raster path override.")
    parser.add_argument("--attribute", help="Optional attribute to burn into the raster.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Defaults depend on --mode.",
    )
    parser.add_argument(
        "--fraction-subgrid",
        type=int,
        default=10,
        help="For glacier polygons, rasterize on a finer grid and average back to 0.5 degree cells.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_dataset(mode: str) -> str:
    """Return the default dataset key for each mode."""
    if mode == "glacier":
        return "rgi"
    if mode == "hydrobasins":
        return "hydrobasins"
    return "human_composite"


def default_output(mode: str, dataset: str) -> Path:
    """Return the default output path for a given mode and dataset."""
    if mode == "glacier" or dataset == "rgi":
        return PROJECT_ROOT / "data_processed" / "adaptive_weights" / "glacier_fraction.nc"
    if mode == "hydrobasins" or dataset == "hydrobasins":
        return PROJECT_ROOT / "data_processed" / "grid_05deg" / "hydrobasins_mask.nc"
    if dataset == "human_composite":
        return PROJECT_ROOT / "data_processed" / "adaptive_weights" / "human_activity_index.nc"
    if dataset == "gmia_aei":
        return PROJECT_ROOT / "data_processed" / "adaptive_weights" / "gmia_irrigated_area_fraction.nc"
    if dataset == "gmia_aeigw":
        return PROJECT_ROOT / "data_processed" / "adaptive_weights" / "gmia_groundwater_irrigation_fraction.nc"
    if dataset == "gmia_aeisw":
        return PROJECT_ROOT / "data_processed" / "adaptive_weights" / "gmia_surfacewater_irrigation_fraction.nc"
    if dataset == "gdw_reservoirs":
        return PROJECT_ROOT / "data_processed" / "adaptive_weights" / "gdw_reservoirs_mask.nc"
    if dataset == "gdw_barriers":
        return PROJECT_ROOT / "data_processed" / "adaptive_weights" / "gdw_barriers_mask.nc"
    return PROJECT_ROOT / "data_processed" / "adaptive_weights" / "human_activity_index.nc"


def default_attribute(dataset: str) -> str | None:
    """Return the default attribute to burn for a dataset when applicable."""
    return {
        "gmia_aei": "PCT_AEI",
        "gmia_aeigw": "PCT_AEIGW",
        "gmia_aeisw": "PCT_AEISW",
    }.get(dataset)


def extract_zip_if_needed(zip_path: Path, extract_dir: Path) -> Path:
    """Extract a zip archive once and return the extraction directory."""
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    shp_present = list(extract_dir.rglob("*.shp"))
    if shp_present or any(extract_dir.iterdir() if extract_dir.exists() else []):
        return extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def prepare_rgi_paths() -> list[Path]:
    """Extract the RGI global bundle into regional shapefile directories and return all shapefiles."""
    rgi_root = PROJECT_ROOT / "data_raw" / "rgi_glacier"
    global_zip = rgi_root / "RGI2000-v7.0-G-global.zip"
    outer_dir = rgi_root / "RGI2000-v7.0-G-global"
    extract_zip_if_needed(global_zip, outer_dir)

    regional_shps = sorted(outer_dir.rglob("*.shp"))
    if regional_shps:
        return regional_shps

    inner_zips = sorted(outer_dir.glob("RGI2000-v7.0-G-*.zip"))
    if not inner_zips:
        raise FileNotFoundError("No regional RGI zip archives were found after extracting the global bundle.")
    for inner_zip in inner_zips:
        region_dir = outer_dir / inner_zip.stem
        extract_zip_if_needed(inner_zip, region_dir)
    regional_shps = sorted(outer_dir.rglob("*.shp"))
    if not regional_shps:
        raise FileNotFoundError("No RGI regional shapefiles were found after nested extraction.")
    return regional_shps


def prepare_hydrobasins_paths() -> list[Path]:
    """Return all preferred HydroBASINS level-06 shapefiles across available domains."""
    hydro_root = PROJECT_ROOT / "data_raw" / "hydrobasins"
    matches = sorted(hydro_root.rglob("*lev06*.shp"))
    if not matches:
        matches = sorted(hydro_root.rglob("*.shp"))
    if not matches:
        raise FileNotFoundError("No HydroBASINS shapefiles were found.")
    return matches


def prepare_gmia_paths(dataset: str) -> list[Path]:
    """Extract the GMIA shapefile bundle and return the target shapefile path."""
    gmia_root = PROJECT_ROOT / "data_raw" / "human_activity" / "irrigation_gmia"
    extract_dir = gmia_root / "gmia_v5_shp"
    extract_zip_if_needed(gmia_root / "gmia_v5_shp.zip", extract_dir)
    shapefile_name = {
        "gmia_aei": "gmia_v5_aei_pct_cellarea.shp",
        "gmia_aeigw": "gmia_v5_aeigw_pct_aei.shp",
        "gmia_aeisw": "gmia_v5_aeisw_pct_aei.shp",
    }[dataset]
    path = extract_dir / shapefile_name
    if not path.exists():
        raise FileNotFoundError(path)
    return [path]


def prepare_gdw_paths(dataset: str) -> list[Path]:
    """Extract the GDW shapefile bundle and return the selected shapefile path."""
    gdw_root = PROJECT_ROOT / "data_raw" / "human_activity" / "reservoirs_grand"
    extract_dir = gdw_root / "GDW_v1_0_shp"
    extract_zip_if_needed(gdw_root / "GDW_v1_0_shp.zip", extract_dir)
    shapefile_name = {
        "gdw_reservoirs": "GDW_reservoirs_v1_0.shp",
        "gdw_barriers": "GDW_barriers_v1_0.shp",
    }[dataset]
    path = extract_dir / "GDW_v1_0_shp" / shapefile_name
    if not path.exists():
        raise FileNotFoundError(path)
    return [path]


def resolve_default_inputs(mode: str, dataset: str) -> list[Path]:
    """Resolve one or more default input paths for a mode/dataset pair."""
    if dataset == "human_composite":
        return []
    if dataset == "rgi":
        return prepare_rgi_paths()
    if dataset == "hydrobasins":
        return prepare_hydrobasins_paths()
    if dataset in {"gmia_aei", "gmia_aeigw", "gmia_aeisw"}:
        return prepare_gmia_paths(dataset)
    if dataset in {"gdw_reservoirs", "gdw_barriers"}:
        return prepare_gdw_paths(dataset)
    raise ValueError(f"Unsupported dataset selector: {dataset}")


def latlon_from_array(array: np.ndarray) -> xr.DataArray:
    """Build a DataArray from a rasterized global array."""
    lat_desc = np.arange(89.75, -90.0, -0.5)
    lon = np.arange(-179.75, 180.0, 0.5)
    da = xr.DataArray(array, dims=("lat", "lon"), coords={"lat": lat_desc, "lon": lon})
    return da.sortby("lat")


def rasterize_fraction(gdf: gpd.GeoDataFrame, factor: int) -> xr.DataArray:
    """Approximate per-cell polygon coverage fraction using a finer subgrid."""
    fine_res = 0.5 / factor
    fine_transform = from_origin(-180.0, 90.0, fine_res, fine_res)
    height = int(180.0 / fine_res)
    width = int(360.0 / fine_res)
    fine = rasterio.features.rasterize(
        [(geom, 1) for geom in gdf.geometry if geom and not geom.is_empty],
        out_shape=(height, width),
        transform=fine_transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    coarse = fine.reshape(360, factor, 720, factor).mean(axis=(1, 3)).astype("float32")
    return latlon_from_array(coarse)


def rasterize_simple(gdf: gpd.GeoDataFrame, attribute: str | None) -> xr.DataArray:
    """Rasterize geometries directly to the 0.5 degree grid."""
    shapes = []
    for _, row in gdf.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        value = 1 if attribute is None else row.get(attribute, 1)
        shapes.append((row.geometry, value))
    array = rasterio.features.rasterize(
        shapes,
        out_shape=(360, 720),
        transform=TRANSFORM_05,
        fill=0,
        all_touched=True,
        dtype="float32",
    )
    return latlon_from_array(array)


def read_vector_inputs(paths: list[Path]) -> gpd.GeoDataFrame:
    """Read and concatenate one or more vector inputs."""
    frames = []
    for path in paths:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")
        frames.append(gdf)
    if not frames:
        raise FileNotFoundError("No vector inputs were available to read.")
    if len(frames) == 1:
        return frames[0]
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


def raster_input_to_grid(input_path: Path) -> xr.DataArray:
    """Resample an existing raster-like file onto the 0.5 degree grid."""
    ds = open_dataset_any(input_path)
    variable = next(iter(ds.data_vars))
    target = make_target_grid()
    return ds[variable].interp(lat=target["lat"], lon=target["lon"], method="nearest")


def rasterize_dataset(mode: str, dataset: str, attribute: str | None, factor: int, input_override: Path | None = None) -> xr.DataArray:
    """Rasterize one dataset selector to the common grid and return its DataArray."""
    if input_override is not None:
        input_paths = [input_override]
    else:
        input_paths = resolve_default_inputs(mode, dataset)

    if len(input_paths) == 1 and input_paths[0].suffix.lower() in {".tif", ".tiff", ".asc", ".nc", ".nc4"}:
        return raster_input_to_grid(input_paths[0])

    gdf = read_vector_inputs(input_paths)
    if mode == "glacier":
        out = rasterize_fraction(gdf, factor)
        out.name = "glacier_fraction"
        return out
    out = rasterize_simple(gdf, attribute)
    if mode == "hydrobasins":
        out.name = attribute or "hydrobasins_mask"
        out.attrs["description"] = "HydroBASINS coverage rasterized onto the shared 0.5 degree grid."
        out.attrs["source_count"] = len(input_paths)
    elif dataset == "gmia_aei":
        out.name = "gmia_irrigated_area_fraction"
    elif dataset == "gmia_aeigw":
        out.name = "gmia_groundwater_irrigation_fraction"
    elif dataset == "gmia_aeisw":
        out.name = "gmia_surfacewater_irrigation_fraction"
    elif dataset == "gdw_reservoirs":
        out.name = "gdw_reservoirs_mask"
    elif dataset == "gdw_barriers":
        out.name = "gdw_barriers_mask"
    else:
        out.name = attribute or input_paths[0].stem
    return out


def build_human_composite(factor: int) -> xr.DataArray:
    """Combine irrigated-area intensity and reservoir presence into one human-activity index."""
    gmia = rasterize_dataset("human", "gmia_aei", default_attribute("gmia_aei"), factor)
    gdw = rasterize_dataset("human", "gdw_reservoirs", None, factor)
    gmia_scaled = (gmia / 100.0).clip(0, 1).fillna(0)
    gdw_scaled = gdw.clip(0, 1).fillna(0)
    fused = (0.7 * gmia_scaled + 0.3 * gdw_scaled).clip(0, 1)
    fused.name = "human_activity_index"
    fused.attrs["description"] = (
        "Composite human-activity index from GMIA irrigated-area fraction (70%) "
        "and GDW reservoir presence mask (30%)."
    )
    fused.attrs["components"] = "GMIA PCT_AEI, GDW reservoirs"
    return fused


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger = configure_logging()
    dataset = args.dataset or default_dataset(args.mode)
    output_path = args.output or default_output(args.mode, dataset)
    attribute = args.attribute if args.attribute is not None else default_attribute(dataset)
    if output_path.exists() and not args.overwrite:
        logger.info("Output already exists: %s", output_path)
        return

    if dataset == "human_composite":
        logger.info("Using dataset=%s with fused GMIA + GDW inputs.", dataset)
        out = build_human_composite(args.fraction_subgrid)
    else:
        if args.input is not None:
            input_paths = [args.input]
        else:
            input_paths = resolve_default_inputs(args.mode, dataset)
        logger.info("Using dataset=%s with %d input path(s).", dataset, len(input_paths))
        out = rasterize_dataset(args.mode, dataset, attribute, args.fraction_subgrid, args.input)

    if out.name is None:
        out.name = dataset
    write_dataset(out.to_dataset(name=out.name), output_path)
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
