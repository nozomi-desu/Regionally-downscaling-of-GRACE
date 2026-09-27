"""Build 0.5-degree masks for named large basins from HydroBASINS level 4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import shapefile
import xarray as xr
from shapely import contains_xy, union_all
from shapely.geometry import Point, shape


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
BASINS = {
    "Amazon": {"region": "sa", "outlet_lon": -52.0, "outlet_lat": -1.5},
    "Mississippi": {"region": "na", "outlet_lon": -91.0, "outlet_lat": 30.5},
    "Ganges-Brahmaputra": {"region": "as", "outlet_lon": 90.5, "outlet_lat": 23.5},
    "Yangtze": {"region": "as", "outlet_lon": 121.0, "outlet_lat": 31.0},
    "Nile": {"region": "af", "outlet_lon": 31.25, "outlet_lat": 30.5},
    "Murray-Darling": {"region": "au", "outlet_lon": 139.35, "outlet_lat": -35.55},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hydrobasins-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_region(path: Path) -> tuple[list[dict[str, object]], list[object]]:
    reader = shapefile.Reader(str(path))
    field_names = [field[0] for field in reader.fields[1:]]
    records: list[dict[str, object]] = []
    geometries: list[object] = []
    for shape_record in reader.iterShapeRecords():
        records.append(dict(zip(field_names, shape_record.record, strict=True)))
        geometries.append(shape(shape_record.shape.__geo_interface__))
    return records, geometries


def select_main_basin(
    records: list[dict[str, object]],
    geometries: list[object],
    outlet_lon: float,
    outlet_lat: float,
) -> tuple[object, object, str]:
    point = Point(outlet_lon, outlet_lat)
    nearby = [index for index, geometry in enumerate(geometries) if geometry.distance(point) <= 2.0]
    if nearby:
        selected = max(nearby, key=lambda index: float(records[index].get("UP_AREA", 0.0)))
        method = "maximum_upstream_area_within_2_degrees_of_outlet"
    else:
        selected = min(range(len(geometries)), key=lambda index: geometries[index].distance(point))
        method = "nearest_polygon_fallback"
    main_id = records[selected].get("MAIN_BAS")
    if main_id is None:
        raise KeyError("HydroBASINS MAIN_BAS field is absent.")
    members = [geometry for record, geometry in zip(records, geometries, strict=True) if record.get("MAIN_BAS") == main_id]
    return main_id, union_all(members), method


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    template = xr.open_zarr(args.template.resolve())
    lon_grid, lat_grid = np.meshgrid(template.lon.values, template.lat.values)
    land = np.asarray(template["land_mask"].values, dtype=bool)
    radians = np.pi / 180.0
    radius_km = 6371.0088
    cell_area = (
        radius_km**2
        * radians * 0.5
        * radians * 0.5
        * np.cos(np.deg2rad(lat_grid))
    )
    loaded: dict[str, tuple[list[dict[str, object]], list[object]]] = {}
    masks: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for name, basin in BASINS.items():
        region = str(basin["region"])
        if region not in loaded:
            shp = (
                args.hydrobasins_dir.resolve()
                / f"hybas_{region}_lev04_v1c"
                / f"hybas_{region}_lev04_v1c.shp"
            )
            loaded[region] = load_region(shp)
        records, geometries = loaded[region]
        main_id, basin_geometry, method = select_main_basin(
            records,
            geometries,
            float(basin["outlet_lon"]),
            float(basin["outlet_lat"]),
        )
        mask = contains_xy(basin_geometry, lon_grid, lat_grid) & land
        masks.append(mask.astype(np.uint8))
        bounds = basin_geometry.bounds
        area_km2 = float(np.sum(cell_area[mask]))
        metadata.append(
            {
                "basin": name,
                "region": region,
                "main_bas_id": int(main_id),
                "selection_method": method,
                "outlet_lon": basin["outlet_lon"],
                "outlet_lat": basin["outlet_lat"],
                "grid_cell_count": int(mask.sum()),
                "approx_grid_area_km2": area_km2,
                "bounds": [float(value) for value in bounds],
            }
        )
        logger.info("%s MAIN_BAS=%s cells=%d area=%.0f km2", name, main_id, mask.sum(), area_km2)
    output = xr.Dataset(
        {
            "basin_mask": (("basin", "lat", "lon"), np.stack(masks)),
        },
        coords={"basin": list(BASINS), "lat": template.lat.values, "lon": template.lon.values},
        attrs={
            "protocol_id": "named_hydrobasins_level04_main_basin_masks_v1",
            "source": "HydroBASINS standard v1c level 4",
            "selection": "MAIN_BAS group identified from a predeclared outlet point",
        },
    )
    write_dataset(output, args.output.resolve())
    args.output.with_suffix(".summary.json").write_text(
        json.dumps({"protocol_id": output.attrs["protocol_id"], "basins": metadata}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
