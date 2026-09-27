"""Shared helpers for the GRACE data download and preprocessing pipeline."""

from __future__ import annotations

import csv
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
import xarray as xr
import yaml
from tqdm import tqdm

LOGGER_NAME = "grace_pipeline"
INVENTORY_COLUMNS = [
    "dataset_id",
    "status",
    "local_path",
    "download_date",
    "time_range",
    "spatial_resolution",
    "notes",
]
DEFAULT_RESOLUTION = 0.5
DEFAULT_LON = np.arange(-179.75, 180.0, DEFAULT_RESOLUTION)
DEFAULT_LAT = np.arange(-89.75, 90.0, DEFAULT_RESOLUTION)


def project_root() -> Path:
    """Return the repository root inferred from this file location."""
    return Path(__file__).resolve().parents[1]


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return a shared logger."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if needed and return it as a Path."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def read_yaml(path: Path | str) -> dict:
    """Read a YAML file into a Python dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def append_csv_row(path: Path | str, row: dict[str, object]) -> None:
    """Append one row to a CSV file, writing the header if needed."""
    csv_path = Path(path)
    ensure_dir(csv_path.parent)
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def update_inventory(
    dataset_id: str,
    status: str,
    local_path: str | Path = "",
    download_date: str | None = None,
    time_range: str = "",
    spatial_resolution: str = "",
    notes: str = "",
    inventory_path: Path | str | None = None,
) -> None:
    """Insert or update one dataset record in metadata/data_inventory.csv."""
    inventory = Path(inventory_path or project_root() / "metadata" / "data_inventory.csv")
    ensure_dir(inventory.parent)
    if inventory.exists():
        frame = pd.read_csv(inventory)
    else:
        frame = pd.DataFrame(columns=INVENTORY_COLUMNS)
    for column in INVENTORY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
        else:
            frame[column] = frame[column].astype("object")
    row = {
        "dataset_id": dataset_id,
        "status": status,
        "local_path": str(local_path) if local_path else "",
        "download_date": download_date or datetime.utcnow().date().isoformat(),
        "time_range": time_range,
        "spatial_resolution": spatial_resolution,
        "notes": notes,
    }
    if "dataset_id" in frame.columns and dataset_id in frame["dataset_id"].values:
        frame.loc[frame["dataset_id"] == dataset_id, INVENTORY_COLUMNS] = list(row.values())
    else:
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame.to_csv(inventory, index=False)


def find_existing_paths(patterns: Iterable[str]) -> list[Path]:
    """Resolve glob patterns relative to the project root and return existing paths."""
    root = project_root()
    matches: list[Path] = []
    for pattern in patterns:
        for candidate in root.glob(pattern):
            if candidate.exists():
                matches.append(candidate.resolve())
    deduped = []
    seen = set()
    for item in matches:
        item_key = str(item)
        if item_key not in seen:
            deduped.append(item)
            seen.add(item_key)
    return deduped


def maybe_copy_to_target(source: Path, target: Path, overwrite: bool = False) -> Path:
    """Copy a file into the target location if requested and needed."""
    ensure_dir(target.parent)
    if target.exists() and not overwrite:
        return target
    shutil.copy2(source, target)
    return target


def download_file(
    url: str,
    destination: Path,
    timeout: int = 120,
    overwrite: bool = False,
    headers: dict[str, str] | None = None,
) -> Path:
    """Download one file with streamed progress reporting."""
    ensure_dir(destination.parent)
    if destination.exists() and not overwrite and destination.stat().st_size > 0:
        return destination
    response = requests.get(url, stream=True, timeout=timeout, headers=headers)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with destination.open("wb") as handle, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=destination.name,
    ) as progress:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if not chunk:
                continue
            handle.write(chunk)
            progress.update(len(chunk))
    return destination


def scrape_links(
    page_url: str,
    include_patterns: Iterable[str] | None = None,
    suffixes: Iterable[str] | None = None,
    timeout: int = 60,
) -> list[str]:
    """Scrape candidate links from an HTML page."""
    response = requests.get(page_url, timeout=timeout)
    response.raise_for_status()
    html = response.text
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    links = []
    normalized_patterns = [p.lower() for p in include_patterns or []]
    normalized_suffixes = [s.lower() for s in suffixes or []]
    for href in hrefs:
        absolute = urljoin(page_url, href)
        href_lower = absolute.lower()
        suffix_ok = not normalized_suffixes or any(
            href_lower.endswith(suffix) for suffix in normalized_suffixes
        )
        pattern_ok = not normalized_patterns or all(pattern in href_lower for pattern in normalized_patterns)
        if suffix_ok and pattern_ok:
            links.append(absolute)
    deduped = []
    seen = set()
    for link in links:
        if link not in seen:
            deduped.append(link)
            seen.add(link)
    return deduped


def write_manual_instructions(path: Path | str, title: str, lines: Iterable[str]) -> Path:
    """Write a Markdown note with manual-download instructions."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    content = [f"# {title}", ""]
    content.extend(lines)
    path_obj.write_text("\n".join(content).strip() + "\n", encoding="utf-8")
    return path_obj


def has_earthdata_credentials() -> bool:
    """Return True if Earthdata auth is likely configured locally."""
    netrc_path = Path.home() / ".netrc"
    return (
        netrc_path.exists()
        or bool(os.environ.get("EARTHDATA_USERNAME"))
        or bool(os.environ.get("EARTHDATA_PASSWORD"))
    )


def has_cds_credentials() -> bool:
    """Return True if CDS auth is likely configured locally."""
    cdsapirc = Path.home() / ".cdsapirc"
    return cdsapirc.exists() or bool(os.environ.get("CDSAPI_KEY"))


def earthaccess_search_and_download(
    output_dir: Path | str,
    short_names: Iterable[str] | None = None,
    keywords: Iterable[str] | None = None,
    temporal: tuple[str, str] | None = None,
    version: str | None = None,
    count: int = 200,
) -> list[Path]:
    """Search and download NASA Earthdata granules with earthaccess."""
    import earthaccess

    output_dir = ensure_dir(output_dir)
    earthaccess.login(strategy="netrc")
    results = []
    for short_name in short_names or []:
        kwargs: dict[str, object] = {"short_name": short_name, "count": count}
        if temporal:
            kwargs["temporal"] = temporal
        if version:
            kwargs["version"] = version
        try:
            results = earthaccess.search_data(**kwargs)
        except Exception:
            continue
        if results:
            break
    if not results:
        for keyword in keywords or []:
            try:
                datasets = earthaccess.search_datasets(keyword=keyword, count=10)
            except Exception:
                continue
            candidate_short_names = []
            for dataset in datasets:
                short_name = dataset.get("umm", {}).get("ShortName")
                if short_name and short_name not in candidate_short_names:
                    candidate_short_names.append(short_name)
            for short_name in candidate_short_names:
                kwargs = {"short_name": short_name, "count": count}
                if temporal:
                    kwargs["temporal"] = temporal
                if version:
                    kwargs["version"] = version
                try:
                    results = earthaccess.search_data(**kwargs)
                except Exception:
                    continue
                if results:
                    break
            if results:
                break
    if not results:
        return []
    earthaccess.download(results, local_path=str(output_dir))
    return sorted(path.resolve() for path in output_dir.glob("*") if path.is_file())


def guess_coord_name(dataset: xr.Dataset | xr.DataArray, candidates: Iterable[str]) -> str:
    """Return the first matching coordinate or dimension name."""
    all_names = list(dataset.coords) + list(dataset.dims)
    for candidate in candidates:
        if candidate in all_names:
            return candidate
    lowered = {name.lower(): name for name in all_names}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise KeyError(f"Could not find any of {list(candidates)} in dataset.")


def standardize_lat_lon(
    ds: xr.Dataset | xr.DataArray,
    lon_name: str | None = None,
    lat_name: str | None = None,
) -> xr.Dataset | xr.DataArray:
    """Rename, normalize, and sort latitude and longitude coordinates."""
    lon_name = lon_name or guess_coord_name(ds, ["lon", "longitude", "x"])
    lat_name = lat_name or guess_coord_name(ds, ["lat", "latitude", "y"])
    renamed = ds.rename({lon_name: "lon", lat_name: "lat"})
    if renamed["lon"].max() > 180:
        lon = ((renamed["lon"] + 180) % 360) - 180
        renamed = renamed.assign_coords(lon=lon).sortby("lon")
    renamed = renamed.sortby("lat").sortby("lon")
    return renamed


def standardize_time(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Rename the time coordinate to `time` if needed."""
    try:
        time_name = guess_coord_name(ds, ["time", "valid_time", "month", "date"])
    except KeyError:
        return ds
    if time_name != "time":
        ds = ds.rename({time_name: "time"})
    return ds


def normalize_monthly_time_values(values: Iterable[object]) -> np.ndarray:
    """Convert datetime-like monthly values to month-start numpy datetime64 values."""
    normalized = []
    for value in values:
        if hasattr(value, "year") and hasattr(value, "month"):
            normalized.append(np.datetime64(f"{int(value.year):04d}-{int(value.month):02d}-01"))
        else:
            ts = pd.Timestamp(value)
            normalized.append(np.datetime64(f"{ts.year:04d}-{ts.month:02d}-01"))
    return np.asarray(normalized, dtype="datetime64[ns]")


def open_dataset_any(path: Path | str) -> xr.Dataset:
    """Open NetCDF, Zarr, or raster data as an xarray Dataset."""
    path_obj = Path(path)
    suffixes = "".join(path_obj.suffixes).lower()
    if path_obj.is_dir() and path_obj.suffix.lower() == ".zarr":
        ds = xr.open_zarr(path_obj)
    elif suffixes.endswith(".tif") or suffixes.endswith(".tiff"):
        import rioxarray  # noqa: F401

        da = xr.open_dataarray(path_obj, engine="rasterio")
        if "band" in da.dims and da.sizes.get("band", 1) == 1:
            da = da.squeeze("band", drop=True)
        ds = da.to_dataset(name=path_obj.stem)
    else:
        ds = xr.open_dataset(path_obj)
    ds = standardize_lat_lon(standardize_time(ds))
    return ds


def guess_data_var(
    ds: xr.Dataset,
    preferred: str | None = None,
    aliases: Iterable[str] | None = None,
) -> str:
    """Guess the most relevant data variable name."""
    if preferred and preferred in ds.data_vars:
        return preferred
    for alias in aliases or []:
        if alias in ds.data_vars:
            return alias
    lowered = {name.lower(): name for name in ds.data_vars}
    for alias in aliases or []:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    raise KeyError(
        f"Could not infer data variable. Available variables: {list(ds.data_vars)}"
    )


def make_target_grid() -> xr.Dataset:
    """Build the canonical 0.5 degree target grid."""
    return xr.Dataset(coords={"lat": DEFAULT_LAT, "lon": DEFAULT_LON})


def write_dataset(ds: xr.Dataset, output_path: Path | str) -> Path:
    """Write a dataset to NetCDF or Zarr based on the output suffix."""
    output = Path(output_path)
    ensure_dir(output.parent)
    ds_to_write = ds.copy(deep=False)
    for name in ds_to_write.variables:
        ds_to_write[name].encoding = {}
    if output.suffix.lower() == ".zarr":
        ds_to_write.to_zarr(output, mode="w", consolidated=True)
    else:
        temp_output = output.with_name(f"{output.stem}.tmp{output.suffix}")
        if temp_output.exists():
            temp_output.unlink()
        ds_to_write.to_netcdf(temp_output)
        temp_output.replace(output)
    return output


def infer_time_range(ds: xr.Dataset) -> str:
    """Create a human-readable time range string."""
    if "time" not in ds.coords:
        return ""
    values = normalize_monthly_time_values(ds["time"].values)
    if len(values) == 0:
        return ""
    values = pd.to_datetime(values)
    return f"{values.min():%Y-%m} to {values.max():%Y-%m}"


def infer_grid_resolution(ds: xr.Dataset) -> str:
    """Estimate regular latitude and longitude spacing."""
    if "lat" not in ds.coords or "lon" not in ds.coords:
        return ""
    lat_values = np.asarray(ds["lat"].values)
    lon_values = np.asarray(ds["lon"].values)
    if lat_values.size < 2 or lon_values.size < 2:
        return ""
    lat_res = float(np.round(np.abs(np.diff(lat_values)).mean(), 6))
    lon_res = float(np.round(np.abs(np.diff(lon_values)).mean(), 6))
    return f"{lat_res} x {lon_res} degree"


def filename_from_url(url: str) -> str:
    """Return a reasonable filename for a URL."""
    parsed = urlparse(url)
    name = Path(parsed.path).name
    return name or "download.bin"
