"""Download and extract official HydroBASINS v1c level-4 continent files."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests


BASE_URL = "https://data.hydrosheds.org/file/hydrobasins/standard"
VALID_REGIONS = ("af", "as", "au", "eu", "na", "sa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--regions", nargs="+", choices=VALID_REGIONS, default=list(VALID_REGIONS))
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "GRACE-recoverability-revision/1.0"
    for region in args.regions:
        filename = f"hybas_{region}_lev04_v1c.zip"
        destination = output_dir / filename
        if not destination.exists() or destination.stat().st_size < 100_000:
            response = session.get(f"{BASE_URL}/{filename}", timeout=args.timeout)
            response.raise_for_status()
            destination.write_bytes(response.content)
            print(f"DOWNLOADED {filename} bytes={destination.stat().st_size}")
        extract_dir = output_dir / destination.stem
        marker = extract_dir / f"hybas_{region}_lev04_v1c.shp"
        if not marker.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(destination) as archive:
                archive.extractall(extract_dir)
            print(f"EXTRACTED {filename} to {extract_dir}")
        else:
            print(f"READY {marker}")


if __name__ == "__main__":
    main()
