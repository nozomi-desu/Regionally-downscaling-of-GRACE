"""Download PO.DAAC GLDAS-Noah v3.3 monthly TWS-anomaly granules.

The collection is aligned to nominal GRACE/GRACE-FO months and supplies a
direct ``TWS_monthly`` variable in millimetres on a 1-degree grid.  Earthdata
credentials are read by ``requests`` from ``~/.netrc`` or from the standard
``EARTHDATA_USERNAME``/``EARTHDATA_PASSWORD`` environment pair.  Credentials
are never accepted as command-line arguments or written to logs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests


COLLECTION_ID = "C2036877565-POCLOUD"
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", default="2002-04-01T00:00:00Z")
    parser.add_argument("--end", default="2022-12-31T23:59:59Z")
    parser.add_argument("--max-files", type=int, help="Download only the first N granules.")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "GRACE-recoverability-revision/1.0"})
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if bool(username) != bool(password):
        raise RuntimeError("Set both EARTHDATA_USERNAME and EARTHDATA_PASSWORD, or neither.")
    if username and password:
        session.auth = (username, password)
    return session


def granule_urls(session: requests.Session, start: str, end: str, timeout: float) -> list[str]:
    response = session.get(
        CMR_URL,
        params={
            "collection_concept_id": COLLECTION_ID,
            "page_size": 2000,
            "temporal": f"{start},{end}",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    entries = response.json().get("feed", {}).get("entry", [])
    urls: list[str] = []
    for entry in entries:
        candidates = [
            link.get("href", "")
            for link in entry.get("links", [])
            if link.get("href", "").startswith(
                "https://archive.podaac.earthdata.nasa.gov/"
            )
            and link.get("href", "").endswith(".nc")
        ]
        if candidates:
            urls.append(candidates[0])
    if not urls:
        raise RuntimeError("CMR returned no PO.DAAC GLDAS-Noah NetCDF granules.")
    return sorted(set(urls))


def validate_netcdf(path: Path) -> None:
    if path.stat().st_size < 100_000:
        raise RuntimeError(f"Downloaded file is unexpectedly small: {path}")
    with path.open("rb") as handle:
        signature = handle.read(8)
    if signature not in {b"\x89HDF\r\n\x1a\n", b"CDF\x01\x00\x00\x00\x00", b"CDF\x02\x00\x00\x00\x00"}:
        raise RuntimeError(f"Downloaded response is not a recognised NetCDF/HDF5 file: {path}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session = build_session()
    urls = granule_urls(session, args.start, args.end, args.timeout)
    if args.max_files is not None:
        urls = urls[: args.max_files]
    for index, url in enumerate(urls, start=1):
        destination = output_dir / Path(url).name
        if destination.exists() and destination.stat().st_size > 100_000:
            validate_netcdf(destination)
            print(f"SKIP {index}/{len(urls)} {destination.name}")
            continue
        response = session.get(url, stream=True, timeout=args.timeout, allow_redirects=True)
        response.raise_for_status()
        if args.check_only:
            print(
                f"ACCESS_OK {index}/{len(urls)} {destination.name} "
                f"bytes={response.headers.get('Content-Length', 'unknown')}"
            )
            response.close()
            continue
        partial = destination.with_suffix(destination.suffix + ".part")
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        response.close()
        validate_netcdf(partial)
        partial.replace(destination)
        print(f"DOWNLOADED {index}/{len(urls)} {destination.name} bytes={destination.stat().st_size}")


if __name__ == "__main__":
    main()

