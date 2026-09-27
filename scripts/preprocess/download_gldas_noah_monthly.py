"""Download monthly GLDAS Noah 2.1 files without exposing credentials.

NASA GES DISC data require an Earthdata account.  Authentication is read from
``~/.netrc`` (recommended) or the ``EARTHDATA_USERNAME`` and
``EARTHDATA_PASSWORD`` environment variables.  Passwords are never accepted on
the command line or written to logs.
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

import requests


DEFAULT_BASE_URL = (
    "https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_M.2.1"
)


def parse_month(value: str) -> date:
    try:
        year, month = (int(part) for part in value.split("-"))
        return date(year, month, 1)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Month must use YYYY-MM format.") from exc


def iter_months(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current = date(current.year + (current.month == 12), 1 if current.month == 12 else current.month + 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=parse_month, default=parse_month("2002-04"))
    parser.add_argument("--end", type=parse_month, default=parse_month("2022-12"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-files", type=int, help="Download/check only the first N files.")
    parser.add_argument("--check-only", action="store_true", help="Verify access without downloading bodies.")
    parser.add_argument("--timeout", type=float, default=120.0)
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


def remote_url(base_url: str, month: date) -> str:
    filename = f"GLDAS_NOAH025_M.A{month:%Y%m}.021.nc4"
    return f"{base_url.rstrip('/')}/{month:%Y}/{filename}"


def validate_response(response: requests.Response, url: str) -> None:
    if response.status_code == 401:
        raise RuntimeError(
            "Earthdata authorization failed (HTTP 401). Configure ~/.netrc for "
            "urs.earthdata.nasa.gov and run the access check again."
        )
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" in content_type:
        raise RuntimeError(f"Received an HTML login/error page instead of NetCDF: {url}")


def main() -> None:
    args = parse_args()
    if args.start > args.end:
        raise ValueError("--start must not be after --end.")
    months = list(iter_months(args.start, args.end))
    if args.max_files is not None:
        months = months[: args.max_files]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session = build_session()

    for index, month in enumerate(months, start=1):
        url = remote_url(args.base_url, month)
        destination = output_dir / Path(url).name
        if args.check_only:
            response = session.get(url, stream=True, timeout=args.timeout, allow_redirects=True)
            validate_response(response, url)
            expected = response.headers.get("Content-Length", "unknown")
            response.close()
            print(f"ACCESS_OK {index}/{len(months)} {month:%Y-%m} bytes={expected}")
            continue
        if destination.exists() and destination.stat().st_size > 1_000_000:
            print(f"SKIP {index}/{len(months)} {destination.name}")
            continue
        partial = destination.with_suffix(destination.suffix + ".part")
        response = session.get(url, stream=True, timeout=args.timeout, allow_redirects=True)
        validate_response(response, url)
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        response.close()
        if partial.stat().st_size < 1_000_000:
            raise RuntimeError(f"Downloaded file is unexpectedly small: {partial}")
        partial.replace(destination)
        print(f"DOWNLOADED {index}/{len(months)} {destination.name} bytes={destination.stat().st_size}")


if __name__ == "__main__":
    main()

