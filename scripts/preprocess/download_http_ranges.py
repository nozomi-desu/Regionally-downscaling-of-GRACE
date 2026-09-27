"""Download one public HTTP object with verified parallel byte ranges.

This utility is intended for large, immutable research files served with
``Accept-Ranges: bytes``.  It refuses silent full-object responses to a range
request and verifies the assembled byte count before replacing the target.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import time
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=4)
    return parser.parse_args()


def object_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise RuntimeError("Server does not advertise byte-range support.")
        return int(response.headers["Content-Length"])


def fetch_range(url: str, start: int, end: int, part: Path) -> tuple[int, int]:
    data: bytes | None = None
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=180) as response:
                if response.status != 206:
                    raise RuntimeError(
                        f"Expected HTTP 206 for {start}-{end}; received {response.status}."
                    )
                data = response.read()
            break
        except Exception as error:  # network retries retain already verified parts
            last_error = error
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 16))
    if data is None:
        raise RuntimeError(f"Range {start}-{end} failed: {last_error}")
    expected = end - start + 1
    if len(data) != expected:
        raise RuntimeError(f"Range {start}-{end}: expected {expected} bytes, received {len(data)}.")
    part.write_bytes(data)
    return start, len(data)


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.chunk_mib < 1:
        raise ValueError("--workers and --chunk-mib must be positive.")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    size = object_size(args.url)
    if output.exists() and output.stat().st_size == size:
        print(f"Already complete: {output} ({size} bytes)")
        return

    chunk = args.chunk_mib * 1024 * 1024
    ranges = [(start, min(start + chunk - 1, size - 1)) for start in range(0, size, chunk)]
    part_dir = output.with_name(output.name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[int, int, Path]] = []
    for index, (start, end) in enumerate(ranges):
        part = part_dir / f"{index:05d}.part"
        if not part.exists() or part.stat().st_size != end - start + 1:
            pending.append((start, end, part))
    print(f"Object: {size} bytes; ranges: {len(ranges)}; pending: {len(pending)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch_range, args.url, start, end, part) for start, end, part in pending]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            start, count = future.result()
            print(f"Completed {completed}/{len(pending)}: offset={start}, bytes={count}", flush=True)

    temporary = output.with_name(output.name + ".assembling")
    with temporary.open("wb") as destination:
        for index, (start, end) in enumerate(ranges):
            part = part_dir / f"{index:05d}.part"
            if part.stat().st_size != end - start + 1:
                raise RuntimeError(f"Invalid part before assembly: {part}")
            with part.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    destination.write(block)
    if temporary.stat().st_size != size:
        raise RuntimeError(f"Assembled size mismatch: {temporary.stat().st_size} != {size}")
    os.replace(temporary, output)
    for part in part_dir.glob("*.part"):
        part.unlink()
    part_dir.rmdir()
    print(f"Wrote {output} ({size} bytes)")


if __name__ == "__main__":
    main()
