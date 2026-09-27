"""Verify the published file inventory using only the Python standard library."""
from pathlib import Path
import csv
import hashlib

def main():
    root = Path(__file__).resolve().parents[1]
    rows = list(csv.DictReader((root/'provenance/published_files.csv').open(encoding='utf-8', newline='')))
    failures = []
    for row in rows:
        path = root/row['path']
        if not path.is_file() or path.stat().st_size != int(row['bytes']):
            failures.append(row['path'])
            continue
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block)
        if h.hexdigest() != row['sha256']:
            failures.append(row['path'])
    if failures:
        raise SystemExit('Verification failed: '+', '.join(failures))
    print(f'PASS: {len(rows)} published files match their SHA-256 inventory.')

if __name__ == '__main__':
    main()
