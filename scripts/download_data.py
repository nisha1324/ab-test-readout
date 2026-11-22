"""Download the ASOS Digital Experiments Dataset into data/raw/ and verify it.

The CSV (2.7 MB, CC BY 4.0) is already committed, so this is only needed to
check the committed copy against the source, or to fetch it again.
The SHA-256 is pinned in abtest/source.py; a file that differs stops here
instead of silently changing the analysis.

    python scripts/download_data.py            # verify the copy on disk (fetch if missing)
    python scripts/download_data.py --force    # download again and verify
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abtest import source  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = ap.parse_args()
    path = source.RAW_PATH
    if path.exists() and not args.force:
        data, how = path.read_bytes(), "on disk"
    else:
        data, how = source.fetch(), "downloaded"
    digest = source.sha256(data)
    if digest != source.CSV_SHA256:
        if how == "downloaded":
            path.with_suffix(".csv.part").write_bytes(data)
        print(f"MISMATCH  {path.name}: expected {source.CSV_SHA256[:12]}…, got {digest[:12]}… ({how})",
              file=sys.stderr)
        return 1
    if how == "downloaded":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(f"ok  {path.name}  {len(data):,} bytes  ({how})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
