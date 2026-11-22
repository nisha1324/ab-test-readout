"""Where the data comes from, and the exact file this project was built on.

ASOS Digital Experiments Dataset, version 1 (October 2021).
C. H. Bryan Liu, Ângelo Cardoso, Paul Couturier and Emma J. McCoy,
"Datasets for Online Controlled Experiments", NeurIPS 2021 Datasets and
Benchmarks. https://openreview.net/forum?id=79shW3z5Eaq

Project page: https://osf.io/64jsb/  (DOI 10.17605/OSF.IO/64JSB)
Licence: CC BY 4.0, stated on the project page and in the datasheet.
No key or sign-up is needed.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

PROJECT_URL = "https://osf.io/64jsb/"
CSV_URL = "https://osf.io/download/hq8sc/"
CSV_NAME = "asos_digital_experiments_dataset.csv"
# SHA-256 of the file as downloaded on 2026-11-22. It equals the hash OSF lists for the file.
CSV_SHA256 = "aa0cc3f8304fac545f232ec0bc363373f932919f6e72e28908958bc10651ce22"

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / CSV_NAME


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str = CSV_URL, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ab-test-readout (python urllib)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()
