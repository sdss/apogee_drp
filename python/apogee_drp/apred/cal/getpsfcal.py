"""Select or build the closest PSF calibration."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np


def _parse_psflibrary(output):
    lines = output.splitlines()
    try:
        start = next(i for i, line in enumerate(lines)
                     if line.startswith("PSF FLAT RESULTS:"))
    except StopIteration:
        return None
    for line in lines[start + 1 :]:
        if not line.strip():
            break
        fields = line.split()
        if len(fields) >= 2 and fields[1].isdigit():
            return int(fields[1])
    return None


def getpsfcal(
    num,
    *,
    mjd,
    telescope,
    psf_files=(),
    psflibrary=None,
    exposure_rows=(),
    makecal_func=None,
    verify_files_func=None,
    unlock=False,
):
    """Return the best PSF calibration ID for an exposure.

    Database discovery and path construction are supplied by the caller.  This
    keeps the selection algorithm testable while allowing the production DRP
    to connect its configured database and ``ApLoad`` instance.
    """
    num = int(np.atleast_1d(num)[0])
    fps = int(mjd) >= 59556
    if psflibrary is None:
        psflibrary = fps
    if psflibrary:
        observatory = telescope[:3].lower()
        result = subprocess.run(
            ["psflibrary", observatory, "--ims", str(num)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0:
            selected = _parse_psflibrary(result.stdout)
            if selected is not None:
                return selected

    candidates = []
    for item in psf_files:
        path = Path(item)
        match = re.search(r"PSF-[abc]-([0-9]{8})", path.name, re.IGNORECASE)
        if match:
            candidates.append((int(match.group(1)), item))
    if candidates:
        ids = np.array([item[0] for item in candidates])
        return int(ids[np.argmin(np.abs(ids - num))])

    rows = [
        row for row in exposure_rows
        if str(row["exptype"]).upper() == "QUARTZFLAT"
    ]
    if not rows:
        return -1
    ids = np.asarray([int(row["num"]) for row in rows])
    selected = int(ids[np.argmin(np.abs(ids - num))])
    if makecal_func is None:
        return -1
    makecal_func(psf=selected, unlock=unlock)
    if verify_files_func is not None and not verify_files_func(selected):
        return -1
    return selected

