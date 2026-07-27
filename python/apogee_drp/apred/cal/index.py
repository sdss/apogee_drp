"""Master calibration-index parsing and nightly selection.

This is the public home for the former ``readcal``, ``readcalstr``, and
``getcal`` IDL routines.  The main parser remains in :mod:`apogee_drp.apred.mkcal`
for compatibility with existing DRP callers.
"""

from __future__ import annotations

import numpy as np


def readcal(*args, **kwargs):
    """Read the master calibration parameter file."""
    from apogee_drp.apred.mkcal import readcal as implementation
    return implementation(*args, **kwargs)


def getcal(*args, **kwargs):
    """Return the calibration products valid for an MJD."""
    from apogee_drp.apred.mkcal import getcal as implementation
    return implementation(*args, **kwargs)


def getnums(*args, **kwargs):
    """Expand comma-separated frame numbers and inclusive ranges."""
    from apogee_drp.apred.mkcal import getnums as implementation
    return implementation(*args, **kwargs)


def readcalstr(records, mjd, *, verbose=True):
    """Return the last calibration name whose inclusive MJD range matches."""
    if records is None or len(records) == 0:
        return 0
    names = records.dtype.names
    if names is None or not {"mjd1", "mjd2", "name"}.issubset(names):
        raise ValueError("records must have mjd1, mjd2, and name fields")
    selected = np.flatnonzero(
        (mjd >= records["mjd1"]) & (mjd <= records["mjd2"])
    )
    if selected.size == 0:
        return 0
    index = int(selected[-1])
    value = records["name"][index]
    if selected.size > 1 and verbose:
        print(
            f"Multiple cal products found for mjd {mjd}; "
            f"will use last: {value}"
        )
    if isinstance(value, bytes):
        value = value.decode().strip()
    if isinstance(value, str):
        value = value.strip()
        try:
            return int(value)
        except ValueError:
            return value
    return value.item() if isinstance(value, np.generic) else value


__all__ = ["getcal", "getnums", "readcal", "readcalstr"]

