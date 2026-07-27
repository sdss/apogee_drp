"""APOGEE calibration builders and shared utilities.

Major calibration builders remain in their own modules.  Smaller routines are
organized by purpose in :mod:`index`, :mod:`detector`, :mod:`diagnostics`, and
:mod:`utils`.
"""

from .detector import aplincorr
from .index import getcal, getnums, readcal, readcalstr
from .utils import flatsmooth, robust_slope


__all__ = [
    "aplincorr",
    "flatsmooth",
    "getcal",
    "getnums",
    "readcal",
    "readcalstr",
    "robust_slope",
]
