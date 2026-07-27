"""Robust slope estimator translated from ``robust_slope.pro``."""

from __future__ import annotations

import numpy as np


def robust_slope(x, y):
    """Estimate a slope from opposite quartiles of the data.

    The algorithm intentionally follows the IDL routine rather than replacing
    it with a different robust-regression estimator.
    """
    xx = np.asarray(x, dtype=float).ravel()
    yy = np.asarray(y, dtype=float).ravel()
    if xx.size != yy.size:
        raise ValueError("x and y must contain the same number of values")
    good = np.isfinite(xx) & np.isfinite(yy)
    xx, yy = xx[good], yy[good]
    if xx.size < 4:
        raise ValueError("robust_slope requires at least four finite points")

    order = np.argsort(xx, kind="stable")
    n = xx.size
    half, quarter = n // 2, n // 4
    first = order[:quarter]
    fourth = order[n - quarter - 1 :]
    x1, y1 = np.median(xx[first]), np.median(yy[first])
    x4, y4 = np.median(xx[fourth]), np.median(yy[fourth])

    with np.errstate(divide="ignore", invalid="ignore"):
        slope34 = (yy[order[half:]] - y1) / (xx[order[half:]] - x1)
        slope12 = (yy[order[:half]] - y4) / (xx[order[:half]] - x4)
    slopes = np.concatenate((slope12, slope34))
    slopes = slopes[np.isfinite(slopes)]
    return float(np.median(slopes)) if slopes.size else np.nan

