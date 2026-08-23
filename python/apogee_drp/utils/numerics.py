"""Numerical helpers shared across APOGEE reduction stages."""

import numpy as np


def median_absolute_deviation(values):
    """Return the median absolute deviation of finite values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    center = np.median(values)
    return float(np.median(np.abs(values - center)))


def robust_polyfit(x, y, degree, maxiter=5, clip=5.0):
    """Fit a polynomial with iterative MAD clipping.

    Coefficients use increasing-power ``numpy.polynomial`` ordering.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if isinstance(degree, (bool, np.bool_)) or int(degree) != degree or degree < 0:
        raise ValueError("degree must be a non-negative integer")
    if isinstance(maxiter, (bool, np.bool_)) or int(maxiter) != maxiter or maxiter < 0:
        raise ValueError("maxiter must be a non-negative integer")
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be positive and finite")
    degree, maxiter = int(degree), int(maxiter)
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) <= degree:
        raise ValueError("too few finite points for polynomial fit")
    for _ in range(maxiter):
        coefficients = np.polynomial.polynomial.polyfit(x[good], y[good], degree)
        residual = y - np.polynomial.polynomial.polyval(x, coefficients)
        center = np.nanmedian(residual[good])
        scatter = np.nanmedian(np.abs(residual[good] - center))
        if not np.isfinite(scatter) or scatter == 0:
            break
        keep = good & (np.abs(residual - center) <= clip * scatter)
        if np.count_nonzero(keep) <= degree or np.array_equal(keep, good):
            break
        good = keep
    return np.polynomial.polynomial.polyfit(x[good], y[good], degree)


__all__ = ["median_absolute_deviation", "robust_polyfit"]
