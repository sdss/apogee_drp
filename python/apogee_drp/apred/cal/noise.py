"""Photon-transfer/noise analysis utilities translated from ``noise.pro``."""

from __future__ import annotations

import numpy as np


def noise(stack, error=None, *, bad=None, bins=40):
    """Measure empirical variance as a function of mean signal.

    Parameters
    ----------
    stack : array-like
        Repeated images with shape ``(nimage, ny, nx)``.
    error : array-like, optional
        Pipeline errors with the same shape.
    bad : array-like, optional
        Two-dimensional bad-pixel mask.
    bins : int or array-like
        Logarithmic bin count or explicit signal edges.

    Returns
    -------
    dict
        Binned signal, variance, difference variance, pixel counts, and
        optional median pipeline variance.
    """
    data = np.asarray(stack, dtype=float)
    if data.ndim != 3 or data.shape[0] < 2:
        raise ValueError("stack must be (nimage, ny, nx) with nimage >= 2")
    mean = np.nanmean(data, axis=0)
    variance = np.nanvar(data, axis=0, ddof=1)
    diff_variance = (data[0] - data[1]) ** 2 / 2.0
    good = np.isfinite(mean) & np.isfinite(variance) & (mean > 0)
    if bad is not None:
        good &= ~np.asarray(bad, dtype=bool)
    if np.isscalar(bins):
        lo, hi = np.nanpercentile(mean[good], [1, 99])
        edges = np.geomspace(max(lo, np.finfo(float).tiny), hi, int(bins) + 1)
    else:
        edges = np.asarray(bins, dtype=float)
    index = np.digitize(mean, edges) - 1
    output = {name: np.full(len(edges) - 1, np.nan) for name in
              ("signal", "variance", "difference_variance")}
    output["npix"] = np.zeros(len(edges) - 1, dtype=int)
    if error is not None:
        errors = np.asarray(error, dtype=float)
        if errors.shape != data.shape:
            raise ValueError("error must have the same shape as stack")
        model_variance = np.nanmedian(errors**2, axis=0)
        output["model_variance"] = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        selected = good & (index == i)
        output["npix"][i] = selected.sum()
        if selected.any():
            output["signal"][i] = np.nanmedian(mean[selected])
            output["variance"][i] = np.nanmedian(variance[selected])
            output["difference_variance"][i] = np.nanmedian(
                diff_variance[selected]
            )
            if error is not None:
                output["model_variance"][i] = np.nanmedian(
                    model_variance[selected]
                )
    output["edges"] = edges
    return output

