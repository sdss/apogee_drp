"""Linearity correction for raw APOGEE detector slices.

This is a direct translation of ``aplincorr.pro``.  The historical IDL
routine only writes the first read because ``slice_out[0:2047]`` addresses the
first column in IDL.  That behavior is available as ``legacy=True`` for exact
regression work; the default applies the calculated correction to every read.
"""

from __future__ import annotations

import numpy as np


def aplincorr(slice_in, lindata, *, legacy=False):
    """Apply an APOGEE polynomial linearity correction.

    Parameters
    ----------
    slice_in : array-like
        Detector slice with shape ``(2048, nreads)``.
    lindata : array-like
        Coefficients with shape ``(4, ncoeff)``; one polynomial per
        512-column detector output.
    legacy : bool, optional
        Reproduce the IDL assignment bug and correct only read zero.

    Returns
    -------
    numpy.ndarray
        Corrected slice, with floating-point dtype.
    """
    data = np.asarray(slice_in)
    coef = np.asarray(lindata)
    if data.ndim != 2 or data.shape[0] != 2048:
        raise ValueError("slice_in must have shape (2048, nreads)")
    if data.shape[1] < 3:
        raise ValueError("aplincorr requires at least three reads")
    if coef.ndim != 2 or coef.shape[0] != 4:
        raise ValueError("lindata must have shape (4, ncoeff)")

    work = data.astype(np.result_type(data, coef, np.float32), copy=False)
    corr = np.empty_like(work)
    nreads = work.shape[1]

    for quadrant in range(4):
        slc = slice(512 * quadrant, 512 * (quadrant + 1))
        level = work[slc].copy()
        reads = np.arange(2, nreads)
        level[:, reads] = (
            (level[:, reads] - level[:, [1]])
            * (reads + 1.0)[None, :]
            / (reads - 1.0)[None, :]
        )
        level[~np.isfinite(level)] = 0.0

        qcorr = np.full_like(level, coef[quadrant, 0])
        term = level.copy()
        for power in range(1, coef.shape[1]):
            qcorr += coef[quadrant, power] * term
            term *= level
        qcorr[:, :2] = qcorr[:, [2]]
        corr[slc] = qcorr

    out = work.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        if legacy:
            out[:, 0] = work[:, 0] / corr[:, 0]
        else:
            out = work / corr
    return out

