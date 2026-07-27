"""Large-scale flat-field smoothing translated from ``flatsmooth.pro``."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from .robust_slope import robust_slope


def flatsmooth(
    image,
    *,
    xstep=32,
    xbin=101,
    ystep=32,
    ybin=101,
    lobad=0.1,
    hibad=1.5,
):
    """Fit local planes and interpolate their centers across an image.

    Array axes follow NumPy convention ``(y, x)``.  Edge regions are allowed
    to shrink, fixing an implicit fixed-size assumption in the IDL code.
    """
    im = np.asarray(image, dtype=float)
    if im.ndim != 2:
        raise ValueError("image must be two-dimensional")
    nyim, nxim = im.shape
    xstep, ystep, xbin, ybin = map(int, (xstep, ystep, xbin, ybin))
    if min(xstep, ystep, xbin, ybin) < 1:
        raise ValueError("step and bin sizes must be positive")

    samples = []
    for xmid in range(xstep // 2, nxim, xstep):
        x0, x1 = max(0, xmid - xbin // 2), min(nxim, xmid - xbin // 2 + xbin)
        for ymid in range(ystep // 2, nyim, ystep):
            y0 = max(0, ymid - ybin // 2)
            y1 = min(nyim, ymid - ybin // 2 + ybin)
            sub = im[y0:y1, x0:x1]
            yy, xx = np.indices(sub.shape)
            xx, yy = xx + x0, yy + y0
            good = np.isfinite(sub) & (sub > lobad) & (sub < hibad)
            if good.sum() < 4:
                continue

            sx = robust_slope(xx[good], sub[good])
            sy = robust_slope(yy[good], sub[good])
            intercept = np.median(sub[good] - sx * xx[good] - sy * yy[good])
            resid = sub - (intercept + sx * xx + sy * yy)
            mad = np.median(np.abs(resid[good] - np.median(resid[good])))
            sigma = 1.4826 * mad
            if np.isfinite(sigma) and sigma > 0:
                good &= np.abs(resid) < 3 * sigma

            design = np.column_stack(
                (np.ones(good.sum()), xx[good], yy[good])
            )
            pars, *_ = np.linalg.lstsq(design, sub[good], rcond=None)
            value = pars[0] + pars[1] * xmid + pars[2] * ymid
            samples.append((xmid, ymid, value))

    if len(samples) < 3:
        raise ValueError("not enough valid regions to smooth image")
    samples = np.asarray(samples)
    yy, xx = np.indices(im.shape)
    linear = LinearNDInterpolator(samples[:, :2], samples[:, 2])
    result = linear(xx, yy)
    missing = ~np.isfinite(result)
    if missing.any():
        # IDL TRIGRID,/EXTRAPOLATE continues the boundary surface.  A plane
        # fit to the grid centers provides the same behavior for the intended
        # slowly varying flat field and avoids nearest-neighbor plateaus.
        design = np.column_stack(
            (np.ones(len(samples)), samples[:, 0], samples[:, 1])
        )
        pars, *_ = np.linalg.lstsq(design, samples[:, 2], rcond=None)
        result[missing] = (
            pars[0] + pars[1] * xx[missing] + pars[2] * yy[missing]
        )
    return result
