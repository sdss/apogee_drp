"""Build the APOGEE Littrow-ghost calibration mask.

This is the Python implementation of ``mklittrow.pro``. NumPy images use
``(y, x)`` ordering, so IDL's ``image[1200:1500, *]`` becomes
``image[:, 1200:1501]``.
"""

from __future__ import annotations

import getpass
from pathlib import Path
import platform
import shutil
from typing import Sequence

import numpy as np
from astropy.io import fits
from scipy.ndimage import distance_transform_edt, median_filter

from ...utils import apload
from ...utils.bitmask import PixelBitMask
from .psfcal import build_psf
from .utils import product_build_lock

__all__ = ["build_littrow", "make_littrow_mask", "subtract_scattered_light"]


def subtract_scattered_light(image, *, x_range=(100, 1948),
                             bottom_rows=(5, 11), top_rows=(2038, 2043)):
    """Subtract the constant edge background used by ``scat_remove,/scat=1``.

    Bounds follow IDL's inclusive convention. The input is not modified.
    """
    flux = np.asarray(image, dtype=float).copy()
    if flux.ndim != 2:
        raise ValueError("image must be two-dimensional")
    x0, x1 = map(int, x_range)
    b0, b1 = map(int, bottom_rows)
    t0, t1 = map(int, top_rows)
    ny, nx = flux.shape
    if not (0 <= x0 <= x1 < nx and 0 <= b0 <= b1 < ny
            and 0 <= t0 <= t1 < ny):
        raise ValueError("scattered-light regions fall outside the image")
    bottom_region = flux[b0:b1 + 1, x0:x1 + 1]
    top_region = flux[t0:t1 + 1, x0:x1 + 1]
    if not np.any(np.isfinite(bottom_region)) or not np.any(np.isfinite(top_region)):
        raise ValueError("cannot measure a finite scattered-light level")
    bottom = np.nanmedian(bottom_region)
    top = np.nanmedian(top_region)
    level = 0.5 * (bottom + top)
    if not np.isfinite(level):
        raise ValueError("cannot measure a finite scattered-light level")
    return flux - level, float(level)


def _fill_nonfinite_nearest(values):
    array = np.asarray(values, dtype=float)
    bad = ~np.isfinite(array)
    if not np.any(bad):
        return array
    if np.all(bad):
        raise ValueError("Littrow search region contains no finite pixels")
    indices = distance_transform_edt(bad, return_distances=False,
                                     return_indices=True)
    return array[tuple(indices)]


def make_littrow_mask(flux, model, pixel_mask=None, *, threshold=10.0,
                      median_width=20, search_columns=(1200, 1500),
                      output_columns=(1250, 1450), bad_pixel_bits=None):
    """Detect positive residuals and place them in the Littrow mask band."""
    image = np.asarray(flux, dtype=float).copy()
    model = np.asarray(model, dtype=float)
    if image.ndim != 2 or model.shape != image.shape:
        raise ValueError("flux and model must be matching two-dimensional arrays")
    if median_width < 1:
        raise ValueError("median_width must be positive")
    if pixel_mask is not None:
        mask = np.asarray(pixel_mask)
        if mask.shape != image.shape:
            raise ValueError("pixel_mask must match flux")
        bits = PixelBitMask().badval() if bad_pixel_bits is None else int(bad_pixel_bits)
        image[(mask.astype(np.uint64) & bits) != 0] = np.nan

    sx0, sx1 = map(int, search_columns)
    ox0, ox1 = map(int, output_columns)
    ny, nx = image.shape
    if not (0 <= sx0 <= sx1 < nx and 0 <= ox0 <= ox1 < nx):
        raise ValueError("Littrow column bounds fall outside the image")
    search_width = sx1 - sx0 + 1
    output_width = ox1 - ox0 + 1
    offset = ox0 - sx0
    if offset < 0 or offset + output_width > search_width:
        raise ValueError("output_columns must map inside search_columns")

    residual = _fill_nonfinite_nearest(
        image[:, sx0:sx1 + 1] - model[:, sx0:sx1 + 1])
    smoothed = median_filter(residual, size=int(median_width), mode="nearest")
    detected = smoothed > float(threshold)
    result = np.zeros((ny, nx), dtype=np.int16)
    result[:, ox0:ox1 + 1] = detected[:, offset:offset + output_width]
    return result


def _run_empirical_extraction(load, frameid, *, unlock=False, verbose=False):
    from .. import ap2d

    twod = load.filename("2D", num=frameid, chip="b")
    psf = load.filename("PSF", num=frameid, chip="b")
    oned = load.filename("1D", num=frameid, chip="b")
    return ap2d.ap2dproc(
        str(Path(twod).parent / f"{int(frameid):08d}"),
        str(Path(psf).parent / f"{int(frameid):08d}"),
        extract_type=4, load=load, outdir=str(Path(oned).parent),
        wavefile=None, chips=[1], clobber=True, unlock=unlock,
        verbose=verbose)


def _write_littrow(filename, mask, *, apred, frameid, scatter_level):
    header = fits.Header()
    header["EXTNAME"] = "LITTROW MASK"
    header["APRED"] = str(apred)
    header["LITID"] = int(frameid)
    header["SCATLEV"] = (float(scatter_level), "Subtracted scattered light")
    header.add_history("MKLITTROW: Python calibration builder")
    header.add_history(f"MKLITTROW: {getpass.getuser()} on {platform.node()}")
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(filename, np.asarray(mask, dtype=np.int16), header,
                 overwrite=True)


def _move_auxiliary_files(load, frameid, destination,
                          extra_files: Sequence[str] = ()):
    """Move temporary PSF/extraction products beside the Littrow mask."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    candidates = []
    for kind in ("PSF", "EPSF", "ETrace", "1D", "2Dmodel"):
        directory = Path(load.filename(
            kind, num=frameid, directory=True))
        candidates.extend(directory.glob(
            f"*{kind}*{int(frameid):08d}*.fits"))
    candidates.extend(Path(filename) for filename in extra_files)
    moved = []
    for source in dict.fromkeys(candidates):
        if not source.is_file() or source.parent == destination:
            continue
        target = destination / source.name
        if target.exists():
            target.unlink()
        shutil.move(str(source), str(target))
        moved.append(str(target))
    return moved


def build_littrow(frameid, *, apred="daily", telescope="apo25m",
                   darkid=None, flatid=None, bpmid=None, sparseid=None,
                   fiberid=None, threshold=10.0, median_width=20,
                   clobber=False, unlock=False, verbose=False,
                   keep_auxiliary=True):
    """Build the chip-b Littrow ghost mask from one calibration exposure."""
    frameid = int(frameid)
    load = apload.ApLoad(apred=apred, telescope=telescope)
    with product_build_lock(load, "littrow", frameid, clobber=clobber,
                            unlock=unlock, verbose=verbose) as (build, outputs):
        if not build:
            return

        output = outputs[0]
        build_psf(
            frameid, apred=apred, telescope=telescope, darkid=darkid,
            flatid=flatid, bpmid=bpmid, sparseid=sparseid,
            fiberid=fiberid, average=200, clobber=True, unlock=unlock,
            verbose=verbose)
        _, models = _run_empirical_extraction(
            load, frameid, unlock=unlock, verbose=verbose)
        reduced = load.frame(frameid, chip="b")
        image, scatter_level = subtract_scattered_light(reduced["flux"])
        model = None if models is None else models.get(1)
        if model is None:
            model = fits.getdata(
                load.filename("2Dmodel", num=frameid, chip="b"), 0)
        littrow = make_littrow_mask(
            image, model, reduced["mask"], threshold=threshold,
            median_width=median_width)
        _write_littrow(output, littrow, apred=apred, frameid=frameid,
                       scatter_level=scatter_level)
        if keep_auxiliary:
            _move_auxiliary_files(load, frameid, Path(output).parent)
