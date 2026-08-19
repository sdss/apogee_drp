"""Build APOGEE static persistence masks from dark/flat exposures."""

from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter

from ...utils import lock
from ...utils.bitmask import PixelBitMask

CHIPS = ("a", "b", "c")
__all__ = ["build_persist", "make_persistence_mask", "product_files"]


def _make_load(*, apred, telescope):
    from ...utils.apload import ApLoad
    return ApLoad(apred=apred, telescope=telescope)


def _chip_filename(load, kind, number, chip):
    template = load.filename(kind, num=int(number), chips=True)
    return template.replace(f"{kind}-", f"{kind}-{chip}-")


def product_files(load, number):
    return [_chip_filename(load, "Persist", number, chip) for chip in CHIPS]


def make_persistence_mask(dark_flux, flat_flux, dark_mask=None, flat_mask=None,
                          *, threshold=0.1, smooth_size=(10, 10),
                          bad_pixel_bits=None):
    """Return the IDL-compatible severity mask and smoothed dark/flat rate."""
    dark = np.asarray(dark_flux, float)
    flat = np.asarray(flat_flux, float)
    if dark.ndim != 2 or dark.shape != flat.shape:
        raise ValueError("dark_flux and flat_flux must be matching 2-D arrays")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if (len(smooth_size) != 2 or
            any(int(value) != value or int(value) <= 0 for value in smooth_size)):
        raise ValueError("smooth_size must contain two positive integers")
    bits = PixelBitMask().badval() if bad_pixel_bits is None else int(bad_pixel_bits)
    bad = ~np.isfinite(dark) | ~np.isfinite(flat) | (flat == 0)
    for mask in (dark_mask, flat_mask):
        if mask is not None:
            if np.shape(mask) != dark.shape:
                raise ValueError("input masks must match the flux arrays")
            bad |= (np.asarray(mask).astype(np.uint64) & bits) != 0
    ratio = np.zeros_like(dark, dtype=float)
    np.divide(dark, flat, out=ratio, where=~bad)
    # ZAP(r,[10,10]) in the IDL code is a running median.
    rate = median_filter(ratio, size=tuple(map(int, smooth_size)), mode="nearest")
    severity = np.zeros(dark.shape, dtype=np.int16)
    severity[rate > threshold / 4] = 4
    severity[rate > threshold / 2] = 2
    severity[rate > threshold] = 1
    return severity, rate.astype(np.float32)


def _load_2d(load, number, chip):
    filename = _chip_filename(load, "2D", number, chip)
    return {"header": fits.getheader(filename, 0),
            "flux": fits.getdata(filename, 1),
            "mask": fits.getdata(filename, 3)}


def _process(load, frames, *, cmjd, darkid, flatid, clobber, unlock, verbose):
    from ..process import process
    return process(frames, load=load, cmjd=cmjd, darkid=darkid,
                   flatid=flatid, nfs=1, doap3dproc=True, clobber=clobber,
                   unlock=unlock, verbose=verbose)


def build_persist(persistid, dark, flat, *, apred="daily", telescope="apo25m",
                  cmjd=None, darkid=None, flatid=None, sparseid=None,
                  fiberid=None, psfid=None, threshold=0.1, thresh=None,
                  clobber=False, unlock=False, verbose=False):
    """Build the three chip-level Persist mask/rate products."""
    del sparseid, fiberid, psfid
    if thresh is not None:
        threshold = thresh
    load = _make_load(apred=apred, telescope=telescope)
    outputs = product_files(load, persistid)
    target = outputs[2]
    lock.lock(target, waittime=10, unlock=unlock)
    if all(Path(f).is_file() and Path(f).stat().st_size > 0 for f in outputs) and not clobber:
        if verbose: print(f" persist file: {target} already made")
        return outputs
    lock.lock(target, lock=True)
    try:
        for filename in outputs:
            path = Path(filename)
            if path.exists():
                path.unlink()
            path.parent.mkdir(parents=True, exist_ok=True)
        _process(load, [int(dark), int(flat)], cmjd=cmjd, darkid=darkid,
                 flatid=flatid, clobber=clobber, unlock=unlock, verbose=verbose)
        for chip, output in zip(CHIPS, outputs):
            dark_frame = _load_2d(load, dark, chip)
            flat_frame = _load_2d(load, flat, chip)
            mask, rate = make_persistence_mask(
                dark_frame["flux"], flat_frame["flux"], dark_frame["mask"],
                flat_frame["mask"], threshold=float(threshold))
            header = dark_frame["header"].copy()
            header["EXTNAME"] = "PERSIST"
            header["PTHRESH"] = float(threshold)
            header["APRED"] = str(apred)
            fits.HDUList([fits.PrimaryHDU(mask, header),
                          fits.ImageHDU(rate, name="PERSIST_RATE")]).writeto(output, overwrite=True)
        return outputs
    finally:
        lock.lock(target, clear=True)
