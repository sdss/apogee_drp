"""APOGEE bad-pixel-mask construction and product writing."""

import os

import numpy as np
from astropy.io import fits
import socket
import getpass
import platform
from datetime import datetime

from ...utils import apload, utils
from ...utils.bitmask import PixelBitMask
from .utils import product_build_lock

CHIPS = ("a", "b", "c")

__all__ = ["CHIPS", "build_bpm", "combine_bpm_masks"]


def combine_bpm_masks(darkmask, flatmask=None, badrows=None, pixmask=None):
    """
    Construct a bad-pixel mask from dark and flat calibration masks.

    This function performs only the numerical mask construction. It does
    not read or write files and does not use APOGEE directory structures.

    Parameters
    ----------
    darkmask : ndarray
        Input dark-frame mask. Nonzero pixels are flagged as BADDARK
        and BADPIX.

    flatmask : ndarray, optional
        Input flat-frame mask. Nonzero pixels are flagged as BADFLAT
        and BADPIX. If omitted, no pixels are flagged from the flat.

    badrows : array-like, optional
        Detector-row indices to flag as BADPIX.

    pixmask : PixelBitMask, optional
        APOGEE pixel bit-mask object. A new PixelBitMask is constructed
        when omitted.

    Returns
    -------
    mask : ndarray
        Constructed integer bad-pixel mask with the same shape as
        ``darkmask``.
    """
    darkmask = np.asarray(darkmask)

    if darkmask.ndim != 2:
        raise ValueError("darkmask must be a two-dimensional array")

    if flatmask is None:
        flatmask = np.zeros_like(darkmask)
    else:
        flatmask = np.asarray(flatmask)

        if flatmask.shape != darkmask.shape:
            raise ValueError("flatmask and darkmask must have identical shapes")

    if pixmask is None:
        pixmask = PixelBitMask()

    mask = np.zeros(darkmask.shape,dtype=np.int64)

    dark_bad = darkmask != 0
    flat_bad = flatmask != 0

    mask[dark_bad] |= pixmask.getval("BADDARK")
    mask[flat_bad] |= pixmask.getval("BADFLAT")

    # Preserve the historical general BADPIX flag.
    any_bad = dark_bad | flat_bad
    mask[any_bad] |= pixmask.getval("BADPIX")

    if badrows is not None:
        badrows = np.asarray(badrows,dtype=int)
        if np.any(badrows < 0) or np.any(badrows >= darkmask.shape[0]):
            raise ValueError("badrows contains an index outside the detector")

        # IDL uses mask[*, row] with (x, y) storage. NumPy images are (y, x).
        mask[badrows, :] |= pixmask.getval("BADPIX")

    return mask


def _chip_badrows(badrow, ichip):
    """Return bad-row indices associated with one detector chip."""
    if badrow is None:
        return None

    rows = []

    for entry in badrow:
        # Support structured arrays, Astropy rows, and simple objects.
        try:
            entry_chip = entry["chip"]
            entry_row = entry["row"]
        except (IndexError, KeyError, TypeError):
            entry_chip = entry.chip
            entry_row = entry.row

        if entry_chip == ichip:
            rows.append(int(entry_row))

    return np.asarray(rows, dtype=int)


def _add_provenance(header, darkfile, flatfile):
    """Record input products, software versions, and execution details."""
    header["DARKFILE"] = (os.path.basename(darkfile), "dark file")
    header["FLATFILE"] = (
        os.path.basename(flatfile) if flatfile else "NONE", "flat file"
    )
    gitvers = utils.software_version()
    softvers = utils.reduction_version()
    header["V_APRED"] = (gitvers, "apogee software version")
    header["APRED"] = (softvers, "apogee reduction version")
    module = __name__.split(".")[-1]
    header.add_history(
        f"{module}: "
        + datetime.now().astimezone().strftime("%a %b %d %H:%M:%S %Y")
    )
    header.add_history(f"{module}: {getpass.getuser()} on {socket.gethostname()}")
    header.add_history(
        f"{module}: Python {platform.python_version()} "
        f"{platform.system().lower()} {platform.machine()}"
    )
    header.add_history(
        f"{module}: APOGEE Reduction Pipeline Version: {softvers}"
    )


def build_bpm(bpmid, apred="daily", telescope="apo25m", darkid=None,
              flatid=None, badrow=None, clobber=False, unlock=False,
              verbose=False):
    """
    Create APOGEE bad-pixel-mask calibration files.

    This wrapper handles APOGEE filenames, locking, FITS I/O, and the
    three detector chips. The numerical mask construction is performed
    by :func:`combine_bpm_masks`.
    """
    if darkid is None:
        raise ValueError("darkid must be supplied")

    load = apload.ApLoad(apred=apred, telescope=telescope)

    with product_build_lock(load, "bpm", bpmid, clobber=clobber,
                            unlock=unlock, verbose=verbose) as (build, output_files):
        if not build:
            return

        if verbose:
            print(f"Making BPM: {int(bpmid):08d}")

        pixmask = PixelBitMask()
        for ichip, chip in enumerate(CHIPS):
            darkfile = load.filename("Dark", num=darkid, chip=chip)
            darkmask = fits.getdata(darkfile,ext=3)

            if flatid is not None and int(flatid) > 0:
                flatfile = load.filename("Flat", num=flatid, chip=chip)
                flatmask = fits.getdata(flatfile,ext=3)
            else:
                flatfile = None
                flatmask = None

            rows = _chip_badrows(badrow,ichip)

            mask = combine_bpm_masks(
                darkmask, flatmask=flatmask, badrows=rows, pixmask=pixmask
            )
            mask = mask.astype(np.int16)
            
            primary = fits.PrimaryHDU(mask)
            primary.header["EXTNAME"] = "BPM"

            _add_provenance(primary.header, darkfile, flatfile)
            
            fits.HDUList([primary]).writeto(
                output_files[ichip], overwrite=True
            )
