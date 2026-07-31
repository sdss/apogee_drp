import os

import numpy as np
from astropy.io import fits
import socket
import getpass
import platform
import time
from datetime import datetime

from apogee_drp.utils import apload, lock, utils
from apogee_drp.utils.bitmask import PixelBitMask

def make_bpm(darkmask,flatmask=None,badrows=None,pixmask=None):
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
        if np.any(badrows < 0) or np.any(badrows >= darkmask.shape[1]):
            raise ValueError("badrows contains an index outside the detector")

        # This preserves the orientation in the original code:
        # mask[:, row].
        mask[:, badrows] |= pixmask.getval("BADPIX")

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


def mkbpm(bpmid,apred="daily",telescope="apo25m",darkid=None,flatid=None,
          badrow=None,clobber=False,unlock=False):
    """
    Create APOGEE bad-pixel-mask calibration files.

    This wrapper handles APOGEE filenames, locking, FITS I/O, and the
    three detector chips. The numerical mask construction is performed
    by :func:`make_bpm`.
    """
    if darkid is None:
        raise ValueError("darkid must be supplied")

    load = apload.ApLoad(apred=apred,telescope=telescope)

    chips = ("a", "b", "c")
    bpmid_string = f"{bpmid:08d}"

    representative_file = load.filename("BPM",num=bpmid,chips=True)
    lockfile = representative_file + ".lock"
    lock.lock(lockfile,waittime=10,unlock=unlock)

    output_files = [load.filename("BPM",num=bpmid,chips=True).replace('BPM-',f'BPM-{chip}-') for chip in chips]

    if (all(os.path.exists(path) for path in output_files) and not clobber):
        print(f"BPM file {representative_file} already made")
        return output_files

    for path in output_files:
        if os.path.exists(path):
            os.remove(path)

    print(f"Making BPM: {bpmid_string}")
    lock.lock(lockfile, lock=True)

    now = datetime.now()
    start = time.time()
    print ("Start: ",now.strftime("%Y-%m-%d %H:%M:%S"))
    
    pixmask = PixelBitMask()

    try:
        for ichip, chip in enumerate(chips):
            darkfile = load.filename("Dark",chips=True,num=darkid).replace('Dark-',f'Dark-{chip}-')
            darkmask = fits.getdata(darkfile,ext=3)

            if flatid is not None:
                flatfile = load.filename("Flat",chips=True,num=flatid).replace('Flat-',f'Flat-{chip}-')
                flatmask = fits.getdata(flatfile,ext=3)
            else:
                flatmask = None

            rows = _chip_badrows(badrow,ichip)

            mask = make_bpm(darkmask,flatmask=flatmask,
                            badrows=rows,pixmask=pixmask)
            mask = mask.astype(np.int16)
            
            outfile = output_files[ichip]
            os.makedirs(os.path.dirname(outfile),exist_ok=True)

            primary = fits.PrimaryHDU(mask)
            primary.header["EXTNAME"] = "BPM"

            # Add dark/fits
            primary.header["darkfile"] = (darkfile,'dark file')
            primary.header["flatfile"] = (flatfile,'flat file')
            gitvers = utils.software_version()
            softvers = utils.reduction_version()
            primary.header["V_APRED"] = (softvers, "apogee software version")
            primary.header["APRED"] = (gitvers, "apogee reduction version")
            
            module = __name__.split('.')[-1]
            primary.header.add_history(f"{module}: "
                + datetime.now().astimezone().strftime("%a %b %d %H:%M:%S %Y")
            )
            primary.header.add_history(f"{module}: {getpass.getuser()} on {socket.gethostname()}")
            primary.header.add_history(
                f"{module}: Python {platform.python_version()} "
                f"{platform.system().lower()} {platform.machine()}"
            )
            primary.header.add_history(
                f"{module}: APOGEE Reduction Pipeline Version: {softvers}"
            )
            
            fits.HDUList([primary]).writeto(outfile,overwrite=True)

    finally:
        # Always clear the lock, including when reading or writing fails.
        lock.lock(lockfile,clear=True)

    now = datetime.now()
    print ("End: ",now.strftime("%Y-%m-%d %H:%M:%S"))
    print("elapsed: ",time.time()-start)
        
    return output_files
