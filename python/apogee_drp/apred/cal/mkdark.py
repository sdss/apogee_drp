"""Build APOGEE superdark calibration products."""

import getpass
import os
import platform
import socket
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_dilation, median_filter

from ...utils import apload, apzip, lock, utils
from .. import ap3d
from .darkhtml import darkhtml
from .darkplot import darkplot


CHIPS = ("a", "b", "c")


def apvariance(data, nread=1, gain=1.9, readnoise=18.0):
    """Return the IDL APOGEE Poisson-plus-readnoise variance."""
    variance = np.maximum(np.asarray(data, dtype=float) / gain, 0.0)
    variance += nread * (readnoise / gain) ** 2
    return variance


def make_dark(ramps, gain=1.9, readnoise=18.0, maxrate=10.0, row_block=32):
    """Combine corrected dark ramps and construct calibration arrays.

    Parameters
    ----------
    ramps : array-like, shape (nframe, nread, ny, nx)
        Reference-corrected ramps with read 1 subtracted from reads 1 onward.
        A NumPy memmap is accepted, allowing the wrapper to avoid holding all
        input ramps in memory.
    gain, readnoise : float, optional
        Values used by the historical apvariance calculation.
    maxrate : float, optional
        Pixels accumulating faster than this many counts/read are hot.
    row_block : int, optional
        Number of detector rows processed at once.

    Returns
    -------
    dark, chi2, mask, rate, statistics
        dark and chi2 have shape (nread, ny, nx). mask and rate have shape
        (ny, nx).
    """
    if np.ndim(ramps) != 4:
        raise ValueError("ramps must have shape (nframe, nread, ny, nx)")
    nframe, nread, ny, nx = np.shape(ramps)
    if nframe < 1:
        raise ValueError("At least one ramp is required")
    if nread < 3:
        raise ValueError("Dark ramps require at least three reads")
    row_block = int(row_block)
    if row_block < 1:
        raise ValueError("row_block must be at least 1")

    dark = np.empty((nread, ny, nx), dtype=np.float32)
    chi2 = np.zeros((nread, ny, nx), dtype=np.float32)

    for ylo in range(0, ny, row_block):
        yhi = min(ylo + row_block, ny)
        samples = np.asarray(ramps[:, :, ylo:yhi, :], dtype=np.float32)
        model = np.nanmedian(samples, axis=0)
        dark[:, ylo:yhi, :] = model
        variance = apvariance(model, nread=1, gain=gain, readnoise=readnoise)
        good = np.isfinite(samples) & np.isfinite(model[None, :, :, :])
        contribution = np.zeros(samples.shape, dtype=np.float32)
        np.divide((samples - model[None, :, :, :]) ** 2, variance[None, :, :, :],
                  out=contribution, where=good)
        number = np.sum(good, axis=0)
        np.divide(np.sum(contribution, axis=0), number, out=chi2[:, ylo:yhi, :],
                  where=number > 0)
        chi2[:, ylo:yhi, :][number == 0] = 0.0

    rate = (dark[-1] - dark[1]) / float(nread - 2)
    mask = np.zeros((ny, nx), dtype=np.uint8)
    nonfinite_rate = ~np.isfinite(rate)
    hot = np.isfinite(rate) & (rate > maxrate)
    mask[nonfinite_rate] |= np.uint8(1)
    mask[hot] |= np.uint8(2)

    neighbor_seed = hot | nonfinite_rate
    neighbor_candidates = np.isfinite(rate) & (rate > maxrate / 4.0)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    hot_neighbors = binary_dilation(neighbor_seed, structure=cross)
    hot_neighbors &= neighbor_candidates & ~neighbor_seed
    mask[hot_neighbors] |= np.uint8(4)

    nbad = int(np.count_nonzero(~np.isfinite(dark)))
    dark[~np.isfinite(dark)] = 0.0
    chi2[~np.isfinite(chi2)] = 0.0
    rate[~np.isfinite(rate)] = 0.0
    medrate = float(np.median(rate))

    for ylo in range(0, ny, row_block):
        yhi = min(ylo + row_block, ny)
        dark[:, ylo:yhi, :] = median_filter(
            dark[:, ylo:yhi, :], size=(7, 1, 1), mode="nearest"
        )

    negative = dark < -10
    nneg = int(np.count_nonzero(negative))
    dark[negative] = 0.0
    statistics = {
        "nframes": int(nframe), "nreads": int(nread),
        "nsat": int(np.count_nonzero(nonfinite_rate)),
        "nhot": int(np.count_nonzero(hot)),
        "nhotneigh": int(np.count_nonzero(hot_neighbors)),
        "nbad": nbad, "medrate": medrate, "nneg": nneg,
    }
    return dark, chi2, mask, rate, statistics


def _read_corrected_ramp(filename, max_read=None, verbose=False):
    """Read an APOGEE ramp, decompressing APZ input when necessary."""
    filename = Path(filename)
    temporary_directory = None
    try:
        rampfile = filename
        if filename.suffix.lower() == ".apz":
            temporary_directory = tempfile.TemporaryDirectory(prefix="mkdark-")
            apzip.unzip(str(filename), clobber=True, delete=False, silent=not verbose,
                        fitsdir=temporary_directory.name)
            rampfile = Path(temporary_directory.name) / f"{filename.stem}.fits"
            if not rampfile.exists():
                raise RuntimeError(f"APZ decompression did not create {rampfile}")
        cube, header = ap3d.read_ramp(rampfile, max_read=max_read, verbose=verbose)
        corrected, _, _, _ = ap3d.reference_correct(cube, header, indiv=3)
        baseline = corrected[1].copy()
        corrected[1:] -= baseline[None, :, :]
        return corrected.astype(np.float32, copy=False), header
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()


def _load_ramps(load, images, chip, directory, max_read=None, verbose=False):
    """Store corrected ramps in a disk-backed array and return it."""
    ramps = None
    header = None
    filename = os.path.join(directory, f"ramps-{chip}.npy")
    for iframe, number in enumerate(images):
        rawfile = load.filename("R", num=int(number), chip=chip)
        if verbose:
            print(f"{iframe + 1}/{len(images)} {chip} {int(number)}")
        ramp, current_header = _read_corrected_ramp(rawfile, max_read=max_read,
                                                     verbose=verbose)
        if ramps is None:
            shape = (len(images),) + ramp.shape
            ramps = np.lib.format.open_memmap(filename, mode="w+", dtype=np.float32,
                                              shape=shape)
            header = current_header.copy()
        elif ramp.shape != ramps.shape[1:]:
            raise ValueError(
                f"All ramps must have the same shape; expected {ramps.shape[1:]}, "
                f"received {ramp.shape} for exposure {number}"
            )
        ramps[iframe] = ramp
    if ramps is None:
        raise RuntimeError("No dark ramps were loaded")
    ramps.flush()
    return ramps, header


def _add_provenance(header, darkid, load):
    """Add software and execution provenance to a FITS header."""
    gitvers = utils.software_version()
    softvers = utils.reduction_version()
    header["DARKID"] = (int(darkid), "dark calibration ID")
    header["V_APRED"] = (gitvers, "apogee software version")
    header["APRED"] = (softvers, "apogee reduction version")
    lead = "APMKDARK: "
    header.add_history(lead + time.asctime())
    header.add_history(lead + getpass.getuser() + " on " + socket.gethostname())
    header.add_history(
        lead + f"Python {platform.python_version()} {platform.system()} "
        f"{platform.release()} {platform.machine()}"
    )
    header.add_history(lead + "APOGEE Reduction Pipeline Version: " + load.apred)


def mkdark(ims, apred="daily", telescope="apo25m", psfid=None, step=None,
           clobber=False, unlock=False,verbose=False):
    """Make APOGEE superdark calibration products."""
    images = np.atleast_1d(ims).astype(np.int64)
    if images.size == 0:
        raise ValueError("ims must contain at least one exposure number")
    if step not in (None, 0, 1):
        warnings.warn("step is obsolete and is ignored", DeprecationWarning,
                      stacklevel=2)
    if psfid is not None:
        raise NotImplementedError(
            "Thermal-trace subtraction with psfid has not yet been ported. "
            "The production makecal path calls mkdark without psfid."
        )

    darkid = int(images[0])
    load = apload.ApLoad(apred=apred, telescope=telescope)
    cmjd = load.cmjd(darkid)
    output_files = [load.filename("Dark", num=darkid, chip=chip) for chip in CHIPS]
    darkdir = os.path.dirname(output_files[0])
    os.makedirs(darkdir, exist_ok=True)
    summary_file = os.path.join(darkdir, f"{load.prefix}Dark-{darkid:08d}.tab")
    rate_files = [
        os.path.join(darkdir, f"{load.prefix}DarkRate-{chip}-{darkid:08d}.fits")
        for chip in CHIPS
    ]
    required_files = output_files + [summary_file]

    lock.lock(summary_file, waittime=10, unlock=unlock)
    if all(os.path.exists(filename) for filename in required_files) and not clobber:
        print("Dark file:", summary_file, "already made")
        return output_files

    lock.lock(summary_file, lock=True)
    try:
        for filename in required_files + rate_files:
            if os.path.exists(filename):
                os.remove(filename)
        dtype = np.dtype([
            ("num", np.int64), ("chip", "S1"), ("nframes", np.int32),
            ("nreads", np.int32), ("nsat", np.int64), ("nhot", np.int64),
            ("nhotneigh", np.int64), ("nbad", np.int64),
            ("medrate", np.float64), ("psfid", np.int64), ("nneg", np.int64),
        ])
        darklog = np.zeros(3, dtype=dtype)
        plotdir = os.path.join(darkdir, "plots")
        os.makedirs(plotdir, exist_ok=True)

        for ichip, chip in enumerate(CHIPS):
            started = time.time()
            with tempfile.TemporaryDirectory(prefix=f"mkdark-{chip}-") as workdir:
                ramps, header = _load_ramps(load, images, chip, workdir,
                                             verbose=verbose)
                dark, chi2, mask, rate, stats = make_dark(ramps)
                del ramps
            _add_provenance(header, darkid, load)
            fits.HDUList([
                fits.PrimaryHDU(header=header),
                fits.ImageHDU(dark, name="DARK"),
                fits.ImageHDU(chi2, name="CHI-SQUARED"),
                fits.ImageHDU(mask, name="MASK"),
            ]).writeto(output_files[ichip], overwrite=True)
            fits.writeto(rate_files[ichip], rate.astype(np.float32), overwrite=True)
            plotbase = os.path.join(
                plotdir, os.path.splitext(os.path.basename(output_files[ichip]))[0]
            )
            darkplot(np.moveaxis(dark, 0, -1), mask, plotbase)
            darklog[ichip] = (
                darkid, chip.encode(), stats["nframes"], stats["nreads"],
                stats["nsat"], stats["nhot"], stats["nhotneigh"], stats["nbad"],
                stats["medrate"], 0, stats["nneg"],
            )
            if verbose:
                print(f"Done {chip} in {time.time() - started:.1f} seconds")

        fits.BinTableHDU(darklog, name="DARKLOG").writeto(summary_file,
                                                          overwrite=True)
        html_rows = []
        for row in darklog:
            html_rows.append({
                "num": int(row["num"]), "chip": row["chip"].decode(),
                "nreads": int(row["nreads"]), "nframes": int(row["nframes"]),
                "medrate": float(row["medrate"]), "nsat": int(row["nsat"]),
                "nhot": int(row["nhot"]), "nhotneigh": int(row["nhotneigh"]),
                "nbad": int(row["nbad"]), "nneg": int(row["nneg"]),
            })
        darkhtml(darkdir, html_rows)
    finally:
        lock.lock(summary_file, clear=True)
    return output_files
