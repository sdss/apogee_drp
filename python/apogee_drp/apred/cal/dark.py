"""APOGEE dark-ramp combination and calibration-product construction."""

import getpass
import os
import platform
import socket
import tempfile
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_dilation, median_filter

from ...utils import apload, utils
from .. import ap3d
from .utils import product_build_lock
from .darkhtml import darkhtml
from .darkplot import darkplot


CHIPS = ("a", "b", "c")

__all__ = ["CHIPS", "build_dark", "combine_dark_ramps", "dark_variance"]


def dark_variance(data, nread=1, gain=1.9, readnoise=18.0):
    """Return the IDL APOGEE Poisson-plus-readnoise variance."""
    variance = np.maximum(np.asarray(data, dtype=float) / gain, 0.0)
    variance += nread * (readnoise / gain) ** 2
    return variance


def combine_dark_ramps(ramps, gain=1.9, readnoise=18.0, maxrate=10.0,
                       row_block=32):
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
    if gain <= 0 or readnoise < 0:
        raise ValueError("gain must be positive and readnoise non-negative")
    if maxrate <= 0:
        raise ValueError("maxrate must be positive")

    dark = np.empty((nread, ny, nx), dtype=np.float32)
    chi2 = np.zeros((nread, ny, nx), dtype=np.float32)

    for ylo in range(0, ny, row_block):
        yhi = min(ylo + row_block, ny)
        samples = np.asarray(ramps[:, :, ylo:yhi, :], dtype=np.float32)
        model = utils.idl_median(samples, axis=0).astype(np.float32, copy=False)
        dark[:, ylo:yhi, :] = model
        variance = dark_variance(model, nread=1, gain=gain,
                                 readnoise=readnoise)
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

    # Reproduce the flattened-index neighbor handling in mkdark.pro.
    flat_rate = rate.ravel()
    flat_mask = mask.ravel()
    npixel = flat_rate.size
    hot_indices = np.flatnonzero(hot)
    bad_indices = np.flatnonzero(nonfinite_rate)
    hot_neighbors = np.zeros(rate.shape, dtype=bool)
    flat_hot_neighbors = hot_neighbors.ravel()
    for seed_indices in (hot_indices, bad_indices):
        for offset in (-1, 1, -nx, nx):
            neighbors = seed_indices + offset
            on_detector = ( (neighbors >= 0) & (neighbors < npixel) )
            neighbors = neighbors[on_detector]
            selected = (np.isfinite(flat_rate[neighbors]) &
                        (flat_rate[neighbors] > maxrate / 4.0) )
            selected_neighbors = neighbors[selected]
            flat_mask[selected_neighbors] |= np.uint8(4)
            flat_hot_neighbors[selected_neighbors] = True

    nbad = int(np.count_nonzero(~np.isfinite(dark)))
    dark[~np.isfinite(dark)] = 0.0
    chi2[~np.isfinite(chi2)] = 0.0
    medrate = float(utils.idl_median(rate))
    rate[~np.isfinite(rate)] = 0.0

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


def _load_ramps(load, images, chip, directory, max_read=None, unlock=False,
                verbose=False):
    """Store corrected ramps in a disk-backed array and return it."""
    ramps = None
    header = None
    filename = os.path.join(directory, f"ramps-{chip}.npy")
    for iframe, number in enumerate(images):
        rawfile = load.filename("R", num=int(number), chip=chip)
        if verbose:
            print(f"{iframe + 1}/{len(images)} {chip} {int(number)}")
        cube, current_header = ap3d.load_raw_ramp(
            rawfile,
            max_read=max_read,
            temporary_directory=directory,
            unlock=unlock,
            verbose=verbose,
        )
        ramp, _, _, _ = ap3d.reference_correct(
            cube, current_header, indiv=3
        )
        baseline = ramp[1].copy()
        ramp[1:] -= baseline[None, :, :]
        ramp = ramp.astype(np.float32, copy=False)
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


def build_dark(ims, apred="daily", telescope="apo25m", psfid=None,
               clobber=False, unlock=False, verbose=False):
    """Create three APOGEE Dark products from raw dark exposures.

    Thermal-trace subtraction remains explicitly unavailable until the EPSF
    extraction path has been validated against the IDL implementation.
    """
    images = np.atleast_1d(ims).astype(np.int64)
    if images.size == 0:
        raise ValueError("ims must contain at least one exposure number")
    if psfid is not None:
        raise NotImplementedError(
            "Thermal-trace subtraction with psfid has not yet been ported. "
            "The production makecal path calls build_dark without psfid."
        )

    darkid = int(images[0])
    load = apload.ApLoad(apred=apred, telescope=telescope)

    with product_build_lock(load, "dark", darkid, clobber=clobber,
                            unlock=unlock, verbose=verbose) as (build, files):
        if not build:
            return

        output_files = files[:3]
        summary_file = files[3]
        darkdir = load.filename("Dark", num=darkid, directory=True)
        rate_files = [
            os.path.join(
                darkdir, f"{load.prefix}DarkRate-{chip}-{darkid:08d}.fits"
            )
            for chip in CHIPS
        ]

        # DarkRate files are diagnostics rather than registered product
        # components, so product_build_lock() intentionally does not know
        # about them. Remove stale copies whenever the Dark is rebuilt.
        for filename in rate_files:
            path = Path(filename)
            if path.exists() or path.is_symlink():
                path.unlink()

        dtype = np.dtype([("NUM", np.int64), ("NFRAMES", np.int32),
                          ("NREADS", np.int32), ("NSAT", np.int64), ("NHOT", np.int64),
                          ("NHOTNEIGH", np.int64), ("NBAD", np.int64),
                          ("MEDRATE",np.float64), ("PSFID", np.int64), ("NNEG", np.int64)])
        darklog = np.zeros(3, dtype=dtype)
        plotdir = os.path.join(darkdir, "plots")
        os.makedirs(plotdir, exist_ok=True)

        for ichip, chip in enumerate(CHIPS):
            started = time.time()
            with tempfile.TemporaryDirectory(prefix=f"mkdark-{chip}-") as workdir:
                ramps, header = _load_ramps(
                    load, images, chip, workdir, unlock=unlock,
                    verbose=verbose,
                )
                dark, chi2, mask, rate, stats = combine_dark_ramps(ramps)
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
                darkid, stats["nframes"], stats["nreads"],
                stats["nsat"], stats["nhot"], stats["nhotneigh"], stats["nbad"],
                stats["medrate"], 0, stats["nneg"],
            )
            if verbose:
                print(f"Done {chip} in {time.time() - started:.1f} seconds")

        fits.BinTableHDU(darklog, name="DARKLOG").writeto(summary_file,
                                                          overwrite=True)


        html_rows = []
        for ichip, row in enumerate(darklog):
            html_rows.append({
                "num": int(row["NUM"]),
                "chip": CHIPS[ichip],
                "nreads": int(row["NREADS"]),
                "nframes": int(row["NFRAMES"]),
                "medrate": float(row["MEDRATE"]),
                "nsat": int(row["NSAT"]),
                "nhot": int(row["NHOT"]),
                "nhotneigh": int(row["NHOTNEIGH"]),
                "nbad": int(row["NBAD"]),
                "nneg": int(row["NNEG"]),
            })
        darkhtml(darkdir, html_rows)
