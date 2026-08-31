"""APOGEE flat combination and calibration-product construction."""

import getpass
import os
import platform
import socket
from datetime import datetime

import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_dilation

from ...utils import apload, utils
from ...utils.bitmask import PixelBitMask
from .. import ap3d
from .utils import product_build_lock,nan_uniform_filter,safe_divide
from .flathtml import flathtml
from .flatplot import flatplot


CHIPS = ("a", "b", "c")
DETECTOR_SHAPE = (2048, 2048)
NORM_SLICE = (slice(800, 1000), slice(800, 1200))
A_OUTER_SLICE = (slice(500, 1501), slice(1950, 2045))
B_LEFT_SLICE = (slice(500, 1501), slice(5, 101))
C_INNER_SLICE = (slice(500, 1501), slice(5, 101))
B_RIGHT_SLICE = (slice(500, 1501), slice(1950, 2045))

__all__ = [
    "CHIPS", "build_flat", "combine_flat_frames", "make_flat_chip",
    "normalize_flat_chips",
]

def _normalization_median(array, region, label):
    """Return a valid normalization median."""
    value = np.nanmedian(np.asarray(array)[region])
    if not np.isfinite(value) or value == 0:
        raise ValueError(f"Invalid {label} normalization: {value}")
    return float(value)


def normalize_flat_chips(flatsum):
    """Normalize three summed detector flats across their chip boundaries."""
    flatsum = np.asarray(flatsum, dtype=float).copy()
    if flatsum.shape != DETECTOR_SHAPE + (3,):
        raise ValueError(
            "flatsum must have shape (2048, 2048, 3); "
            f"received {flatsum.shape}"
        )

    middle_norm = _normalization_median(
        flatsum[:, :, 1], NORM_SLICE, "middle-chip"
    )
    flatsum[:, :, 1] /= middle_norm

    # IDL images are indexed (x, y), whereas Python/FITS arrays are (y, x).
    a_outer = _normalization_median(
        flatsum[:, :, 0],
        A_OUTER_SLICE,
        "chip-a outer edge",
    )
    b_left = _normalization_median(
        flatsum[:, :, 1],
        B_LEFT_SLICE,
        "chip-b left edge",
    )
    c_inner = _normalization_median(
        flatsum[:, :, 2],
        C_INNER_SLICE,
        "chip-c inner edge",
    )
    b_right = _normalization_median(
        flatsum[:, :, 1],
        B_RIGHT_SLICE,
        "chip-b right edge",
    )

    flatsum[:, :, 0] /= a_outer / b_left
    flatsum[:, :, 2] /= c_inner / b_right
    return flatsum


def make_flat_chip(flat,flatmask,dithered=False,kludge=False,bad_pixel_bits=None):
    """Create the flat, spectral-flat, and mask arrays for one chip.

    This function contains no APOGEE path handling or FITS I/O, making it
    suitable for direct unit testing with synthetic arrays.
    """
    flat = np.asarray(flat, dtype=float).copy()
    flatmask = np.asarray(flatmask)
    if flat.shape != flatmask.shape:
        raise ValueError("flat and flatmask must have identical shapes")
    if flat.ndim != 2:
        raise ValueError("flat and flatmask must be two-dimensional")

    if bad_pixel_bits is None:
        bad_pixel_bits = PixelBitMask().badval()

    mask = np.zeros(flat.shape, dtype=np.uint8)
    reduction_bad = (flatmask.astype(np.uint64) & bad_pixel_bits) != 0
    flat[reduction_bad] = np.nan
    mask[reduction_bad] |= np.uint8(1)

    # IDL used ZAP(flat, [100, 10]) as a local reference image.  A
    # NaN-aware boxcar gives the same intended large-scale comparison
    # without the enormous cost of a 100x10 running median.
    reference = nan_uniform_filter(flat, size=(100, 10))
    localflat = safe_divide(flat, reference)

    rejected = (
        (localflat < 0.85)
        | (localflat > 1.25)
        | (flat < 0.1)
    )
    mask[rejected] |= np.uint8(2)

    # Grow only into neighboring pixels that exceed the looser thresholds.
    loose_rejection = (localflat < 0.95) | (localflat > 1.05)
    structure = np.ones((3, 3), dtype=bool)
    for _ in range(11):
        neighbors = binary_dilation(mask != 0, structure=structure)
        new_neighbors = neighbors & loose_rejection & (mask == 0)
        if not np.any(new_neighbors):
            break
        mask[new_neighbors] |= np.uint8(4)

    zero = np.isfinite(flat) & (flat == 0)
    mask[zero] |= np.uint8(8)
    flat[zero] = np.nan

    if dithered:
        smooth = nan_uniform_filter(flat, size=100)
        row_level = np.nanmean(smooth, axis=1)[:, None]
        column_level = np.nanmean(smooth, axis=0)[None, :]

        spectral_flat = nan_uniform_filter(flat, size=(1, 50))
        flat = safe_divide(flat, spectral_flat)
        flat *= safe_divide(
            smooth,
            row_level * column_level,
        )

        if kludge:
            width = min(14, flat.shape[0] // 2)
            for row in list(range(width)) + list(
                range(flat.shape[0] - width, flat.shape[0])
            ):
                bad = ~np.isfinite(flat[row, :])
                flat[row, bad] = 1.0
                mask[row, bad] = 0
    else:
        # Preserve the historical behavior: record an estimate of the
        # spectral signature, but do not divide it out of the flat.
        profile = np.nanmedian(flat, axis=0)
        x = np.arange(flat.shape[1], dtype=float)
        good = np.isfinite(profile)
        if np.count_nonzero(good) >= 3:
            coefficients = np.polyfit(x[good], profile[good], 2)
            profile = np.polyval(coefficients, x)
        else:
            profile = np.ones(flat.shape[1], dtype=float)
        spectral_flat = np.broadcast_to(
            profile[None, :], flat.shape
        ).copy()

    flat[~np.isfinite(flat)] = 0.0
    spectral_flat[~np.isfinite(spectral_flat)] = 0.0
    return flat, spectral_flat, mask


def combine_flat_frames(load, images, nrep):
    """Median groups of ap2D frames and sum the resulting groups."""
    flatsum = np.zeros(DETECTOR_SHAPE + (3,), dtype=np.float64)
    flatmasks = np.zeros(DETECTOR_SHAPE + (3,), dtype=np.uint32)
    first_header = None

    for start in range(0, len(images), nrep):
        group = images[start : start + nrep]
        group_frames = {
            int(number): load.frame(int(number))
            for number in group
        }
        for ichip, chip in enumerate(CHIPS):
            samples = []
            for number in group:
                frame = group_frames[int(number)][chip]
                if first_header is None:
                    first_header = frame["header"].copy()
                samples.append(
                    np.asarray(frame["flux"], dtype=np.float32))
                flatmasks[:, :, ichip] = frame["mask"]

            if len(samples) == 1:
                flatsum[:, :, ichip] += samples[0]
            else:
                flatsum[:, :, ichip] += np.nanmedian(np.stack(samples), axis=0)

    if first_header is None:
        raise RuntimeError("No ap2D flat frames were read")
    return flatsum, flatmasks, first_header


def _add_provenance(header, darkfile, flatid):
    """Add calibration and software provenance to a FITS header."""
    header["DARKFILE"] = (
        os.path.basename(darkfile) if darkfile else "NONE", "dark file"
    )
    header["FLATID"] = (int(flatid), "flat calibration ID")

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
    header.add_history(f"{module}: APOGEE Reduction Pipeline Version: {softvers}")


def build_flat(images, apred="daily", telescope="apo25m", detid=None,
               darkid=None, clobber=False, kludge=False, nrep=1,
               dithered=False, unlock=False, verbose=False):
    """Make APOGEE superflat calibration files from individual flat ramps."""
    images = np.atleast_1d(images).astype(np.int64)
    if images.size == 0:
        raise ValueError("images must contain at least one exposure number")
    nrep = int(nrep)
    if nrep < 1:
        raise ValueError("nrep must be at least 1")

    flatid = int(images[0])
    load = apload.ApLoad(apred=apred, telescope=telescope)

    with product_build_lock(load, "flat", flatid, clobber=clobber,
                            unlock=unlock, verbose=verbose) as (build, files):
        if not build:
            return

        output_files = files[:3]
        summary_file = files[3]
        flatdir = load.filename("Flat", num=flatid, directory=True)

        ap3d.process_exposures(
            images,
            load=load,
            detectorid=detid,
            darkid=darkid,
            overwrite=clobber,
            verbose=verbose,
            detect_cosmic_rays=False,
            up_the_ramp=False,
            nfowler=1,
        )
        flatsum, flatmasks, header = combine_flat_frames(load, images, nrep)
        flatsum = normalize_flat_chips(flatsum)

        dtype = np.dtype([("NAME", "S256"), ("NUM", np.int64), ("NFRAMES", np.int32)])
        flatlog = np.zeros(3, dtype=dtype)
        plotdir = os.path.join(flatdir, "plots")
        os.makedirs(plotdir, exist_ok=True)

        for ichip, chip in enumerate(CHIPS):
            flat, spectral_flat, mask = make_flat_chip(flatsum[:, :, ichip],
                                                       flatmasks[:, :, ichip],
                                                       dithered=dithered,
                                                       kludge=kludge)
            chip_header = header.copy()
            darkfile = (
                load.filename("Dark", num=darkid, chip=chip)
                if darkid is not None and int(darkid) > 0
                else None
            )
            _add_provenance(chip_header,darkfile,flatid)

            outfile = output_files[ichip]
            hdul = fits.HDUList(
                [
                    fits.PrimaryHDU(header=chip_header),
                    fits.ImageHDU(flat.astype(np.float32), name="FLAT"),
                    fits.ImageHDU(
                        spectral_flat.astype(np.float32),
                        name="SPECTRAL FLAT",
                    ),
                    fits.ImageHDU(mask, name="MASK"),
                ]
            )
            hdul.writeto(outfile, overwrite=True)

            flatplot(flat,os.path.join(plotdir,
                                       os.path.splitext(os.path.basename(outfile))[0]))
            flatlog[ichip] = (outfile.encode(),flatid,len(images))

        fits.BinTableHDU(flatlog, name="FLATLOG").writeto(summary_file, overwrite=True)
        flathtml(flatdir,[{"num": flatid, "nframes": len(images)}])
