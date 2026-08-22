"""Detector characterization and calibration-product construction.

This module owns the complete detector-calibration workflow. A linearity
exposure is characterized directly from its raw ramp; that result is stored
with the fixed gain and read-noise values in the Detector products. Linearity
measurement therefore has no Dark, BPM, or Detector prerequisites.
"""

from __future__ import annotations

import os
import time
import numpy as np
from astropy.io import fits
from datetime import datetime
from pathlib import Path
from typing import Callable
from .aplincorr import aplincorr
from .getrn import fowler_sample, getrn, rnhtml
from .noise import noise
from ...utils import apload, apzip, lock, utils
from .utils import file_build_lock, product_build_lock

__all__ = ["LINEARITY_DTYPE", "aplincorr", "build_detector",
           "fit_linearity", "fowler_sample", "getrn", "measure_linearity",
           "noise", "rnhtml", "sample_linearity"]

CHIPS = ("a", "b", "c")

LINEARITY_DTYPE = np.dtype([
    ("read", np.int32), ("chip", np.int16), ("ix", np.int16),
    ("iy", np.int16), ("counts", np.float64), ("rate", np.float64),
    ("instantaneous_rate", np.float64),
])


DETECTOR_CONSTANTS = {"apo": (1.9, {"a":13.0, "b":11.0, "c":10.0}),
                      "lco": (3.0, {"a":7.0, "b":8.0, "c":4.0})}

def sample_linearity(cube: np.ndarray, chip: int, *, nskip: int = 4,
                     reference_counts: float = 3000.0) -> np.ndarray:
    """Measure relative count rates in regions of one corrected ramp.

    ``cube`` uses standard DRP ordering ``(nread, ny, nx)``.
    """
    data = np.asarray(cube)
    if data.ndim != 3 or data.shape[1] < 1984 or data.shape[2] < 1984:
        raise ValueError("cube must have shape (nread, >=1984, >=1984)")
    if data.shape[0] < 4:
        raise ValueError("at least four reads are required")
    if nskip < 1:
        raise ValueError("nskip must be positive")

    records = []
    reads = np.arange(2, data.shape[0], nskip, dtype=int)
    reference_counts = np.float32(reference_counts)
    for ix in range(0, 40, 5):
        x1 = 24 + ix * 50
        for iy in range(0, 40, 5):
            y1 = 24 + iy * 50
            # IDL ix1:ix2 and iy1:iy2 are inclusive: 11 × 11 pixels.
            region = np.asarray(data[:, y1:y1 + 11, x1:x1 + 11],
                                dtype=np.float32)
            counts = np.zeros(len(reads), dtype=np.float32)
            rates = np.zeros(len(reads), dtype=np.float32)
            instantaneous_rates = np.zeros(len(reads),dtype=np.float32)

            for index, read in enumerate(reads):
                counts[index] = np.median( region[read] - region[1] )
                rates[index] = counts[index] / np.float32(read - 1)
                # Correct to the nominal zero read after calculating rate.
                counts[index] *= ( np.float32(read + 1) /
                                   np.float32(read - 1) )
                instantaneous_rates[index] = np.median(region[read] -
                                                       region[read - 1] )
            use = ( (counts > reference_counts - np.float32(2000.0)) &
                    (counts < reference_counts + np.float32(2000.0)) )
            if np.count_nonzero(use) <= 2:
                continue
            # mklinearity.pro requires measurements to bracket cref.
            if ( np.min(counts[use]) > reference_counts or
                 np.max(counts[use]) < reference_counts ):
                continue
            local = np.polynomial.polynomial.polyfit(counts[use], rates[use], 2)
            reference_rate = np.polynomial.polynomial.polyval(reference_counts, local)
            if not np.isfinite(reference_rate) or reference_rate == 0:
                continue
            for read, count, rate, instantaneous in zip(reads, counts,
                                                        rates, instantaneous_rates):
                records.append((read, chip, ix, iy, count, rate /
                                reference_rate, instantaneous / reference_rate) )
    return np.asarray(records, dtype=LINEARITY_DTYPE)


def fit_linearity(measurements: np.ndarray, *, telescope: str = "apo25m",
                  minread: int = 2, order: int = 2) -> np.ndarray:
    """Fit correction coefficients in the order expected by ``aplincorr``."""
    data = np.asarray(measurements)
    required = set(LINEARITY_DTYPE.names)
    if data.dtype.names is None or not required.issubset(data.dtype.names):
        raise ValueError(f"measurements must contain fields {sorted(required)}")
    selected = np.zeros(len(data), dtype=bool)
    for chip in range(3):
        ymax = 50
        if telescope.startswith("apo") and chip == 1:
            ymax = 0
        elif telescope.startswith("apo") and chip == 2:
            ymax = 18
        selected |= (data["chip"] == chip) & (data["iy"] < ymax)
    selected &= ((data["read"] >= minread) & (data["counts"] < 50_000)
                 & np.isfinite(data["counts"]) & np.isfinite(data["rate"]))
    if selected.sum() <= order:
        raise ValueError("not enough valid linearity measurements for the fit")
    return np.polynomial.polynomial.polyfit(
        data["counts"][selected], data["rate"][selected], order)


def _read_and_correct_ramp( filename, *, apred, nread, unlock=False, ):
    """Load and reference-correct a raw ramp for linearity measurement."""
    from ..ap3d import load_raw_ramp, reference_correct
    temporary_directory = Path(utils.localdir()) / apred
    cube, header = load_raw_ramp(filename, max_read=nread,
                                 temporary_directory=temporary_directory, unlock=unlock)
    corrected, _, _, _ = reference_correct(cube, header, indiv=0, cds=True)
    return corrected


def measure_linearity(frameid: int, *, apred: str = "daily",
                      telescope: str = "apo25m", chip: int | None = None,
                      nread: int | None = None, minread: int = 2,
                      order: int = 2, nskip: int = 4,
                      clobber: bool = False, unlock: bool = False,
                      verbose: bool = False,
                      ramp_reader: Callable[..., np.ndarray] | None = None,
                      ) -> np.ndarray:
    """Derive linearity directly from a raw internal-flat exposure.

    Existing measurement files are read and refit unless ``clobber`` is set.
    ``ramp_reader`` is injectable for tests and alternate storage backends.
    """
    load = apload.ApLoad(apred=apred, telescope=telescope)
    directory = load.filename("Detector", num=frameid, directory=True)
    filename = os.path.join(directory, f"{load.prefix}Linearity-{int(frameid):08d}.dat")

    with file_build_lock(filename, clobber=clobber, unlock=unlock,
                         verbose=verbose) as build:
        if build:
            if verbose:
                print(f"Measuring linearity from exposure {frameid}")
            reader = ramp_reader or _read_and_correct_ramp
            chip_indices = [chip] if chip is not None else [0, 1, 2]
            pieces = []
            for index in chip_indices:
                if index not in (0, 1, 2):
                    raise ValueError("chip must be 0, 1, or 2")
                raw = load.filename("R", num=frameid, chip=CHIPS[index])
                cube = reader(raw, apred=apred, nread=nread, unlock=unlock)
                pieces.append(sample_linearity(cube, index, nskip=nskip))
            measurements = (np.concatenate(pieces) if pieces
                            else np.empty(0, dtype=LINEARITY_DTYPE))
            # Writing measurements
            values = np.column_stack(
                [measurements[field] for field in LINEARITY_DTYPE.names]
            )
            np.savetxt(
                filename,
                values,
                fmt="%3d %3d %5d %5d %12.4f %12.4f %12.4f",
            )

    # mklinearity.pro writes its measurements and then reads the rounded
    # values back from disk before performing the final polynomial fit.
    measurements = np.atleast_1d(
        np.genfromtxt(filename, dtype=LINEARITY_DTYPE)
    )

    return fit_linearity(measurements, telescope=telescope,
                         minread=minread, order=order)


def build_detector(detid: int, *, linid: int | None = None,
                   apred: str = "daily", telescope: str = "apo25m",
                   unlock: bool = False, clobber: bool = False,
                   verbose: bool = True,
                   linearity_function: Callable[..., np.ndarray] = measure_linearity,
                   ) -> None:
    """Create the three Detector FITS products."""
    now = datetime.now()
    start = time.time()
    if verbose:
        print("Start: "+now.strftime("%Y-%m-%d %H:%M:%S"))
    
    load = apload.ApLoad(apred=apred, telescope=telescope)

    with product_build_lock(load, "detector", detid, clobber=clobber,
                            unlock=unlock, verbose=verbose) as (build, output_files):
        if not build:
            return

        outputs = dict(zip(CHIPS, output_files))

        # linearity
        coefficients = np.array([1.0, 0.0, 0.0])
        if linid is not None and int(linid) > 0:
            coefficients = np.asarray(linearity_function(
                int(linid), apred=apred, telescope=telescope, unlock=unlock,
                clobber=clobber, verbose=verbose), dtype=float)
        linearity = np.tile(coefficients[:, np.newaxis], (1, 4)).astype(np.float32)
        
        gain_value, read_noise_dn = DETECTOR_CONSTANTS[telescope[:3]]
        gain_value = np.float32(gain_value)
        # chip loop
        for chip_name, output in outputs.items():
            gain = np.full(4, gain_value, dtype=np.float32)
            read_noise_value = ( np.float32(read_noise_dn[chip_name]) * gain_value)
            read_noise = np.full(4, read_noise_value,dtype=np.float32)
            
            fits.HDUList([
                fits.PrimaryHDU(), fits.ImageHDU(read_noise, name="READNOISE"),
                fits.ImageHDU(gain, name="GAIN"),
                fits.ImageHDU(linearity, name="LINEARITY CORRECTION"),
            ]).writeto(output, overwrite=True)
            if verbose:
                print('Writing '+output)
                
        if verbose:
            now = datetime.now()
            print("End: "+now.strftime("%Y-%m-%d %H:%M:%S"))
            print("elapsed: %0.1f sec." % (time.time()-start))

        return
