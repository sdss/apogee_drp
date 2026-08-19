"""Detector characterization and calibration-product construction.

This module owns the complete detector-calibration workflow. A linearity
exposure is characterized directly from its raw ramp; that result is stored
with the fixed gain and read-noise values in the Detector products. Linearity
measurement therefore has no Dark, BPM, or Detector prerequisites.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from astropy.io import fits

from .aplincorr import aplincorr
from .getrn import fowler_sample, getrn, rnhtml
from .noise import noise
from ...utils import apload, apzip, lock, utils

__all__ = ["LINEARITY_DTYPE", "aplincorr", "build_detector",
           "fit_linearity", "fowler_sample", "getrn", "measure_linearity",
           "noise", "rnhtml", "sample_linearity"]

LINEARITY_DTYPE = np.dtype([
    ("read", np.int32), ("chip", np.int16), ("ix", np.int16),
    ("iy", np.int16), ("counts", np.float64), ("rate", np.float64),
    ("instantaneous_rate", np.float64),
])

def _make_load(*, apred: str, telescope: str):
    """Construct ``ApLoad`` without importing its large dependency tree here."""
    from ...utils.apload import ApLoad
    return ApLoad(apred=apred, telescope=telescope)

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
    for ix in range(0, 40, 5):
        x1 = 24 + ix * 50
        for iy in range(0, 40, 5):
            y1 = 24 + iy * 50
            region = data[:, y1:y1 + 10, x1:x1 + 10].astype(float)
            counts = np.array([np.nanmedian(region[r] - region[1]) for r in reads])
            elapsed = reads - 1
            rates = counts / elapsed
            counts = counts * (reads + 1) / elapsed  # extrapolate to zero read
            instant = np.array([
                np.nanmedian(region[r] - region[r - 1]) for r in reads
            ])
            use = (np.abs(counts - reference_counts) < 2000)
            use &= np.isfinite(counts) & np.isfinite(rates)
            if use.sum() < 3:
                continue
            local = np.polynomial.polynomial.polyfit(counts[use], rates[use], 2)
            reference_rate = np.polynomial.polynomial.polyval(reference_counts, local)
            if not np.isfinite(reference_rate) or reference_rate == 0:
                continue
            for read, count, rate, instantaneous in zip(reads, counts, rates, instant):
                if np.all(np.isfinite([count, rate, instantaneous])):
                    records.append((read, chip, ix, iy, count,
                                    rate / reference_rate,
                                    instantaneous / reference_rate))
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

def _read_and_correct_ramp(filename: str, *, apred: str,
                           nread: int | None,
                           unlock: bool = False) -> np.ndarray:
    """Unpack, read, and reference-correct one raw APOGEE ramp."""
    from ..ap3d import reference_correct

    local = Path(utils.localdir())
    root = local if local.is_dir() else Path(".")
    outdir = root / apred
    outdir.mkdir(parents=True, exist_ok=True)
    unpacked = outdir / f"{Path(filename).stem}.fits"
    if not unpacked.exists():
        apzip.unzip(filename, fitsdir=str(outdir), unlock=unlock)
    with fits.open(unpacked, memmap=False) as hdul:
        count = min(nread or len(hdul) - 1, len(hdul) - 1)
        header = hdul[0].header.copy()
        cube = np.stack([hdul[index].data for index in range(1, count + 1)])
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
    load = _make_load(apred=apred, telescope=telescope)
    directory = Path(load.filename("Detector", num=0, chips=True)).parent
    directory.mkdir(parents=True, exist_ok=True)
    filename = directory / f"{load.prefix}Linearity-{int(frameid):08d}.dat"
    lock.lock(str(filename), waittime=10, unlock=unlock)
    locked = False
    try:
        if filename.exists() and filename.stat().st_size and not clobber:
            if verbose:
                print(f"Linearity measurements {filename} already exist; reusing them")
                measurements = np.atleast_1d(
                    np.genfromtxt(filename, dtype=LINEARITY_DTYPE)
                )
        else:
            if verbose:
                print(f"Measuring linearity from exposure {frameid}")
            lock.lock(str(filename), lock=True)
            locked = True
            reader = ramp_reader or _read_and_correct_ramp
            chip_indices = [chip] if chip is not None else [0, 1, 2]
            pieces = []
            for index in chip_indices:
                if index not in (0, 1, 2):
                    raise ValueError("chip must be 0, 1, or 2")
                raw = load.filename("R", num=frameid, chips=True)
                raw = raw.replace("R-", f"R-{'abc'[index]}-")
                cube = reader(raw, apred=apred, nread=nread, unlock=unlock)
                pieces.append(sample_linearity(cube, index, nskip=nskip))
            measurements = (np.concatenate(pieces) if pieces
                            else np.empty(0, dtype=LINEARITY_DTYPE))
            # Writing measurements
            values = np.column_stack(
                [measurements[field] for field in LINEARITY_DTYPE.names]
            )
            np.savetxt(filename,values,
                       fmt="%3d %3d %5d %5d %12.4f %12.7f %12.7f",
                       header="read chip ix iy counts rate instantaneous_rate")
        return fit_linearity(measurements, telescope=telescope,
                             minread=minread, order=order)
    finally:
        if locked:
            lock.lock(str(filename), clear=True)


def _detector_constants(telescope: str) -> tuple[float, Sequence[float]]:
    return ((1.9, (13.0, 11.0, 10.0)) if telescope.startswith("apo")
            else (3.0, (7.0, 8.0, 4.0)))


def build_detector(detid: int, *, linid: int | None = None,
                   apred: str = "daily", telescope: str = "apo25m",
                   unlock: bool = False, clobber: bool = False,
                   verbose: bool = False,
                   linearity_function: Callable[..., np.ndarray] = measure_linearity,
                   ) -> list[str]:
    """Create the three Detector FITS products and return their filenames."""
    load = _make_load(apred=apred, telescope=telescope)
    template = load.filename("Detector", num=detid, chips=True)
    outputs = [template.replace("Detector-", f"Detector-{chip}-")
               for chip in "abc"]
    lock.lock(template, waittime=10, unlock=unlock)
    locked = False
    try:
        if load.exists('Detector',num=detid) and not clobber:
            if verbose:
                print(f"Detector {int(detid):08d} already exists")
            return outputs
        lock.lock(template, lock=True)
        locked = True
        coefficients = np.array([1.0, 0.0, 0.0])
        if linid is not None and int(linid) > 0:
            coefficients = np.asarray(linearity_function(
                int(linid), apred=apred, telescope=telescope, unlock=unlock,
                clobber=clobber, verbose=verbose), dtype=float)
        linearity = np.tile(coefficients, (4, 1))
        gain_value, read_noise_dn = _detector_constants(telescope)
        for chip_index, output in enumerate(outputs):
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            gain = np.full(4, gain_value)
            read_noise = np.full(4, read_noise_dn[chip_index] * gain_value)
            fits.HDUList([
                fits.PrimaryHDU(), fits.ImageHDU(read_noise, name="READNOISE"),
                fits.ImageHDU(gain, name="GAIN"),
                fits.ImageHDU(linearity, name="LINEARITY CORRECTION"),
            ]).writeto(output, overwrite=True)
        return outputs
    finally:
        if locked:
            lock.lock(template, clear=True)
