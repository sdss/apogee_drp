"""Build APOGEE Fiber, Sparse, PSF, and ModelPSF calibration products.

The four products share tracing and empirical-PSF algorithms but expose
separate builders and product contracts. Arrays use NumPy/FITS ``(y, x)``
ordering; trace images use ``(fiber, x)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from astropy.io import fits
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .. import ap3d
from ..psf import loadepsf, makeprofilegrid, saveepsf
from ...utils import apload
from ...utils.bitmask import PixelBitMask
from .utils import average_calibration_frames, product_build_lock

CHIPS = ("a", "b", "c")

__all__ = [
    "CHIPS", "TraceSolution", "build_empirical_psf", "build_fiber",
    "build_modelpsf", "build_psf", "build_sparse", "find_traces",
]


@dataclass(frozen=True)
class TraceSolution:
    """Fiber identities and fitted detector positions for one chip."""

    fibers: np.ndarray
    trace: np.ndarray
    coefficients: np.ndarray
    peak_positions: np.ndarray
    mean_distance: float


def _quadratic_peak(profile, index):
    if index <= 0 or index >= len(profile) - 1:
        return float(index)
    values = np.asarray(profile[index - 1:index + 2], float)
    if not np.all(np.isfinite(values)):
        return float(index)
    curvature = values[0] - 2 * values[1] + values[2]
    if curvature == 0:
        return float(index)
    offset = 0.5 * (values[0] - values[2]) / curvature
    return float(index + np.clip(offset, -1, 1))


def _match_peaks(peaks, reference, tolerance=2.0):
    """Match detected peaks to unique reference fibers."""
    candidates = []
    for peak in np.asarray(peaks, float):
        fiber = int(np.argmin(np.abs(reference - peak)))
        distance = float(abs(reference[fiber] - peak))
        if distance < tolerance:
            candidates.append((distance, fiber, peak))
    best = {}
    for distance, fiber, peak in sorted(candidates):
        best.setdefault(fiber, (distance, peak))
    fibers = np.array(sorted(best), dtype=int)
    positions = np.array([best[fiber][1] for fiber in fibers], dtype=float)
    distances = np.array([best[fiber][0] for fiber in fibers], dtype=float)
    return fibers, positions, distances


def _centroid(profile, center, half_width=2):
    lo = max(int(round(center)) - half_width, 0)
    hi = min(int(round(center)) + half_width + 1, profile.size)
    values = np.asarray(profile[lo:hi], float)
    coordinates = np.arange(lo, hi, dtype=float)
    good = np.isfinite(values) & (values > 0)
    total = np.sum(values[good])
    if total <= 0:
        return np.nan
    return float(np.sum(coordinates[good] * values[good]) / total)


def find_traces(frame, reference_positions, *, average=50, threshold=0.05,
                match_tolerance=2.0, bad_pixel_bits=None):
    """Find, identify, follow, and quadratically fit fiber traces."""
    flux = np.asarray(frame["flux"], float).copy()
    mask = np.asarray(frame.get("mask", np.zeros_like(flux)))
    if flux.ndim != 2 or mask.shape != flux.shape:
        raise ValueError("frame flux and mask must be matching 2-D arrays")
    reference = np.asarray(reference_positions, float)
    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("reference_positions must be a nonempty 1-D array")
    if average < 1 or average * 2 >= flux.shape[1]:
        raise ValueError("average is incompatible with the detector width")
    bits = PixelBitMask().badval() if bad_pixel_bits is None else bad_pixel_bits
    flux[(mask.astype(np.uint64) & int(bits)) != 0] = np.nan
    values = np.where(np.isfinite(flux), flux, 0.0)
    weights = np.isfinite(flux).astype(float)
    smooth_values = uniform_filter1d(values, average, axis=1, mode="nearest")
    smooth_weights = uniform_filter1d(weights, average, axis=1, mode="nearest")
    smooth = np.full_like(flux, np.nan)
    np.divide(smooth_values, smooth_weights, out=smooth,
              where=smooth_weights > 0)

    nx = flux.shape[1]
    center = nx // 2
    lo, hi = max(center - 100, 0), min(center + 101, nx)
    profile = np.nansum(smooth[:, lo:hi], axis=1)
    high = float(np.nanmax(profile))
    minimum = max(threshold * high, 5000.0)
    peak_indices, _ = find_peaks(profile, height=minimum, distance=3)
    refined = np.array([_quadratic_peak(profile, index)
                        for index in peak_indices])
    fibers, centers, distances = _match_peaks(
        refined, reference, tolerance=match_tolerance)
    if fibers.size == 0:
        raise ValueError("No detected traces matched the reference fibers")
    if fibers.size > reference.size:
        raise ValueError("More traces were matched than reference fibers")

    trace = np.full((fibers.size, nx), np.nan, dtype=float)
    trace[:, center] = centers
    previous = centers.copy()
    for column in range(center + 1, nx):
        for index in range(fibers.size):
            measured = _centroid(smooth[:, column], previous[index])
            if np.isfinite(measured):
                trace[index, column] = measured
                previous[index] = measured
    previous = centers.copy()
    for column in range(center - 1, -1, -1):
        for index in range(fibers.size):
            measured = _centroid(smooth[:, column], previous[index])
            if np.isfinite(measured):
                trace[index, column] = measured
                previous[index] = measured

    x = np.arange(nx, dtype=float)
    coefficients = np.zeros((fibers.size, 3), dtype=float)
    fitted = np.empty_like(trace)
    edge = min(average, max(nx // 4, 1))
    for index in range(fibers.size):
        good = np.isfinite(trace[index]) & (x > edge) & (x < nx - edge)
        if good.sum() < 3:
            raise ValueError(f"Insufficient valid centroids for fiber {fibers[index]}")
        coefficients[index] = np.polynomial.polynomial.polyfit(
            x[good], trace[index, good], 2)
        fitted[index] = np.polynomial.polynomial.polyval(
            x, coefficients[index])
    return TraceSolution(
        fibers=fibers, trace=fitted, coefficients=coefficients,
        peak_positions=centers,
        mean_distance=float(np.mean(distances)),
    )


def build_empirical_psf(frame, solution, *, half_width=7,
                        smooth_columns=50):
    """Construct compact, column-normalized empirical profiles."""
    flux = np.asarray(frame["flux"], float).copy()
    mask = np.asarray(frame.get("mask", np.zeros_like(flux)))
    flux[(mask != 0) | ~np.isfinite(flux)] = 0.0
    flux = uniform_filter1d(flux, smooth_columns, axis=1, mode="nearest")
    ny, nx = flux.shape
    output = []
    for fiber, trace in zip(solution.fibers, solution.trace):
        row_min = max(int(np.floor(np.min(trace))) - half_width, 0)
        row_max = min(int(np.ceil(np.max(trace))) + half_width, ny - 1)
        image = np.zeros((nx, row_max - row_min + 1), dtype=np.float32)
        for column, center in enumerate(trace):
            rows = np.arange(row_min, row_max + 1)
            use = np.abs(rows - center) <= half_width
            values = np.maximum(flux[rows, column], 0.0) * use
            total = values.sum()
            if total > 0:
                image[column] = values / total
        output.append({
            "fiber": int(fiber), "cent": trace.astype(np.float32),
            "lo": row_min, "hi": row_max, "img": image,
        })
    return output


def _reference_positions(load, chip_index, *, fiberid=None, yshift=0.0):
    if fiberid is not None and int(fiberid) > 0:
        trace = fits.getdata(load.filename(
            "Fiber", num=fiberid, chip=CHIPS[chip_index]))
        return np.asarray(trace)[:, np.asarray(trace).shape[1] // 2]
    root = os.environ.get("APOGEE_DRP_DIR")
    if root is None:
        raise RuntimeError("APOGEE_DRP_DIR is required for reference fiber positions")
    filename = Path(root) / "data" / "cal" / f"{load.instrument}_fiber_positions.fits"
    return np.asarray(fits.getdata(filename))[chip_index] + float(yshift)


def _reduce(load, exposures, *, darkid=None, flatid=None, bpmid=None,
            littrowid=None, maxread=None, clobber=False, verbose=False):
    for chip_index, chip in enumerate(CHIPS):
        limits = np.asarray(maxread) if maxread is not None else np.asarray(None)
        limit = (limits[chip_index] if limits.ndim > 0 else maxread)
        calibrations = {
            "dark": load.filename("Dark", num=darkid, chip=chip)
                    if darkid and int(darkid) > 0 else None,
            "flat": load.filename("Flat", num=flatid, chip=chip)
                    if flatid and int(flatid) > 0 else None,
            "bpm": load.filename("BPM", num=bpmid, chip=chip)
                   if bpmid and int(bpmid) > 0 else None,
            "littrow": load.filename("Littrow", num=littrowid, chip=chip)
                       if littrowid and int(littrowid) > 0 else None,
        }
        for exposure in exposures:
            raw = load.filename("R", num=exposure, chip=chip)
            output = load.filename("2D", num=exposure, chip=chip)
            if Path(output).exists() and not clobber:
                continue
            ap3d.process_file(
                raw, output, overwrite=clobber, max_read=limit,
                detect_cosmic_rays=False, up_the_ramp=False, nfowler=1,
                verbose=verbose, **calibrations)


def _write_trace(filename, solution, header):
    hdr = header.copy()
    hdr["NTRACE"] = len(solution.fibers)
    hdr["AVGDIST"] = solution.mean_distance
    hdr["EXTNAME"] = "ETRACE"
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(filename, solution.trace.astype(np.float32), hdr,
                 overwrite=True)


def _combine_profiles(dense_profiles, sparse_profiles):
    """Use dense cores and sparse wings on a common absolute-pixel grid."""
    sparse_by_fiber = {int(profile["fiber"]): profile
                       for profile in sparse_profiles}
    combined = []
    for dense in dense_profiles:
        sparse = sparse_by_fiber.get(int(dense["fiber"]))
        if sparse is None:
            combined.append(dense)
            continue
        lo = min(int(dense["lo"]), int(sparse["lo"]))
        hi = max(int(dense["hi"]), int(sparse["hi"]))
        nx = dense["img"].shape[0]
        image = np.zeros((nx, hi - lo + 1), dtype=np.float32)
        sparse_slice = slice(int(sparse["lo"]) - lo, int(sparse["hi"]) - lo + 1)
        dense_slice = slice(int(dense["lo"]) - lo, int(dense["hi"]) - lo + 1)
        image[:, sparse_slice] = sparse["img"]
        image[:, dense_slice] = dense["img"]
        centers = np.asarray(dense["cent"], float)[:, None]
        rows = np.arange(lo, hi + 1, dtype=float)[None, :]
        core = np.abs(rows - centers) <= 5.0
        sparse_image = np.zeros_like(image)
        sparse_image[:, sparse_slice] = sparse["img"]
        image[~core] = sparse_image[~core]
        totals = image.sum(axis=1, keepdims=True)
        np.divide(image, totals, out=image, where=totals > 0)
        combined.append({**dense, "lo": lo, "hi": hi, "img": image})
    return combined


def _write_psf(filename, frame, solution, profiles):
    nx = frame["flux"].shape[1]
    trace_dtype = np.dtype([
        ("fiber", np.int16), ("coef", np.float64, (3,)),
        ("fwhm", np.float32), ("fluxcoef", np.float64, (5,)),
        ("gaussy", np.float32),
    ])
    table = np.zeros(len(solution.fibers), dtype=trace_dtype)
    table["fiber"] = solution.fibers
    table["coef"] = solution.coefficients
    table["fwhm"] = 2.355
    table["gaussy"] = solution.trace[:, nx // 2]
    x = np.arange(nx, dtype=float)
    psf_image = np.zeros_like(frame["flux"], dtype=np.float32)
    for index, (trace, profile) in enumerate(zip(solution.trace, profiles)):
        spectrum = np.array([
            np.sum(frame["flux"][max(int(round(y))-2, 0):int(round(y))+3, col])
            for col, y in enumerate(trace)
        ])
        table["fluxcoef"][index] = np.polynomial.polynomial.polyfit(
            x, np.nan_to_num(spectrum, nan=np.nanmedian(spectrum)), 4)
        lo, hi = profile["lo"], profile["hi"]
        psf_image[lo:hi + 1] += profile["img"].T
    header = frame["header"].copy()
    header["NTRACES"] = len(solution.fibers)
    fits.HDUList([
        fits.PrimaryHDU(header=header),
        fits.BinTableHDU(table, name="TRACE"),
        fits.ImageHDU(psf_image, name="PSF"),
        fits.ImageHDU(solution.coefficients.T, name="TRACESET"),
        fits.ImageHDU(np.ones_like(solution.coefficients.T), name="WIDTHSET"),
    ]).writeto(filename, overwrite=True)


def build_fiber(frameid, *, apred="daily", telescope="apo25m", darkid=None,
                flatid=None, bpmid=None, yshift=None, average=50,
                clobber=False, unlock=False, verbose=False,
                reference_positions: Mapping[str, np.ndarray] | None = None):
    """Build the three trace-only Fiber calibration files."""
    load = apload.ApLoad(apred=apred, telescope=telescope)
    with product_build_lock(load, "fiber", frameid, clobber=clobber,
                            unlock=unlock, verbose=verbose) as (build, outputs):
        if not build:
            return

        shifts = np.broadcast_to(0.0 if yshift is None else yshift, (3,))
        _reduce(load, [frameid], darkid=darkid, flatid=flatid, bpmid=bpmid,
                clobber=clobber, verbose=verbose)
        frames = load.frame(frameid)
        for index, chip in enumerate(CHIPS):
            frame = frames[chip]
            reference = (reference_positions[chip] if reference_positions
                         is not None else _reference_positions(
                             load, index, yshift=shifts[index]))
            solution = find_traces(frame, reference, average=average)
            _write_trace(outputs[index], solution, frame["header"])


def build_sparse(frames, *, apred="daily", telescope="apo25m", darkid=None,
                 flatid=None, bpmid=None, fiberid=None, darkframes=None,
                 maxread=None, dmax=21, average=50, threshold=0.2,
                 clobber=False, unlock=False, verbose=False):
    """Build the combined Sparse image and its three empirical PSFs."""
    exposures = np.atleast_1d(frames).astype(int).tolist()
    if not exposures:
        raise ValueError("frames must contain at least one exposure")
    load = apload.ApLoad(apred=apred, telescope=telescope)
    outid = exposures[0]
    with product_build_lock(load, "sparse", outid, clobber=clobber,
                            unlock=unlock, verbose=verbose) as (build, outputs):
        if not build:
            return

        _reduce(load, exposures, darkid=darkid, flatid=flatid, bpmid=bpmid,
                maxread=maxread, clobber=clobber, verbose=verbose)
        dark_exposures = ([] if darkframes is None else [
            value for value in np.atleast_1d(darkframes).astype(int).tolist()
            if value > 0
        ])
        if dark_exposures:
            _reduce(load, dark_exposures, darkid=darkid, bpmid=bpmid,
                    maxread=maxread, clobber=clobber, verbose=verbose)
        frames_by_exposure = {
            exposure: load.frame(exposure) for exposure in exposures
        }
        dark_frames_by_exposure = {
            exposure: load.frame(exposure) for exposure in dark_exposures
        }
        chip_images = []
        for index, chip in enumerate(CHIPS):
            frame = average_calibration_frames(
                frames_by_exposure[exposure][chip]
                for exposure in exposures)
            if dark_exposures:
                dark_frame = average_calibration_frames(
                    dark_frames_by_exposure[exposure][chip]
                    for exposure in dark_exposures)
                frame["flux"] -= dark_frame["flux"]
            chip_images.append(frame["flux"].astype(np.float32))
            reference = _reference_positions(load, index, fiberid=fiberid)
            solution = find_traces(
                frame, reference, average=average, threshold=threshold)
            profiles = build_empirical_psf(
                frame, solution, half_width=int(dmax),
                smooth_columns=average)
            saveepsf(outputs[index + 1], profiles, header=frame["header"],
                     compress=False)
        fits.writeto(outputs[0], np.stack(chip_images), overwrite=True)


def build_psf(frameid, *, apred="daily", telescope="apo25m", darkid=None,
              flatid=None, bpmid=None, sparseid=None, fiberid=None,
              littrowid=None, average=50, clobber=False, unlock=False,
              verbose=False):
    """Build complete PSF, EPSF, and ETrace products for three chips."""
    if sparseid is None or int(sparseid) <= 0:
        raise ValueError("sparseid is required to build a full PSF calibration")
    load = apload.ApLoad(apred=apred, telescope=telescope)
    with product_build_lock(load, "psf", frameid, clobber=clobber,
                            unlock=unlock, verbose=verbose) as (build, outputs):
        if not build:
            return

        psf_files = outputs[0:3]
        epsf_files = outputs[3:6]
        trace_files = outputs[6:9]
        _reduce(load, [frameid], darkid=darkid, flatid=flatid, bpmid=bpmid,
                littrowid=littrowid, clobber=clobber, verbose=verbose)
        frames = load.frame(frameid)
        for index, chip in enumerate(CHIPS):
            frame = frames[chip]
            reference = _reference_positions(load, index, fiberid=fiberid)
            solution = find_traces(frame, reference, average=average)
            profiles = build_empirical_psf(
                frame, solution, half_width=7, smooth_columns=average)
            sparse_profiles = loadepsf(
                load.filename("EPSF", num=sparseid, chip=chip))
            profiles = _combine_profiles(profiles, sparse_profiles)
            _write_psf(psf_files[index], frame, solution, profiles)
            saveepsf(epsf_files[index], profiles, header=frame["header"],
                     compress=False)
            _write_trace(trace_files[index], solution, frame["header"])


def _validate_model_grid(profiles, labels, offsets):
    profiles = np.asarray(profiles, dtype=float)
    labels = np.asarray(labels, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    if profiles.ndim != 3 or profiles.shape[:2] != labels.shape[1:]:
        raise ValueError("profiles and labels have incompatible grid dimensions")
    if labels.shape[0] != 2 or offsets.ndim != 1:
        raise ValueError("labels must contain X/Y grids and offsets must be 1-D")
    if profiles.shape[2] != offsets.size:
        raise ValueError("profile length must match the offset grid")
    if not np.all(np.isfinite(profiles)) or not np.all(np.isfinite(labels)):
        raise ValueError("ModelPSF grid contains nonfinite values")
    if np.any(profiles < 0):
        raise ValueError("ModelPSF profiles cannot be negative")
    # np.trapezoid was added in NumPy 2.0; the production DRP environment
    # still uses an older NumPy where np.trapz is the compatible spelling.
    normalization = np.trapz(profiles, offsets, axis=2)
    if np.any(normalization <= 0):
        raise ValueError("ModelPSF contains an empty profile")
    normalized = profiles / normalization[:, :, None]
    # PSF.gridinterp() converts query coordinates to float64.  Keeping the
    # profile grid float64 ensures that its direct-corner and interpolated
    # branches return the same dtype, which is required by older Numba.
    return (normalized.astype(np.float64), labels.astype(np.float32),
            offsets.astype(np.float32))


def _write_model_grid(filename, profiles, labels, offsets, *, apred,
                      sparseid, psfid, nfbin, ncbin):
    profiles, labels, offsets = _validate_model_grid(
        profiles, labels, offsets)
    primary = fits.PrimaryHDU(profiles)
    primary.header["TYPE"] = "grid"
    primary.header["LOG"] = False
    primary.header["EXTNAME"] = "DATA"
    primary.header["APRED"] = str(apred)
    primary.header["SPARSEID"] = str(sparseid)
    primary.header["PSFID"] = str(psfid)
    primary.header["NFBIN"] = int(nfbin)
    primary.header["NCBIN"] = int(ncbin)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([
        primary, fits.ImageHDU(labels, name="LABELS"),
        fits.ImageHDU(offsets, name="X"),
    ]).writeto(filename, overwrite=True)


def build_modelpsf(name, *, sparseid, psfid, apred="daily",
                   telescope="apo25m", nfbin=5, ncbin=200,
                   clobber=False, unlock=False, verbose=False):
    """Build three detector-wide ModelPSF profile grids synchronously."""
    if sparseid is None or int(sparseid) <= 0:
        raise ValueError("sparseid is required")
    if psfid is None or int(psfid) <= 0:
        raise ValueError("psfid is required")
    if int(nfbin) <= 0 or int(ncbin) <= 0:
        raise ValueError("nfbin and ncbin must be positive")
    load = apload.ApLoad(apred=apred, telescope=telescope)
    with product_build_lock(load, "modelpsf", name, clobber=clobber,
                            unlock=unlock, verbose=verbose) as (build, outputs):
        if not build:
            return

        sparse_file = load.filename("Sparse", num=sparseid)
        epsf_files = [load.filename("EPSF", num=psfid, chip=chip)
                      for chip in CHIPS]
        missing = [
            filename for filename in [sparse_file, *epsf_files]
            if (not Path(filename).is_file()
                or Path(filename).stat().st_size == 0)
        ]
        if missing:
            raise FileNotFoundError(
                "Missing ModelPSF inputs: " + ", ".join(missing))

        for chip, epsf_file, output in zip(CHIPS, epsf_files, outputs):
            _, mean_x, mean_y, profiles, offsets, _ = makeprofilegrid(
                epsf_file, sparse_file, nfbin=nfbin, ncbin=ncbin,
                verbose=verbose)
            labels = np.stack((mean_x, mean_y))
            _write_model_grid(
                output, profiles, labels, offsets, apred=apred,
                sparseid=sparseid, psfid=psfid, nfbin=nfbin, ncbin=ncbin)
            if verbose:
                print(f" writing ModelPSF chip {chip}: {output}")
        if not all(Path(filename).is_file() and Path(filename).stat().st_size > 0
                   for filename in outputs):
            raise RuntimeError(f"ModelPSF {name} did not create all chip files")
