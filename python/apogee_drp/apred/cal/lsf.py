"""Measure and build APOGEE line-spread-function calibrations.

The implementation replaces the ``mklsf.pro``/``aplsf.pro`` orchestration
with a NumPy-native Gaussian line-width fit.  The output parameter vectors
retain the historical ``LSF_GH`` layout and are therefore compatible with
the existing APOGEE LSF readers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks
from scipy.stats import theilslopes

from ...utils import apload
from .utils import product_build_lock

CHIPS = ("a", "b", "c")
BADERR = 1.0e10

__all__ = [
    "CHIPS", "build_lsf", "evaluate_gauss_hermite_lsf",
    "fit_gauss_hermite_chip", "fit_sigma_model", "gaussian_lsf_array",
    "measure_line_widths", "pack_ghlsf_parameters",
]


def _sanitize_frame(frame):
    flux = np.asarray(frame["flux"], dtype=float).copy()
    err = np.asarray(frame["err"], dtype=float).copy()
    mask = np.asarray(frame.get("mask", np.zeros_like(flux)), dtype=np.uint32).copy()
    if flux.ndim != 2 or err.shape != flux.shape or mask.shape != flux.shape:
        raise ValueError("flux, err, and mask must be matching 2-D arrays")
    bad = ~np.isfinite(flux) | ~np.isfinite(err) | (err <= 0)
    flux[bad] = 0
    err[bad] = BADERR
    mask[bad] |= 1
    return {**frame, "flux": flux, "err": err, "mask": mask}


def combine_frames(frames):
    """Add same-dither spectra and propagate their independent errors."""
    if not frames:
        raise ValueError("at least one frame is required")
    clean = [_sanitize_frame(frame) for frame in frames]
    shape = clean[0]["flux"].shape
    if any(frame["flux"].shape != shape for frame in clean):
        raise ValueError("all input frames must have the same shape")
    dithers = [float(frame.get("header", {}).get("DITHPIX", 0.0)) for frame in clean]
    if np.ptp(dithers) >= 0.01:
        raise NotImplementedError(
            "LSF combination of different dither positions is not yet supported")
    return {
        "flux": np.sum([frame["flux"] for frame in clean], axis=0),
        "err": np.sqrt(np.sum([frame["err"] ** 2 for frame in clean], axis=0)),
        "mask": np.bitwise_or.reduce([frame["mask"] for frame in clean]),
        "header": clean[0].get("header", fits.Header()).copy(),
    }


def remove_continuum(flux, width=101):
    if int(width) < 3 or int(width) % 2 == 0:
        raise ValueError("continuum width must be an odd integer of at least 3")
    values = np.asarray(flux, dtype=float)
    return values - median_filter(values, size=(1, int(width)), mode="nearest")


def measure_line_widths(flux, err, mask=None, *, threshold_sigma=10.0,
                        half_width=4, minimum_distance=5):
    """Measure emission-line centers and Gaussian-equivalent widths."""
    values = np.asarray(flux, dtype=float)
    errors = np.asarray(err, dtype=float)
    if values.ndim != 1 or errors.shape != values.shape:
        raise ValueError("flux and err must be matching one-dimensional arrays")
    bad = ~np.isfinite(values) | ~np.isfinite(errors) | (errors <= 0)
    if mask is not None:
        if np.shape(mask) != values.shape:
            raise ValueError("mask must match flux")
        bad |= np.asarray(mask) != 0
    usable_error = errors[~bad & (errors < BADERR)]
    if usable_error.size == 0:
        return np.empty((0, 4), dtype=float)
    noise = max(float(np.nanmedian(usable_error)), np.finfo(float).eps)
    profile = values.copy()
    profile[bad] = 0
    peaks, properties = find_peaks(
        profile, height=float(threshold_sigma) * noise,
        distance=int(minimum_distance))
    measurements = []
    for peak, height in zip(peaks, properties["peak_heights"]):
        lo, hi = max(peak - half_width, 0), min(peak + half_width + 1, len(profile))
        x = np.arange(lo, hi, dtype=float)
        y = np.maximum(profile[lo:hi], 0.0)
        good = ~bad[lo:hi]
        total = np.sum(y[good])
        if good.sum() < 3 or total <= 0:
            continue
        center0 = np.sum(x[good] * y[good]) / total
        variance = np.sum((x[good] - center0) ** 2 * y[good]) / total
        sigma0 = np.sqrt(max(variance, 0.3 ** 2))
        def gaussian(xx, amplitude, center, sigma, baseline):
            return baseline + amplitude * np.exp(-0.5 * ((xx - center) / sigma) ** 2)
        try:
            fitted, _ = curve_fit(
                gaussian, x[good], profile[lo:hi][good],
                p0=(height, center0, sigma0, 0.0),
                bounds=([0, lo - 1, 0.3, -np.inf],
                        [np.inf, hi, 8.0, np.inf]), maxfev=2000)
            center, sigma = fitted[1:3]
        except (RuntimeError, ValueError):
            center, sigma = center0, sigma0
        if 0.3 <= sigma <= 8.0:
            measurements.append((center, sigma, float(height), total))
    return np.asarray(measurements, dtype=float).reshape(-1, 4)


def fit_sigma_model(measurements, *, npix=2048, order=1,
                    rejection_sigma=4.0):
    """Robustly fit LSF sigma as a polynomial in detector position."""
    lines = np.asarray(measurements, dtype=float)
    if lines.ndim != 2 or lines.shape[1] < 2:
        raise ValueError("measurements must have columns center and sigma")
    good = np.isfinite(lines[:, 0]) & np.isfinite(lines[:, 1]) & (lines[:, 1] > 0)
    if good.sum() < order + 1:
        raise ValueError("not enough valid lines to fit the sigma model")
    xoffset = -(int(npix) - 1) / 2.0
    x = lines[:, 0] + xoffset
    for iteration in range(3):
        if iteration == 0 and order == 1 and good.sum() >= 3:
            slope, intercept, _, _ = theilslopes(lines[good, 1], x[good])
            coefficient = np.array([intercept, slope])
        else:
            coefficient = np.polynomial.polynomial.polyfit(
                x[good], lines[good, 1], order)
        residual = lines[:, 1] - np.polynomial.polynomial.polyval(x, coefficient)
        residual_center = np.nanmedian(residual[good])
        centered = residual - residual_center
        scale = 1.4826 * np.nanmedian(np.abs(centered[good]))
        if not np.isfinite(scale):
            break
        if scale == 0:
            scale = np.finfo(float).eps * max(1.0, np.nanmax(lines[good, 1]))
        new_good = good & (np.abs(centered) < rejection_sigma * scale)
        if new_good.sum() < order + 1 or np.array_equal(new_good, good):
            break
        good = new_good
    return coefficient, good, xoffset


def _parameter_vector(coefficient, xoffset):
    """Pack a Gaussian model into the historical LSF_GH parameter layout."""
    coefficient = np.asarray(coefficient, dtype=float)
    return np.concatenate((
        [1.0, float(xoffset), 0.0, float(len(coefficient) - 1)], coefficient))


def gaussian_lsf_array(parameters, npix, *, nlsfpix=None):
    """Evaluate normalized binned-pixel Gaussian LSFs across a detector."""
    parameters = np.asarray(parameters, dtype=float)
    order = int(parameters[3])
    coefficient = parameters[4:5 + order]
    x = np.arange(int(npix), dtype=float)
    sigma = np.polynomial.polynomial.polyval(x + parameters[1], coefficient)
    sigma = np.clip(sigma, 0.3, None)
    if nlsfpix is None:
        nlsfpix = int(2 * np.ceil(np.nanmax(sigma) * 5) + 1)
    nlsfpix = max(int(nlsfpix) | 1, 3)
    offsets = np.arange(nlsfpix, dtype=float) - nlsfpix // 2
    result = np.exp(-0.5 * (offsets[:, None] / sigma[None, :]) ** 2)
    result /= result.sum(axis=0, keepdims=True)
    return result.astype(np.float32)


def pack_ghlsf_parameters(binsize, xoffset, porder, ghcoefs, *,
                          wproftype=1, wporder=(0, 0),
                          wcoefs=(0.03, 3.5)):
    """Pack coefficients into the historical APOGEE LSF_GH layout."""
    porder = np.asarray(porder)
    wporder = np.asarray(wporder)
    if (porder.ndim != 1 or porder.size == 0 or
            np.any(porder < 0) or np.any(porder != porder.astype(int))):
        raise ValueError("porder must contain nonnegative integers")
    if (wporder.ndim != 1 or np.any(wporder < 0) or
            np.any(wporder != wporder.astype(int))):
        raise ValueError("wporder must contain nonnegative integers")
    porder = porder.astype(int)
    wporder = wporder.astype(int)
    if wporder.size != 2:
        raise ValueError(
            "Doppler-compatible APOGEE LSFs require two wing parameters")

    def flatten_coefficients(values, orders, name):
        values = np.asarray(values, dtype=float)
        required = int(np.sum(orders + 1))
        if values.ndim == 1:
            if values.size != required:
                raise ValueError(
                    f"{name} has {values.size} values; expected {required}")
            return values
        if values.ndim != 2 or values.shape[0] != len(orders):
            raise ValueError(
                f"{name} must have one row for each polynomial order")
        maximum = max((int(order) for order in orders), default=-1)
        if values.shape[1] < maximum + 1:
            raise ValueError(f"{name} does not contain enough coefficients")
        return np.concatenate([
            values[index, :order + 1]
            for index, order in enumerate(orders)
        ])

    ghflat = flatten_coefficients(ghcoefs, porder, "ghcoefs")
    parts = [
        np.array([
            float(binsize), float(xoffset), float(len(porder) - 1)
        ]),
        porder.astype(float),
        ghflat,
    ]
    wflat = flatten_coefficients(wcoefs, wporder, "wcoefs")
    parts.extend([
        np.array([float(wproftype), float(len(wporder))]),
        wporder.astype(float),
        wflat,
    ])
    return np.concatenate(parts)


def evaluate_gauss_hermite_lsf(parameters, centers, offsets, *,
                                positive=False, normalize=False):
    """Evaluate an APOGEE Gauss-Hermite LSF using Doppler."""
    from doppler.lsf import ghlsf

    centers = np.atleast_1d(np.asarray(centers, dtype=float))
    offsets = np.asarray(offsets, dtype=float)
    if centers.ndim != 1 or offsets.ndim != 1:
        raise ValueError("centers and offsets must be one-dimensional")
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(offsets)):
        raise ValueError("centers and offsets must be finite")

    result = np.asarray(ghlsf(offsets, centers, parameters), dtype=float)
    if result.ndim == 1:
        result = result[None, :]
    if positive:
        result = np.maximum(result, 0.0)
    if normalize:
        total = np.sum(result, axis=1, keepdims=True)
        if np.any(~np.isfinite(total)) or np.any(total <= 0):
            raise ValueError("Gauss-Hermite LSF has invalid normalization")
        result = result / total
    return result[0] if len(centers) == 1 else result


def fit_lsf_chip(frame, *, fibers=None, threshold_sigma=10.0,
                 continuum_width=101, polynomial_order=1, nlsfpix=None):
    """Fit Gaussian LSF parameters and arrays for one detector chip."""
    clean = _sanitize_frame(frame)
    residual = remove_continuum(clean["flux"], continuum_width)
    nfiber, npix = residual.shape
    selected = np.arange(nfiber) if fibers is None else np.asarray(fibers, dtype=int)
    if np.any((selected < 0) | (selected >= nfiber)):
        raise ValueError("fibers contains an out-of-range index")
    parameters = np.full((nfiber, 5 + polynomial_order), np.nan, dtype=float)
    diagnostics = []
    for fiber in selected:
        lines = measure_line_widths(
            residual[fiber], clean["err"][fiber], clean["mask"][fiber],
            threshold_sigma=threshold_sigma)
        if len(lines) < polynomial_order + 1:
            continue
        coefficient, used, xoffset = fit_sigma_model(
            lines, npix=npix, order=polynomial_order)
        parameters[fiber] = _parameter_vector(coefficient, xoffset)
        for line, accepted in zip(lines, used):
            diagnostics.append((fiber, *line, bool(accepted)))
    valid = np.flatnonzero(np.isfinite(parameters[:, 0]))
    if valid.size == 0:
        raise ValueError("No fibers had enough emission lines for an LSF fit")
    for fiber in range(nfiber):
        if not np.isfinite(parameters[fiber, 0]):
            nearest = valid[np.argmin(abs(valid - fiber))]
            parameters[fiber] = parameters[nearest]
    max_sigma = max(np.polynomial.polynomial.polyval(
        np.array([parameters[f, 1], parameters[f, 1] + npix - 1]),
        parameters[f, 4:]).max() for f in range(nfiber))
    width = nlsfpix or int(2 * np.ceil(max_sigma * 5) + 1)
    arrays = np.zeros((width, nfiber, npix), dtype=np.float32)
    for fiber in range(nfiber):
        arrays[:, fiber] = gaussian_lsf_array(
            parameters[fiber], npix, nlsfpix=width)
    dtype = [("fiber", np.int16), ("center", np.float32),
             ("sigma", np.float32), ("height", np.float32),
             ("flux", np.float32), ("accepted", bool)]
    return parameters, arrays, np.asarray(diagnostics, dtype=dtype)


def _initial_gh_coefficients(measurements, porder, wporder):
    """Return IDL-like starting coefficients for a full LSF fit."""
    gh = np.zeros(int(np.sum(porder + 1)), dtype=float)
    gh[0] = np.nanmedian(measurements[:, 1])
    wings = np.zeros(int(np.sum(wporder + 1)), dtype=float)
    if len(wporder):
        wings[0] = 0.03
    if len(wporder) > 1:
        wings[wporder[0] + 1] = 3.5
    return gh, wings


def _shared_parameter_bounds(porder, wporder, xscale):
    """Construct conservative bounds for GH and wing polynomials."""
    lower, upper = [], []
    for parameter, order in enumerate(porder):
        for degree in range(order + 1):
            scale = max(float(xscale) ** degree, 1.0)
            if parameter == 0 and degree == 0:
                lo, hi = 0.3, 8.0
            elif parameter == 0:
                lo, hi = -8.0 / scale, 8.0 / scale
            else:
                lo, hi = -1.0 / scale, 1.0 / scale
            lower.append(lo)
            upper.append(hi)
    for parameter, order in enumerate(wporder):
        for degree in range(order + 1):
            scale = max(float(xscale) ** degree, 1.0)
            if parameter == 0 and degree == 0:
                lo, hi = 0.0, 0.5
            elif parameter == 1 and degree == 0:
                lo, hi = 1.5, 20.0
            elif parameter == 0:
                lo, hi = -0.5 / scale, 0.5 / scale
            else:
                lo, hi = -20.0 / scale, 20.0 / scale
            lower.append(lo)
            upper.append(hi)
    return np.asarray(lower), np.asarray(upper)


def _fit_gh_fiber(residual, error, mask, *, porder, wporder,
                  wproftype, binsize, xoffset, threshold_sigma,
                  nsigfit, max_nfev, initial_shared=None):
    """Simultaneously fit emission lines and one fiber's full LSF."""
    measurements = measure_line_widths(
        residual, error, mask, threshold_sigma=threshold_sigma)
    minimum_lines = max(2, len(porder))
    if len(measurements) < minimum_lines:
        raise ValueError("not enough emission lines for Gauss-Hermite fit")

    segments = []
    for line in measurements:
        center, sigma = line[:2]
        half_width = max(4, int(np.ceil(float(nsigfit) * sigma)))
        lo = max(0, int(np.floor(center)) - half_width)
        hi = min(len(residual), int(np.floor(center)) + half_width + 1)
        pixel = np.arange(lo, hi, dtype=float)
        good = (
            np.isfinite(residual[lo:hi])
            & np.isfinite(error[lo:hi])
            & (error[lo:hi] > 0)
            & (error[lo:hi] < BADERR)
            & (np.asarray(mask[lo:hi]) == 0)
        )
        if np.count_nonzero(good) >= 5:
            segments.append((
                line,
                pixel[good],
                residual[lo:hi][good],
                error[lo:hi][good],
            ))
    if len(segments) < minimum_lines:
        raise ValueError("not enough uncontaminated lines for Gauss-Hermite fit")

    measurements = np.asarray([segment[0] for segment in segments])
    nlines = len(segments)
    gh0, wing0 = _initial_gh_coefficients(
        measurements, porder, wporder)
    if initial_shared is not None:
        initial_shared = np.asarray(initial_shared, dtype=float)
        if initial_shared.size == gh0.size + wing0.size:
            gh0 = initial_shared[:gh0.size].copy()
            wing0 = initial_shared[gh0.size:].copy()
    line0 = np.empty(3 * nlines, dtype=float)
    line_lower = np.empty_like(line0)
    line_upper = np.empty_like(line0)
    for index, segment in enumerate(segments):
        measurement, _, _, segment_error = segment
        center, _, _, flux = measurement
        noise = max(float(np.nanmedian(segment_error)), np.finfo(float).eps)
        line0[3 * index:3 * index + 3] = [
            max(float(flux), noise), float(center), 0.0]
        line_lower[3 * index:3 * index + 3] = [
            0.0, center - 1.5, -5.0 * noise]
        line_upper[3 * index:3 * index + 3] = [
            np.inf, center + 1.5, 5.0 * noise]

    xscale = max(abs(xoffset), abs(len(residual) - 1 + xoffset), 1.0)
    shared_lower, shared_upper = _shared_parameter_bounds(
        porder, wporder, xscale)
    initial = np.concatenate((line0, gh0, wing0))
    lower = np.concatenate((line_lower, shared_lower))
    upper = np.concatenate((line_upper, shared_upper))
    ngh = len(gh0)

    def residual_function(values):
        gh = values[3 * nlines:3 * nlines + ngh]
        wings = values[3 * nlines + ngh:]
        packed = pack_ghlsf_parameters(
            binsize, xoffset, porder, gh, wproftype=wproftype,
            wporder=wporder, wcoefs=wings)
        output = []
        for index, (_, pixel, observed, uncertainty) in enumerate(segments):
            area, center, background = values[
                3 * index:3 * index + 3]
            offsets = pixel - center
            profile = evaluate_gauss_hermite_lsf(
                packed, center, offsets)
            if (not np.all(np.isfinite(profile)) or
                    np.any(profile < 0)):
                output.append(np.full(len(pixel), 1.0e6))
            else:
                output.append(
                    (observed - (area * profile + background))
                    / uncertainty)
        return np.concatenate(output)

    result = least_squares(
        residual_function, initial, bounds=(lower, upper),
        loss="soft_l1", f_scale=2.0, max_nfev=int(max_nfev),
        x_scale="jac")
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(
            "Gauss-Hermite optimization failed: " + result.message)

    gh = result.x[3 * nlines:3 * nlines + ngh]
    wings = result.x[3 * nlines + ngh:]
    packed = pack_ghlsf_parameters(
        binsize, xoffset, porder, gh, wproftype=wproftype,
        wporder=wporder, wcoefs=wings)
    return packed, measurements, result


def fit_gauss_hermite_chip(
        frame, *, fibers=None, threshold_sigma=10.0,
        continuum_width=101, porder=(1, 1, 1, 1, 1, 0),
        wporder=(0, 0), wproftype=1, binsize=1.0, nsigfit=8.0,
        nlsfpix=None, max_nfev=2000):
    """Fit Doppler-compatible Gauss-Hermite LSFs for one detector chip."""
    from doppler.lsf import unpack_ghlsf_params

    clean = _sanitize_frame(frame)
    residual = remove_continuum(clean["flux"], continuum_width)
    nfiber, npix = residual.shape
    selected = (
        np.arange(nfiber) if fibers is None
        else np.asarray(fibers, dtype=int))
    if np.any((selected < 0) | (selected >= nfiber)):
        raise ValueError("fibers contains an out-of-range index")
    raw_porder = np.asarray(porder)
    raw_wporder = np.asarray(wporder)
    if (raw_porder.ndim != 1 or raw_porder.size == 0 or
            np.any(raw_porder < 0) or
            np.any(raw_porder != raw_porder.astype(int))):
        raise ValueError("porder must contain nonnegative integers")
    if (raw_wporder.ndim != 1 or np.any(raw_wporder < 0) or
            np.any(raw_wporder != raw_wporder.astype(int))):
        raise ValueError("wporder must contain nonnegative integers")
    porder = raw_porder.astype(int)
    wporder = raw_wporder.astype(int)
    if wporder.size != 2:
        raise ValueError(
            "the supported wing profiles require two wing parameters")
    if int(wproftype) != wproftype or not 1 <= int(wproftype) <= 5:
        raise ValueError("wproftype must be an integer from 1 through 5")
    if not np.isfinite(binsize) or binsize <= 0:
        raise ValueError("binsize must be positive and finite")
    if not np.isfinite(nsigfit) or nsigfit <= 0:
        raise ValueError("nsigfit must be positive and finite")
    if int(max_nfev) != max_nfev or int(max_nfev) <= 0:
        raise ValueError("max_nfev must be a positive integer")

    xoffset = -(npix - 1) / 2.0
    fitted = {}
    diagnostics = []
    previous_shared = None
    for fiber in selected:
        try:
            parameters, lines, result = _fit_gh_fiber(
                residual[fiber], clean["err"][fiber],
                clean["mask"][fiber], porder=porder,
                wporder=wporder, wproftype=int(wproftype),
                binsize=float(binsize), xoffset=xoffset,
                threshold_sigma=threshold_sigma, nsigfit=nsigfit,
                max_nfev=max_nfev, initial_shared=previous_shared)
        except ValueError:
            continue
        fitted[int(fiber)] = parameters
        nline_parameters = 3 * len(lines)
        previous_shared = result.x[nline_parameters:].copy()
        accepted = bool(result.success)
        diagnostics.extend([
            (fiber, *line, accepted) for line in lines
        ])
    if not fitted:
        raise ValueError(
            "No fibers had enough emission lines for a Gauss-Hermite fit")

    valid = np.array(sorted(fitted), dtype=int)
    template = fitted[int(valid[0])]
    parameters = np.empty((nfiber, len(template)), dtype=float)
    for fiber in range(nfiber):
        nearest = int(valid[np.argmin(np.abs(valid - fiber))])
        parameters[fiber] = fitted[nearest]

    if nlsfpix is None:
        maximum_sigma = 0.0
        pixels = np.array([0.0, float(npix - 1)])
        for row in parameters:
            unpacked = unpack_ghlsf_params(row)
            sigma = np.polynomial.polynomial.polyval(
                pixels + unpacked["Xoffset"],
                unpacked["GHcoefs"][0])
            maximum_sigma = max(maximum_sigma, float(np.nanmax(sigma)))
        nlsfpix = int(2 * np.ceil(5.0 * maximum_sigma) + 1)
    nlsfpix = max(3, int(nlsfpix) | 1)
    offsets = np.arange(nlsfpix, dtype=float) - nlsfpix // 2
    centers = np.arange(npix, dtype=float)
    arrays = np.empty((nlsfpix, nfiber, npix), dtype=np.float32)
    for fiber in range(nfiber):
        profile = evaluate_gauss_hermite_lsf(
            parameters[fiber], centers, offsets,
            positive=True, normalize=True)
        arrays[:, fiber] = profile.T.astype(np.float32)
        if fiber > 0:
            difference = np.max(np.abs(
                arrays[:, fiber, npix // 2]
                - arrays[:, fiber - 1, npix // 2]))
            if difference > 0.03:
                parameters[fiber] = parameters[fiber - 1]
                arrays[:, fiber] = arrays[:, fiber - 1]

    dtype = [("fiber", np.int16), ("center", np.float32),
             ("sigma", np.float32), ("height", np.float32),
             ("flux", np.float32), ("accepted", bool)]
    return parameters, arrays, np.asarray(diagnostics, dtype=dtype)


def _write_chip(filename, parameters, array, header, *, apred, frameid,
                method):
    primary = fits.PrimaryHDU(parameters.T.astype(np.float32), header=header.copy())
    primary.header["EXTNAME"] = "LSFPARS"
    primary.header["LSFMETH"] = str(method)
    primary.header["APRED"] = str(apred)
    primary.header["LSFID"] = int(frameid)
    image = fits.ImageHDU(array, name="LSF ARRAY")
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, image]).writeto(filename, overwrite=True)


def build_lsf(lsfid, waveid, *, apred="daily", telescope="apo25m",
              darkid=None, flatid=None, psfid=None, fiberid=None,
              fibers=None, threshold_sigma=10.0, continuum_width=101,
              polynomial_order=1, clobber=False, full=False,
              porder=(1, 1, 1, 1, 1, 0), wporder=(0, 0),
              wproftype=1, binsize=1.0, nsigfit=8.0,
              nlsfpix=None, max_nfev=2000,
              newwave=False, plot=False, nowait=False, unlock=False,
              verbose=False):
    """Process lamp frames and build three APOGEE LSF calibration files.

    Set full=True to fit the Doppler-backed Gauss-Hermite model used by
    the historical IDL calibration. The default retains the faster Gaussian
    model.
    """
    del fiberid, newwave, plot  # retained for API compatibility
    frames = [int(value) for value in np.atleast_1d(lsfid)]
    if not frames:
        raise ValueError("lsfid must contain at least one exposure")
    if psfid is None or int(psfid) <= 0:
        raise ValueError("psfid is required")
    if waveid is None:
        raise ValueError("waveid is required")
    load = apload.ApLoad(apred=apred, telescope=telescope)
    with product_build_lock(
        load, "lsf", frames[0], clobber=clobber, unlock=unlock,
        waittime=(0 if nowait else 10), verbose=verbose,
    ) as (build, outputs):
        if not build:
            return
        if len(outputs) != len(CHIPS) + 1:
            raise RuntimeError(
                f"LSF product {frames[0]} resolved to {len(outputs)} files; "
                f"expected {len(CHIPS) + 1}")
        from ..process import process
        process(
            frames,
            load=load,
            darkid=darkid,
            flatid=flatid,
            psfid=int(psfid),
            waveid=waveid,
            fluxid=None,
            doproc=True,
            skywave=True,
            clobber=clobber,
            onedclobber=clobber,
            unlock=unlock,
            verbose=verbose,
        )

        spectra = [load.spectrum(frame) for frame in frames]
        all_diagnostics = []
        for chip, output in zip(CHIPS, outputs[:3]):
            # ApLoad uses [pixel, fiber], while the LSF fitter uses
            # [fiber, pixel].
            chip_frames = [
                {
                    "header": spectrum[chip]["header"],
                    "flux": spectrum[chip]["flux"].T,
                    "err": spectrum[chip]["err"].T,
                    "mask": spectrum[chip]["mask"].T,
                }
                for spectrum in spectra
            ]
            combined = combine_frames(chip_frames)
            if full:
                parameters, array, diagnostics = fit_gauss_hermite_chip(
                    combined, fibers=fibers,
                    threshold_sigma=threshold_sigma,
                    continuum_width=continuum_width,
                    porder=porder, wporder=wporder,
                    wproftype=wproftype, binsize=binsize,
                    nsigfit=nsigfit, nlsfpix=nlsfpix,
                    max_nfev=max_nfev)
                method = "GAUSS-HERMITE"
            else:
                parameters, array, diagnostics = fit_lsf_chip(
                    combined, fibers=fibers,
                    threshold_sigma=threshold_sigma,
                    continuum_width=continuum_width,
                    polynomial_order=polynomial_order,
                    nlsfpix=nlsfpix)
                method = "GAUSSIAN"
            _write_chip(output, parameters, array, combined["header"],
                        apred=apred, frameid=frames[0], method=method)
            if len(diagnostics):
                table = Table(diagnostics)
                table["chip"] = np.full(len(table), chip)
                all_diagnostics.append(table)
        diagnostic_hdus = [fits.PrimaryHDU()]
        for table in all_diagnostics:
            diagnostic_hdus.append(fits.table_to_hdu(table))
            diagnostic_hdus[-1].header["EXTNAME"] = f"LINES-{table['chip'][0]}"
        fits.HDUList(diagnostic_hdus).writeto(outputs[3], overwrite=True)
        if not load.product_exists("lsf", frames[0]):
            raise RuntimeError(
                f"LSF {frames[0]} did not create all registered files")
