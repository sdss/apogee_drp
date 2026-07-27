"""Dither-shift measurement translated from ``apdithershift.pro``.

The public :func:`dither_shift` routine supports both algorithms in the IDL
source: cross-correlation of every fiber/chip spectrum and matching fitted
emission-line centroids.  APOGEE Python arrays use ``[fiber, pixel]`` order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


@dataclass
class DitherShiftResult:
    """Result fields corresponding to the IDL ``shiftstr`` structure."""

    type: str
    shiftfit: np.ndarray
    shifterr: np.float32
    chipshift: np.ndarray
    chipfit: np.ndarray
    shiftarr: np.ndarray | None = None


@dataclass
class LinePeak:
    """One fitted line centroid and the errors used by APDITHERSHIFT."""

    fiber: int
    gaussx: float
    center_error: float
    height_error: float


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        for key in (name, name.lower(), name.upper()):
            if key in value:
                return value[key]
    for key in (name, name.lower(), name.upper()):
        if hasattr(value, key):
            return getattr(value, key)
    raise ValueError(f"required field {name!r} is missing")


def _chips(frame: Any) -> list[Any]:
    return [_field(frame, name) for name in ("chipa", "chipb", "chipc")]


def _validate_frames(frame1: Any, frame2: Any) -> tuple[list[Any], list[Any]]:
    chips1, chips2 = _chips(frame1), _chips(frame2)
    shape = None
    for chip in chips1 + chips2:
        for name in ("header", "flux", "err", "mask"):
            _field(chip, name)
        flux = np.asarray(_field(chip, "flux"))
        if flux.ndim != 2:
            raise ValueError("chip flux arrays must have [fiber, pixel] shape")
        if shape is None:
            shape = flux.shape
        elif flux.shape != shape:
            raise ValueError("all chip flux arrays must have the same shape")
    return chips1, chips2


def _robust_mean(values: Any) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    good = np.isfinite(values)
    if not np.any(good):
        return np.nan, np.nan
    for _ in range(5):
        sample = values[good]
        center = np.median(sample)
        mad = np.median(np.abs(sample - center))
        if mad == 0 or not np.isfinite(mad):
            break
        keep = np.isfinite(values) & (np.abs(values - center) <= 5.0 * 1.4826 * mad)
        if np.array_equal(keep, good):
            break
        good = keep
    sample = values[good]
    return float(np.mean(sample)), float(np.std(sample))


def _robust_line(x: Any, y: Any) -> np.ndarray:
    """Return IDL polynomial ordering ``[constant, slope]``."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) < 2:
        return np.full(2, np.nan, dtype=np.float32)
    for _ in range(5):
        slope, constant = np.polyfit(x[good], y[good], 1)
        residual = y - (constant + slope * x)
        mad = np.median(np.abs(residual[good] - np.median(residual[good])))
        if mad == 0 or not np.isfinite(mad):
            break
        keep = good & (np.abs(residual) <= 5.0 * mad)
        if np.count_nonzero(keep) < 2 or np.array_equal(keep, good):
            break
        good = keep
    slope, constant = np.polyfit(x[good], y[good], 1)
    return np.asarray([constant, slope], dtype=np.float32)


def _gaussian_polynomial(
    x: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
    c0: float,
    c1: float,
    c2: float,
    c3: float,
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + (
        c0 + x * (c1 + x * (c2 + x * c3))
    )


def _fit_correlation_peak(lags: np.ndarray, correlation: np.ndarray) -> float:
    if not np.all(np.isfinite(correlation)) or np.sum(correlation) == 0:
        return np.nan
    baseline = np.polyfit(lags, correlation, 1)
    center0 = float(lags[np.argmax(correlation)])
    p0 = [
        max(float(np.max(correlation) - np.median(correlation)), 1e-12),
        center0,
        2.0,
        baseline[1],
        baseline[0],
        0.0,
        0.0,
    ]
    scale = max(float(np.max(np.abs(correlation))), 1.0)
    try:
        pars, _ = curve_fit(
            _gaussian_polynomial,
            lags,
            correlation,
            p0=p0,
            bounds=(
                [0.0, -11.0, 0.05, -np.inf, -np.inf, -np.inf, -np.inf],
                [np.inf, 11.0, 20.0, np.inf, np.inf, np.inf, np.inf],
            ),
            x_scale=[scale, 1, 1, scale, scale, scale, scale],
            maxfev=20_000,
        )
        use = np.abs(lags - pars[1]) < 3.0 * abs(pars[2])
        if np.count_nonzero(use) >= 7:
            pars, _ = curve_fit(
                _gaussian_polynomial,
                lags[use],
                correlation[use],
                p0=pars,
                bounds=(
                    [0.0, -11.0, 0.05, -np.inf, -np.inf, -np.inf, -np.inf],
                    [np.inf, 11.0, 20.0, np.inf, np.inf, np.inf, np.inf],
                ),
                maxfev=20_000,
            )
        return float(pars[1])
    except (RuntimeError, ValueError, FloatingPointError):
        peak = int(np.argmax(correlation))
        if peak == 0 or peak == correlation.size - 1:
            return np.nan
        left, middle, right = correlation[peak - 1 : peak + 2]
        denominator = left - 2.0 * middle + right
        if denominator == 0:
            return float(lags[peak])
        return float(lags[peak] + 0.5 * (left - right) / denominator)


def _plugged_fibers(plugmap: Any | None, nfibers: int) -> np.ndarray:
    if plugmap is None:
        return np.arange(nfibers, dtype=int)
    fiberdata = _field(plugmap, "fiberdata")
    spectrograph = np.asarray(_field(fiberdata, "spectrographid"))
    fiberid = np.asarray(_field(fiberdata, "fiberid"))
    fibers = 300 - fiberid[spectrograph == 2].astype(int)
    if np.any((fibers < 0) | (fibers >= nfibers)):
        raise ValueError("plugmap contains a fiber outside the frame")
    return fibers


def _cross_correlation_shifts(
    chips1: list[Any],
    chips2: list[Any],
    fibers: np.ndarray,
    *,
    normalize_object: bool,
) -> np.ndarray:
    npix = np.asarray(_field(chips1[0], "flux")).shape[1]
    lags = np.arange(-10, 11, dtype=np.float64)
    lo, stop = 21, npix - 21
    if stop <= lo:
        raise ValueError("spectra must contain at least 43 pixels")
    output = np.full((fibers.size, 3), -100.0, dtype=np.float32)
    for chip_index, (chip1, chip2) in enumerate(zip(chips1, chips2)):
        first = np.asarray(_field(chip1, "flux"), dtype=np.float64)[fibers].copy()
        second = np.asarray(_field(chip2, "flux"), dtype=np.float64)[fibers].copy()
        if normalize_object:
            first /= np.maximum(
                median_filter(first, size=(1, 100), mode="nearest"), 1.0
            )
            second /= np.maximum(
                median_filter(second, size=(1, 100), mode="nearest"), 1.0
            )
        correlations = np.empty((fibers.size, lags.size), dtype=np.float64)
        for column, lag in enumerate(lags.astype(int)):
            correlations[:, column] = np.sum(
                first[:, lo:stop] * second[:, lo + lag : stop + lag], axis=1
            )
        for row in range(fibers.size):
            measured = _fit_correlation_peak(lags, correlations[row])
            if np.isfinite(measured):
                output[row, chip_index] = np.float32(measured)
    return output


def _default_peak_finder(chip: Any) -> list[LinePeak]:
    """Approximate APPEAKFIT with local Gaussian fits for isolated lines."""

    flux = np.asarray(_field(chip, "flux"), dtype=np.float64)
    peaks: list[LinePeak] = []
    xall = np.arange(flux.shape[1], dtype=np.float64)
    for fiber, spectrum in enumerate(flux):
        continuum = median_filter(spectrum, size=31, mode="nearest")
        residual = spectrum - continuum
        noise = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        if not np.isfinite(noise) or noise <= 0:
            continue
        locations, _ = find_peaks(residual, height=5.0 * noise, distance=4)
        for location in locations:
            lo, hi = max(0, location - 3), min(spectrum.size, location + 4)
            x, y = xall[lo:hi], residual[lo:hi]
            if x.size < 5:
                continue

            def model(xx: np.ndarray, amp: float, center: float, sigma: float, base: float):
                return amp * np.exp(-0.5 * ((xx - center) / sigma) ** 2) + base

            try:
                pars, covariance = curve_fit(
                    model,
                    x,
                    y,
                    p0=[max(residual[location], noise), location, 1.0, 0.0],
                    bounds=([0, location - 2, 0.2, -np.inf], [np.inf, location + 2, 3, np.inf]),
                    maxfev=5000,
                )
                errors = np.sqrt(np.maximum(np.diag(covariance), 0))
                peaks.append(
                    LinePeak(fiber, float(pars[1]), float(errors[1]), float(errors[0]))
                )
            except (RuntimeError, ValueError, FloatingPointError):
                continue
    return peaks


def _match_lines(first: list[LinePeak], second: list[LinePeak]) -> list[tuple[LinePeak, LinePeak]]:
    matches: list[tuple[LinePeak, LinePeak]] = []
    for fiber in sorted({peak.fiber for peak in first} & {peak.fiber for peak in second}):
        one = [peak for peak in first if peak.fiber == fiber]
        two = [peak for peak in second if peak.fiber == fiber]
        candidates = sorted(
            (
                (abs(a.gaussx - b.gaussx), ai, bi)
                for ai, a in enumerate(one)
                for bi, b in enumerate(two)
                if abs(a.gaussx - b.gaussx) < 1.0
            )
        )
        used_one: set[int] = set()
        used_two: set[int] = set()
        for _, ai, bi in candidates:
            if ai not in used_one and bi not in used_two:
                matches.append((one[ai], two[bi]))
                used_one.add(ai)
                used_two.add(bi)
    return matches


def _line_shift(
    chips1: list[Any],
    chips2: list[Any],
    peak_finder: Callable[[Any], list[LinePeak]],
) -> DitherShiftResult:
    accepted: list[float] = []
    for chip_index, (chip1, chip2) in enumerate(zip(chips1, chips2), start=1):
        for first, second in _match_lines(peak_finder(chip1), peak_finder(chip2)):
            persistence_ok = chip_index < 3 or second.fiber < 200
            if persistence_ok and first.center_error < 2 and second.center_error < 2:
                accepted.append(second.gaussx - first.gaussx)
    if not accepted:
        raise ValueError("no valid matched emission lines were found")
    shift, scatter = _robust_mean(accepted)
    error = scatter / np.sqrt(len(accepted))
    return DitherShiftResult(
        type="lines",
        shiftfit=np.asarray([shift, 0.0], dtype=np.float32),
        shifterr=np.float32(error),
        chipshift=np.zeros((3, 2), dtype=np.float32),
        chipfit=np.zeros(4, dtype=np.float32),
    )


def dither_shift(
    frame1: Any,
    frame2: Any,
    *,
    xcorr: bool = False,
    lines: bool | None = None,
    object_spectra: bool = False,
    plugmap: Any | None = None,
    nofit: bool = False,
    mjd: int = 999999,
    return_shiftarr: bool = False,
    peak_finder: Callable[[Any], list[LinePeak]] | None = None,
) -> DitherShiftResult:
    """Measure the pixel shift from ``frame1`` to ``frame2``.

    Positive shifts put frame 2 to the right of frame 1.  As in IDL, line
    fitting is the default; ``xcorr=True`` explicitly selects correlation.
    """

    if xcorr and lines:
        raise ValueError("xcorr and lines are mutually exclusive")
    chips1, chips2 = _validate_frames(frame1, frame2)
    if not xcorr:
        return _line_shift(chips1, chips2, peak_finder or _default_peak_finder)

    nfibers = np.asarray(_field(chips1[0], "flux")).shape[0]
    fibers = _plugged_fibers(plugmap, nfibers)
    measured = _cross_correlation_shifts(
        chips1, chips2, fibers, normalize_object=object_spectra
    )
    valid = measured > -99
    if not np.any(valid):
        raise ValueError("cross-correlation failed for every fiber and chip")
    mean_shift, scatter = _robust_mean(measured[valid])
    shifterr = scatter / np.sqrt(fibers.size * 3)
    chipshift = np.zeros((3, 2), dtype=np.float32)

    if nofit:
        offsets = [0.0]
        for chip_index in range(3):
            value, _ = _robust_mean(measured[valid[:, chip_index], chip_index])
            offsets.append(value)
        shiftfit = np.asarray([mean_shift, 0.0], dtype=np.float32)
        chipfit = np.asarray(offsets, dtype=np.float32)
    else:
        fiber_grid = np.broadcast_to(fibers[:, None], measured.shape)
        chip_grid = np.broadcast_to(np.arange(3)[None, :], measured.shape)

        # This exclusion is unconditional in the IDL single-line fit.
        global_good = valid & ~((chip_grid == 2) & (fiber_grid > 200))
        shiftfit = _robust_line(fiber_grid[global_good], measured[global_good])

        fit_values = measured.copy()
        if mjd < 56860:
            fit_values[(chip_grid == 2) & (fiber_grid > 200)] = -100.0
        for chip_index in range(3):
            good = fit_values[:, chip_index] > -99
            chipshift[chip_index] = _robust_line(
                fibers[good], fit_values[good, chip_index]
            )

        for _ in range(3):
            good = fit_values > -99
            rows, columns = np.nonzero(good)
            if rows.size < 4:
                raise ValueError("too few valid correlations for chip-offset fit")
            design = np.zeros((rows.size, 4), dtype=np.float64)
            design[:, 0] = fibers[rows]
            design[np.arange(rows.size), columns + 1] = 1.0
            values = fit_values[rows, columns]
            pars, *_ = np.linalg.lstsq(design, values, rcond=None)
            residual = values - design @ pars
            mad = np.median(np.abs(residual - np.median(residual)))
            if not np.isfinite(mad) or mad == 0:
                break
            reject = np.abs(residual) > 5.0 * mad
            if not np.any(reject):
                break
            fit_values[rows[reject], columns[reject]] = -100.0
        chipfit = np.asarray(pars, dtype=np.float32)

    return DitherShiftResult(
        type="xcorr",
        shiftfit=shiftfit,
        shifterr=np.float32(shifterr),
        chipshift=chipshift,
        chipfit=chipfit,
        shiftarr=measured.copy() if return_shiftarr else None,
    )
