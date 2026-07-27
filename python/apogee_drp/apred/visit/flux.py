"""Rough visit flux calibration translated from ``ap1dfluxing.pro``."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np

from .io import BADERR

BADMASK = 16639
H_ZEROPOINT_FLAMBDA = 1.133e-10


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        for key in (name, name.lower(), name.upper()):
            if key in value:
                return value[key]
        for key in value:
            if str(key).lower() == name.lower():
                return value[key]
    for key in (name, name.lower(), name.upper()):
        if hasattr(value, key):
            return getattr(value, key)
    raise ValueError(f"required field {name!r} is missing")


def _set_field(value: Any, name: str, data: Any) -> None:
    if isinstance(value, dict):
        for key in value:
            if str(key).lower() == name.lower():
                value[key] = data
                return
        value[name] = data
        return
    setattr(value, name, data)


def _chips(frame: Any) -> list[Any]:
    return [_field(frame, f"chip{letter}") for letter in "abc"]


def _column(fiberdata: Any, name: str) -> np.ndarray:
    return np.asarray(_field(fiberdata, name))


def _mad(values: Any) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    center = np.median(values)
    return float(np.median(np.abs(values - center)))


def _robust_polyfit(x: Any, y: Any, degree: int) -> np.ndarray:
    """Polynomial coefficients in IDL order: constant through highest power."""

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) <= degree:
        raise ValueError("too few finite samples for response polynomial")
    for _ in range(5):
        coefficients = np.polynomial.polynomial.polyfit(x[good], y[good], degree)
        residual = y - np.polynomial.polynomial.polyval(x, coefficients)
        scatter = _mad(residual[good])
        if scatter == 0 or not np.isfinite(scatter):
            break
        keep = good & (np.abs(residual) <= 5.0 * scatter)
        if np.count_nonzero(keep) <= degree or np.array_equal(keep, good):
            break
        good = keep
    return np.polynomial.polynomial.polyfit(x[good], y[good], degree)


def _validate_frame(frame: Any) -> tuple[list[Any], int, int]:
    chips = _chips(frame)
    shape: tuple[int, int] | None = None
    for chip in chips:
        for name in ("header", "flux", "err", "mask", "wavelength", "sky", "skyerr"):
            _field(chip, name)
        flux = np.asarray(_field(chip, "flux"))
        if flux.ndim != 2:
            raise ValueError("chip arrays must have [fiber, pixel] shape")
        if shape is None:
            shape = flux.shape
        elif flux.shape != shape:
            raise ValueError("all chip flux arrays must have the same shape")
        for name in ("err", "mask", "wavelength", "sky", "skyerr"):
            if np.asarray(_field(chip, name)).shape != flux.shape:
                raise ValueError(f"{name} must have the same shape as flux")
    assert shape is not None
    return chips, shape[0], shape[1]


def _fiber_rows(fiberid: np.ndarray, nfibers: int) -> np.ndarray:
    rows = 300 - np.asarray(fiberid, dtype=int)
    if np.any((rows < 0) | (rows >= nfibers)):
        raise ValueError("plugmap fiber IDs are outside the frame")
    return rows


def _add_header_metadata(
    chips: list[Any], coefficients: np.ndarray, telluric_fibers: np.ndarray
) -> None:
    history = (
        "AP1DFLUXING: 5th order polynomial fit to telluric stars",
        "AP1DFLUXING: fit to log10 flux",
        "AP1DFLUXING: x = wavelength(A) - 16000.0",
        f"AP1DFLUXING: {telluric_fibers.size} telluric stars",
        "AP1DFLUXING: " + ",".join(str(value) for value in telluric_fibers),
    )
    for chip in chips:
        header = _field(chip, "header")
        if hasattr(header, "add_history"):
            for line in history:
                header.add_history(line)
        elif isinstance(header, dict):
            old = header.get("HISTORY", [])
            if isinstance(old, str):
                old = [old]
            header["HISTORY"] = [*old, *history]
        for index, coefficient in enumerate(coefficients, start=1):
            key = f"FLXPAR{index}"
            if isinstance(header, dict):
                header[key] = float(coefficient)
            else:
                header[key] = float(coefficient)


def _fit_relative_response(
    chips: list[Any],
    telluric_rows: np.ndarray,
    telluric_fibers: np.ndarray,
    npix: int,
) -> np.ndarray:
    xlo = 100
    xhi = (npix // 10) * 10 - 100
    sample_pixels = np.arange(xlo, xhi + 1, 10)
    nppix = sample_pixels.size
    if nppix < 7:
        raise ValueError("spectra are too short for AP1DFLUXING response fit")

    medflux = np.empty((3, nppix, telluric_rows.size), dtype=np.float64)
    for chip_index, chip in enumerate(chips):
        flux = np.asarray(_field(chip, "flux"), dtype=np.float64)
        blocks = flux[telluric_rows, xlo : xhi + 10].reshape(
            telluric_rows.size, nppix, 10
        )
        medflux[chip_index] = np.median(blocks, axis=2).T
    with np.errstate(divide="ignore", invalid="ignore"):
        logmedflux = np.log10(medflux)
    star_offsets = np.nanmedian(logmedflux, axis=(0, 1))
    delta = logmedflux - star_offsets[None, None, :]
    initial_flux = np.nanmedian(delta, axis=2)
    center_row = min(150, np.asarray(_field(chips[0], "flux")).shape[0] - 1)
    initial_x = np.stack(
        [
            np.asarray(_field(chip, "wavelength"), dtype=np.float64)[
                center_row, sample_pixels
            ]
            - 16000.0
            for chip in chips
        ]
    )
    initial_coefficients = _robust_polyfit(initial_x, initial_flux, 5)

    design_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    for star_index, row in enumerate(telluric_rows):
        for chip in chips:
            wavelength = np.asarray(_field(chip, "wavelength"), dtype=np.float64)[
                row, sample_pixels
            ]
            x = wavelength - 16000.0
            block = np.zeros((nppix, 5 + telluric_rows.size), dtype=np.float64)
            block[:, :5] = np.column_stack([x**5, x**4, x**3, x**2, x])
            block[:, 5 + star_index] = 1.0
            with np.errstate(divide="ignore", invalid="ignore"):
                logflux = np.log10(
                    np.asarray(_field(chip, "flux"), dtype=np.float64)[
                        row, sample_pixels
                    ]
                )
            model = (
                np.polynomial.polynomial.polyval(x, initial_coefficients)
                + star_offsets[star_index]
            )
            threshold = max(3.0 * _mad(logflux - model), 0.1)
            logflux[np.abs(logflux - model) > threshold] = np.nan
            design_parts.append(block)
            value_parts.append(logflux)

    design = np.concatenate(design_parts)
    values = np.concatenate(value_parts)
    finite = np.isfinite(values) & np.all(np.isfinite(design), axis=1)
    if np.count_nonzero(finite) < design.shape[1]:
        raise ValueError("too few valid telluric samples for response fit")
    # The raw x**5 column is about 10**15 larger than a star-intercept column.
    # Column scaling leaves the mathematical model unchanged while avoiding
    # numerical rank loss in LAPACK.
    fit_design = design[finite]
    scales = np.linalg.norm(fit_design, axis=0)
    scales[scales == 0] = 1.0
    scaled_parameters, *_ = np.linalg.lstsq(
        fit_design / scales, values[finite], rcond=None
    )
    parameters = scaled_parameters / scales

    coefficients = parameters[:5]
    for chip in chips:
        wavelength = np.asarray(_field(chip, "wavelength"), dtype=np.float64)
        x = wavelength - 16000.0
        log_response = sum(
            coefficients[index] * x ** (5 - index) for index in range(5)
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            log_response += 4.0 * np.log10(wavelength / 16000.0)
            response = 10.0**log_response
        if np.any(~np.isfinite(response) | (response <= 0)):
            raise ValueError("relative response is non-finite or non-positive")
        for name in ("flux", "err", "sky", "skyerr"):
            original = np.asarray(_field(chip, name))
            bad_error = (original == BADERR) if name == "err" else None
            calibrated = original / response
            if bad_error is not None:
                calibrated[bad_error] = BADERR
            _set_field(chip, name, calibrated.astype(original.dtype, copy=False))

    _add_header_metadata(chips, coefficients, telluric_fibers)
    return np.asarray(coefficients, dtype=np.float64)


def _set_fluxnorm(chips: list[Any], value: float) -> None:
    for chip in chips:
        header = _field(chip, "header")
        if isinstance(header, dict):
            header["FLUXNORM"] = float(value)
        else:
            header["FLUXNORM"] = (float(value), "median flux normalization factor")


def flux_calibrate(
    frame: Any,
    plugmap: Any,
    *,
    badmask: int = BADMASK,
    copy_frame: bool = True,
) -> Any:
    """Apply the relative and absolute calibrations from ``AP1DFLUXING``.

    The returned frame gains a ``fluxcorr`` vector.  Pixels with ``BADERR``
    retain that sentinel through both calibration stages.
    """

    original_chips, nfibers, npix = _validate_frame(frame)
    output = copy.deepcopy(frame) if copy_frame else frame
    chips, _, _ = _validate_frame(output)
    fiberdata = _field(plugmap, "fiberdata")
    spectrograph = _column(fiberdata, "spectrographid").astype(int)
    holetype = np.char.upper(_column(fiberdata, "holetype").astype(str))
    objtype = np.char.upper(_column(fiberdata, "objtype").astype(str))
    fiberid = _column(fiberdata, "fiberid").astype(int)
    magnitude = np.asarray(_field(fiberdata, "mag"), dtype=np.float64)
    if magnitude.ndim != 2 or magnitude.shape[1] < 2:
        raise ValueError("plugmap fiberdata.mag must contain J and H magnitudes")
    nrows = fiberid.size
    if not all(array.size == nrows for array in (spectrograph, holetype, objtype)):
        raise ValueError("plugmap fiberdata columns have inconsistent lengths")

    hot = (
        (spectrograph == 2) & (holetype == "OBJECT") & (objtype == "HOT_STD")
    )
    hot_indices = np.flatnonzero(hot)
    good_telluric_fibers: list[int] = []
    for index in hot_indices:
        row = _fiber_rows(np.asarray([fiberid[index]]), nfibers)[0]
        mask = np.asarray(_field(chips[1], "mask"))[row]
        if np.count_nonzero((mask.astype(np.int64) & int(badmask)) > 0) < 500:
            good_telluric_fibers.append(int(fiberid[index]))
    telluric_fibers = np.asarray(good_telluric_fibers, dtype=int)
    if telluric_fibers.size:
        telluric_rows = _fiber_rows(telluric_fibers, nfibers)
        _fit_relative_response(
            chips, telluric_rows, telluric_fibers, npix
        )

    sky_selection = (
        (spectrograph == 2) & (holetype == "OBJECT") & (objtype == "SKY")
    )
    sky_indices = np.flatnonzero(sky_selection)
    if sky_indices.size:
        sky_rows = _fiber_rows(fiberid[sky_indices], nfibers)
        medsky = float(
            np.median(np.asarray(_field(original_chips[1], "flux"))[sky_rows])
        )
    else:
        medsky = 0.0

    object_selection = (
        (spectrograph == 2) & (holetype == "OBJECT") & (objtype != "SKY")
    )
    object_indices = np.flatnonzero(object_selection)
    norms = np.full(object_indices.size, np.nan, dtype=np.float64)
    for position, index in enumerate(object_indices):
        fiber = fiberid[index]
        if 0 < fiber <= 300:
            row = _fiber_rows(np.asarray([fiber]), nfibers)[0]
            hmag = magnitude[index, 1]
            medflux = float(
                np.median(np.asarray(_field(original_chips[1], "flux"))[row])
            )
            if hmag != 0 and hmag < 30 and medflux - medsky > 100:
                denominator = max(medflux - medsky, 1.0)
                norms[position] = (
                    10.0 ** (-0.4 * hmag)
                    * H_ZEROPOINT_FLAMBDA
                    / denominator
                )
    finite_norms = norms[np.isfinite(norms)]
    mednorm = float(np.median(finite_norms)) if finite_norms.size else 1.0
    fluxcorr = np.zeros(nfibers, dtype=np.float32)

    for position, index in enumerate(object_indices):
        fiber = fiberid[index]
        if not (0 < fiber <= 300):
            continue
        row = _fiber_rows(np.asarray([fiber]), nfibers)[0]
        norm = norms[position] if np.isfinite(norms[position]) else mednorm
        fluxcorr[row] = np.float32(norm)
        for chip in chips:
            for name in ("flux", "err", "sky", "skyerr"):
                original = np.asarray(_field(chip, name))
                bad_error = (original[row] == BADERR) if name == "err" else None
                original[row] *= norm
                if bad_error is not None:
                    original[row, bad_error] = BADERR

    for index in sky_indices:
        fiber = fiberid[index]
        if not (0 < fiber <= 300):
            continue
        row = _fiber_rows(np.asarray([fiber]), nfibers)[0]
        fluxcorr[row] = np.float32(mednorm)
        for chip in chips:
            for name in ("flux", "err", "sky", "skyerr"):
                original = np.asarray(_field(chip, name))
                bad_error = (original[row] == BADERR) if name == "err" else None
                original[row] *= mednorm
                if bad_error is not None:
                    original[row, bad_error] = BADERR

    _set_fluxnorm(chips, mednorm)
    _set_field(output, "fluxcorr", fluxcorr)
    return output
