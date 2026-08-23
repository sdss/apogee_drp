"""Nearest-fiber sky subtraction translated from ``apskysub.pro``."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter, uniform_filter1d

from ...utils.bitmask import PixelBitMask
from ..visit.io import BADERR
from .model import (
    AirglowModel,
    build_airglow_model,
    fit_sky_lines,
    load_airglow_lines,
    synthesize_airglow,
)

BADMASK = 16639
BITS = PixelBitMask()


@dataclass
class SkyMetrics:
    nskies: int
    median_sky: float
    median_abs_residual: float
    line_rms: float
    line_zrms: float
    full_rms: float


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        for key in value:
            if str(key).lower() == name.lower():
                return value[key]
    for key in (name, name.lower(), name.upper()):
        if hasattr(value, key):
            return getattr(value, key)
    raise ValueError(f"required field {name!r} is missing")


def _optional(value: Any, name: str, default: Any = None) -> Any:
    try:
        return _field(value, name)
    except ValueError:
        return default


def _set_field(value: Any, name: str, data: Any) -> None:
    if isinstance(value, dict):
        for key in value:
            if str(key).lower() == name.lower():
                value[key] = data
                return
        value[name] = data
    else:
        setattr(value, name, data)


def _chips(frame: Any) -> list[Any]:
    return [_field(frame, f"chip{letter}") for letter in "abc"]


def _validate(frame: Any) -> tuple[list[Any], int, int]:
    _field(frame, "shift")
    chips = _chips(frame)
    shape = None
    for chip in chips:
        for name in ("header", "flux", "err", "mask", "wavelength", "lsfcoef", "wcoef"):
            _field(chip, name)
        flux = np.asarray(_field(chip, "flux"))
        if flux.ndim != 2:
            raise ValueError("flux must have [fiber, pixel] shape")
        if shape is None:
            shape = flux.shape
        elif flux.shape != shape:
            raise ValueError("all chips must have the same flux shape")
        for name in ("err", "mask", "wavelength"):
            if np.asarray(_field(chip, name)).shape != flux.shape:
                raise ValueError(f"{name} must have the same shape as flux")
    assert shape is not None
    return chips, shape[0], shape[1]


def _spherical_distance(ra: np.ndarray, dec: np.ndarray, ra0: float, dec0: float) -> np.ndarray:
    ra1, dec1 = np.deg2rad(ra), np.deg2rad(dec)
    ra0, dec0 = np.deg2rad(ra0), np.deg2rad(dec0)
    cosine = np.sin(dec1) * np.sin(dec0) + np.cos(dec1) * np.cos(dec0) * np.cos(ra1 - ra0)
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _mad(values: Any) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.median(np.abs(values - np.median(values))))


def _interpolate_sky(
    wavelength: np.ndarray,
    sky_wavelength: np.ndarray,
    sky_flux: np.ndarray,
    sky_error: np.ndarray,
    sky_bad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(sky_wavelength)
    swave = np.asarray(sky_wavelength, dtype=np.float64)[order]
    unique = np.concatenate(([True], np.diff(swave) > 0))
    swave = swave[unique]
    sflux = np.asarray(sky_flux, dtype=np.float64)[order][unique]
    serr = np.asarray(sky_error, dtype=np.float64)[order][unique]
    sbad = np.asarray(sky_bad, dtype=np.float64)[order][unique]
    output = np.zeros(wavelength.size, dtype=np.float64)
    error = np.full(wavelength.size, np.nanmedian(serr), dtype=np.float64)
    bad = np.ones(wavelength.size, dtype=np.float64)
    inside = (
        np.isfinite(wavelength)
        & (wavelength >= swave[0])
        & (wavelength <= swave[-1])
    )
    if swave.size >= 4 and np.any(inside):
        output[inside] = CubicSpline(swave, sflux)(wavelength[inside])
        error[inside] = CubicSpline(swave, serr)(wavelength[inside])
        bad[inside] = CubicSpline(swave, sbad)(wavelength[inside])
    return output, error, bad


def _adjacent_is_faint(
    candidate_index: int,
    threshold: float,
    fiberid: np.ndarray,
    spectrograph: np.ndarray,
    magnitude: np.ndarray,
) -> bool:
    candidate = fiberid[candidate_index]
    for neighbor in (candidate + 1, candidate - 1):
        found = np.flatnonzero((spectrograph == 2) & (fiberid == neighbor))
        if found.size and magnitude[int(found[0]), 1] <= threshold:
            return False
    return True


def _candidate_skies(
    target_fiber: int,
    sky_indices: np.ndarray,
    fiberid: np.ndarray,
    spectrograph: np.ndarray,
    magnitude: np.ndarray,
) -> np.ndarray:
    nearby = sky_indices[np.abs(fiberid[sky_indices] - target_fiber) < 75]
    if nearby.size == 0:
        return nearby
    threshold = 9.5
    accepted = np.array([], dtype=int)
    while accepted.size <= 1 and threshold > 5:
        accepted = np.asarray(
            [
                index
                for index in nearby
                if _adjacent_is_faint(
                    int(index), threshold, fiberid, spectrograph, magnitude
                )
            ],
            dtype=int,
        )
        threshold -= 0.25
    return np.unique(accepted) if threshold > 5 else np.array([], dtype=int)


def _skyline_pixels(
    flux: np.ndarray,
    sky: np.ndarray,
    *,
    width: int = 5,
    threshold: float = 1.5,
    minimum: float = 100.0,
) -> np.ndarray:
    smoothed_sky = uniform_filter1d(
        sky - np.median(sky), size=width, mode="nearest"
    )
    continuum = median_filter(flux, size=51, mode="nearest")
    return np.flatnonzero(smoothed_sky > np.maximum(threshold * continuum, minimum))


def _add_header_metrics(chips: list[Any], metrics: SkyMetrics) -> None:
    values = {
        "NSKIES": metrics.nskies,
        "SKYMED": metrics.median_sky,
        "SMEDABS": metrics.median_abs_residual,
        "SLNRMS": metrics.line_rms,
        "SLNZRMS": metrics.line_zrms,
        "SFULRMS": metrics.full_rms,
    }
    for chip in chips:
        header = _field(chip, "header")
        for key, value in values.items():
            if isinstance(header, dict):
                header[key] = value
            else:
                header[key] = value


def _metrics(chips: list[Any], sky_rows: np.ndarray) -> SkyMetrics:
    chip = chips[1]
    mask = np.asarray(_field(chip, "mask"))[sky_rows]
    model = np.asarray(_field(chip, "sky"))[sky_rows]
    residual = np.asarray(_field(chip, "flux"))[sky_rows]
    error = np.asarray(_field(chip, "err"))[sky_rows]
    med = median_filter(residual, size=(1, 51), mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        zresidual = residual / error
    zmed = median_filter(zresidual, size=(1, 51), mode="nearest")
    good = (mask.astype(np.int64) & BADMASK) == 0
    full = residual[good]
    full_rms = float(np.sqrt(np.nanmean(full**2))) if full.size else np.nan
    median_abs = float(np.median(np.median(np.abs(residual), axis=1)))
    line = good & (model > 1000)
    line_values = (residual - med)[line]
    line_zvalues = (zresidual - zmed)[line]
    line_rms = (
        float(np.sqrt(np.nanmean(line_values**2))) if line_values.size else np.nan
    )
    line_zrms = (
        float(np.sqrt(np.nanmean(line_zvalues**2)))
        if line_zvalues.size
        else np.nan
    )
    return SkyMetrics(
        len(sky_rows),
        float(np.median(model)),
        median_abs,
        line_rms,
        line_zrms,
        full_rms,
    )


def sky_subtract(
    frame: Any,
    plugmap: Any,
    *,
    suboption: int = 1,
    force: bool = False,
    telescope: str = "apo25m",
    sky_selector: Callable[[Any], np.ndarray] | None = None,
    airglow: Any | None = None,
    copy_frame: bool = True,
    return_metrics: bool = False,
) -> Any | tuple[Any, SkyMetrics] | tuple[Any, SkyMetrics, AirglowModel]:
    """Subtract a weighted average of nearby sky fibers.

    ``SUBOPTION=1`` is the production nearest-fiber mode. ``SUBOPTION=2`` fits
    the supplied APOGEE airglow line list using the Gauss-Hermite LSF.
    """

    if suboption not in (1, 2):
        raise ValueError("suboption must be 1 (nearest) or 2 (line model)")
    if suboption == 2 and airglow is None:
        candidates = []
        if os.environ.get("APOGEE_DRP_DIR"):
            candidates.append(
                Path(os.environ["APOGEE_DRP_DIR"]) / "data/skylines/airglow.txt"
            )
        candidates.append(
            Path(__file__).resolve().parents[4] / "data/skylines/airglow.txt"
        )
        airglow = next((path for path in candidates if path.exists()), None)
        if airglow is None:
            raise ValueError("suboption=2 requires an airglow line list")
    source_frame = frame if copy_frame else copy.deepcopy(frame)
    source_chips, nfibers, npix = _validate(source_frame)
    output = copy.deepcopy(frame) if copy_frame else frame
    chips = _chips(output)
    for chip in chips:
        _set_field(chip, "sky", np.zeros((nfibers, npix), dtype=np.float32))
        _set_field(chip, "skyerr", np.zeros((nfibers, npix), dtype=np.float32))
        _set_field(chip, "skyscale", np.float32(0))

    fiberdata = _field(plugmap, "fiberdata")
    fiberid = np.asarray(_field(fiberdata, "fiberid"), dtype=int)
    spectrograph = np.asarray(_field(fiberdata, "spectrographid"), dtype=int)
    holetype = np.char.upper(np.asarray(_field(fiberdata, "holetype")).astype(str))
    objtype = np.char.upper(np.asarray(_field(fiberdata, "objtype")).astype(str))
    ra = np.asarray(_field(fiberdata, "ra"), dtype=float)
    dec = np.asarray(_field(fiberdata, "dec"), dtype=float)
    magnitude = np.asarray(_field(fiberdata, "mag"), dtype=float)
    if sky_selector is None:
        selected_sky = objtype == "SKY"
    else:
        selected_sky = np.asarray(sky_selector(fiberdata), dtype=bool)
    sky_indices = np.flatnonzero(
        (spectrograph == 2) & (holetype == "OBJECT") & selected_sky
    )
    if sky_indices.size == 0 and force:
        middle = np.asarray(_field(source_chips[1], "flux"))
        lo, hi = min(900, npix - 1), min(1101, npix)
        med = np.median(middle[:, lo:hi], axis=1)
        plug_rows = 300 - fiberid
        valid = (
            (spectrograph == 2)
            & (holetype == "OBJECT")
            & (plug_rows >= 0)
            & (plug_rows < nfibers)
        )
        order = np.argsort(med[plug_rows[valid]])
        valid_indices = np.flatnonzero(valid)
        sky_indices = valid_indices[order[: min(31, order.size)]]
    if sky_indices.size == 0:
        raise ValueError("no sky fibers; sky subtraction cannot be performed")
    sky_rows = 300 - fiberid[sky_indices]
    if np.any((sky_rows < 0) | (sky_rows >= nfibers)):
        raise ValueError("sky fiber IDs are outside the frame")

    if suboption == 2:
        lines = load_airglow_lines(airglow)
        measurements = fit_sky_lines(source_chips, sky_rows, lines, _field)
        if not measurements:
            raise ValueError("no airglow lines could be fitted in the sky fibers")
        zeta = np.asarray(_field(fiberdata, "zeta"), dtype=np.float64)[sky_indices]
        eta = np.asarray(_field(fiberdata, "eta"), dtype=np.float64)[sky_indices]
        airglow_model = build_airglow_model(
            measurements, lines, sky_rows, zeta, eta
        )
        if not airglow_model.lines:
            raise ValueError("no airglow line was detected in enough sky fibers")
        lookup = {
            int(fid): int(index)
            for index, fid in enumerate(fiberid)
            if spectrograph[index] == 2
        }
        all_zeta = np.asarray(_field(fiberdata, "zeta"), dtype=np.float64)
        all_eta = np.asarray(_field(fiberdata, "eta"), dtype=np.float64)
        for row in range(nfibers):
            plug_index = lookup.get(300 - row)
            if plug_index is None:
                continue
            for chip, source_chip in zip(chips, source_chips):
                model_spectrum, model_error = synthesize_airglow(
                    np.asarray(_field(source_chip, "wavelength"))[row],
                    np.asarray(_field(source_chip, "lsfcoef"))[row],
                    all_zeta[plug_index],
                    all_eta[plug_index],
                    airglow_model,
                )
                source_flux = np.asarray(_field(source_chip, "flux"))[row]
                _field(chip, "flux")[row] = source_flux - model_spectrum
                _field(chip, "sky")[row] = model_spectrum
                _field(chip, "skyerr")[row] = model_error
        for chip in chips:
            header = _field(chip, "header")
            for species_index, species in enumerate(("OH", "O2"), start=1):
                if species not in airglow_model.species_coefficients:
                    continue
                coefficients = airglow_model.species_coefficients[species]
                errors = airglow_model.species_errors[species]
                for index, (coefficient, error) in enumerate(
                    zip(coefficients, errors), start=1
                ):
                    header[f"SKPR{species_index}_{index}"] = float(coefficient)
                    header[f"SKER{species_index}_{index}"] = float(error)
        metrics = _metrics(chips, sky_rows)
        _add_header_metrics(chips, metrics)
        if return_metrics:
            return output, metrics, airglow_model
        return output

    fps = int(_field(plugmap, "mjd")) >= 59556
    fpi_rows = (87, 218) if telescope == "lco25m" else (75, 225)
    lookup = {
        int(fid): int(index)
        for index, fid in enumerate(fiberid)
        if spectrograph[index] == 2
    }
    nosky_bit = np.int16(BITS.getval("NOSKY"))
    skyline_bit = np.int16(BITS.getval("SIG_SKYLINE"))

    for row in range(nfibers):
        target_fiber = 300 - row
        plug_index = lookup.get(target_fiber)
        if plug_index is None or (fps and row in fpi_rows):
            continue
        candidates = _candidate_skies(
            target_fiber, sky_indices, fiberid, spectrograph, magnitude
        )
        if candidates.size < 2:
            continue
        distances = _spherical_distance(
            ra[candidates], dec[candidates], ra[plug_index], dec[plug_index]
        )
        order = np.argsort(distances)
        if objtype[plug_index] == "SKY":
            order = np.asarray(
                [index for index in order if candidates[index] != plug_index]
            )
        chosen = candidates[order[:5]]
        chosen_rows = 300 - fiberid[chosen]

        for chip, source_chip in zip(chips, source_chips):
            fiber = np.asarray(_field(source_chip, "flux"))[row].astype(np.float64)
            fiber_error = np.asarray(_field(source_chip, "err"))[row].astype(np.float64)
            mask = np.asarray(_field(chip, "mask"))[row]
            wavelength = np.asarray(_field(source_chip, "wavelength"))[row]
            fiber_lines = fiber - median_filter(fiber, size=150, mode="nearest")
            fiber_lines[(mask.astype(np.int64) & BADMASK) > 0] = 0
            total = np.zeros(npix, dtype=np.float64)
            total_weight = np.zeros(npix, dtype=np.float64)
            for sky_row in chosen_rows:
                sky_flux = np.asarray(_field(source_chip, "flux"))[sky_row]
                sky_error = np.asarray(_field(source_chip, "err"))[sky_row]
                sky_wave = np.asarray(_field(source_chip, "wavelength"))[sky_row]
                sky_bad = (
                    np.asarray(_field(source_chip, "mask"))[sky_row].astype(np.int64)
                    & BADMASK
                ) > 0
                interpolated, interpolated_error, interpolated_bad = _interpolate_sky(
                    wavelength, sky_wave, sky_flux, sky_error, sky_bad
                )
                good = (
                    (interpolated != 0)
                    & (np.abs(interpolated_bad) <= 0.01)
                    & np.isfinite(interpolated_error)
                    & (interpolated_error > 0)
                )
                weight = np.zeros(npix)
                weight[good] = 1.0 / interpolated_error[good] ** 2
                total += interpolated * weight
                total_weight += weight

            valid = total_weight > 0
            sky_spectrum = np.zeros(npix)
            sky_error = np.full(npix, BADERR, dtype=np.float64)
            sky_spectrum[valid] = total[valid] / total_weight[valid]
            sky_error[valid] = np.sqrt(1.0 / total_weight[valid])
            sky_continuum = median_filter(sky_spectrum, size=251, mode="nearest")
            sky_lines = sky_spectrum - sky_continuum
            scatter = _mad(sky_lines)
            reference = np.array([], dtype=int)
            for factor in (10, 5, 2):
                reference = np.flatnonzero(sky_lines > factor * scatter)
                if reference.size:
                    break
            if reference.size == 0:
                continue
            ratios = fiber_lines[reference] / sky_lines[reference]
            central = ratios[(ratios > 0.5) & (ratios < 1.5)]
            scale = float(np.median(central if central.size else ratios))
            sky_spectrum = sky_continuum + scale * sky_lines
            subtracted = fiber - sky_spectrum
            combined_error = np.sqrt(fiber_error**2 + sky_error**2)
            subtracted[~valid] = 0
            sky_spectrum[~valid] = 0
            combined_error[~valid] = BADERR
            sky_error[~valid] = BADERR
            mask[~valid] |= nosky_bit
            _field(chip, "flux")[row] = subtracted
            _field(chip, "err")[row] = combined_error
            _field(chip, "sky")[row] = sky_spectrum
            _field(chip, "skyerr")[row] = sky_error
            _set_field(chip, "skyscale", np.float32(scale))
            if objtype[plug_index] != "SKY":
                header = _field(chip, "header")
                exptime = float(_optional(header, "EXPTIME", 373))
                high = _skyline_pixels(
                    np.asarray(_field(chip, "flux"))[row],
                    np.asarray(_field(chip, "sky"))[row],
                    threshold=1.5,
                    minimum=100 * exptime / 373,
                )
                mask[high] |= skyline_bit

    metrics = _metrics(chips, sky_rows)
    _add_header_metrics(chips, metrics)
    return (output, metrics) if return_metrics else output
