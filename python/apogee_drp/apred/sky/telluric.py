"""Telluric fitting and correction translated from APOGEE ``APTELLURIC``.

This module separates the numerical parts of the monolithic IDL routine:
single-star spectral fitting, spatial fitting of species scales, and applying
the correction.  Model arrays are expected to contain continuum-normalized
transmissions for CH4, CO2, and H2O in their last dimension.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import median_filter
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares, minimize_scalar

from ...utils.bitmask import PixelBitMask

SPECIES = ("CH4", "CO2", "H2O")
TELLURIC_BIT = int(PixelBitMask().getval("SIG_TELLURIC"))


def _spatial_basis(x: Any, y: Any, npars: int) -> np.ndarray:
    """Return ``FUNC_POLY2D`` terms for the orders used by APTELLURIC."""

    x, y = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    )
    if npars == 1:
        terms = (np.ones_like(x),)
    elif npars == 3:
        terms = (np.ones_like(x), x, y)
    elif npars == 6:
        terms = (np.ones_like(x), x, x**2, x * y, y, y**2)
    else:
        raise ValueError("APTELLURIC spatial fits require 1, 3, or 6 terms")
    return np.stack(terms, axis=-1)


@dataclass
class TelluricStarFit:
    fiber: int
    scale: np.ndarray
    snr: float
    rchisq: float
    status: int
    best_model: np.ndarray
    continuum: np.ndarray
    transmission: np.ndarray


@dataclass
class TelluricSpatialFit:
    species: str
    coefficients: np.ndarray
    errors: np.ndarray
    scatter: float
    npoints: int
    status: int


@dataclass
class TelluricCorrection:
    flux: np.ndarray
    error: np.ndarray
    transmission: np.ndarray
    transmission_error: np.ndarray
    scale: np.ndarray
    scale_error: np.ndarray


@dataclass
class TelluricFrameResult:
    frame: Any
    tellstar: dict[str, Any]
    star_fits: list[TelluricStarFit]
    spatial_fits: list[TelluricSpatialFit]


def telluric_transmission(models: Any, scales: Any) -> np.ndarray:
    """Return the exact multiplicative model used by ``FIT_TELLURIC``."""

    models = np.asarray(models, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if models.shape[-1] != 3 or scales.shape[-1] != 3:
        raise ValueError("models and scales must contain CH4, CO2, and H2O")
    return np.prod(1.0 + scales * (models - 1.0), axis=-1)


def _mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    med = np.nanmedian(values)
    return float(np.nanmedian(np.abs(values - med)))


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, min(int(width), values.size))
    if width % 2 == 0:
        width = width - 1 if width == values.size else width + 1
    return median_filter(values, size=width, mode="nearest")


def fit_telluric_star(
    flux: Any,
    error: Any,
    models: Any,
    *,
    mask: Any | None = None,
    sky: Any | None = None,
    fiber: int = -1,
    continuum_width: int = 201,
    max_iterations: int = 12,
) -> TelluricStarFit:
    """Fit three species scales to one standard-star spectrum.

    This implements the iterative continuum removal of
    ``APTELLURIC_SPECFIT`` option 1. Multiple models may be supplied as
    ``(pixel, species, model)``; the lowest robust residual model is selected
    independently for each species before the joint least-squares fit.
    """

    flux = np.asarray(flux, dtype=np.float64)
    error = np.maximum(np.asarray(error, dtype=np.float64), 1.0)
    models = np.asarray(models, dtype=np.float64)
    if models.ndim == 2:
        models = models[..., None]
    if models.shape[:2] != (flux.size, 3):
        raise ValueError("models must have shape (pixel, 3[, nmodel])")
    good = np.isfinite(flux) & np.isfinite(error) & (flux > 0)
    if mask is not None:
        good &= np.asarray(mask) == 0
    if sky is not None:
        sky = np.asarray(sky, dtype=np.float64)
        sky_lines = sky - _smooth(sky, 51)
    else:
        sky_lines = np.zeros_like(flux)
    snr = float(np.nanmedian(flux[good] / error[good])) if good.any() else 0.0
    if good.sum() < 10 or snr < 10:
        blank = np.ones_like(flux)
        return TelluricStarFit(
            fiber, np.ones(3), snr, np.inf, 0, np.full(3, -1), blank, blank
        )

    continuum0 = np.maximum(_smooth(flux, 101), 1.0)
    normalized0 = flux / continuum0
    best = np.zeros(3, dtype=int)
    selected = np.empty((flux.size, 3))
    for species in range(3):
        scores = []
        for model_index in range(models.shape[2]):
            model = models[:, species, model_index]
            pixels = good & (model < 0.99) & (normalized0 < 1.5)
            if pixels.sum() < 10:
                scores.append(np.inf)
                continue
            result = minimize_scalar(
                lambda scale: np.nanmedian(
                    np.abs(
                        normalized0[pixels]
                        / (1 + scale * (model[pixels] - 1))
                        - 1
                    )
                ),
                bounds=(0.0, 10.0),
                method="bounded",
            )
            scores.append(result.fun)
        best[species] = int(np.argmin(scores))
        selected[:, species] = models[:, species, best[species]]

    scales = np.ones(3)
    continuum = continuum0
    fit_pixels = good
    for _ in range(max_iterations):
        transmission = telluric_transmission(selected, scales)
        continuum = np.maximum(_smooth(flux / transmission, continuum_width), 1.0)
        normalized = flux / continuum
        normalized_error = np.maximum(error / continuum, 0.02)
        candidate = (
            good
            & (transmission < 0.99)
            & (normalized > 0)
            & (sky_lines < 0.5 * continuum)
        )
        if candidate.sum() < 10:
            break
        fit_pixels = candidate
        result = least_squares(
            lambda pars: (
                normalized[fit_pixels]
                - telluric_transmission(selected[fit_pixels], pars)
            )
            / normalized_error[fit_pixels],
            scales,
            bounds=(0.0, 10.0),
        )
        change = np.max(np.abs(result.x - scales) / np.maximum(result.x, 1e-8))
        scales = result.x
        if change < 1e-3:
            break
    transmission = telluric_transmission(selected, scales)
    residual = (flux / continuum - transmission) / np.maximum(
        error / continuum, 0.02
    )
    rchisq = float(np.mean(residual[fit_pixels] ** 2))
    return TelluricStarFit(
        fiber, scales, snr, rchisq, 1, best, continuum, transmission
    )


def fit_spatial_scales(
    zeta: Any,
    eta: Any,
    scales: Any,
    *,
    single: bool = False,
    reject_sigma: float = 2.5,
) -> list[TelluricSpatialFit]:
    """Fit the plate-wide scale surfaces used by ``APTELLURIC``."""

    x = np.asarray(zeta, dtype=np.float64)
    y = np.asarray(eta, dtype=np.float64)
    values = np.asarray(scales, dtype=np.float64)
    if values.shape != (x.size, 3):
        raise ValueError("scales must have shape (nstar, 3)")
    output = []
    for species in range(3):
        npars = 1 if single else (6 if species == 2 and x.size >= 10 else 3)
        design = _spatial_basis(x, y, npars)
        good = np.isfinite(values[:, species]) & np.all(
            np.isfinite(design), axis=1
        )
        if good.sum() < npars:
            raise ValueError(f"not enough stars to fit {SPECIES[species]}")
        coefficients, _, _, _ = np.linalg.lstsq(
            design[good], values[good, species], rcond=None
        )
        residual = values[:, species] - design @ coefficients
        scatter = _mad(residual[good])
        if scatter > 0 and good.sum() > npars:
            clipped = good & (np.abs(residual) <= reject_sigma * scatter)
            if clipped.sum() >= npars:
                good = clipped
                coefficients, _, _, _ = np.linalg.lstsq(
                    design[good], values[good, species], rcond=None
                )
                residual = values[:, species] - design @ coefficients
                scatter = _mad(residual[good])
        dof = good.sum() - npars
        covariance = np.linalg.pinv(design[good].T @ design[good])
        variance = (
            np.sum(residual[good] ** 2) / dof
            if dof > 0
            else scatter**2
        )
        errors = np.sqrt(np.maximum(np.diag(covariance) * variance, 0))
        output.append(
            TelluricSpatialFit(
                SPECIES[species],
                coefficients,
                errors,
                scatter,
                int(good.sum()),
                1,
            )
        )
    return output


def evaluate_spatial_scales(
    zeta: float, eta: float, fits: Sequence[TelluricSpatialFit]
) -> tuple[np.ndarray, np.ndarray]:
    scales = np.empty(3)
    errors = np.empty(3)
    for index, fit in enumerate(fits):
        active = _spatial_basis(
            zeta, eta, fit.coefficients.size
        ).ravel()
        scales[index] = active @ fit.coefficients
        errors[index] = np.sqrt(np.sum((active * fit.errors) ** 2))
    return scales, errors


def apply_telluric_correction(
    flux: Any,
    error: Any,
    models: Any,
    zeta: float,
    eta: float,
    spatial_fits: Sequence[TelluricSpatialFit],
) -> TelluricCorrection:
    """Correct one fiber and propagate the IDL normalization uncertainty."""

    flux = np.asarray(flux, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    models = np.asarray(models, dtype=np.float64)
    scales, scale_error = evaluate_spatial_scales(zeta, eta, spatial_fits)
    transmission = telluric_transmission(models, scales)
    derivatives = np.empty((flux.size, 3))
    for species in range(3):
        others = [item for item in range(3) if item != species]
        derivatives[:, species] = (models[:, species] - 1) * np.prod(
            1 + scales[others] * (models[:, others] - 1), axis=1
        )
    transmission_error = np.sqrt(
        np.sum((derivatives * scale_error) ** 2, axis=1)
    )
    corrected = flux / transmission
    corrected_error = np.maximum(corrected, 1.0) * np.sqrt(
        (error / np.maximum(flux, 1.0)) ** 2
        + (transmission_error / transmission) ** 2
    )
    return TelluricCorrection(
        corrected,
        corrected_error,
        transmission,
        transmission_error,
        scales,
        scale_error,
    )


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        for key in value:
            if str(key).lower() == name.lower():
                return value[key]
    names = getattr(value, "colnames", None)
    if names is None:
        names = getattr(getattr(value, "dtype", None), "names", None)
    if names is not None:
        for key in names:
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
    else:
        setattr(value, name, data)


def _chips(frame: Any) -> list[Any]:
    return [_field(frame, f"chip{letter}") for letter in "abc"]


def normalize_preconvolved_models(
    models: Any, *, nfibers: int, npix: int
) -> np.ndarray:
    """Normalize common preconvolved layouts to ``[fiber,pixel,species,model]``.

    Accepted layouts are the IDL concatenated form
    ``[3*npix,nfiber,3[,nmodel]]`` and the Python chip form
    ``[nfiber,3,npix,3[,nmodel]]``.
    """

    array = np.asarray(models, dtype=np.float64)
    if array.ndim == 3:
        array = array[..., None]
    if array.ndim == 4 and array.shape[:3] == (3 * npix, nfibers, 3):
        return np.moveaxis(array, 1, 0)
    if array.ndim == 4 and array.shape[:3] == (nfibers, 3 * npix, 3):
        return array
    if array.ndim == 4 and array.shape == (nfibers, 3, npix, 3):
        return array.reshape(nfibers, 3 * npix, 3, 1)
    if array.ndim == 5 and array.shape[:4] == (nfibers, 3, npix, 3):
        return array.reshape(nfibers, 3 * npix, 3, array.shape[-1])
    raise ValueError(
        "preconvolved models must be [3*npix,nfiber,3[,nmodel]] or "
        "[nfiber,3,npix,3[,nmodel]]"
    )


def load_preconvolved_telluric(
    files: Sequence[str],
    frame: Any,
    *,
    fibers: Any | None = None,
) -> np.ndarray:
    """Read three ``apTelluric`` files and sample them onto a frame.

    This translates the read/interpolate half of ``APTELLURIC_CONVOLVE``:
    select or linearly interpolate the two airmass extensions, then spline
    each fiber/species/model spectrum onto that fiber's wavelength solution.
    """

    from astropy.io import fits

    if len(files) != 3:
        raise ValueError("exactly three chip telluric files are required")
    chips = _chips(frame)
    nfibers, npix = np.asarray(_field(chips[0], "flux")).shape
    selected_fibers = (
        np.arange(nfibers, dtype=int)
        if fibers is None
        else np.asarray(fibers, dtype=int)
    )
    altitude = 60.0
    header = _field(chips[0], "header")
    try:
        altitude = float(header.get("ALT", 60.0))
    except AttributeError:
        altitude = 60.0
    if altitude < 5:
        altitude = 60.0
    airmass = 1.0 / np.cos(np.deg2rad(90.0 - altitude))
    result = None
    for chip_index, filename in enumerate(files):
        with fits.open(filename, memmap=False) as hdul:
            primary = hdul[0].header
            wavelength_grid = np.asarray(hdul[0].data, dtype=np.float64)
            if wavelength_grid.shape[0] == nfibers:
                wavelength_grid = wavelength_grid.T
            if wavelength_grid.shape[1] != nfibers:
                raise ValueError("telluric wavelength grid has wrong fiber count")
            air0 = float(primary.get("AIR0", 0.0))
            dair = float(primary.get("DAIR", 1.0))
            if air0 == 0 or len(hdul) == 2:
                lower = 0
            else:
                lower = max(0, int((airmass - air0) / dair))
                lower = min(lower, len(hdul) - 2)
            data1 = np.asarray(hdul[1 + lower].data, dtype=np.float64)
            if 2 + lower < len(hdul):
                data2 = np.asarray(hdul[2 + lower].data, dtype=np.float64)
                weight = np.clip(
                    (airmass - (air0 + lower * dair)) / dair, 0.0, 1.0
                )
                data = data1 + weight * (data2 - data1)
            else:
                data = data1
            # FITS reverses the IDL [wave,fiber,species,scale] axes.
            if data.ndim == 2:
                data = data[None, None, ...]
            elif data.ndim == 3:
                data = data[None, ...]
            if data.shape[-2:] != (nfibers, wavelength_grid.shape[0]):
                raise ValueError("telluric data axes do not match wavelength grid")
            data = np.transpose(data, (3, 2, 1, 0))
            nspecies, nmodel = data.shape[2:]
            if nspecies != 3:
                raise ValueError("telluric products must contain three species")
            if result is None:
                result = np.ones(
                    (nfibers, 3 * npix, 3, nmodel), dtype=np.float64
                )
            elif result.shape[-1] != nmodel:
                raise ValueError("telluric files have inconsistent model counts")
            target_wavelength = np.asarray(
                _field(chips[chip_index], "wavelength"), dtype=np.float64
            )
            for fiber in selected_fibers:
                order = np.argsort(wavelength_grid[:, fiber])
                source_wave = wavelength_grid[order, fiber]
                unique = np.concatenate(([True], np.diff(source_wave) > 0))
                source_wave = source_wave[unique]
                target = target_wavelength[fiber]
                target_order = np.argsort(target)
                section = slice(chip_index * npix, (chip_index + 1) * npix)
                for species in range(3):
                    for model_index in range(nmodel):
                        source = data[order, fiber, species, model_index][unique]
                        sampled = CubicSpline(
                            source_wave, source, extrapolate=False
                        )(target[target_order])
                        sampled = np.clip(
                            np.nan_to_num(sampled, nan=1.0), 0.0, np.inf
                        )
                        restored = np.empty(npix)
                        restored[target_order] = sampled
                        result[fiber, section, species, model_index] = restored
    assert result is not None
    return result


def select_telluric_standards(
    plugmap: Any,
    *,
    starfitopt: int = 1,
    force: bool = False,
    chip_b_flux: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return plug-map rows and zero-based detector rows used by APTELLURIC."""

    fiberdata = _field(plugmap, "fiberdata")
    spectrograph = np.asarray(_field(fiberdata, "spectrographid"), dtype=int)
    holetype = np.char.upper(np.asarray(_field(fiberdata, "holetype")).astype(str))
    objtype = np.char.upper(np.asarray(_field(fiberdata, "objtype")).astype(str))
    fiberid = np.asarray(_field(fiberdata, "fiberid"), dtype=int)
    if starfitopt == 1:
        allowed = objtype == "HOT_STD"
    elif starfitopt == 2:
        allowed = np.isin(objtype, ("HOT_STD", "STAR"))
    else:
        raise ValueError("starfitopt must be 1 (HOT_STD) or 2 (all stars)")
    selected = (spectrograph == 2) & (holetype == "OBJECT") & allowed
    rows = np.flatnonzero(selected)
    detector = 300 - fiberid[rows]
    valid = (detector >= 0) & (
        detector < (np.asarray(chip_b_flux).shape[0] if chip_b_flux is not None else 300)
    )
    rows, detector = rows[valid], detector[valid]
    if rows.size == 0 and force and chip_b_flux is not None:
        flux = np.asarray(chip_b_flux)
        lo, hi = min(900, flux.shape[1]), min(1101, flux.shape[1])
        median_flux = np.median(flux[:, lo:hi], axis=1)
        allowed = np.isin(objtype, ("HOT_STD", "STAR"))
        candidate = np.flatnonzero(
            (spectrograph == 2) & (holetype == "OBJECT") & allowed
        )
        candidate_detector = 300 - fiberid[candidate]
        valid = (
            (candidate_detector >= 0)
            & (candidate_detector < flux.shape[0])
            & (median_flux[np.clip(candidate_detector, 0, flux.shape[0] - 1)] > 100)
        )
        rows, detector = candidate[valid], candidate_detector[valid]
    return rows, detector


def _good_star_fits(fits: Sequence[TelluricStarFit]) -> np.ndarray:
    status = np.asarray([fit.status for fit in fits])
    snr = np.asarray([fit.snr for fit in fits])
    rchisq = np.asarray([fit.rchisq for fit in fits])
    scales = np.asarray([fit.scale for fit in fits])
    valid = (status > 0) & np.isfinite(rchisq)
    if not valid.any():
        return valid
    medchi = np.median(rchisq[valid])
    sigchi = max(_mad(rchisq[valid]), 1.0)
    medscale = np.median(scales[valid], axis=0)
    sigscale = np.median(np.abs(scales[valid] - medscale), axis=0)
    scale_ok = np.all(
        np.abs(scales - medscale)
        < 4 * np.maximum(sigscale, np.finfo(float).eps),
        axis=1,
    )
    good = valid & (snr > 20) & (rchisq <= medchi + 2.5 * sigchi) & scale_ok
    if good.sum() < 4:
        good = valid & (snr > 20) & (rchisq <= medchi + 2.5 * sigchi)
    if good.sum() < 4:
        good = valid & (snr > 10)
    return good


def telluric_correct_frame(
    frame: Any,
    plugmap: Any,
    preconvolved_models: Any,
    *,
    starfitopt: int = 1,
    force: bool = False,
    single: bool = False,
    telescope: str = "apo25m",
    copy_frame: bool = True,
) -> TelluricFrameResult:
    """Run the native preconvolved-model APTELLURIC path on a full frame."""

    output = copy.deepcopy(frame) if copy_frame else frame
    chips = _chips(output)
    fluxes = [np.asarray(_field(chip, "flux")) for chip in chips]
    nfibers, npix = fluxes[0].shape
    if any(flux.shape != (nfibers, npix) for flux in fluxes):
        raise ValueError("all chip flux arrays must share [fiber,pixel] shape")
    for chip in chips:
        for name in ("err", "mask", "sky"):
            if np.asarray(_field(chip, name)).shape != (nfibers, npix):
                raise ValueError(f"{name} must match the flux shape")
        _set_field(chip, "telluric", np.ones((nfibers, npix), np.float32))
        _set_field(chip, "telluricerr", np.zeros((nfibers, npix), np.float32))

    models = normalize_preconvolved_models(
        preconvolved_models, nfibers=nfibers, npix=npix
    )
    fiberdata = _field(plugmap, "fiberdata")
    plug_rows, detector_rows = select_telluric_standards(
        plugmap,
        starfitopt=starfitopt,
        force=force,
        chip_b_flux=fluxes[1],
    )
    if detector_rows.size < (1 if single else 2):
        raise ValueError("not enough telluric standards")

    star_fits = []
    for detector in detector_rows:
        combined_flux = np.concatenate(
            [np.asarray(_field(chip, "flux"))[detector] for chip in chips]
        )
        combined_error = np.concatenate(
            [np.asarray(_field(chip, "err"))[detector] for chip in chips]
        )
        combined_mask = np.concatenate(
            [np.asarray(_field(chip, "mask"))[detector] for chip in chips]
        )
        combined_sky = np.concatenate(
            [np.asarray(_field(chip, "sky"))[detector] for chip in chips]
        )
        star_fits.append(
            fit_telluric_star(
                combined_flux,
                combined_error,
                models[detector],
                mask=combined_mask,
                sky=combined_sky,
                fiber=int(detector),
            )
        )
    successful = [fit for fit in star_fits if fit.status > 0]
    if not successful:
        raise ValueError("no good telluric spectrum fits")
    best_model = np.rint(
        np.median(np.asarray([fit.best_model for fit in successful]), axis=0)
    ).astype(int)

    if models.shape[-1] > 1:
        fixed = np.stack(
            [models[:, :, species, best_model[species]] for species in range(3)],
            axis=2,
        )[..., None]
        refitted = []
        for plug_row, detector in zip(plug_rows, detector_rows):
            combined_flux = np.concatenate(
                [np.asarray(_field(chip, "flux"))[detector] for chip in chips]
            )
            combined_error = np.concatenate(
                [np.asarray(_field(chip, "err"))[detector] for chip in chips]
            )
            combined_mask = np.concatenate(
                [np.asarray(_field(chip, "mask"))[detector] for chip in chips]
            )
            combined_sky = np.concatenate(
                [np.asarray(_field(chip, "sky"))[detector] for chip in chips]
            )
            refitted.append(
                fit_telluric_star(
                    combined_flux,
                    combined_error,
                    fixed[detector],
                    mask=combined_mask,
                    sky=combined_sky,
                    fiber=int(detector),
                )
            )
        star_fits = refitted

    good = _good_star_fits(star_fits)
    minimum = 1 if single else 4
    if good.sum() < minimum:
        raise ValueError(f"not enough good telluric spectrum fits: {good.sum()}")
    zeta_all = np.asarray(_field(fiberdata, "zeta"), dtype=float)
    eta_all = np.asarray(_field(fiberdata, "eta"), dtype=float)
    fit_scales = np.asarray([fit.scale for fit in star_fits])
    spatial_fits = fit_spatial_scales(
        zeta_all[plug_rows][good],
        eta_all[plug_rows][good],
        fit_scales[good],
        single=single,
    )

    fiberid = np.asarray(_field(fiberdata, "fiberid"), dtype=int)
    spectrograph = np.asarray(_field(fiberdata, "spectrographid"), dtype=int)
    objtype = np.char.upper(np.asarray(_field(fiberdata, "objtype")).astype(str))
    magnitude = np.asarray(_field(fiberdata, "mag"), dtype=float)
    mjd = int(np.asarray(_field(plugmap, "mjd")).ravel()[0])
    fpi = (87, 218) if telescope.lower() == "lco25m" else (75, 225)
    tellstar = {
        "im": np.int32(0),
        "scale": np.zeros((nfibers, 3), np.float32),
        "sig": np.asarray([fit.scatter for fit in spatial_fits], np.float32),
        "nfit": np.asarray([fit.npoints for fit in spatial_fits], np.int16),
        "bestmod": best_model.astype(np.int16),
        "fitpars": np.zeros((3, 6), np.float32),
        "fitscale": np.zeros((nfibers, 3), np.float32),
        "rchisq": np.zeros(nfibers, np.float32),
        "status": np.zeros(nfibers, np.int16),
        "zeta": np.zeros(nfibers, np.float32),
        "eta": np.zeros(nfibers, np.float32),
        "mag": np.zeros((nfibers, magnitude.shape[1]), np.float32),
    }
    for species, fit in enumerate(spatial_fits):
        tellstar["fitpars"][species, : fit.coefficients.size] = fit.coefficients
        for chip in chips:
            header = _field(chip, "header")
            for coefficient, value in enumerate(fit.coefficients):
                header[f"TLPR{species + 1}_{coefficient + 1}"] = float(value)
                header[f"TLER{species + 1}_{coefficient + 1}"] = float(
                    fit.errors[coefficient]
                )
            try:
                header.add_history(
                    f"APTELLURIC: {fit.species} spatial fit "
                    f"N={fit.npoints} SIG={fit.scatter:.5f}"
                )
            except AttributeError:
                history = header.setdefault("HISTORY", [])
                if isinstance(history, str):
                    history = [history]
                    header["HISTORY"] = history
                history.append(
                    f"APTELLURIC: {fit.species} spatial fit "
                    f"N={fit.npoints} SIG={fit.scatter:.5f}"
                )
    for fit in star_fits:
        tellstar["fitscale"][fit.fiber] = fit.scale
        tellstar["rchisq"][fit.fiber] = fit.rchisq
        tellstar["status"][fit.fiber] = fit.status

    row_by_detector = {
        300 - int(fid): row
        for row, fid in enumerate(fiberid)
        if spectrograph[row] == 2 and objtype[row] != "SKY"
    }
    for detector, plug_row in row_by_detector.items():
        if not 0 <= detector < nfibers:
            continue
        if mjd >= 59556 and detector in fpi:
            continue
        chosen_models = np.stack(
            [
                models[detector, :, species, best_model[species]]
                for species in range(3)
            ],
            axis=1,
        )
        combined_flux = np.concatenate(
            [np.asarray(_field(chip, "flux"))[detector] for chip in chips]
        )
        combined_error = np.concatenate(
            [np.asarray(_field(chip, "err"))[detector] for chip in chips]
        )
        corrected = apply_telluric_correction(
            combined_flux,
            combined_error,
            chosen_models,
            float(zeta_all[plug_row]),
            float(eta_all[plug_row]),
            spatial_fits,
        )
        tellstar["scale"][detector] = corrected.scale
        tellstar["zeta"][detector] = zeta_all[plug_row]
        tellstar["eta"][detector] = eta_all[plug_row]
        tellstar["mag"][detector] = magnitude[plug_row]
        for chip_index, chip in enumerate(chips):
            section = slice(chip_index * npix, (chip_index + 1) * npix)
            np.asarray(_field(chip, "flux"))[detector] = corrected.flux[section]
            np.asarray(_field(chip, "err"))[detector] = corrected.error[section]
            np.asarray(_field(chip, "telluric"))[detector] = corrected.transmission[
                section
            ]
            np.asarray(_field(chip, "telluricerr"))[
                detector
            ] = corrected.transmission_error[section]
            deep = _smooth(corrected.transmission[section], 5) < 0.9
            mask = np.asarray(_field(chip, "mask"))
            mask[detector, deep] |= TELLURIC_BIT
    _set_field(output, "tellstar", tellstar)
    return TelluricFrameResult(output, tellstar, star_fits, spatial_fits)
