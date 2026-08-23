"""Airglow line-model components used by APSKYSUB ``SUBOPTION=2``."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.optimize import curve_fit
from scipy.special import erf, hermitenorm


@dataclass(frozen=True)
class AirglowLine:
    id: int
    wave: float
    species: str
    doublet: bool = False
    separation: float = 0.0
    emission: float = 0.0


@dataclass
class MeasuredSkyLine:
    skyfiber: int
    fiber: int
    chip: int
    model_id: int
    model_wave: float
    species: str
    flux: float
    flux_error: float
    fitted_wave: float
    status: int
    fiber_species_norm: float = 1.0


@dataclass
class AirglowModel:
    lines: list[AirglowLine]
    median_flux: dict[int, float]
    line_scatter: dict[int, float]
    line_count: dict[int, int]
    species_coefficients: dict[str, np.ndarray]
    species_errors: dict[str, np.ndarray]
    species_scatter: dict[str, float]
    measurements: list[MeasuredSkyLine]


def load_airglow_lines(source: Any) -> list[AirglowLine]:
    """Read the APOGEE ``airglow.txt`` format or normalize line records."""

    if isinstance(source, (str, Path)):
        rows = np.genfromtxt(
            source,
            comments="#",
            names=(
                "id",
                "chipnum",
                "xpix",
                "wave",
                "emission",
                "doublet",
                "dbl_wsep",
                "name",
                "uselsf",
                "usewave",
                "source",
            ),
            dtype=None,
            encoding=None,
        )
        return [
            AirglowLine(
                int(row["id"]),
                float(row["wave"]),
                str(row["name"]).upper(),
                bool(row["doublet"]),
                float(row["dbl_wsep"]),
                float(row["emission"]),
            )
            for row in np.atleast_1d(rows)
        ]
    output = []
    for row in source:
        if isinstance(row, AirglowLine):
            output.append(row)
        elif isinstance(row, dict):
            output.append(
                AirglowLine(
                    int(row["id"]),
                    float(row["wave"]),
                    str(row.get("species", row.get("name", ""))).upper(),
                    bool(row.get("doublet", False)),
                    float(row.get("separation", row.get("dbl_wsep", 0))),
                    float(row.get("emission", 0)),
                )
            )
        else:
            output.append(AirglowLine(*row))
    return output


def poly2d_basis(x: Any, y: Any) -> np.ndarray:
    """The exact 11 terms in IDL ``FUNC_POLY2D`` ordering."""

    x, y = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    )
    return np.stack(
        (
            np.ones_like(x),
            x,
            x**2,
            x**3,
            x * y,
            x**2 * y,
            x * y**2,
            x**2 * y**2,
            y,
            y**2,
            y**3,
        ),
        axis=-1,
    )


def evaluate_poly2d(x: Any, y: Any, coefficients: Any) -> np.ndarray:
    return poly2d_basis(x, y) @ np.asarray(coefficients, dtype=np.float64)


def _unpack_lsf(parameters: Any, center: float) -> tuple[float, np.ndarray, int, np.ndarray]:
    pars = np.asarray(parameters, dtype=np.float64).ravel()
    if pars.size < 5:
        raise ValueError("LSF coefficient vector is incomplete")
    binsize, xoffset, horder = pars[0], pars[1], int(round(pars[2]))
    porder = pars[3 : horder + 4].astype(int)
    cursor = horder + 4
    gh_values = []
    evaluation_x = center + xoffset
    for order in porder:
        coefficients = pars[cursor : cursor + order + 1]
        gh_values.append(np.polynomial.polynomial.polyval(evaluation_x, coefficients))
        cursor += order + 1
    wing_type = 0
    wing_values = np.array([], dtype=np.float64)
    if cursor + 1 < pars.size:
        wing_type = int(round(pars[cursor]))
        nwing = int(round(pars[cursor + 1]))
        cursor += 2
        wing_orders = pars[cursor : cursor + nwing].astype(int)
        cursor += nwing
        values = []
        for order in wing_orders:
            coefficients = pars[cursor : cursor + order + 1]
            values.append(
                np.polynomial.polynomial.polyval(evaluation_x, coefficients)
            )
            cursor += order + 1
        wing_values = np.asarray(values)
    return binsize, np.asarray(gh_values), wing_type, wing_values


def _power_integrals(w1: np.ndarray, w2: np.ndarray, maximum: int) -> np.ndarray:
    integrals = np.zeros((maximum + 1, w1.size), dtype=np.float64)
    e1, e2 = np.exp(-0.5 * w1**2), np.exp(-0.5 * w2**2)
    integrals[0] = np.sqrt(np.pi / 2) * (
        erf(w2 / np.sqrt(2)) - erf(w1 / np.sqrt(2))
    )
    if maximum >= 1:
        integrals[1] = e1 - e2
    for power in range(2, maximum + 1):
        integrals[power] = (
            w1 ** (power - 1) * e1
            - w2 ** (power - 1) * e2
            + (power - 1) * integrals[power - 2]
        )
    return integrals


def lsf_gh(x: Any, center: float, parameters: Any) -> np.ndarray:
    """Normalized binned Gauss–Hermite LSF translated from ``LSF_GH``.

    The commonly used Gaussian wing profile is supported. Other historical
    wing types raise explicitly instead of silently changing the LSF.
    """

    x = np.asarray(x, dtype=np.float64)
    binsize, gh_values, wing_type, wing_values = _unpack_lsf(parameters, center)
    if binsize <= 0 or gh_values.size == 0 or gh_values[0] <= 0:
        raise ValueError("LSF requires positive binsize and sigma")
    sigma = gh_values[0]
    hcoeff = np.zeros(gh_values.size)
    hcoeff[0] = 1.0
    if gh_values.size > 1:
        hcoeff[1:] = gh_values[1:]
    wing_norm = float(wing_values[0]) if wing_values.size else 0.0
    hcoeff *= 1.0 - wing_norm
    w1 = (x - 0.5 * binsize - center) / sigma
    w2 = (x + 0.5 * binsize - center) / sigma
    integrals = _power_integrals(w1, w2, len(hcoeff) - 1)
    profile = np.zeros_like(x)
    for order, coefficient in enumerate(hcoeff):
        polynomial = hermitenorm(order)
        powers = polynomial.c[::-1]
        integral = sum(
            value * integrals[power] for power, value in enumerate(powers)
        )
        profile += coefficient * integral / np.sqrt(
            factorial(order) * 2 * np.pi
        )
    if wing_values.size:
        if wing_type != 1 or wing_values.size < 2:
            raise NotImplementedError(
                f"LSF wing profile type {wing_type} is not implemented"
            )
        wing_sigma = wing_values[1]
        ww1 = (x - 0.5 * binsize - center) / wing_sigma
        ww2 = (x + 0.5 * binsize - center) / wing_sigma
        profile += 0.5 * wing_norm * (
            erf(ww2 / np.sqrt(2)) - erf(ww1 / np.sqrt(2))
        )
    return profile


def _continuum_removed(flux: np.ndarray) -> np.ndarray:
    first = median_filter(flux, size=101, mode="nearest")
    scatter = np.median(np.abs(flux - np.median(flux)))
    temporary = flux.copy()
    bad = maximum_filter1d(
        (np.abs(temporary - first) > 2 * scatter).astype(np.uint8),
        size=5,
        mode="nearest",
    ).astype(bool)
    temporary[bad] = np.nan

    def fill(values):
        values = values.copy()
        missing = ~np.isfinite(values)
        if np.any(missing) and np.any(~missing):
            values[missing] = np.interp(
                np.flatnonzero(missing),
                np.flatnonzero(~missing),
                values[~missing],
            )
        return values

    second = median_filter(fill(temporary), size=101, mode="nearest")
    temporary = flux.copy()
    bad = maximum_filter1d(
        (np.abs(temporary - second) > 2 * scatter).astype(np.uint8),
        size=5,
        mode="nearest",
    ).astype(bool)
    temporary[bad] = np.nan
    final = median_filter(fill(temporary), size=101, mode="nearest")
    return flux - final


def fit_sky_lines(
    chips: list[Any],
    sky_rows: np.ndarray,
    lines: list[AirglowLine],
    field,
) -> list[MeasuredSkyLine]:
    """Fit fixed-LSF line flux and a ±3-pixel center offset."""

    measurements: list[MeasuredSkyLine] = []
    for sky_number, row in enumerate(sky_rows):
        for chip_index, chip in enumerate(chips):
            flux = np.asarray(field(chip, "flux"), dtype=np.float64)[row]
            error = np.asarray(field(chip, "err"), dtype=np.float64)[row].copy()
            wavelength = np.asarray(field(chip, "wavelength"), dtype=np.float64)[row]
            lsfcoef = np.asarray(field(chip, "lsfcoef"), dtype=np.float64)[row]
            clean = _continuum_removed(flux)
            difference = np.roll(flux, 1) - flux
            difference_scatter = np.median(np.abs(difference))
            median_error = np.median(error)
            added_error = np.sqrt(
                max(difference_scatter**2 - median_error**2, 0.0)
            )
            if added_error > 5:
                error = np.sqrt(error**2 + added_error**2)
            order = np.argsort(wavelength)
            pixel = np.arange(wavelength.size, dtype=np.float64)
            for line in lines:
                if not np.nanmin(wavelength) <= line.wave <= np.nanmax(wavelength):
                    continue
                center0 = float(np.interp(line.wave, wavelength[order], pixel[order]))
                use = np.flatnonzero(np.abs(pixel - center0) < 16)
                if use.size < 7:
                    continue

                def model(x, amplitude, center):
                    if line.doublet:
                        dw = np.median(np.abs(np.diff(wavelength)))
                        separation = line.separation / dw
                        return amplitude * (
                            lsf_gh(x, center - 0.5 * separation, lsfcoef)
                            + lsf_gh(x, center + 0.5 * separation, lsfcoef)
                        )
                    return amplitude * lsf_gh(x, center, lsfcoef)

                sigma = np.maximum(error[use], 1.0)
                initial = max(float(clean[int(round(center0))]), 1.0)
                try:
                    pars, covariance = curve_fit(
                        model,
                        pixel[use],
                        clean[use],
                        p0=(initial, center0),
                        sigma=sigma,
                        absolute_sigma=True,
                        bounds=((0, center0 - 3), (np.inf, center0 + 3)),
                        maxfev=10_000,
                    )
                    perror = np.sqrt(np.maximum(np.diag(covariance), 0))
                    fitted_wave = float(
                        np.interp(pars[1], pixel[order], wavelength[order])
                    )
                    measurements.append(
                        MeasuredSkyLine(
                            sky_number,
                            int(row),
                            chip_index,
                            line.id,
                            line.wave,
                            line.species,
                            float(pars[0]),
                            float(perror[0]),
                            fitted_wave,
                            1,
                        )
                    )
                except (RuntimeError, ValueError, FloatingPointError):
                    continue
    return measurements


def _weighted_location(values: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    center = np.median(values)
    mad = max(np.median(np.abs(values - center)), 1.0)
    good = np.abs(values - center) < 3 * mad
    valid_error = np.isfinite(errors) & (errors > 0)
    good &= valid_error
    if np.any(good):
        weights = 1.0 / errors[good] ** 2
        mean = float(np.sum(values[good] * weights) / np.sum(weights))
        scatter = float(np.sqrt(np.sum(weights * (values[good] - mean) ** 2) / np.sum(weights)))
        return mean, scatter
    return float(center), float(mad)


def _spatial_fit(
    zeta: np.ndarray, eta: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    design = poly2d_basis(zeta, eta)
    if values.size < 11:
        coefficients = np.zeros(11)
        coefficients[0] = np.median(values)
        scatter = float(np.median(np.abs(values - coefficients[0])))
        errors = np.zeros(11)
        errors[0] = scatter
        return coefficients, errors, scatter
    good = np.isfinite(values)
    coefficients, *_ = np.linalg.lstsq(design[good], values[good], rcond=None)
    residual = values - design @ coefficients
    scatter = np.median(np.abs(residual[good] - np.median(residual[good])))
    if scatter > 0:
        good &= np.abs(residual) <= 2.5 * scatter
        coefficients, *_ = np.linalg.lstsq(
            design[good], values[good], rcond=None
        )
        residual = values[good] - design[good] @ coefficients
        scatter = float(np.median(np.abs(residual - np.median(residual))))
    dof = max(np.count_nonzero(good) - 11, 1)
    covariance = np.linalg.pinv(design[good].T @ design[good])
    variance = float(np.sum(residual**2) / dof)
    errors = np.sqrt(np.maximum(np.diag(covariance) * variance, 0))
    return coefficients, errors, scatter


def build_airglow_model(
    measurements: list[MeasuredSkyLine],
    lines: list[AirglowLine],
    sky_rows: np.ndarray,
    zeta: np.ndarray,
    eta: np.ndarray,
) -> AirglowModel:
    """Iterate line medians and fiber/species normalizations as in IDL."""

    species = ("OH", "O2")
    nsky = len(sky_rows)
    norms = np.ones((nsky, len(species)))
    median_flux: dict[int, float] = {}
    line_scatter: dict[int, float] = {}
    line_count: dict[int, int] = {}
    line_lookup = {line.id: line for line in lines}
    threshold = nsky / 2
    previous = None
    for _ in range(11):
        for line_id in sorted({measurement.model_id for measurement in measurements}):
            selected = [
                measurement
                for measurement in measurements
                if measurement.model_id == line_id and measurement.status > 0
            ]
            line_count[line_id] = len(selected)
            if len(selected) <= 1:
                continue
            values = np.asarray(
                [
                    measurement.flux
                    / norms[
                        measurement.skyfiber,
                        species.index(measurement.species),
                    ]
                    for measurement in selected
                ]
            )
            errors = np.asarray(
                [
                    max(measurement.flux_error, 1.0)
                    / norms[
                        measurement.skyfiber,
                        species.index(measurement.species),
                    ]
                    for measurement in selected
                ]
            )
            median_flux[line_id], line_scatter[line_id] = _weighted_location(
                values, errors
            )
        for species_index, name in enumerate(species):
            for sky_number in range(nsky):
                selected = [
                    measurement
                    for measurement in measurements
                    if measurement.skyfiber == sky_number
                    and measurement.species == name
                    and line_count.get(measurement.model_id, 0) > threshold
                    and not (
                        sky_rows[sky_number] > 200 and measurement.chip >= 2
                    )
                    and measurement.model_id in median_flux
                ]
                if not selected:
                    continue
                fractions = np.asarray(
                    [
                        measurement.flux / median_flux[measurement.model_id]
                        for measurement in selected
                    ]
                )
                errors = np.asarray(
                    [
                        measurement.flux_error / median_flux[measurement.model_id]
                        for measurement in selected
                    ]
                )
                norms[sky_number, species_index], _ = _weighted_location(
                    fractions, errors
                )
                for measurement in selected:
                    measurement.fiber_species_norm = norms[
                        sky_number, species_index
                    ]
        state = np.concatenate(
            ([median_flux.get(line.id, 0) for line in lines], norms.ravel())
        )
        if previous is not None:
            denominator = np.maximum(np.abs(state), 1e-12)
            if np.max(np.abs(state - previous) / denominator * 100) < 0.01:
                break
        previous = state.copy()

    coefficients: dict[str, np.ndarray] = {}
    coefficient_errors: dict[str, np.ndarray] = {}
    species_scatter: dict[str, float] = {}
    for species_index, name in enumerate(species):
        if not any(line.species == name for line in lines):
            continue
        result = _spatial_fit(zeta, eta, norms[:, species_index])
        coefficients[name], coefficient_errors[name], species_scatter[name] = result
    used_lines = [
        line
        for line in lines
        if line_count.get(line.id, 0) > threshold and line.id in median_flux
    ]
    return AirglowModel(
        used_lines,
        median_flux,
        line_scatter,
        line_count,
        coefficients,
        coefficient_errors,
        species_scatter,
        measurements,
    )


def synthesize_airglow(
    wavelength: np.ndarray,
    lsfcoef: np.ndarray,
    zeta: float,
    eta: float,
    model: AirglowModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize the fitted airglow spectrum on one fiber/chip grid."""

    wavelength = np.asarray(wavelength, dtype=np.float64)
    pixel = np.arange(wavelength.size, dtype=np.float64)
    order = np.argsort(wavelength)
    output = np.zeros(wavelength.size)
    error = np.zeros(wavelength.size)
    pixel_width = abs(float(np.median(np.diff(wavelength))))
    scale = 2 if wavelength.size == 4096 else 1
    for line in model.lines:
        if not wavelength.min() - 2 <= line.wave <= wavelength.max() + 2:
            continue
        spatial = evaluate_poly2d(
            zeta, eta, model.species_coefficients[line.species]
        )
        flux = model.median_flux[line.id] * float(spatial)
        centers = [line.wave]
        if line.doublet:
            centers = [
                line.wave - 0.5 * line.separation,
                line.wave + 0.5 * line.separation,
            ]
        for wave_center in centers:
            center = float(
                np.interp(wave_center, wavelength[order], pixel[order])
            ) * scale
            use = np.flatnonzero(abs(wavelength - wave_center) <= 50 * pixel_width)
            if use.size == 0:
                continue
            profile = lsf_gh(pixel[use] * scale, center, lsfcoef)
            contribution = flux * profile
            output[use] += contribution
            # Preserve the IDL accumulation, which adds sqrt(line) directly.
            error[use] += np.sqrt(np.maximum(contribution, 0))
    return output, error
