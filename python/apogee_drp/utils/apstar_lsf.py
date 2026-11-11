"""Propagate native apVisit line profiles to the rest-frame apStar pixel scale.

Native pixel-integrated profiles use the original APOGEE IDL normalization;
a lightweight adapter reads and writes the Doppler/APOGEE parameter format.
They are passed through the actual APOGEE damped-sinc interpolation formula;
the output profiles are approximated by one shared polynomial coefficient set
for the GH-plus-Gaussian-wing fields. This is a response calculation, not a
coordinate relabeling of the input polynomial coefficients.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import hashlib
import time

import numpy as np
from astropy.io import fits
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import least_squares
from scipy.special import ndtr, eval_hermitenorm
from numpy.polynomial import chebyshev as cheb

from .lsf_parameters import unpack_parameters, pack_parameters

CSPEED = 299792.458


def wave_to_pixel(wavelength, native_wave):
    """Invert a chip wavelength solution with the same spline as rv.visitcomb.

    Both increasing and decreasing native wavelength arrays are accepted.
    Values outside the chip return NaN. The result uses zero-based *array*
    pixels, before conversion to LSF calibration pixels.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    order = np.argsort(native_wave)
    w = np.asarray(native_wave, dtype=float)[order]
    if not np.isfinite(w).all() or np.any(np.diff(w) <= 0):
        raise ValueError("Native wavelengths must be finite and strictly monotonic")
    p = np.arange(len(w), dtype=float)[order]
    baseline = np.polynomial.Polynomial.fit(w, p, 1)
    spline = InterpolatedUnivariateSpline(w, p / baseline(w), k=3)
    result = baseline(wavelength) * spline(wavelength)
    return np.where((wavelength >= w[0]) & (wavelength <= w[-1]), result, np.nan)


def sinc_weights(pixel, nres, npix):
    """Return native indices/weights used by apred.sincint.sincint.

    The existing routine's exact-integer special case is deliberately retained.
    Rows need not sum to precisely one. Out-of-range native indices are assigned
    zero weight, matching the pipeline's truncated interpolation stencil.
    """
    pixel = np.atleast_1d(np.asarray(pixel, dtype=float))
    if not np.isfinite(pixel).all() or nres <= 0:
        raise ValueError("Finite interpolation pixels and positive nres required")
    size = int(21 * nres / 2)
    size += size % 2 == 0
    ix = pixel.astype(int)
    frac = pixel - ix
    offset = np.arange(size) - size // 2
    x = (offset[None, :] - frac[:, None]) / (nres / 2)
    weight = np.exp(-((x / (3.25 * nres / 2)) ** 2)) * np.sinc(x) / (nres / 2)
    weight[x == 0] = 1  # Reproduce sincint; do not silently change its convention.
    index = ix[:, None] + offset
    valid = (index >= 0) & (index < npix)
    weight = np.where(valid, weight, 0.0)
    return np.clip(index, 0, npix - 1), weight


def integrated_basis(offset, sigma, order, binsize=1.0):
    """Unit-area, pixel-integrated probabilists' Hermite basis, H0 through Horder.

    Integration is over one *output* pixel. This function describes the fitted
    output profile; it is never applied as a second integration to the reference
    native profiles that have already been sampled by the visit resampler.
    """
    offset, sigma = np.broadcast_arrays(offset, sigma)
    lo = (offset - 0.5 * binsize) / sigma
    hi = (offset + 0.5 * binsize) / sigma
    phi_lo = np.exp(-0.5 * lo**2) / np.sqrt(2 * np.pi)
    phi_hi = np.exp(-0.5 * hi**2) / np.sqrt(2 * np.pi)
    basis = [ndtr(hi) - ndtr(lo)]
    for m in range(1, order + 1):
        basis.append(
            (
                phi_lo * eval_hermitenorm(m - 1, lo)
                - phi_hi * eval_hermitenorm(m - 1, hi)
            )
            / np.sqrt(math.factorial(m))
        )
    return np.stack(basis, axis=-1)


def _field_profile(fields, offsets, hermite_orders, derivatives=False):
    """Evaluate normalized output kernels and exact derivatives of local fields."""
    sigma, fraction, wing_sigma = fields[0], fields[-2], fields[-1]
    order = max(hermite_orders)
    basis = integrated_basis(offsets, sigma[:, None], order)
    core = basis[..., 0].copy()
    for i, m in enumerate(hermite_orders):
        core += fields[i + 1, :, None] * basis[..., m]
    wing = integrated_basis(offsets, wing_sigma[:, None], 0)[..., 0]
    raw = (1 - fraction[:, None]) * core + fraction[:, None] * wing
    area = raw.sum(axis=1)[:, None]
    value = raw / area
    if not derivatives:
        return value

    def width_derivative(width, maxorder):
        lo = (offsets - 0.5) / width[:, None]
        hi = (offsets + 0.5) / width[:, None]
        left = lo * np.exp(-lo * lo / 2) / np.sqrt(2 * np.pi)
        right = hi * np.exp(-hi * hi / 2) / np.sqrt(2 * np.pi)
        return np.stack(
            [
                (left * eval_hermitenorm(m, lo) - right * eval_hermitenorm(m, hi))
                / (width[:, None] * np.sqrt(math.factorial(m)))
                for m in range(maxorder + 1)
            ],
            axis=-1,
        )

    dbasis = width_derivative(sigma, order)
    dcore = dbasis[..., 0].copy()
    for i, m in enumerate(hermite_orders):
        dcore += fields[i + 1, :, None] * dbasis[..., m]
    parts = [(1 - fraction[:, None]) * dcore]
    parts += [(1 - fraction[:, None]) * basis[..., m] for m in hermite_orders]
    parts += [wing - core, fraction[:, None] * width_derivative(wing_sigma, 0)[..., 0]]
    d_raw = np.stack(parts, axis=-1)
    d_value = (d_raw - value[..., None] * d_raw.sum(axis=1)[:, None, :]) / area[
        ..., None
    ]
    return value, d_value


def evaluate_apogee_gh(offsets, centers, parameters):
    """Evaluate packed APOGEE GH-plus-Gaussian-wing parameters as in IDL.

    Uses the APOGEE parameter layout and applies the unit-area Hermite
    normalization exactly once and retains every requested polynomial term.
    This avoids two translation defects in existing Doppler ghlsf/gausshermitebin.
    Only the Gaussian wing type used by the supplied APOGEE visits is supported;
    other wing types raise NotImplementedError. Negative GH values are retained.
    """

    p = unpack_parameters(np.asarray(parameters))
    if p["Wproftype"] != 1 or p["nWpar"] != 2:
        raise NotImplementedError("Only the APOGEE Gaussian wing model is supported")
    if p["binsize"] <= 0:
        raise ValueError("A positive native pixel integration width is required")
    centers = np.atleast_1d(centers).astype(float)
    offsets = np.broadcast_to(offsets, (len(centers), np.shape(offsets)[-1]))
    coordinate = centers + p["Xoffset"]
    gh = np.array(
        [np.polynomial.polynomial.polyval(coordinate, row) for row in p["GHcoefs"]]
    )
    wings = np.array(
        [np.polynomial.polynomial.polyval(coordinate, row) for row in p["Wcoefs"]]
    )
    if np.any(gh[0] <= 0) or np.any(wings[1] <= 0):
        raise ValueError("Nonpositive native core or wing width")
    basis = integrated_basis(offsets, gh[0, :, None], p["Horder"], p["binsize"])
    core = basis[..., 0].copy()
    for m in range(1, p["Horder"] + 1):
        core += gh[m, :, None] * basis[..., m]
    wing = integrated_basis(offsets, wings[1, :, None], 0, p["binsize"])[..., 0]
    result = (1 - wings[0, :, None]) * core + wings[0, :, None] * wing
    return result[0] if len(centers) == 1 else result


@dataclass
class VisitLSF:
    """Native visit wavelengths, packed Doppler parameters, and resampling inputs.

    ``coordinate_scale`` converts an array-pixel coordinate to the coordinate
    used by the stored LSF polynomials. It does not change ``binsize`` or
    ``Xoffset``, which remain part of the native LSF description.
    """

    filename: str
    wave: np.ndarray
    pars: np.ndarray
    vrel: float
    nres: np.ndarray
    coordinate_scale: np.ndarray
    _inverse: list = field(default_factory=list, init=False, repr=False)

    def pixel_at_wave(self, wavelength, chip):
        """Evaluate a cached inverse wavelength solution, preserving pipeline math."""
        if not self._inverse:
            for wave in self.wave:
                order = np.argsort(wave)
                w = wave[order]
                if not np.isfinite(w).all() or np.any(np.diff(w) <= 0):
                    raise ValueError(
                        "Native wavelengths must be finite and strictly monotonic"
                    )
                pix = np.arange(len(w), dtype=float)[order]
                baseline = np.polynomial.Polynomial.fit(w, pix, 1)
                spline = InterpolatedUnivariateSpline(w, pix / baseline(w), k=3)
                self._inverse.append((baseline, spline, w[0], w[-1]))
        baseline, spline, lo, hi = self._inverse[chip]
        wavelength = np.asarray(wavelength)
        value = baseline(wavelength) * spline(wavelength)
        return np.where((wavelength >= lo) & (wavelength <= hi), value, np.nan)

    @classmethod
    def read(cls, filename, vrel, nres=(5.0, 4.25, 3.5), coordinate_scale=None):
        """Read apVisit HDUs 4 and 10, validating supported pixel conventions.

        Auto-detection follows the older apvisitcomb convention: 4096-sample
        visits with binsize=2 use their array pixels directly; binsize=1 uses
        half the array pixel coordinate. Other conventions require an explicit
        three-element ``coordinate_scale``. No observed-flux fitting is done.
        """
        with fits.open(filename) as hdul:
            wave = np.array(hdul[4].data, dtype=float, copy=True)
            pars = np.array(hdul[10].data, dtype=float, copy=True)
        if wave.ndim != 2 or wave.shape[0] != 3 or pars.shape[0] != 3:
            raise ValueError(
                "Expected three native chips and three LSF parameter arrays"
            )
        if not np.isfinite(vrel) or 1 + vrel / CSPEED <= 0:
            raise ValueError("A finite visit-frame VREL in km/s is required")
        if coordinate_scale is None:
            binsize = pars[:, 0]
            if wave.shape[1] != 4096 or not np.isin(binsize, [1.0, 2.0]).all():
                raise ValueError(
                    "Unrecognized native convention: supply coordinate_scale"
                )
            coordinate_scale = binsize / 2
        scale = np.broadcast_to(np.asarray(coordinate_scale, float), (3,)).copy()
        resolution = np.broadcast_to(np.asarray(nres, float), (3,)).copy()
        if (
            not np.isfinite(scale).all()
            or np.any(scale <= 0)
            or np.any(resolution <= 0)
        ):
            raise ValueError("Coordinate scales and nres must be finite and positive")
        return cls(str(filename), wave, pars, float(vrel), resolution, scale)

    def response(self, star_wave, center, output_pixels, chip):
        """Propagate an unresolved line at fractional apStar pixel ``center``.

        Returns a normalized sampled line response, retaining negative sinc
        lobes. Native profiles are evaluated in their original IDL GH convention,
        including pixel integration and wings. Chip-edge stencils that would
        lose samples raise ValueError instead of being silently normalized.
        """
        star_pixel = np.arange(len(star_wave))
        log_wave = np.log10(star_wave)
        center_wave = 10 ** np.interp(center, star_pixel, log_wave)
        factor = 1 + self.vrel / CSPEED
        x0 = float(self.pixel_at_wave(center_wave * factor, chip))
        xout = self.pixel_at_wave(star_wave[output_pixels] * factor, chip)
        if not np.isfinite(x0) or not np.isfinite(xout).all():
            raise ValueError("Line response extends outside this visit chip")
        index, weight = sinc_weights(xout, self.nres[chip], self.wave.shape[1])
        if index.min() == 0 or index.max() == self.wave.shape[1] - 1:
            raise ValueError("Line response requires a truncated native sinc stencil")
        scale = self.coordinate_scale[chip]
        unique, inverse = np.unique(index, return_inverse=True)
        native = evaluate_apogee_gh(
            (unique * scale - x0 * scale)[None, :],
            np.array([x0 * scale]),
            self.pars[chip],
        )
        profile = np.sum(weight * native[inverse.reshape(index.shape)], axis=1)
        area = profile.sum()
        if not np.isfinite(profile).all() or area <= 0:
            raise ValueError("Non-finite or nonpositive integrated line response")
        return profile / area


@dataclass
class ChipLSF:
    """Smooth output-pixel GH coefficients for one chip segment.

    Polynomial rows are sigma, H1, H3 ... Horder, wing area fraction, wing sigma.
    H0 is one; H2 is normally fixed to zero to remove its width degeneracy.
    With fit_h2=True, H2 is included between H1 and H3.
    All coefficients use Chebyshev polynomials in (pixel-origin)/scale.
    The packed representation returned by ``doppler_parameters`` uses ordinary
    ascending-power polynomials in pixel-origin, as Doppler expects.
    """

    chip: int
    origin: float
    scale: float
    coefficients: np.ndarray
    horder: int
    domain: tuple
    diagnostics: dict

    @property
    def hermite_orders(self):
        return (
            list(range(1, self.horder + 1))
            if self.coefficients.shape[0] == self.horder + 3
            else [1] + list(range(3, self.horder + 1))
        )

    def evaluate(self, centers, offsets):
        """Evaluate sampled fitted profiles at arbitrary centers and offsets.

        ``offsets`` is one-dimensional or has one row per center. Rows are
        normalized over the requested support, matching the saved references.
        Negative fitted GH/sinc-like lobes are retained, not clipped.
        """
        centers = np.atleast_1d(centers).astype(float)
        offset = np.broadcast_to(
            np.asarray(offsets), (len(centers), np.shape(offsets)[-1])
        )
        t = (centers - self.origin) / self.scale
        fields = cheb.chebval(t, self.coefficients.T)
        sigma, wing_fraction, wing_sigma = fields[0], fields[-2], fields[-1]
        if np.any(sigma <= 0) or np.any(wing_sigma <= 0):
            raise ValueError("Fitted LSF has nonpositive width at requested center")
        return _field_profile(fields, offset, self.hermite_orders)

    def evaluate_with_jacobian(self, centers, offsets):
        """Return kernels and analytic derivatives of all fitted coefficients."""
        centers = np.atleast_1d(centers)
        t = (centers - self.origin) / self.scale
        fields = cheb.chebval(t, self.coefficients.T)
        value, local = _field_profile(fields, offsets, self.hermite_orders, True)
        design = cheb.chebvander(t, self.coefficients.shape[1] - 1)
        jacobian = np.einsum("ijm,ik->ijmk", local, design).reshape(value.size, -1)
        return value, jacobian

    def doppler_parameters(self):
        """Return a packed APOGEE/Doppler-format GH array in zero-based *apStar* pixels.

        Binsize is one output pixel; Xoffset=-origin. Use the original apStar
        pixel indices, not indices reset to zero after trimming a chip.
        Requires the IDL-compatible evaluator or the accompanying Doppler fix.
        The output fit already incorporates the effects of native integration
        and resampling; do not subsequently apply another pixel response.
        """

        power = np.zeros_like(self.coefficients)
        for i, row in enumerate(self.coefficients):
            polynomial = cheb.cheb2poly(row)
            power[i, : len(polynomial)] = polynomial
        power = power / self.scale ** np.arange(power.shape[1])[None, :]
        gh = np.zeros((self.horder + 1, power.shape[1]))
        gh[0] = power[0]
        for i, m in enumerate(self.hermite_orders):
            gh[m] = power[1 + i]
        params = dict(
            binsize=1.0,
            Xoffset=-self.origin,
            Horder=self.horder,
            Porder=np.full(self.horder + 1, power.shape[1] - 1),
            GHcoefs=gh,
            Wproftype=1,
            nWpar=2,
            WPorder=np.full(2, power.shape[1] - 1),
            Wcoefs=power[-2:],
        )
        return pack_parameters(params)


def fit_chip(
    chip,
    centers,
    offsets,
    reference,
    degree=3,
    horder=5,
    initial_sigma=1.2,
    initial_wing_sigma=5.0,
    max_nfev=150,
    fit_h2=False,
):
    """Fit smooth, conditioned GH coefficients to propagated reference lines.

    Fits all wavelengths and supplied subpixel phases together. Widths/fractions
    are constrained at a dense set of wavelengths using penalties. A failed
    optimizer raises RuntimeError; residual accuracy is reported, not assumed.
    """
    centers, reference = np.asarray(centers), np.asarray(reference)
    if horder < 3 or horder > 9 or degree < 0 or len(centers) <= degree:
        raise ValueError("Require 3<=horder<=9, degree>=0, enough training centers")
    origin = float((centers.min() + centers.max()) / 2)
    scale = max(float((centers.max() - centers.min()) / 2), 1.0)
    fields = 1 + 1 + (horder - 2) + 2 + int(fit_h2)
    coefs = np.zeros((fields, degree + 1))
    coefs[0, 0] = initial_sigma
    coefs[-2, 0] = 0.03
    coefs[-1, 0] = initial_wing_sigma
    model = ChipLSF(
        chip,
        origin,
        scale,
        coefs,
        horder,
        (float(centers.min()), float(centers.max())),
        {},
    )
    tcheck = np.linspace(-1, 1, 100)

    design_check = cheb.chebvander(tcheck, degree)
    cache = {}

    def evaluate(flat):
        if "x" in cache and np.array_equal(cache["x"], flat):
            return cache["residual"], cache["jacobian"]
        model.coefficients = flat.reshape(coefs.shape)
        fields = cheb.chebval(tcheck, model.coefficients.T)
        constraints = np.array(
            [fields[0] - 0.15, fields[-1] - fields[0], fields[-2], fields[-2] - 0.5]
        )
        active = constraints < 0
        active[3] = constraints[3] > 0
        penalties = np.where(active, constraints, 0.0) * 100
        constraint_jac = np.zeros((4, len(tcheck), fields.shape[0], degree + 1))
        constraint_jac[0, :, 0, :] = design_check
        constraint_jac[1, :, 0, :] = -design_check
        constraint_jac[1, :, -1, :] = design_check
        constraint_jac[2, :, -2, :] = design_check
        constraint_jac[3, :, -2, :] = design_check
        constraint_jac *= active[:, :, None, None] * 100
        if np.any(fields[0] <= 0) or np.any(fields[-1] <= 0):
            values = np.full(reference.size, 10.0)
            jac = np.zeros((reference.size, coefs.size))
        else:
            prediction, jac = model.evaluate_with_jacobian(centers, offsets)
            values = (prediction - reference).ravel()
        residual = np.r_[values, penalties.ravel()]
        jacobian = np.vstack([jac, constraint_jac.reshape(-1, coefs.size)])
        cache.update(x=flat.copy(), residual=residual, jacobian=jacobian)
        return residual, jacobian

    lower = np.full_like(coefs, -2.0)
    upper = np.full_like(coefs, 2.0)
    lower[0, 0], upper[0, 0] = 0.2, 5.0
    lower[-1], upper[-1] = -15.0, 15.0
    lower[-1, 0], upper[-1, 0] = 0.3, 20.0
    lower[-2], upper[-2] = -0.5, 0.5
    fitted = least_squares(
        lambda x: evaluate(x)[0],
        coefs.ravel(),
        jac=lambda x: evaluate(x)[1],
        bounds=(lower.ravel(), upper.ravel()),
        x_scale="jac",
        max_nfev=max_nfev,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    if not fitted.success:
        raise RuntimeError(f"Chip {chip} LSF fit failed: {fitted.message}")
    model.coefficients = fitted.x.reshape(coefs.shape)
    error = model.evaluate(centers, offsets) - reference
    model.diagnostics = dict(
        nfev=fitted.nfev,
        rms=float(np.sqrt(np.mean(error**2))),
        max_absolute=float(np.max(np.abs(error))),
        message=fitted.message,
    )
    return model


def fit_polynomial_profiles(
    profiles,
    degree=3,
    horder=8,
    initial_sigma=1.15,
    initial_wing_sigma=5.0,
    fit_h2=False,
):
    """Fit one shared polynomial solution directly to all supplied line profiles.

    Input chip profiles only define coverage and training/validation samples;
    their polynomial coefficients are not averaged or projected. Diagnostics
    measure agreement with those input references, not empirical calibration
    accuracy. The references may themselves inherit errors in native fits.
    """
    centers = np.concatenate([p["centers"] for p in profiles.values()])
    offsets = np.concatenate([p["offsets"] for p in profiles.values()])
    reference = np.concatenate([p["reference"] for p in profiles.values()])
    training = np.concatenate([p["training"] for p in profiles.values()])
    model = fit_chip(
        -1,
        centers[training],
        offsets[training],
        reference[training],
        degree,
        horder,
        initial_sigma,
        initial_wing_sigma,
        max_nfev=300,
        fit_h2=fit_h2,
    )
    diagnostics = {}
    for chip, p in profiles.items():
        p["model"] = model.evaluate(p["centers"], p["offsets"])
        keep = ~p["training"]
        error = p["model"][keep] - p["reference"][keep]
        diagnostics[str(chip)] = dict(
            validation_rms=float(np.sqrt(np.mean(error**2))),
            validation_max_absolute=float(np.max(np.abs(error))),
            validation_max_peak_relative=float(
                np.max(
                    np.max(np.abs(error), axis=1) / np.max(p["reference"][keep], axis=1)
                )
            ),
            validation_max_centroid_error_pixels=float(
                np.max(np.abs(np.sum(error * p["offsets"][keep], axis=1)))
            ),
        )
    return model, diagnostics


@dataclass
class ApStarLSF:
    """One global apStar LSF coefficient set, reference profiles and provenance."""

    wave: np.ndarray
    chips: list
    profiles: dict
    provenance: dict
    global_model: object = None

    domains: object = None

    def __post_init__(self):
        if self.global_model is None and len(self.chips) == 1:
            self.global_model = self.chips[0]
        if self.domains is None:
            self.domains = np.array([c.domain for c in self.chips])

    @property
    def coefficients(self):
        """Conditioned coefficients of the single global polynomial solution."""
        if self.global_model is None:
            raise ValueError(
                "This legacy file has independent chip solutions; refit jointly"
            )
        return self.global_model.coefficients

    @property
    def parameters(self):
        """One ordinary-polynomial parameter vector in the legacy APOGEE format."""
        if self.global_model is None:
            raise ValueError(
                "This legacy file has no single global polynomial solution"
            )
        return self.global_model.doppler_parameters()

    def __call__(self, pixels, halfwidth=None):
        """Return point-source kernels centered at zero-based apStar pixels.

        Each row samples integer output pixels around floor(center). Unsupported
        chip gaps and unvalidated edge regions return NaN rows. ``halfwidth``
        defaults to the support used to derive this LSF. No clipping is applied.
        """
        pixels = np.atleast_1d(np.asarray(pixels, float))
        if not np.isfinite(pixels).all():
            raise ValueError("Finite apStar pixel centers are required")
        halfwidth = (
            self.provenance["halfwidth"] if halfwidth is None else int(halfwidth)
        )
        if halfwidth < 1:
            raise ValueError("Positive profile halfwidth required")
        offsets = np.arange(-halfwidth, halfwidth + 1)[None, :] - (pixels % 1)[:, None]
        result = np.full(offsets.shape, np.nan)
        if self.global_model is None:
            for chip in self.chips:
                selected = (pixels >= chip.domain[0]) & (pixels <= chip.domain[1])
                if selected.any():
                    result[selected] = chip.evaluate(
                        pixels[selected], offsets[selected]
                    )
        else:
            supported = np.zeros(len(pixels), bool)
            for lo, hi in self.domains:
                supported |= (pixels >= lo) & (pixels <= hi)
            if supported.any():
                result[supported] = self.global_model.evaluate(
                    pixels[supported], offsets[supported]
                )
        return result

    @classmethod
    def read(cls, filename):
        """Load coefficients and saved references written by :meth:`save`."""
        chips, profiles = [], {}
        with fits.open(filename, checksum=True) as hdul:
            if hdul[0].header.get("LSFMODE") != "APSTAR-SINC":
                raise ValueError("Not a derived apStar LSF file")
            wave = hdul["WAVELENGTH"].data.copy()
            provenance = json.loads(hdul["PROVENANCE"].data["JSON"][0])
            version = hdul[0].header.get("LSFVER", 1)
            if version == 2:
                raise ValueError(
                    "The archived spline prototype is not supported; use polynomial output"
                )
            if version == 3:
                h = hdul["PARAMETERS"]
                p = unpack_parameters(h.data)
                origin = -float(p["Xoffset"])
                scale = h.header["SCALE"]
                degree = max(max(p["Porder"]), max(p["WPorder"]))
                orders = (
                    list(range(1, p["Horder"] + 1))
                    if np.any(p["GHcoefs"][2] != 0)
                    else [1] + list(range(3, p["Horder"] + 1))
                )
                power = np.vstack(
                    [p["GHcoefs"][0], *[p["GHcoefs"][m] for m in orders], *p["Wcoefs"]]
                )
                coefs = np.zeros((len(power), degree + 1))
                for i, row in enumerate(power):
                    c = cheb.poly2cheb(row * scale ** np.arange(len(row)))
                    coefs[i, : len(c)] = c
                domains = hdul["DOMAINS"].data.copy()
                model = ChipLSF(
                    -1,
                    origin,
                    scale,
                    coefs,
                    p["Horder"],
                    (float(domains[:, 0].min()), float(domains[:, 1].max())),
                    provenance.get("global_diagnostics", {}),
                )
                for i in range(hdul[0].header["NCHIPS"]):
                    table = hdul[f"PROFILE{i}"].data
                    profiles[i] = dict(
                        centers=table["CENTER"].copy(),
                        offsets=table["OFFSETS"].copy(),
                        reference=table["REFERENCE"].copy(),
                        model=table["MODEL"].copy(),
                        training=table["TRAIN"].copy(),
                    )
                return cls(wave, [], profiles, provenance, model, domains)
            for i in range(hdul[0].header["NCHIPS"]):
                h = hdul[f"COEF{i}"]
                chips.append(
                    ChipLSF(
                        i,
                        h.header["ORIGIN"],
                        h.header["SCALE"],
                        h.data.copy(),
                        h.header["HORDER"],
                        (h.header["QMIN"], h.header["QMAX"]),
                        provenance["diagnostics"][str(i)],
                    )
                )
                table = hdul[f"PROFILE{i}"].data
                profiles[i] = dict(
                    centers=table["CENTER"].copy(),
                    offsets=table["OFFSETS"].copy(),
                    reference=table["REFERENCE"].copy(),
                    model=table["MODEL"].copy(),
                    training=table["TRAIN"].copy(),
                )
        return cls(wave, chips, profiles, provenance)

    def save(self, filename, overwrite=False):
        """Write portable FITS coefficients, training/validation kernels, metadata.

        Original input FITS files are never changed. PARAMETERS contains one
        legacy-format polynomial vector; PROFILEn tables preserve the exact
        resampling reference and its fitted approximation at saved line centers.
        """
        primary = fits.PrimaryHDU()
        primary.header["LSFMODE"] = "APSTAR-SINC"
        primary.header["NCHIPS"] = len(self.profiles)
        primary.header["LSFVER"] = 3
        primary.header["LSFSTAT"] = self.provenance.get("status", "TRIAL")
        primary.header["PIXBASE"] = 0
        hdus = [primary, fits.ImageHDU(self.wave, name="WAVELENGTH")]
        model = self.global_model
        h = fits.ImageHDU(self.parameters, name="PARAMETERS")
        h.header["SCALE"] = model.scale
        h.header["DEGREE"] = model.coefficients.shape[1] - 1
        hdus += [h, fits.ImageHDU(self.domains, name="DOMAINS")]
        for chip_id, p in sorted(self.profiles.items()):
            count = p["reference"].shape[1]
            cols = [
                fits.Column(name="CENTER", format="D", array=p["centers"]),
                fits.Column(name="OFFSETS", format=f"{count}D", array=p["offsets"]),
                fits.Column(name="REFERENCE", format=f"{count}D", array=p["reference"]),
                fits.Column(name="MODEL", format=f"{count}D", array=p["model"]),
                fits.Column(name="TRAIN", format="L", array=p["training"]),
            ]
            hdus.append(fits.BinTableHDU.from_columns(cols, name=f"PROFILE{chip_id}"))
        metadata = json.dumps(self.provenance)
        hdus.append(
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(
                        name="JSON", format=f"{len(metadata)}A", array=[metadata]
                    )
                ],
                name="PROVENANCE",
            )
        )
        fits.HDUList(hdus).writeto(filename, overwrite=overwrite, checksum=True)


def derive_apstar_lsf(
    apstar,
    visitfiles,
    *,
    weights=None,
    coordinate_scales=None,
    degree=3,
    horder=8,
    halfwidth=48,
    centers_per_chip=24,
    fit_h2=False,
    native_calibration=None,
):
    """Derive output-pixel GH coefficients from an apStar and contributing visits.

    Parameters
    ----------
    apstar : path-like
        Supplies the exact logarithmic wavelength grid, SFILE/VREL entries,
        and per-chip sinc widths NRES. VREL is the observed-to-rest shift used
        in rv.visitcomb; VRAD (barycentric velocity) is not a substitute.
    visitfiles : sequence of path-like
        All contributing apVisit files, in any order; matched to SFILE headers.
    weights : array, optional
        Shape (nvisit, nwave), in the order of visitfiles. These are the frozen,
        effective weights of the selected continuum-matched coadd. Required
        for multiple visits: the routine never invents an S/N weighting scheme.
        Zero marks unavailable/unused output pixels. Nonlinear mask repair or
        continuum fitting is not recomputed by this instrumental calculation.
    coordinate_scales : array, optional
        Explicit array-pixel to calibration-pixel scale, (nvisit,3), overriding
        the supported binsize convention inferred by VisitLSF.read.
    degree, horder : int
        Output coefficient polynomial degree and highest Hermite order. A single
        shared solution is fitted directly across all chip reference profiles.
    native_calibration : NativePolynomialLSF, optional
        Override the three visit parameter vectors with a shared polynomial
        fitted to measured calibration lines. Native integration and visit
        resampling remain explicit. Currently limited to a single visit.
    fit_h2 : bool, optional
        Include H2 as a free polynomial. Default False fixes it to zero to
        reduce its strong degeneracy with width. Compare validation errors
        when assessing model flexibility.
    halfwidth : int
        Half-width of the saved sampled kernels, in output pixels.
    centers_per_chip : int
        Number of training wavelengths per chip, each at three subpixel phases.
        Additional wavelengths/phases are reserved for validation.

    Returns
    -------
    ApStarLSF
        One global polynomial coefficient model, exact propagated reference kernels,
        validation diagnostics and input/velocity/coordinate provenance. The
        fit is valid on the reported interior domains; chip-edge truncation is
        excluded. Inspect validation errors before scientific use.
    """
    started = time.perf_counter()
    timings = dict(profile_generation=0.0, nonlinear_fit=0.0)
    if halfwidth < 8 or centers_per_chip < degree + 3:
        raise ValueError("Need halfwidth>=8 and enough centers for training/validation")
    with fits.open(apstar) as hdul:
        header = hdul[0].header.copy()
        wh = hdul[1].header
        nwave = hdul[1].data.shape[-1]
        wave = 10 ** (
            wh["CRVAL1"] + (np.arange(nwave) + 1 - wh.get("CRPIX1", 1)) * wh["CDELT1"]
        )
    nvisit = int(header["NVISITS"])
    if len(visitfiles) != nvisit:
        raise ValueError("Supply every visit listed in this apStar")
    names = [header[f"SFILE{i+1}"].strip() for i in range(nvisit)]
    nres = np.array(header.get("NRES", "5.00 4.25 3.50").split(), float)
    visits = []
    seen = set()
    for i, path in enumerate(visitfiles):
        name = Path(path).name
        if name not in names or name in seen:
            raise ValueError(
                "Missing, duplicate, or mismatched apVisit filename: " + name
            )
        seen.add(name)
        index = names.index(name)
        scale = None if coordinate_scales is None else np.asarray(coordinate_scales)[i]
        visits.append(VisitLSF.read(path, header[f"VREL{index+1}"], nres, scale))
    if native_calibration is not None and nvisit != 1:
        raise ValueError(
            "A measured native calibration override currently requires one visit"
        )
    if native_calibration is not None:
        expected_fiber = native_calibration.diagnostics.get("fiber")
        if expected_fiber is not None:
            with fits.open(visitfiles[0]) as hdul:
                if hdul[0].header.get("FIBERID") != expected_fiber:
                    raise ValueError(
                        "Native calibration fiber does not match the visit"
                    )
    if weights is None:
        if nvisit != 1:
            raise ValueError("Multiple visits require the actual coaddition weights")
        weights = np.ones((1, nwave))
    weights = np.asarray(weights, float)
    if (
        weights.shape != (nvisit, nwave)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError(
            "Weights must be finite, nonnegative and shaped (nvisit,nwave)"
        )
    pixel = np.arange(nwave)
    profiles = {}
    sigmas, wing_sigmas = [], []
    for chip in range(3):
        # Conservative common interior avoids silently fitting truncated kernels.
        intervals = []
        for visit in visits:
            rest = visit.wave[chip] / (1 + visit.vrel / CSPEED)
            cover = pixel[(wave >= rest.min()) & (wave <= rest.max())]
            intervals.append((cover.min(), cover.max()))
        lo = max(x[0] for x in intervals) + halfwidth + 24
        hi = min(x[1] for x in intervals) - halfwidth - 24
        if hi <= lo:
            raise ValueError("No fully covered common interior for chip " + str(chip))
        base = np.linspace(lo + 1, hi - 1, centers_per_chip).astype(int)
        train = np.concatenate([base + phase for phase in [0.0, 0.25, 0.65]])
        validation = (base[:-1] + base[1:]) / 2 + 0.13
        centers = np.r_[train, validation]
        output = np.floor(centers).astype(int)[:, None] + np.arange(
            -halfwidth, halfwidth + 1
        )
        offsets = output - centers[:, None]
        profile_start = time.perf_counter()
        reference = []
        for center, q in zip(centers, output):
            response = np.array(
                [
                    (
                        v.response(wave, center, q, chip)
                        if native_calibration is None
                        else native_calibration.visit_response(v, wave, center, q, chip)
                    )
                    for v in visits
                ]
            )
            w = weights[:, q]
            total = w.sum(axis=0)
            if np.any(total <= 0):
                raise ValueError(
                    "No visit weight in a sampled line window; choose valid windows"
                )
            response = np.sum(response * w / total, axis=0)
            reference.append(response / response.sum())
        reference = np.array(reference)
        timings["profile_generation"] += time.perf_counter() - profile_start
        # Width initialization from the native GH sigma in output-pixel units.

        v = visits[0]
        center = float(np.median(train))
        wc = 10 ** np.interp(
            [center - 0.1, center, center + 0.1], pixel, np.log10(wave)
        )
        xc = (
            wave_to_pixel(wc * (1 + v.vrel / CSPEED), v.wave[chip])
            * v.coordinate_scale[chip]
        )
        slope = abs((xc[2] - xc[0]) / 0.2)
        pars = unpack_parameters(v.pars[chip])
        sig = (
            np.polynomial.polynomial.polyval(
                xc[1] + pars["Xoffset"], pars["GHcoefs"][0]
            )
            / slope
        )
        wsig = (
            np.polynomial.polynomial.polyval(xc[1] + pars["Xoffset"], pars["Wcoefs"][1])
            / slope
        )
        sigmas.append(sig)
        wing_sigmas.append(wsig)
        profiles[chip] = dict(
            centers=centers,
            offsets=offsets,
            reference=reference,
            training=np.arange(len(centers)) < len(train),
        )
    fit_start = time.perf_counter()
    model, diagnostics = fit_polynomial_profiles(
        profiles,
        degree,
        horder,
        initial_sigma=float(np.median(sigmas)),
        initial_wing_sigma=float(np.median(wing_sigmas)),
        fit_h2=fit_h2,
    )
    timings["nonlinear_fit"] = time.perf_counter() - fit_start
    provenance = dict(
        format_version=3,
        native_evaluator="APOGEE-IDL-compatible",
        native_calibration_source=(
            "visit-coefficients"
            if native_calibration is None
            else "measured-global-polynomial"
        ),
        native_calibration_diagnostics=(
            None if native_calibration is None else native_calibration.diagnostics
        ),
        apstar_sha256=hashlib.sha256(Path(apstar).read_bytes()).hexdigest(),
        visit_sha256=[
            hashlib.sha256(Path(v.filename).read_bytes()).hexdigest() for v in visits
        ],
        apstar=str(apstar),
        visits=[v.filename for v in visits],
        vrel_kms=[v.vrel for v in visits],
        nres=nres.tolist(),
        coordinate_scales=[v.coordinate_scale.tolist() for v in visits],
        native_binsizes=[v.pars[:, 0].tolist() for v in visits],
        degree=degree,
        horder=horder,
        fit_h2=fit_h2,
        halfwidth=halfwidth,
        diagnostics=diagnostics,
        global_diagnostics=model.diagnostics,
        status="EXPERIMENTAL",
    )
    domains = np.array(
        [(p["centers"].min(), p["centers"].max()) for p in profiles.values()]
    )
    result = ApStarLSF(wave, [], profiles, provenance, model, domains)
    timings["total"] = time.perf_counter() - started
    timings["other"] = (
        timings["total"] - timings["profile_generation"] - timings["nonlinear_fit"]
    )
    provenance["timings_seconds"] = timings
    provenance["coefficient_format"] = "apogee-gh-polynomial"
    return result
