"""Joint native-pixel Gauss--Hermite calibration across three detectors.

The profile variable, widths and integration bins are native pixels. A chip's
measured wavelength-solution offset joins its pixel centers to the common
coordinate X = pixel + chip_offset, as in APLSF fitmethod=3. Exporting each
chip's standard packed vector only changes Xoffset; no wavelength projection
or refit is involved.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from astropy.io import fits

from .calibration_lsf import (
    NativePolynomialLSF,
    LineWindows,
    LOGW0,
    DLOGW,
    edge_profile,
    fit_native_polynomial,
)
from .lsf_parameters import pack_parameters
from .apstar_lsf import evaluate_apogee_gh


@dataclass
class NativePixelLSF(NativePolynomialLSF):
    """One shared pixel-based GH solution for one fiber.

    Parameters
    ----------
    coefficients : ndarray
        Chebyshev coefficients of sigma, H3..Horder, wing area and wing sigma.
        Widths are in native pixels; H1 and H2 are fixed at zero.
    origin, scale : float
        Conditioning of common native pixel X before evaluating polynomials.
    chip_offsets : ndarray, shape (3,)
        Measured offsets defining X = native pixel + offset for chips a/b/c.
        They affect polynomial variation only, not the local profile offset.
    npix : int
        Number of pixels in the native calibration detector arrays.
    """

    chip_offsets: np.ndarray = field(default_factory=lambda: np.zeros(3))
    npix: int = 2048

    def parameters(self, chip, pixel_scale=1.0, binsize=1.0):
        """Export exact standard APOGEE GH coefficients for a detector chip.

        pixel_scale is target coordinate pixels per original native pixel.
        Sigma and wing sigma scale with this factor; the polynomial argument
        and Xoffset transform algebraically. binsize is in target pixels.
        """
        if not 0 <= int(chip) < 3 or pixel_scale <= 0 or binsize <= 0:
            raise ValueError("Require chip 0..2 and positive pixel_scale/binsize")
        power = np.zeros_like(self.coefficients)
        for i, row in enumerate(self.coefficients):
            converted = cheb.cheb2poly(row)
            power[i, : len(converted)] = converted
        power /= (self.scale * pixel_scale) ** np.arange(power.shape[1])[None, :]
        power[0] *= pixel_scale
        power[-1] *= pixel_scale
        gh = np.zeros((self.horder + 1, power.shape[1]))
        gh[0] = power[0]
        for m in range(3, self.horder + 1):
            gh[m] = power[m - 2]
        order = lambda rows: np.array(
            [np.flatnonzero(r)[-1] if np.any(r) else 0 for r in rows]
        )
        return pack_parameters(
            dict(
                binsize=float(binsize),
                Xoffset=(float(self.chip_offsets[chip]) - self.origin) * pixel_scale,
                Horder=self.horder,
                Porder=order(gh),
                GHcoefs=gh,
                Wproftype=1,
                nWpar=2,
                WPorder=order(power[-2:]),
                Wcoefs=power[-2:],
            )
        )

    def native_profiles(self, centers, offsets, chip):
        """Evaluate the unchanged standard pixel-coordinate GH profile."""
        return evaluate_apogee_gh(offsets, centers, self.parameters(chip))

    def profile(self, lines, shift, derivatives=False):
        """Integrate measured pixel bins, including known pixel-space doublets.

        Each component evaluates polynomial fields at its own center, matching
        separate calls to the standard GH evaluator. Fitted line centers are
        nuisance parameters in native pixels, independent of the LSF widths.
        """
        centers = lines.centers + shift
        pieces = []
        shared = []
        dc = []
        degree = self.coefficients.shape[1] - 1
        for component in (-0.5, 0.5):
            center = centers + component * lines.separation
            fields = self.fields(center)
            if not derivatives:
                pieces.append(
                    edge_profile(lines.lo, lines.hi, center, fields, self.horder)
                )
                continue
            value, local, center_derivative = edge_profile(
                lines.lo, lines.hi, center, fields, self.horder, True
            )
            derivative = (
                cheb.chebval(
                    (center - self.origin) / self.scale,
                    cheb.chebder(self.coefficients.T),
                )
                / self.scale
            )
            design = cheb.chebvander((center - self.origin) / self.scale, degree)
            pieces.append(value)
            shared.append(
                np.einsum("ijk,il->ijkl", local, design).reshape(
                    value.shape[0], value.shape[1], -1
                )
            )
            dc.append(
                center_derivative + np.sum(local * derivative.T[:, None, :], axis=-1)
            )
        value = 0.5 * (pieces[0] + pieces[1])
        if not derivatives:
            return value
        return value, 0.5 * (shared[0] + shared[1]), 0.5 * (dc[0] + dc[1])

    def visit_response(self, visit, star_wave, center, output_pixels, chip):
        """Propagate standard native pixel GH coefficients through visit sinc.

        The existing visit convention supplies the native integration width.
        Dither-combined calibration coordinates are handled by exact parameter
        scaling, not by fitting profiles in a wavelength coordinate.
        """
        from copy import copy

        scale = visit.coordinate_scale[chip] * visit.wave.shape[1] / self.npix
        response_visit = copy(visit)
        # response() uses only this chip's vector; preserve the other rows.
        packed = self.parameters(chip, pixel_scale=scale, binsize=visit.pars[chip, 0])
        response_visit.pars = [np.asarray(p) for p in visit.pars]
        response_visit.pars[chip] = packed
        return response_visit.response(star_wave, center, output_pixels, chip)

    def save(self, filename, overwrite=False):
        """Save an explicitly identified native-pixel model without pickle."""
        with open(filename, "wb" if overwrite else "xb") as stream:
            np.savez(
                stream,
                format_version=2,
                coordinate="native_pixel",
                coefficients=self.coefficients,
                origin=self.origin,
                scale=self.scale,
                horder=self.horder,
                chip_offsets=self.chip_offsets,
                npix=self.npix,
                diagnostics=np.array(json.dumps(self.diagnostics)),
            )

    @classmethod
    def read(cls, filename):
        """Load pixel-native NPZ, rejecting wavelength-native files."""
        with np.load(filename, allow_pickle=False) as data:
            if (
                int(data["format_version"]) != 2
                or str(data["coordinate"]) != "native_pixel"
            ):
                raise ValueError("Not a native pixel LSF model")
            return cls(
                data["coefficients"].copy(),
                float(data["origin"]),
                float(data["scale"]),
                int(data["horder"]),
                json.loads(str(data["diagnostics"])),
                data["chip_offsets"].copy(),
                int(data["npix"]),
            )


def pixel_line_windows(lines, files, fiber, chip_offsets=None):
    """Put identical selected measurements and doublets on native pixel axes.

    Wavelength is used only for identifying catalog lines and their doublet
    separation. The fitted GH profile and coefficient variations use pixels.
    Measured chip offsets come from the first wavelength-solution coefficient,
    matching the historical APLSF global-pixel fit. Explicit offsets can be
    supplied for alternate input formats; offsets are never guessed from gaps.
    """
    wave = []
    offsets = []
    for chip, path in enumerate(files):
        with fits.open(path) as h:
            w = np.array(h[4].data[300 - int(fiber)], float)
            wave.append(w)
            if chip_offsets is None:
                if len(h) < 6 or h[5].data is None:
                    raise ValueError(
                        "Native pixel global fit requires measured chip offsets or ap1D wavelength coefficients"
                    )
                co = np.asarray(h[5].data)
                if co.shape == (14, 300):
                    offsets.append(float(co[0, 300 - int(fiber)]))
                elif co.shape == (300, 14):
                    offsets.append(float(co[300 - int(fiber), 0]))
                else:
                    raise ValueError("Expected 14 wavelength coefficients per fiber")
    if chip_offsets is not None:
        offsets = list(np.asarray(chip_offsets, float))
    offsets = np.asarray(offsets, float)
    if offsets.shape != (3,) or not np.isfinite(offsets).all():
        raise ValueError("Require three finite native chip offsets")
    n = len(wave[0])
    if any(len(w) != n for w in wave):
        raise ValueError("Native chips must have equal pixel counts")
    centers = np.empty_like(lines.centers)
    sep = np.empty_like(lines.separation)
    for chip, w in enumerate(wave):
        forward = CubicSpline(np.arange(n), (np.log10(w) - LOGW0) / DLOGW)
        for i in np.flatnonzero(lines.chip == chip):
            inv = lambda q: brentq(
                lambda p: float(forward(p) - q), 0, n - 1, xtol=1e-11
            )
            p = inv(lines.centers[i])
            centers[i] = p + offsets[chip]
            sep[i] = abs(
                inv(lines.centers[i] + lines.separation[i] / 2)
                - inv(lines.centers[i] - lines.separation[i] / 2)
            )
    # x is the original observed integer sample minus its catalog center.
    # Recovering centers above inverts the same forward mapping to high precision.
    lo = centers[:, None] + lines.x - 0.5
    hi = lo + 1
    provenance = dict(
        lines.provenance,
        coordinate="native_pixel",
        chip_offsets=offsets.tolist(),
        npix=n,
        files=[
            dict(
                path=str(path), q_min=float(offsets[c]), q_max=float(offsets[c] + n - 1)
            )
            for c, path in enumerate(files)
        ],
    )
    result = LineWindows(
        lo,
        hi,
        lines.flux.copy(),
        lines.error.copy(),
        lines.valid.copy(),
        centers,
        sep,
        lines.chip.copy(),
        lines.ids.copy(),
        lines.train.copy(),
        lines.x.copy(),
        provenance,
    )
    return result, offsets, n


def fit_native_pixel(
    lines, files, fiber, degree=3, horder=5, chip_offsets=None, max_nfev=160
):
    """Fit one standard pixel-based GH model to all three detectors jointly."""
    pixel, offsets, npix = pixel_line_windows(lines, files, fiber, chip_offsets)
    factory = lambda *args: NativePixelLSF(*args, chip_offsets=offsets, npix=npix)
    model = fit_native_polynomial(
        pixel, degree=degree, horder=horder, max_nfev=max_nfev, model_factory=factory
    )
    model.diagnostics.update(
        coordinate="native_pixel", chip_offsets=offsets.tolist(), npix=npix
    )
    return model, pixel
