"""Fit shared polynomial LSFs directly to extracted calibration emission lines.

All three native detectors are fitted in a common log-wavelength coordinate.
Native pixel edges, masks, uncertainties, OH doublets and independent line
centers/amplitudes/backgrounds are retained. This is an experimental calibration
comparison; it does not automatically replace existing survey calibrations.
"""

from dataclasses import dataclass
from pathlib import Path
import math, time, json
import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.interpolate import CubicSpline
from scipy.io import readsav
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import ndtr, eval_hermitenorm
from astropy.io import fits

LOGW0 = 4.179
DLOGW = 6e-6
BAD_BITS = sum(1 << i for i in [0, 1, 2, 3, 4, 5, 6, 7, 14, 15])


def edge_profile(lo, hi, center, fields, horder, derivatives=False):
    """Native-pixel probabilities and analytic derivatives in common log lambda.

    H1 and H2 are fixed to zero: independent line centers define the centroid,
    and sigma defines the core width. Remaining fields are sigma, H3..Horder,
    wing area and wing sigma. No extra integration or profile normalization is
    applied after integration over the actual native pixel edges.
    """

    def basis(width, order):
        l = (lo - center[:, None]) / width[:, None]
        r = (hi - center[:, None]) / width[:, None]
        pl = np.exp(-l * l / 2) / np.sqrt(2 * np.pi)
        pr = np.exp(-r * r / 2) / np.sqrt(2 * np.pi)
        b = [ndtr(r) - ndtr(l)]
        for m in range(1, order + 1):
            b.append(
                (pl * eval_hermitenorm(m - 1, l) - pr * eval_hermitenorm(m - 1, r))
                / np.sqrt(math.factorial(m))
            )
        if not derivatives:
            return np.stack(b, axis=-1), None, None
        dw = []
        dc = []
        for m in range(order + 1):
            left = pl * eval_hermitenorm(m, l)
            right = pr * eval_hermitenorm(m, r)
            dc.append((left - right) / (width[:, None] * np.sqrt(math.factorial(m))))
            dw.append(
                (l * left - r * right) / (width[:, None] * np.sqrt(math.factorial(m)))
            )
        return np.stack(b, axis=-1), np.stack(dw, axis=-1), np.stack(dc, axis=-1)

    sigma, fraction, wing_sigma = fields[0], fields[-2], fields[-1]
    b, dw, dc = basis(sigma, horder)
    w, ww, wc = basis(wing_sigma, 0)
    core = b[..., 0].copy()
    if derivatives:
        corew = dw[..., 0].copy()
        corec = dc[..., 0].copy()
    for j, m in enumerate(range(3, horder + 1)):
        core += fields[1 + j, :, None] * b[..., m]
        if derivatives:
            corew += fields[1 + j, :, None] * dw[..., m]
            corec += fields[1 + j, :, None] * dc[..., m]
    value = (1 - fraction[:, None]) * core + fraction[:, None] * w[..., 0]
    if not derivatives:
        return value
    local = [(1 - fraction[:, None]) * corew]
    local += [(1 - fraction[:, None]) * b[..., m] for m in range(3, horder + 1)]
    local += [w[..., 0] - core, fraction[:, None] * ww[..., 0]]
    return (
        value,
        np.stack(local, axis=-1),
        (1 - fraction[:, None]) * corec + fraction[:, None] * wc[..., 0],
    )


@dataclass
class LineWindows:
    lo: np.ndarray
    hi: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    valid: np.ndarray
    centers: np.ndarray
    separation: np.ndarray
    chip: np.ndarray
    ids: np.ndarray
    train: np.ndarray
    x: np.ndarray
    provenance: dict

    def select(self, mask):
        values = {
            k: getattr(self, k)[mask]
            for k in [
                "lo",
                "hi",
                "flux",
                "error",
                "valid",
                "centers",
                "separation",
                "chip",
                "ids",
                "train",
                "x",
            ]
        }
        return LineWindows(**values, provenance=self.provenance)


def load_calibration_windows(files, saved_fit, fiber=22, halfwidth=16, min_snr=15.0):
    """Select clean, isolated line windows using the original saved line list.

    Uses paired flux/error/mask/wavelength arrays in each ap1D file. The saved
    apLSF *_res image is deliberately not treated as an uncertainty image.
    Masks exclude hard bad bits, not persistence warning bits. Catalog doublets
    retain their known separation and the equal-component ratio used by APLSF.
    Unresolved catalog neighbors are rejected; other nearby line cores are masked. Every fourth
    accepted line within each chip is held out of the LSF coefficient fit.
    """
    saved = readsav(saved_fit, python_dict=True)["linestr"]
    row = 300 - int(fiber)
    all_values = []
    metadata = []
    for chip, path in enumerate(files):
        with fits.open(path) as h:
            f = np.array(h[1].data[row], float)
            e = np.array(h[2].data[row], float)
            mask = np.array(h[3].data[row], np.uint32)
            wave = np.array(h[4].data[row], float)
        logmap = CubicSpline(np.arange(len(wave)), np.log10(wave))
        records = saved[(saved.FIBER == row) & (saved.CHIP == chip)]
        positions = np.array([r.X for r in records], float)
        candidates = []
        for r, pos in zip(records, positions):
            if pos < halfwidth + 2 or pos > len(wave) - halfwidth - 3:
                continue
            # Avoid overlapping line windows and independently tabulated blends.
            other = np.abs(positions - pos)
            other[other < 1e-5] = np.inf
            if other.min() < 6:
                continue
            pixel = int(np.rint(pos)) + np.arange(-halfwidth, halfwidth + 1)
            valid = (
                np.isfinite(f[pixel])
                & np.isfinite(e[pixel])
                & (e[pixel] > 0)
                & (e[pixel] < 1e5)
                & ((mask[pixel] & BAD_BITS) == 0)
            )
            for neighbor in positions[other < halfwidth + 5]:
                valid &= np.abs(pixel - neighbor) > 4.0
            if valid.sum() < len(pixel) * 0.5 or not np.all(
                valid[halfwidth - 2 : halfwidth + 3]
            ):
                continue
            background = np.median(f[pixel][valid])
            peak = np.max(f[pixel[halfwidth - 3 : halfwidth + 4]]) - background
            if peak / max(float(np.median(e[pixel][valid])), 1.0) < min_snr:
                continue
            edges1 = (logmap(pixel - 0.5) - LOGW0) / DLOGW
            edges2 = (logmap(pixel + 0.5) - LOGW0) / DLOGW
            center = float((logmap(pos) - LOGW0) / DLOGW)
            sep = float(r.MODEL_DBL_WSEP) if r.MODEL_DOUBLET else 0.0
            separation = (
                np.log10(r.MODEL_WAVE + sep / 2) - np.log10(r.MODEL_WAVE - sep / 2)
            ) / DLOGW
            candidates.append(
                dict(
                    lo=np.minimum(edges1, edges2),
                    hi=np.maximum(edges1, edges2),
                    flux=f[pixel],
                    error=e[pixel],
                    valid=valid,
                    centers=center,
                    separation=separation,
                    chip=chip,
                    ids=int(r.MODEL_ID),
                    x=pixel - pos,
                )
            )
        candidates.sort(key=lambda r: r["centers"])
        for i, r in enumerate(candidates):
            r["train"] = i % 4 != 2
        # Prevent leakage of shared wing pixels from held-out lines into training.
        held = (
            np.concatenate([r["lo"][r["valid"]] for r in candidates if not r["train"]])
            if any(not r["train"] for r in candidates)
            else np.array([])
        )
        for r in candidates:
            if r["train"]:
                r["valid"] &= ~np.isin(np.round(r["lo"], 9), np.round(held, 9))
        candidates = [r for r in candidates if r["valid"].sum() >= 8]
        all_values.extend(candidates)
        metadata.append(
            dict(
                path=str(path),
                chip=chip,
                selected_lines=len(candidates),
                held_out=sum(not r["train"] for r in candidates),
            )
        )
    if any(r["selected_lines"] < 6 for r in metadata):
        raise ValueError("Too few isolated, unflagged lines per chip: " + str(metadata))
    names = [
        "lo",
        "hi",
        "flux",
        "error",
        "valid",
        "centers",
        "separation",
        "chip",
        "ids",
        "train",
        "x",
    ]
    values = {k: np.array([r[k] for r in all_values]) for k in names}
    return LineWindows(
        **values,
        provenance=dict(
            fiber=fiber,
            halfwidth=halfwidth,
            minimum_snr=min_snr,
            files=metadata,
            saved_fit=str(saved_fit),
        )
    )


@dataclass
class NativePolynomialLSF:
    coefficients: np.ndarray
    origin: float
    scale: float
    horder: int
    diagnostics: dict

    def save(self, filename, overwrite=False):
        """Save the native continuous-LSF polynomial and provenance, without pickle.

        This file describes the instrumental response before native pixel
        integration and apStar resampling. Use visit_response/derive_apstar_lsf
        to obtain an effective LSF for a resampled stellar spectrum.
        """
        with open(filename, "wb" if overwrite else "xb") as stream:
            np.savez(
                stream,
                coefficients=self.coefficients,
                origin=self.origin,
                scale=self.scale,
                horder=self.horder,
                format_version=1,
                logw0=LOGW0,
                dlogw=DLOGW,
                diagnostics=np.array(json.dumps(self.diagnostics)),
            )

    @classmethod
    def read(cls, filename):
        """Load a native polynomial saved by save, rejecting unknown conventions."""
        with np.load(filename, allow_pickle=False) as data:
            if (
                int(data["format_version"]) != 1
                or float(data["logw0"]) != LOGW0
                or float(data["dlogw"]) != DLOGW
            ):
                raise ValueError("Unknown native polynomial wavelength convention")
            return cls(
                data["coefficients"].copy(),
                float(data["origin"]),
                float(data["scale"]),
                int(data["horder"]),
                json.loads(str(data["diagnostics"])),
            )

    def fields(self, centers):
        return cheb.chebval((centers - self.origin) / self.scale, self.coefficients.T)

    def visit_response(self, visit, star_wave, center, output_pixels, chip):
        """Propagate this homogeneous measured-line fit through visit resampling.

        Polynomial fields are evaluated in observed-frame log wavelength, so
        the stellar Doppler shift changes their argument rather than being
        absorbed into an arbitrary native-pixel polynomial. Native integration
        and damped-sinc sampling are then applied explicitly.
        """
        from .apstar_lsf import sinc_weights, CSPEED

        logwave = np.log10(star_wave)
        observed_log = float(
            np.interp(center, np.arange(len(star_wave)), logwave)
        ) + np.log10(1 + visit.vrel / CSPEED)
        q0 = (observed_log - LOGW0) / DLOGW
        pixels = visit.pixel_at_wave(
            star_wave[output_pixels] * (1 + visit.vrel / CSPEED), chip
        )
        if not np.isfinite(pixels).all():
            raise ValueError("Response extends outside visit chip")
        index, weights = sinc_weights(pixels, visit.nres[chip], visit.wave.shape[1])
        if index.min() == 0 or index.max() == visit.wave.shape[1] - 1:
            raise ValueError("Truncated native interpolation stencil")
        unique, inverse = np.unique(index, return_inverse=True)
        cache = getattr(visit, "_calibration_forward", None)
        if cache is None:
            cache = {}
            visit._calibration_forward = cache
        if chip not in cache:
            cache[chip] = CubicSpline(
                np.arange(visit.wave.shape[1]), np.log10(visit.wave[chip])
            )
        forward = cache[chip]
        half = visit.pars[chip, 0] / visit.coordinate_scale[chip] / 2
        a = (forward(unique - half) - LOGW0) / DLOGW
        b = (forward(unique + half) - LOGW0) / DLOGW
        value = edge_profile(
            np.minimum(a, b)[None, :],
            np.maximum(a, b)[None, :],
            np.array([q0]),
            self.fields(np.array([q0])),
            self.horder,
        )[0]
        response = np.sum(weights * value[inverse.reshape(index.shape)], axis=1)
        return response / response.sum()

    def profile(self, lines, shift, derivatives=False):
        center = lines.centers + shift
        fields = self.fields(center)
        first = center - lines.separation / 2
        second = center + lines.separation / 2
        if not derivatives:
            return 0.5 * (
                edge_profile(lines.lo, lines.hi, first, fields, self.horder)
                + edge_profile(lines.lo, lines.hi, second, fields, self.horder)
            )
        a, da, ca = edge_profile(lines.lo, lines.hi, first, fields, self.horder, True)
        b, db, cb = edge_profile(lines.lo, lines.hi, second, fields, self.horder, True)
        value = (a + b) / 2
        local = (da + db) / 2
        field_derivative = (
            cheb.chebval(
                (center - self.origin) / self.scale,
                cheb.chebder(self.coefficients.T),
                tensor=True,
            )
            / self.scale
        )
        # For degree zero, chebder returns a zero constant as desired.
        dc = (ca + cb) / 2 + np.sum(local * field_derivative.T[:, None, :], axis=-1)
        design = cheb.chebvander(
            (center - self.origin) / self.scale, self.coefficients.shape[1] - 1
        )
        shared = np.einsum("ijk,il->ijkl", local, design).reshape(
            value.shape[0], value.shape[1], -1
        )
        return value, shared, dc


def project_lines(profile, lines, derivatives=None):
    """Solve line amplitude and linear background exactly, optionally differentiate.

    Exact variable-projection derivatives include the fitted linear nuisance
    parameters. Thus the nonlinear optimizer only handles LSF fields and line
    centers; amplitudes/backgrounds do not increase its parameter dimension.
    """
    inv = np.where(lines.valid, 1 / lines.error, 0.0)
    x = lines.x / max(float(np.max(np.abs(lines.x))), 1.0)
    design = np.stack([profile, np.ones_like(profile), x], axis=-1) * inv[..., None]
    y = lines.flux * inv
    normal = np.einsum("ijk,ijl->ikl", design, design)
    inverse = np.linalg.inv(normal)
    coefficients = np.einsum("ijk,ik->ij", inverse, np.einsum("ijk,ij->ik", design, y))
    prediction = np.einsum("ijk,ik->ij", design, coefficients)
    residual = prediction - y
    if derivatives is None:
        return (
            residual,
            coefficients,
            (
                coefficients[:, 0, None] * profile
                + coefficients[:, 1, None]
                + coefficients[:, 2, None] * x
            ),
        )
    dp = derivatives * inv[..., None]
    u = coefficients[:, 0, None, None] * dp
    projected = u - np.einsum(
        "ijk,ikl,ilm->ijm", design, inverse, np.einsum("ijk,ijm->ikm", design, u)
    )
    left = np.einsum("ijk,ik->ij", design, inverse[:, :, 0])
    projected -= left[..., None] * np.einsum("ijk,ij->ik", dp, residual)[:, None, :]
    return residual, projected, coefficients


def fit_native_polynomial(lines, degree=2, horder=5, max_nfev=160):
    """Fit measured native line pixels with one shared polynomial LSF.

    Uses only lines marked train. Validation lines are assessed separately by
    fitting their own center/amplitude/background while keeping the LSF fixed.
    Native uncertainties are not rescaled to force chi-square to unity.
    """
    start = time.perf_counter()
    data = lines.select(lines.train)
    origin = float((lines.centers.min() + lines.centers.max()) / 2)
    scale = float(np.ptp(lines.centers) / 2)
    fields = horder + 1  # sigma, H3..Horder, wing fraction, wing sigma
    coefs = np.zeros((fields, degree + 1))
    coefs[0, 0] = 1.1
    coefs[-2, 0] = 0.04
    coefs[-1, 0] = 4.0
    model = NativePolynomialLSF(coefs, origin, scale, horder, {})
    nshared = coefs.size
    nline = len(data.centers)
    qcheck = np.linspace(lines.centers.min(), lines.centers.max(), 80)
    vcheck = cheb.chebvander((qcheck - origin) / scale, degree)
    cache = {}

    def evaluate(parameters):
        if "x" in cache and np.array_equal(cache["x"], parameters):
            return cache["r"], cache["j"]
        model.coefficients = parameters[:nshared].reshape(coefs.shape)
        shift = parameters[nshared:]
        field = model.fields(qcheck)
        constraint = np.array(
            [field[0] - 0.2, field[-1] - field[0], field[-2], field[-2] - 0.4]
        )
        active = constraint < 0
        active[3] = constraint[3] > 0
        penalty = np.where(active, constraint, 0.0) * 300
        pj = np.zeros((4, len(qcheck), fields, degree + 1))
        pj[0, :, 0] = vcheck
        pj[1, :, 0] = -vcheck
        pj[1, :, -1] = vcheck
        pj[2, :, -2] = vcheck
        pj[3, :, -2] = vcheck
        pj *= active[:, :, None, None] * 300
        if np.any(field[0] <= 0) or np.any(field[-1] <= 0):
            residual = np.full(data.flux.size, 1e4)
            jac = np.zeros((data.flux.size, nshared + nline))
        else:
            profile, shared, center = model.profile(data, shift, True)
            derivative = np.concatenate([shared, center[..., None]], axis=-1)
            residual, derivative, linear = project_lines(profile, data, derivative)
            jac = np.zeros((data.flux.shape[0], data.flux.shape[1], nshared + nline))
            jac[:, :, :nshared] = derivative[:, :, :nshared]
            for i in range(nline):
                jac[i, :, nshared + i] = derivative[i, :, -1]
            jac = jac.reshape(-1, nshared + nline)
            residual = residual.ravel()
        r = np.r_[residual, penalty.ravel()]
        j = np.vstack(
            [
                jac,
                np.column_stack(
                    [pj.reshape(-1, nshared), np.zeros((penalty.size, nline))]
                ),
            ]
        )
        cache.update(x=parameters.copy(), r=r, j=j)
        return r, j

    low = np.full_like(coefs, -0.7)
    high = np.full_like(coefs, 0.7)
    low[0], high[0] = -2, 2
    low[0, 0], high[0, 0] = 0.3, 4.0
    low[-1], high[-1] = -10, 10
    low[-1, 0], high[-1, 0] = 0.5, 15.0
    low[-2], high[-2] = -0.4, 0.4
    initial = np.r_[coefs.ravel(), np.zeros(nline)]
    result = least_squares(
        lambda x: evaluate(x)[0],
        initial,
        jac=lambda x: evaluate(x)[1],
        bounds=(
            np.r_[low.ravel(), np.full(nline, -0.8)],
            np.r_[high.ravel(), np.full(nline, 0.8)],
        ),
        x_scale="jac",
        max_nfev=max_nfev,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-7,
    )
    if not result.success:
        raise RuntimeError(result.message)
    model.coefficients = result.x[:nshared].reshape(coefs.shape)
    model.diagnostics = dict(
        nfev=result.nfev,
        seconds=time.perf_counter() - start,
        training_lines=nline,
        degree=degree,
        horder=horder,
        fiber=lines.provenance.get("fiber"),
        message=result.message,
        condition="centers free; H1=H2=0",
    )
    return model


def evaluate_held_out(model, lines):
    """Evaluate unseen line profiles, fitting only line-specific nuisance values."""
    data = lines.select(~lines.train)
    results = []
    predictions = []
    for i in range(len(data.centers)):
        line = data.select(np.arange(len(data.centers)) == i)

        def objective(shift):
            r, _, _ = project_lines(model.profile(line, np.array([shift])), line)
            return float(np.sum(r * r))

        optimum = minimize_scalar(
            objective, bounds=(-0.8, 0.8), method="bounded", options={"xatol": 1e-7}
        )
        profile = model.profile(line, np.array([optimum.x]))
        residual, linear, pred = project_lines(profile, line)
        valid = line.valid[0]
        n = int(valid.sum())
        peak = max(float(linear[0, 0] * np.max(profile[0])), 1.0)
        error = (pred[0] - line.flux[0])[valid]
        results.append(
            dict(
                chip=int(line.chip[0]),
                id=int(line.ids[0]),
                q=float(line.centers[0]),
                reduced_chisq=float(np.sum(residual * residual) / max(n - 4, 1)),
                rms_peak_relative=float(np.sqrt(np.mean(error**2)) / peak),
                shift_pixels=float(optimum.x),
                pixels=n,
            )
        )
        predictions.append(pred[0])
    return results, np.array(predictions), data
