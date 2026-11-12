"""Constrain an estimated native LSF's sigma polynomial using its own peaks."""

from copy import deepcopy
import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.optimize import least_squares
from .calibration_lsf import project_lines, evaluate_held_out
from .pixel_lsf import NativePixelLSF


def validation_statistics(model, lines):
    """Assess held-out pixels after fitting only line-specific nuisance values."""
    rows, _, _ = evaluate_held_out(model, lines)
    if not rows:
        return dict(
            lines=0, pooled_reduced_chisq=None, chi_square=0.0, degrees_of_freedom=0
        )
    dof = np.array([max(r["pixels"] - 4, 1) for r in rows])
    return dict(
        lines=len(rows),
        degrees_of_freedom=int(dof.sum()),
        chi_square=float(sum(r["reduced_chisq"] * d for r, d in zip(rows, dof))),
        pooled_reduced_chisq=float(
            sum(r["reduced_chisq"] * d for r, d in zip(rows, dof)) / dof.sum()
        ),
        median_rms_peak_relative=float(
            np.median([r["rms_peak_relative"] for r in rows])
        ),
        per_line=rows,
    )


def refine_sigma(
    model,
    lines,
    *,
    degree=None,
    prior_fraction=0.10,
    max_fractional_change=0.30,
    validation_tolerance=0.02,
    validation_noise_sigma=1.0,
    max_nfev=200
):
    """Refit only sigma's native-pixel polynomial in an estimated LSF.

    Hermite terms, wing fraction and wing width are kept at their adopted
    neighbor-based values. Line centers are free; amplitudes and constant/linear
    backgrounds are eliminated by exact variable projection. Only training
    lines change the polynomial. Held-out lines gate adoption of the candidate.

    The correction has the existing sigma degree by default, increasing no LSF
    polynomial order. A weak Gaussian prior toward the neighbor estimate and a
    full-detector fractional-change limit prevent poorly constrained edge swings.
    A candidate must converge, remain physical across all three detectors and
    not significantly worsen held-out chi-square. The allowance is the larger
    of validation_tolerance times the initial chi-square and
    validation_noise_sigma*sqrt(2*dof), an approximate noise-fluctuation guard.
    This is a conservative adoption heuristic, not a formal significance test.
    Rejection
    returns an unchanged model copy with explicit diagnostics. No input model
    or line arrays are modified.
    """
    if not isinstance(model, NativePixelLSF):
        raise TypeError("Native pixel LSF required")
    ncoef = model.coefficients.shape[1]
    degree = ncoef - 1 if degree is None else int(degree)
    if (
        not 0 <= degree < ncoef
        or prior_fraction <= 0
        or not 0 < max_fractional_change < 1
        or validation_tolerance < 0
        or validation_noise_sigma < 0
    ):
        raise ValueError("Invalid sigma refinement settings")
    result = deepcopy(model)
    before = validation_statistics(model, lines)
    info = dict(
        status="NOT_RUN",
        accepted=False,
        degree=degree,
        training_lines=int(np.sum(lines.train)),
        validation_lines=int(np.sum(~lines.train)),
        before=before,
        prior_fraction=float(prior_fraction),
        max_fractional_change=float(max_fractional_change),
        validation_tolerance=float(validation_tolerance),
        validation_noise_sigma=float(validation_noise_sigma),
        initial_sigma_coefficients=model.coefficients[0].tolist(),
        basis_origin=float(model.origin),
        basis_scale=float(model.scale),
        fitted_fields=["sigma"],
        fixed_fields=["H1", "H2", "H3..Horder", "wing_fraction", "wing_sigma"],
    )

    def finish(status, accepted=False):
        info.update(status=status, accepted=bool(accepted))
        pixels = np.linspace(0, model.npix - 1, 9)
        info["sigma_curves"] = [
            dict(
                chip=c,
                pixel=pixels.tolist(),
                before=model.fields(pixels + offset)[0].tolist(),
                adopted=result.fields(pixels + offset)[0].tolist(),
            )
            for c, offset in enumerate(model.chip_offsets)
        ]
        result.diagnostics["sigma_refinement"] = info
        result.diagnostics["sigma_data_constrained"] = bool(accepted)
        return result, info

    train = lines.select(lines.train)
    k = degree + 1
    if (
        len(train.centers) < k + 2
        or before["lines"] < 3
        or len(np.unique(lines.chip[~lines.train])) < 2
    ):
        return finish("INSUFFICIENT_PEAKS")
    design = cheb.chebvander((train.centers - model.origin) / model.scale, degree)
    if np.linalg.matrix_rank(design) < k or np.linalg.cond(design) > 1e8:
        return finish("INSUFFICIENT_PIXEL_COVERAGE")
    xcheck = np.concatenate(
        [np.linspace(o, o + model.npix - 1, 128) for o in model.chip_offsets]
    )
    bcheck = cheb.chebvander((xcheck - model.origin) / model.scale, degree)
    baseline = model.fields(xcheck)[0]
    wing = model.fields(xcheck)[-1]
    if np.any(baseline <= 0):
        return finish("INVALID_INITIAL_SIGMA")
    candidate = deepcopy(model)
    nline = len(train.centers)
    prior_scale = prior_fraction * baseline / np.sqrt(k / len(xcheck))
    cache = {}

    def evaluate(parameters):
        if "x" in cache and np.array_equal(cache["x"], parameters):
            return cache["r"], cache["j"]
        candidate.coefficients = model.coefficients.copy()
        candidate.coefficients[0, :k] += parameters[:k]
        sigma = baseline + bcheck @ parameters[:k]
        prior = (sigma - baseline) / prior_scale
        lower = baseline * (1 - max_fractional_change)
        upper = np.minimum(baseline * (1 + max_fractional_change), wing)
        low_bad = sigma < lower
        high_bad = sigma > upper
        penalty = (
            np.r_[np.minimum(sigma - lower, 0), np.maximum(sigma - upper, 0)] * 1000
        )
        pj = np.vstack([bcheck * low_bad[:, None], bcheck * high_bad[:, None]]) * 1000
        if np.any(sigma <= 0):
            residual = np.full(train.flux.size, 1e4)
            jac = np.zeros((train.flux.size, k + nline))
        else:
            profile, shared, center = candidate.profile(train, parameters[k:], True)
            # The first native field is sigma, whose coefficients occupy row 0.
            derivative = np.concatenate([shared[:, :, :k], center[:, :, None]], axis=-1)
            residual, dp, _ = project_lines(profile, train, derivative)
            jac = np.zeros((*train.flux.shape, k + nline))
            jac[:, :, :k] = dp[:, :, :k]
            for i in range(nline):
                jac[i, :, k + i] = dp[i, :, -1]
            jac = jac.reshape(-1, k + nline)
            residual = residual.ravel()
        r = np.r_[residual, prior, penalty]
        j = np.vstack(
            [
                jac,
                np.c_[bcheck / prior_scale[:, None], np.zeros((len(xcheck), nline))],
                np.c_[pj, np.zeros((len(penalty), nline))],
            ]
        )
        cache.update(x=parameters.copy(), r=r, j=j)
        return r, j

    try:
        opt = least_squares(
            lambda p: evaluate(p)[0],
            np.zeros(k + nline),
            jac=lambda p: evaluate(p)[1],
            bounds=(
                np.r_[np.full(k, -2.0), np.full(nline, -0.8)],
                np.r_[np.full(k, 2.0), np.full(nline, 0.8)],
            ),
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        info["reason"] = str(exc)
        return finish("FIT_FAILED")
    info.update(nfev=int(opt.nfev), optimizer_message=str(opt.message))
    if not opt.success:
        return finish("FIT_FAILED")
    candidate.coefficients = model.coefficients.copy()
    candidate.coefficients[0, :k] += opt.x[:k]
    refined = candidate.fields(xcheck)[0]
    ratio = refined / baseline
    info.update(
        sigma_ratio_min=float(ratio.min()),
        sigma_ratio_max=float(ratio.max()),
        sigma_ratio_median=float(np.median(ratio)),
        candidate_sigma_coefficients=candidate.coefficients[0].tolist(),
    )
    if (
        not np.isfinite(refined).all()
        or np.any(refined <= 0)
        or np.any(abs(ratio - 1) > max_fractional_change + 1e-5)
        or np.any(refined > wing + 1e-5)
    ):
        return finish("REJECTED_PHYSICAL_BOUNDS")
    after = validation_statistics(candidate, lines)
    info["after"] = after
    allowance = max(
        validation_tolerance * before["chi_square"],
        validation_noise_sigma * np.sqrt(2 * before["degrees_of_freedom"]),
    )
    info["validation_chi_square_allowance"] = float(allowance)
    if (
        not np.isfinite(after["pooled_reduced_chisq"])
        or after["chi_square"] > before["chi_square"] + allowance
    ):
        return finish("REJECTED_VALIDATION")
    result.coefficients = candidate.coefficients.copy()
    return finish("ACCEPTED", True)
