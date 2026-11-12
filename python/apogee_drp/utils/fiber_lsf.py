"""Estimate missing per-fiber native-pixel LSFs without mixing coordinate bases."""

import numpy as np
from numpy.polynomial import Chebyshev
from .pixel_lsf import NativePixelLSF


def fiber_weights(fibers, target, method="linear", neighbors=7, degree=2):
    """Choose measured donors and prediction weights in physical fiber number.

    Linear interpolation uses the nearest measured fiber on either side.
    At an edge it uses the nearest measured fiber, without extrapolating.
    Polynomial estimates use a local least-squares fit; they are provided for
    validation/comparison and may have negative weights. Never pass previously
    filled fibers as donors.
    """
    fibers = np.asarray(sorted(f for f in fibers if int(f) != int(target)), int)
    if not len(fibers):
        raise ValueError("No measured donor fibers")
    if method == "nearest":
        return fibers[[np.argmin(abs(fibers - target))]], np.ones(1), "nearest"
    if method == "linear":
        lower = fibers[fibers < target]
        upper = fibers[fibers > target]
        if not len(lower) or not len(upper):
            return fibers[[np.argmin(abs(fibers - target))]], np.ones(1), "nearest"
        selected = np.array([lower[-1], upper[0]])
        return (
            selected,
            np.array([selected[1] - target, target - selected[0]], float)
            / np.diff(selected)[0],
            "linear",
        )
    if method not in ("polynomial", "local_linear"):
        raise ValueError("Unknown fiber filling method")
    if method == "local_linear":
        degree = 1
    if not np.any(fibers < target) or not np.any(fibers > target):
        return fibers[[np.argmin(abs(fibers - target))]], np.ones(1), "nearest"
    if int(degree) != degree or degree < 0 or neighbors < degree + 1:
        raise ValueError("Insufficient neighbors for polynomial degree")
    selected = fibers[np.argsort(abs(fibers - target), kind="stable")[:neighbors]]
    if len(selected) < degree + 1:
        raise ValueError("Too few measured fibers for polynomial fit")
    x = (selected - target) / max(float(np.max(abs(selected - target))), 1.0)
    design = np.polynomial.polynomial.polyvander(x, degree)
    rhs = np.zeros(degree + 1)
    rhs[0] = 1.0
    weights = np.linalg.lstsq(design.T, rhs, rcond=None)[0]
    return selected, weights, method


def interpolate_fiber_model(
    models,
    target,
    method="linear",
    neighbors=7,
    degree=2,
    chip_offsets=None,
    original_failure=None,
):
    """Predict a native pixel LSF from measured neighbors in a common basis.

    Each donor polynomial is first converted to the same X origin/scale.
    Directly averaging its stored Chebyshev coefficients would be incorrect
    because the conditioning differs between fitted fibers. Chip offsets may
    be measured independently or estimated with the same donor weights.
    """
    donors, weights, actual_method = fiber_weights(
        models, target, method, neighbors, degree
    )
    selected = [models[int(f)] for f in donors]
    if not all(isinstance(m, NativePixelLSF) for m in selected):
        raise TypeError("Fiber filling currently requires native pixel models")
    if len({m.horder for m in selected}) != 1 or len({m.npix for m in selected}) != 1:
        raise ValueError("Donor Hermite order and detector size must agree")
    origin = 0.0
    scale = 4096.0
    count = max(m.coefficients.shape[1] for m in selected)
    co = np.zeros((len(selected), selected[0].coefficients.shape[0], count))
    for j, m in enumerate(selected):
        for i, row in enumerate(m.coefficients):
            converted = (
                Chebyshev(row, domain=[m.origin - m.scale, m.origin + m.scale])
                .convert(domain=[origin - scale, origin + scale])
                .coef
            )
            co[j, i, : len(converted)] = converted
    coefficients = np.tensordot(weights, co, axes=1)
    geometry_estimated = chip_offsets is None
    if chip_offsets is None:
        chip_offsets = np.tensordot(
            weights, np.array([m.chip_offsets for m in selected]), axes=1
        )
    chip_offsets = np.asarray(chip_offsets, float)
    diagnostics = dict(
        fiber=int(target),
        filled=True,
        fill_method=actual_method,
        source_fibers=donors.tolist(),
        source_weights=weights.tolist(),
        original_failure=original_failure,
        geometry_estimated=geometry_estimated,
        requested_fill_method=method,
        neighbors=int(neighbors),
        fiber_polynomial_degree=1 if method == "local_linear" else int(degree),
        coordinate="native_pixel",
        degree=count - 1,
        horder=selected[0].horder,
        npix=selected[0].npix,
        condition="Estimated from measured fiber solutions; not fitted to this fiber",
    )
    result = NativePixelLSF(
        coefficients,
        origin,
        scale,
        selected[0].horder,
        diagnostics,
        chip_offsets.copy(),
        selected[0].npix,
    )
    x = np.concatenate(
        [np.linspace(offset, offset + result.npix - 1, 128) for offset in chip_offsets]
    )
    fields = result.fields(x)
    # Match the soft-constrained measured fits: retain their small negative
    # wing fractions instead of clipping parameters or changing the profile.
    if (
        not np.isfinite(fields).all()
        or np.any(fields[0] <= 0)
        or np.any(fields[-1] <= 0)
        or np.any(fields[-2] < -0.01)
        or np.any(fields[-2] > 1.01)
    ):
        raise ValueError("Interpolated LSF has invalid widths or wing fractions")
    return result
