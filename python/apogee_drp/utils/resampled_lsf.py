"""Build row-matched apStar LSF tables during RV visit combination."""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from .apstar_lsf import derive_apstar_lsf, evaluate_apogee_gh

# sigma, H1, H3, H4, H5, wing fraction, wing sigma. H2 is fixed to zero.
H5_MIXED_DEGREES = (3, 0, 3, 2, 3, 0, 0)  # Preserved wavelength-native comparison.
H5_NATIVE_PIXEL_DEGREES = (3, 0, 4, 1, 2, 2, 4)


def inverse_variance_weights(errors):
    """Finite nonnegative combination weights; invalid errors contribute zero."""
    errors = np.asarray(errors, float)
    weights = np.zeros_like(errors)
    valid = np.isfinite(errors) & (errors > 0)
    np.divide(1.0, errors**2, out=weights, where=valid)
    return weights


def build_apstar_lsf_table(
    apstar, visitfiles, pixel_weights, *, global_weights=None, vrel=None, options=None
):
    """Convert native LSFs into one polynomial vector per apStar flux row.

    For N>1 visits row 0 uses the actual pixel weights, row 1 uses the actual
    smoothed-error weights, and rows 2..N+1 describe individual resampled visits.
    A one-visit apStar has only row 0. Native GLOBAL_LSF extensions are consumed
    automatically; classic visits use their historical native pixel parameters.
    Supply the shifts actually used for resampling via vrel (also for bconly).

    The default 23-coefficient H5 mixed orders are experimental: selected on one fiber and
    calibration exposure. MAXERR records held-out profile errors; a configurable
    max_error (default None) can reject inaccurate approximations. No calibration
    measurements are refitted here. Failed conversions raise rather than being
    silently assigned a generic Gaussian LSF.
    """
    n = len(visitfiles)
    if not n:
        raise ValueError("At least one visit is required")
    wave = np.asarray(apstar.wave, float)
    weights = np.asarray(pixel_weights, float)
    if (
        weights.shape != (n, len(wave))
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("pixel_weights must be finite, nonnegative [nvisit,nwave]")
    if n > 1:
        if global_weights is None:
            raise ValueError("Supply the actual global weighting array")
        global_weights = np.asarray(global_weights, float)
        if (
            global_weights.shape != weights.shape
            or not np.isfinite(global_weights).all()
            or np.any(global_weights < 0)
        ):
            raise ValueError("Invalid global_weights")
    shifts = np.asarray(
        [apstar.header[f"VREL{i+1}"] for i in range(n)] if vrel is None else vrel, float
    )
    if shifts.shape != (n,):
        raise ValueError("One velocity per input visit is required")
    valid = np.isfinite(shifts)
    if not valid.any():
        raise ValueError("No finite visit Doppler shifts")
    opts = dict(degree=4, horder=5, polynomial_degrees=H5_NATIVE_PIXEL_DEGREES)
    if options:
        opts.update(options)
        if "polynomial_degrees" not in options and (
            opts["horder"] != 5 or opts["degree"] < 4 or opts.get("fit_h2", False)
        ):
            opts["polynomial_degrees"] = None
    max_error = opts.pop("max_error", None)
    qa_threshold = opts.pop("qa_threshold", 0.01)
    rows = []
    filled_rows = {}
    sigma_rows = {}

    def fit_row(row, kind, indices, w):
        empty_status = "NO_RV" if not len(indices) else "NO_DATA"
        if len(indices):
            contributes = np.any(w > 0, axis=1)
            indices = indices[contributes]
            w = w[contributes]
        if not len(indices):
            rows.append(
                (
                    row,
                    kind,
                    0,
                    np.array([], float),
                    np.full(6, np.nan),
                    np.nan,
                    empty_status,
                )
            )
            return
        header = apstar.header.copy()
        header["NVISITS"] = len(indices)
        paths = [visitfiles[i] for i in indices]
        for j, i in enumerate(indices):
            header[f"SFILE{j+1}"] = Path(visitfiles[i]).name
            header[f"VREL{j+1}"] = shifts[i]
        target = SimpleNamespace(header=header, wave=wave)
        from .native_lsf import NativeLSFUnavailable

        try:
            result = derive_apstar_lsf(target, paths, weights=w, **opts)
        except NativeLSFUnavailable:
            # A contributing spectrum without a calibration prevents assigning
            # a valid combined LSF; preserve the spectrum and record this fact.
            rows.append(
                (
                    row,
                    kind,
                    0,
                    np.array([], float),
                    np.full(6, np.nan),
                    np.nan,
                    "NO_LSF",
                )
            )
            return
        native_diagnostics = (
            result.provenance.get("native_calibration_diagnostics", []) or []
        )
        if isinstance(native_diagnostics, dict):
            native_diagnostics = [native_diagnostics]
        sigma_rows[row]=any(d and d.get("sigma_data_constrained",False) for d in native_diagnostics)
        filled_rows[row] = any(d and d.get("filled", False) for d in native_diagnostics)
        error = max(
            d["validation_max_peak_relative"]
            for d in result.provenance["diagnostics"].values()
        )
        if max_error is not None and error > max_error:
            raise ValueError(
                f"apStar LSF row {row} ({kind}) error {error:.3%} exceeds {max_error:.3%}"
            )
        pars = result.parameters
        rows.append(
            (
                row,
                kind,
                len(pars),
                pars,
                result.domains.ravel(),
                error,
                "OK" if error <= qa_threshold else "WARN",
            )
        )

    good = np.flatnonzero(valid)
    fit_row(0, "PIXEL", good, weights[good])
    if n > 1:
        fit_row(1, "GLOBAL", good, global_weights[good])
        for i in range(n):
            fit_row(
                i + 2,
                "VISIT",
                np.array([i]) if valid[i] else np.array([], int),
                np.ones((1, len(wave))),
            )
    expected = 1 if n == 1 else n + 2
    if np.asarray(apstar.flux).shape[0] != expected:
        raise ValueError("apStar flux row count does not match visit list")
    npar = max(1, max(r[2] for r in rows))
    packed = np.full((len(rows), npar), np.nan)
    for i, r in enumerate(rows):
        packed[i, : r[2]] = r[3]
    table = Table(
        dict(
            ROW=[r[0] for r in rows],
            KIND=[r[1] for r in rows],
            NPARS=[r[2] for r in rows],
            PARAMETERS=packed,
            DOMAINS=np.array([r[4] for r in rows]).reshape(-1, 3, 2),
            MAXERR=[r[5] for r in rows],
            STATUS=[r[6] for r in rows],
            NATIVE_FILLED=[bool(filled_rows.get(r[0], False)) for r in rows],
            NATIVE_SIGMA_REFINED=[bool(sigma_rows.get(r[0],False)) for r in rows],
        )
    )
    table.meta.update(
        LSFVER=1,
        LSFMODE="APSTAR-SINC",
        LSFCOORD="APSTAR_PIXEL",
        HORDER=int(opts["horder"]),
        LSFFRAME="REST",
        PIXORIG=0,
        QALIMIT=float(qa_threshold),
        COMMENT="ROW matches zero-based FLUX row; DOMAINS excludes gaps and edges.",
    )
    return table


def evaluate_lsf_table(table, row, centers, offsets):
    """Evaluate a row-matched packed apStar LSF; return NaN outside its domains."""
    matches = np.flatnonzero(np.asarray(table["ROW"]) == row)
    if len(matches) != 1:
        raise ValueError("No unique LSF entry for flux row")
    r = table[matches[0]]
    if str(r["STATUS"]) not in ("OK", "WARN"):
        raise ValueError(f'LSF row unavailable: {r["STATUS"]}')
    centers = np.atleast_1d(centers).astype(float)
    offsets = np.broadcast_to(
        np.asarray(offsets, float), (len(centers), np.shape(offsets)[-1])
    )
    domains = np.asarray(r["DOMAINS"]).reshape(3, 2)
    valid = np.any(
        (centers[:, None] >= domains[:, 0]) & (centers[:, None] <= domains[:, 1]),
        axis=1,
    )
    result = np.full(offsets.shape, np.nan)
    if valid.any():
        result[valid] = evaluate_apogee_gh(
            offsets[valid], centers[valid], r["PARAMETERS"][: r["NPARS"]]
        )
    return result
