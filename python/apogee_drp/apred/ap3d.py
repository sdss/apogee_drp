"""APOGEE raw 3-D ramp to calibrated 2-D image processing.

This is a Python translation of the core algorithms in ``ap3dproc.pro`` and
``aprefcorr.pro``.  Arrays use the normal NumPy/FITS ordering
``(nread, ny, nx)``.  The returned 2-D arrays use ``(ny, nx)``.

The :func:`ap3d` plan-level wrapper interprets APOGEE plan files, constructs
survey filenames, optionally requests calibration creation, and dispatches
each exposure/chip to :func:`process_file`. Pipeline locking remains the
responsibility of the surrounding batch system.

The implementation keeps the original algorithmic choices:

* reference-output, vertical-ramp, and horizontal-ramp subtraction;
* bad-read rejection and interpolation;
* per-read linearity and dark correction;
* CR detection in differences of reads;
* saturated-read extrapolation;
* Fowler or up-the-ramp collapse;
* Poisson, read-noise, saturation, and CR error propagation;
* flat-fielding and APOGEE-compatible FLUX/ERROR/MASK FITS extensions.

The persistence *mask* is supported.  Persistence-model subtraction is exposed
as an input image because the IDL ``APPERSISTMODEL`` routine belongs to a
separate calibration subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import getpass
import os
from pathlib import Path
import platform
import socket
import subprocess
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta
#from apogee_drp.utils import apzip
#from apogee_drp.utils.bitmask import PixelBitMask
from dlnpyutils import utils as dln
from scipy.ndimage import median_filter
from ..utils import apzip,utils
from ..utils.bitmask import PixelBitMask

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised in no-Numba environments
    njit = None

__all__ = [
    "AP3D_VERSION",
    "NUMBA_AVAILABLE",
    "PIXMASK",
    "CosmicRay",
    "ProcessResult",
    "PlanProcessRecord",
    "reference_correct",
    "detect_and_fix_cosmic_rays",
    "fowler_sampling",
    "up_the_ramp_sampling",
    "process_array",
    "process_cube",
    "read_ramp",
    "load_raw_ramp",
    "read_calibrations",
    "write_ap2d",
    "process_file",
    "ap3d",
]

NUMBA_AVAILABLE = njit is not None
PIXMASK = PixelBitMask()
BAD_VARIANCE = np.float32(99_999_999.0)
NONFINITE_ERROR = np.float32(1.0e10)


def _log(message: str, started: float | None = None) -> None:
    """Print a UTC timestamped message with optional elapsed wall time."""

    timestamp = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    elapsed = (
        f" [{perf_counter() - started:.2f} s]" if started is not None else ""
    )
    print(f"{timestamp}{elapsed} {message}", flush=True)

    
@dataclass
class CosmicRay:
    """Description of a cosmic ray detected in one detector pixel.

    Attributes use zero-based NumPy coordinates and read numbers.  ``counts``
    is the excess signal in the affected read difference, while ``fix_error``
    is the additional variance contribution when the event was repaired.
    """

    x: int
    y: int = -1
    read: int = 0
    counts: float = 0.0
    nsigma: float = 0.0
    global_sigma: float = 0.0
    fixed: bool = False
    local_sigma: float = 0.0
    fix_error: float = 0.0
    neighbor_checked: bool = False


@dataclass
class ProcessResult:
    """Calibrated 2-D products and auxiliary reduction information.

    ``flux``, ``error``, and ``mask`` have shape ``(2048, 2048)``.  The
    optional fields describe detected events, saturation and read state.
    ``fixed_cube`` is retained only when :func:`process_array` is called with
    ``return_cube=True``.
    """

    flux: np.ndarray
    error: np.ndarray
    mask: np.ndarray
    header: fits.Header
    cosmic_rays: list[CosmicRay] = field(default_factory=list)
    saturation: np.ndarray | None = None
    fixed_cube: np.ndarray | None = None
    persistence_model: np.ndarray | None = None
    read_mask: np.ndarray | None = None
    global_variability: float = -1.0


@dataclass(frozen=True)
class PlanProcessRecord:
    """Outcome for one chip of one exposure processed by :func:`ap3d`.

    Keeping only filenames and status prevents the plan-level wrapper from
    retaining several full 2048-by-2048 result arrays in memory.
    """

    planfile: str
    exposure: int
    flavor: str
    chip: str
    input: str
    output: str
    status: str
    elapsed_seconds: float
    error: str | None = None


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _rolling_nanmedian_numba_float32(
        values: np.ndarray,
        width: int,
    ) -> np.ndarray:
        """Numba kernel for IDL-compatible float32 rolling nanmedians."""

        nrow, nvalue = values.shape
        result = values.copy()
        if width <= 1:
            return result
        left = width // 2
        right = width - left
        scratch = np.empty(width, dtype=np.float32)

        for row in range(nrow):
            for index in range(left, nvalue - right):
                count = 0
                start = index - left
                for offset in range(width):
                    value = values[row, start + offset]
                    if np.isfinite(value):
                        # Insertion sorting is efficient for APOGEE's
                        # eleven-element windows and avoids allocations.
                        position = count
                        while (
                            position > 0
                            and scratch[position - 1] > value
                        ):
                            scratch[position] = scratch[position - 1]
                            position -= 1
                        scratch[position] = value
                        count += 1
                if count == 0:
                    result[row, index] = np.nan
                elif count % 2 == 1:
                    result[row, index] = scratch[count // 2]
                else:
                    result[row, index] = (
                        scratch[count // 2 - 1] + scratch[count // 2]
                    ) / np.float32(2.0)

            for index in range(left):
                result[row, index] = result[row, left]
            last = nvalue - right - 1
            for index in range(nvalue - right, nvalue):
                result[row, index] = result[row, last]
        return result

else:

    def _rolling_nanmedian_numba_float32(
        values: np.ndarray,
        width: int,
    ) -> np.ndarray:
        """Raise when the optional Numba rolling-median kernel is unavailable."""

        raise RuntimeError("Numba is not available")


def _rolling_nanmedian(a: np.ndarray, width: int) -> np.ndarray:
    """Median-filter the last axis using IDL ``MEDFILT2D`` edge behavior.

    This reproduces ``MEDFILT2D(array, width, DIM=2, /EDGE_COPY, /EVEN)``
    from the APOGEE IDL pipeline.  Only positions containing a complete
    ``width``-element window are filtered.  The first ``width // 2`` output
    values are set to the first complete-window median.  The final
    ``width - width // 2`` values are set to the last complete-window
    median.  Consequently, an odd-width filter has one more copied value at
    the end than at the beginning.

    NumPy's median convention for an even number of finite samples—averaging
    the two middle values—matches IDL's ``/EVEN`` keyword.  NaNs are ignored
    here because saturated or otherwise rejected read differences are
    represented by NaNs in the Python implementation.

    Parameters
    ----------
    a
        Input array.  Filtering is performed along its final axis.
    width
        Number of samples in each complete median window.  It must not exceed
        the length of the final axis.

    Returns
    -------
    numpy.ndarray
        Median-filtered array with the same shape as ``a``.
    """

    values = np.asarray(a)
    normalized_width = max(1, int(width))
    if (
        NUMBA_AVAILABLE
        and values.ndim == 2
        and values.dtype == np.dtype(np.float32)
        and normalized_width < values.shape[-1]
    ):
        return _rolling_nanmedian_numba_float32(
            values,
            normalized_width,
        )
    return utils.idl_median_filter_1d(
        values,
        normalized_width,
        edge_copy=True,
    )


def _expand_quadrants(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Expand four detector-output values into a full image."""

    values = np.asarray(values, dtype=np.float32).ravel()
    if values.size != 4:
        raise ValueError("Expected four detector-output values")
    ny, nx = shape
    if nx != 2048:
        raise ValueError("Four-output expansion requires nx=2048")
    out = np.empty(shape, dtype=np.float32)
    for q, value in enumerate(values):
        out[:, q * 512 : (q + 1) * 512] = value
    return out


def _reference_subtract_image(image: np.ndarray, ref: np.ndarray) -> None:
    """In-place equivalent of IDL ``aprefcorr_sub``."""

    image[:, 0:512] -= ref
    image[:, 512:1024] -= ref[:, ::-1]
    image[:, 1024:1536] -= ref
    image[:, 1536:2048] -= ref[:, ::-1]


def _reference_subtract_image_long(
    image: np.ndarray, ref: np.ndarray
) -> None:
    """IDL ``APREFCORR_SUB`` with assignment back into a LONG image."""

    image[:, 0:512] = (image[:, 0:512] - ref).astype(np.int32)
    image[:, 512:1024] = (image[:, 512:1024] - ref[:, ::-1]).astype(np.int32)
    image[:, 1024:1536] = (image[:, 1024:1536] - ref).astype(np.int32)
    image[:, 1536:2048] = (image[:, 1536:2048] - ref[:, ::-1]).astype(np.int32)


def reference_correct(
    cube: np.ndarray,
    header: Mapping[str, Any] | None = None,
    *,
    indiv: int = 3,
    cds: bool = True,
    vertical: bool = True,
    horizontal: bool = True,
    noflip: bool = False,
    q3fix: bool = False,
    keep_reference: bool = False,
    saturation: float = 55_000.0,
    verbose: bool = False,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Apply the APOGEE reference-output and reference-pixel correction.

    Parameters
    ----------
    cube
        Raw ramp with shape ``(nread, 2048, 2560)``.  The first 2048 columns
        are science pixels and the final 512 are the reference output.
    indiv
        Positive: subtract each reference output, median-filtered with this
        width when greater than one.  Negative: subtract the mean reference
        pattern.  Zero: do not subtract the reference output.
    Returns
    -------
    corrected, mask, read_mask, last_good
        ``corrected`` has 2048 columns unless ``keep_reference`` is true, in
        which case the corrected 512-column reference output is appended.
    """

    data = np.asarray(cube)
    if data.ndim != 3 or data.shape[1:] != (2048, 2560):
        raise ValueError("reference_correct expects (nread, 2048, 2560)")
    nread = data.shape[0]
    hdr = header or {}
    mask = np.zeros((2048, 2048), dtype=np.uint16)
    read_mask = np.zeros(nread, dtype=bool)

    mean_ref = np.zeros((2048, 512), dtype=np.float64)
    nref = np.zeros((2048, 512), dtype=np.int32)
    for i in range(nread):
        ref = data[i, :, 2048:2560].astype(np.float64)
        core = ref[128:-128, 128:-128]
        mean, std = np.mean(core), np.std(core)
        high = np.max(ref[128 : 2048 - 256, 128:-128])
        ref[ref >= saturation] = np.nan
        iread = int(hdr.get(f"SLICE{i:03d}", i + 1))
        good_read = iread > 1 and std > 0 and mean / std > 10
        if indiv <= 0:
            good_read &= high < 65_530
        if debug:
            snr = mean / std if std > 0 else np.inf
            _log(
                f"  reference read {i + 1:3d}: SLICE={iread:3d}, "
                f"mean={mean:10.3f}, std={std:9.3f}, "
                f"mean/std={snr:8.2f}, max={high:9.1f}, "
                f"{'accepted' if good_read else 'REJECTED'}"
            )
        if good_read:
            good = np.isfinite(ref)
            mean_ref[good] += ref[good] - mean
            nref[good] += 1
        else:
            read_mask[i] = True
    if verbose:
        rejected = np.flatnonzero(read_mask) + 1
        _log(
            "Reference-pattern reads rejected: "
            + (", ".join(map(str, rejected)) if rejected.size else "none")
        )
    np.divide(mean_ref, nref, out=mean_ref, where=nref > 0)

    # APREFCORR explicitly uses LONARR output.  Its science working image is
    # also LONG, so every in-place floating correction is converted back to
    # int32.  Preserving those conversions is required for IDL-level numerical
    # agreement in the collapsed image.
    out = np.empty((nread, 2048, 2048), dtype=np.int32)
    refout = (
        np.empty((nread, 2048, 512), dtype=np.int32)
        if keep_reference
        else None
    )
    cds_ref = data[1, :, :2048].astype(np.int32) if cds and nread > 1 else 0
    yfrac = np.arange(2048, dtype=np.float32) / np.float32(2048.0)
    xfrac = np.arange(2048, dtype=np.float32) / np.float32(2048.0)
    last_good = nread - 1
    nsat0 = 0

    for i in range(nread):
        red = data[i, :, :2048].astype(np.int32)
        saturated = red > saturation
        nsat = int(saturated.sum())
        if i == 0:
            nsat0 = nsat
        if nsat > nsat0 + 2000 and last_good == nread - 1:
            last_good = i - 1
        mask[red == 0] |= PIXMASK.getval("BADPIX")
        mask[saturated] |= PIXMASK.getval("SATPIX")
        red[saturated] = 65_535

        if read_mask[i]:
            # This placeholder is overwritten by rejected-read interpolation
            # in process_array.  IDL assigns NaN to a LONG array here.
            out[i] = 0
            if refout is not None:
                refout[i] = 0
            continue

        if cds:
            red = (red - cds_ref).astype(np.int32)
        ref = data[i, :, 2048:2560].astype(np.float32)
        if indiv == 1:
            correction = ref
        elif indiv > 1:
            correction = utils.idl_median_filter_2d(ref, indiv)
        elif indiv < 0:
            correction = mean_ref
        else:
            correction = None
        if correction is not None:
            _reference_subtract_image_long(red, correction)
            ref = (ref - correction).astype(np.int32)

        if vertical:
            for q in range(4):
                sl = slice(q * 512, (q + 1) * 512)
                rlo = utils.idl_mean_float(red[2:4, sl],ignore_nonfinite=True)
                rhi = utils.idl_mean_float(red[2045:2047, sl],ignore_nonfinite=True)
                red[:,sl] = (red[:,sl] - rlo * (np.float32(1.0) - yfrac[:,None])).astype(np.int32)
                red[:,sl] = (red[:,sl] - rhi * yfrac[:,None]).astype(np.int32)

        if horizontal:
            clo = utils.idl_mean_float(red[:, 1:4],axis=1,ignore_nonfinite=True,)
            chi = utils.idl_mean_float(red[:, 2044:2047],axis=1,ignore_nonfinite=True)
            slo = utils.idl_median_filter_1d(clo[None, :], 7, edge_copy=True)[0]
            shi = utils.idl_median_filter_1d(chi[None, :], 7, edge_copy=True)[0]
            if noflip:
                red = (red- (slo[:, None]* (np.float32(1.0) - xfrac[None, :])
                        + shi[:, None] * xfrac[None, :])).astype(np.int32)
            else:
                bias = np.minimum(slo, shi)[:, None] * np.ones(
                    (1, 2048), dtype=np.float32)
                bias[:, 512:1024] = bias[:, 512:1024][:, ::-1]
                bias[:, 1536:2048] = bias[:, 1536:2048][:, ::-1]
                red = (red - bias).astype(np.int32)

        if q3fix:
            offset = 0.5 * (np.median(red[:, 923:1024], axis=1)
                            - np.median(red[:, 1024:1125], axis=1)
                            + np.median(red[:, 1536:1637], axis=1)
                            - np.median(red[:, 1435:1536], axis=1))
            offset = utils.idl_median_filter_1d(offset[None, :], 7, edge_copy=True)[0]
            red[:, 1024:1536] = (red[:, 1024:1536] + offset[:, None]).astype(np.int32)

        red[saturated] = 65_535
        out[i] = red
        if refout is not None:
            refout[i] = ref

    mask[:4, :] |= PIXMASK.getval("BADPIX")
    mask[-4:, :] |= PIXMASK.getval("BADPIX")
    mask[:, :4] |= PIXMASK.getval("BADPIX")
    mask[:, -4:] |= PIXMASK.getval("BADPIX")
    if refout is not None:
        out = np.concatenate((out, refout), axis=2)
    return out, mask, read_mask, last_good


def _detect_bad_reads(
    cube: np.ndarray, *, debug: bool = False
) -> np.ndarray:
    """Reproduce the IDL reference-pixel/reference-output RMS test."""

    nread, ny, nx = cube.shape
    first = cube[: min(4, nread)]
    edge0 = np.concatenate(
        (np.median(first[:, :4, :2048], axis=0).ravel(),
         np.median(first[:, :, :4], axis=0).ravel(),
         np.median(first[:, :, 2044:2048], axis=0).ravel(),
         np.median(first[:, -4:, :2048], axis=0).ravel()))
    edge_rms = np.empty(nread)
    ref_rms = np.empty(nread) if nx == 2560 else None
    if nx == 2560:
        ref0 = np.median(first[:, :, 2048:], axis=0)
    for i in range(nread):
        edge = np.concatenate(
            (cube[i, :4, :2048].ravel(),
             cube[i, :, :4].ravel(),
             cube[i, :, 2044:2048].ravel(),
             cube[i, -4:, :2048].ravel()))
        edge_rms[i] = np.sqrt(np.mean((edge.astype(float) - edge0) ** 2))
        if ref_rms is not None:
            diff = cube[i, :, 2048:].astype(float) - ref0
            ref_rms[i] = np.sqrt(np.mean(diff[100:1950] ** 2))
    series = ref_rms if ref_rms is not None else edge_rms
    if nread > 2:
        local = median_filter(series, size=min(11, nread), mode="nearest")
    else:
        local = np.full(nread, np.median(series))
    sigma = max(float(dln.mad(series)), 1.0)
    rejected = series - local > 10.0 * sigma
    if debug:
        source = "reference-output RMS" if ref_rms is not None else "reference-pixel RMS"
        _log(f"Initial bad-read diagnostic ({source}; sigma={sigma:.3f}):")
        for i, (value, baseline, bad) in enumerate(zip(series, local, rejected)):
            _log(
                f"  read {i + 1:3d}: RMS={value:10.3f}, "
                f"local={baseline:10.3f}, delta={value - baseline:9.3f}, "
                f"{'REJECTED' if bad else 'accepted'}"
            )
    return rejected


def _interpolate_bad_reads(cube: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Replace rejected reads by linear interpolation along the read axis.

    End-point rejections are replaced by the nearest good read.  The input is
    copied so callers retain the unmodified ramp.
    """

    result = cube.astype(np.float32, copy=True)
    good = np.flatnonzero(~bad)
    if good.size < 2:
        raise ValueError("At least two good reads are required")
    for index in np.flatnonzero(bad):
        below = good[good < index]
        above = good[good > index]
        if below.size == 0:
            lo, hi = above[:2]
        elif above.size == 0:
            lo, hi = below[-2:]
        else:
            lo, hi = below[-1], above[0]
        interpolated = result[lo] + (index - lo) * (result[hi] - result[lo]) / (hi - lo)
        # AP3DPROC stores ROUND(im0) back into its LONG cube.
        result[index] = np.rint(interpolated)
    return result


def _apply_linearity(
    cube: np.ndarray,
    coefficients: np.ndarray,
    *,
    mode: str = "idl",
) -> np.ndarray:
    """Apply the APOGEE detector linearity response correction.

    Parameters
    ----------
    cube
        Observed ramp with shape ``(nread, ny, nx)``.
    coefficients
        Polynomial coefficients ordered from constant to highest power.
        Supported shapes are ``(ny, nx, ncoef)``, ``(4, ncoef)``, and
        ``(ncoef, 4)``.  The latter two forms provide one polynomial per
        512-column detector output.
    mode
        ``"idl"`` reproduces ``aplincorr.pro`` exactly, including its
        first-read-only indexing bug. ``"all"`` applies the intended
        correction to every read. ``"none"`` returns an unchanged copy.

    Returns
    -------
    numpy.ndarray
        A floating-point copy of the ramp.  Corrected samples are divided by
        the response polynomial, as in the IDL routine.

    Notes
    -----
    For read indices two and above, the polynomial argument is
    ``(read[j] - read[1]) * (j + 1) / (j - 1)``.  Reads zero and one use the
    read-two argument.  This unusual convention is intentionally retained for
    compatibility with the operational IDL implementation.
    """

    selected_mode = str(mode).lower()
    allowed_modes = {"idl", "all", "none"}
    if selected_mode not in allowed_modes:
        raise ValueError(
            "linearity mode must be 'idl', 'all', or 'none'; "
            f"received {mode!r}"
        )

    result = np.asarray(cube, dtype=np.float32).copy()
    if selected_mode == "none":
        return result
    if result.ndim != 3:
        raise ValueError("cube must have shape (nread, ny, nx)")

    c = np.asarray(coefficients, dtype=np.float32)
    per_pixel = c.ndim == 3 and c.shape[:2] == result.shape[1:]
    if c.ndim == 2 and c.shape[0] != 4 and c.shape[1] == 4:
        c = c.T
    per_output = c.ndim == 2 and c.shape[0] == 4
    if not per_pixel and not per_output:
        raise ValueError(
            "Linearity coefficients must be (ny,nx,ncoef), (4,ncoef), "
            f"or (ncoef,4); received {c.shape}"
        )
    if per_output and result.shape[2] != 2048:
        raise ValueError(
            "Four-output linearity coefficients require 2048 detector columns"
        )
    if result.shape[0] < 3:
        raise ValueError(
            "APOGEE linearity correction requires at least three reads"
        )

    count_level = result.copy()
    for read_index in range(2, result.shape[0]):
        count_level[read_index] = ((result[read_index] - result[1])
                                   * (read_index + 1.0) / (read_index - 1.0))
    count_level[~np.isfinite(count_level)] = 0.0
    count_level[0] = count_level[2]
    count_level[1] = count_level[2]

    factor = np.zeros_like(result)
    if per_pixel:
        for coefficient in np.moveaxis(c, -1, 0)[::-1]:
            factor = factor * count_level + coefficient[None, :, :]
    else:
        for output_index in range(4):
            columns = slice(output_index * 512, (output_index + 1) * 512)
            output_factor = np.zeros_like(result[:, :, columns])
            for coefficient in c[output_index, ::-1]:
                output_factor = (output_factor * count_level[:, :, columns] +
                                 coefficient)
            factor[:, :, columns] = output_factor

    with np.errstate(divide="ignore", invalid="ignore"):
        if selected_mode == "all":
            result /= factor
        else:
            # This reproduces IDL's slice_out[0:2047] assignment: with IDL's
            # one-dimensional indexing of a 2-D slice, only read zero changes.
            result[0] /= factor[0]
    return result


def detect_and_fix_cosmic_rays(
    dcounts: np.ndarray,
    saturation: np.ndarray,
    *,
    noise: float = 17.0,
    sigma_threshold: float = 10.0,
    fix: bool = True,
    only_read: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[CosmicRay]]:
    """Detect CRs in ``(npixel, ndifference)`` read differences."""

    dc = np.asarray(dcounts, dtype=np.float32)
    fixed = dc.copy()
    median = np.nanmedian(dc, axis=1)
    median[~np.isfinite(median)] = 0
    ngood = np.isfinite(dc).sum(axis=1)
    two = np.flatnonzero(ngood == 2)
    for p in two:
        lo, hi = np.nanmin(dc[p]), np.nanmax(dc[p])
        if (hi - lo) / max(lo, 1e-4) > 0.3:
            median[p] = max(lo, 1e-4)
    model = np.broadcast_to(median[:, None], dc.shape).copy()
    width = min(11, dc.shape[1])
    if dc.shape[1] > width:
        model = _rolling_nanmedian(dc, width)
        bad = ~np.isfinite(model)
        model[bad] = np.broadcast_to(median[:, None], dc.shape)[bad]
    variability = dln.mad(dc - median[:, None], axis=1, zero=True) / np.maximum(median, 0.001)
    variability[~np.isfinite(variability)] = 0
    variability[two] = 0.5
    sigma = np.maximum(dln.mad(dc - model, axis=1, zero=True), noise)
    sigma[~np.isfinite(sigma)] = noise
    sigma[two] = np.maximum(0.3 * median[two], noise)
    nsigma = (dc - model) / sigma[:, None]
    candidates = np.argwhere((nsigma > max(sigma_threshold, 3.0))
                             & (dc > noise * max(sigma_threshold, 3.0)))
    events: list[CosmicRay] = []
    for pixel, diff_index in candidates:
        read = int(diff_index + 1)
        if only_read is not None and abs(read - only_read) > 1:
            continue
        local_med = float(median[pixel])
        local_sigma = float(sigma[pixel])
        fix_error = local_sigma / np.sqrt(max(width - 1, 1))
        if fix:
            fixed[pixel, diff_index] = local_med
        events.append(
            CosmicRay(x=int(pixel),read=read,
                counts=float(dc[pixel, diff_index] - model[pixel, diff_index]),
                nsigma=float(nsigma[pixel, diff_index]),
                global_sigma=float(sigma[pixel]),fixed=fix,
                local_sigma=local_sigma,
                fix_error=fix_error if fix else 0.0))
    return fixed, median, variability, events


def fowler_sampling(
    cube: np.ndarray,
    good_reads: np.ndarray,
    readnoise: np.ndarray | float,
    *,
    nfowler: int = 10,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Collapse a corrected ramp using Fowler sampling.

    The mean of the first ``nfowler`` good reads is subtracted from the mean
    of the last ``nfowler`` good reads.  If fewer reads are available,
    ``nfowler`` is reduced to half the number of good reads, matching the IDL
    implementation.

    Parameters
    ----------
    cube
        Corrected ramp with read number on axis zero.
    good_reads
        One-dimensional integer indices of usable reads, in time order.
    readnoise
        Per-pixel read noise in ADU, or a scalar broadcastable to the output
        image.
    nfowler
        Requested number of reads in each of the beginning and ending samples.

    Returns
    -------
    image, sample_noise, nfowler_used
        Fowler-difference image, sampled read noise in ADU, and the number of
        reads actually used for each sample.
    """

    data = np.asarray(cube)
    reads = np.asarray(good_reads, dtype=int).ravel()
    if data.ndim != 3:
        raise ValueError("cube must have shape (nread, ny, nx)")
    if reads.size < 2:
        raise ValueError("Fowler sampling requires at least two good reads")
    if np.any(reads < 0) or np.any(reads >= data.shape[0]):
        raise ValueError("good_reads contains an index outside the ramp")
    requested = int(nfowler)
    if requested < 1:
        raise ValueError("nfowler must be at least one")

    nfowler_used = min(requested, reads.size // 2)
    if nfowler_used < 1:
        raise ValueError("No valid Fowler samples")
    beginning = reads[:nfowler_used]
    ending = reads[-nfowler_used:]
    image = data[ending].mean(axis=0) - data[beginning].mean(axis=0)
    sample_noise = np.asarray(readnoise, dtype=np.float32) * np.sqrt( 2.0 / nfowler_used)
    return image, sample_noise, nfowler_used


def up_the_ramp_sampling(
    cube: np.ndarray,
    good_reads: np.ndarray,
    readnoise: np.ndarray | float,
    gain: np.ndarray | float,
    *,
    science_nx: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a corrected ramp with the IDL sequential linear fit.

    A slope is fitted against the original read indices and multiplied by one
    less than the number of good reads to produce integrated counts.  The
    sampled-noise expression follows Equation 1 of Rauscher et al. (2007) with
    ``m=1`` and therefore already includes the source Poisson contribution.

    Parameters
    ----------
    cube
        Corrected ramp with shape ``(nread, ny, nx)``.
    good_reads
        One-dimensional integer indices of usable reads, in time order.
    readnoise
        Scalar or ``(ny, science_nx)`` read-noise array in ADU.
    gain
        Scalar or ``(ny, science_nx)`` gain in electron/ADU.
    science_nx
        Number of leading columns containing science pixels.  Extra reference
        columns may remain in ``cube`` but are excluded from ``sample_noise``.

    Returns
    -------
    image, sample_noise
        Integrated image (including any reference columns) and the
        ``(ny, science_nx)`` sampled noise in ADU.
    """

    data = np.asarray(cube)
    reads = np.asarray(good_reads, dtype=int).ravel()
    if data.ndim != 3:
        raise ValueError("cube must have shape (nread, ny, nx)")
    if reads.size < 2:
        raise ValueError("Up-the-ramp sampling requires at least two good reads")
    if np.any(reads < 0) or np.any(reads >= data.shape[0]):
        raise ValueError("good_reads contains an index outside the ramp")
    if science_nx < 1 or science_nx > data.shape[2]:
        raise ValueError("science_nx must select columns present in cube")

    # These accumulator types and their update order intentionally follow
    # ap3dproc.pro. Algebraically equivalent centered/vectorized fits do not
    # reproduce IDL's FLTARR rounding at the bit level.
    shape = data.shape[1:]
    sumts = np.zeros(shape, dtype=np.float32)
    sums = np.zeros(shape, dtype=np.float32)
    count = np.zeros(shape, dtype=np.int16)
    sumt = np.zeros(shape, dtype=np.float32)
    sumt2 = np.zeros(shape, dtype=np.float32)
    for read in reads:
        sample = np.asarray(data[read], dtype=np.float32)
        good = np.isfinite(sample)
        tread = np.float32(read)
        sumts[good] += tread * sample[good]
        sums[good] += sample[good]
        count[good] += 1
        sumt[good] += tread
        sumt2[good] += np.float32(read * read)

    count_float = count.astype(np.float32)
    numerator = count_float * sumts - sumt * sums
    denominator = count_float * sumt2 - sumt**2
    if not np.any(denominator > 0):
        raise ValueError("Good reads do not span more than one read index")
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = numerator / denominator
    image = slope * np.float32(reads.size - 1)

    n_good = np.float32(reads.size)
    n_total = np.float32(data.shape[0])
    gain_image = np.asarray(gain, dtype=np.float32)
    noise_image = np.asarray(readnoise, dtype=np.float32)
    signal = image[:, :science_nx]
    sample_noise = np.sqrt(np.float32(12.0) * (n_good - np.float32(1.0))
                           / (n_total * (n_good + np.float32(1.0)))
                           * noise_image**2 + np.float32(6.0)
                           * (n_good**2 + np.float32(1.0))
                           / ( np.float32(5.0) * n_good * (n_good + np.float32(1.0)))
                           * signal * gain_image ) / gain_image
    return image, sample_noise


def process_array(
    raw_cube: np.ndarray,
    header: fits.Header | None = None,
    *,
    gain: np.ndarray | float | None = None,
    readnoise: np.ndarray | float | None = None,
    linearity: np.ndarray | None = None,
    linearity_mode: str = "idl",
    bpm: np.ndarray | None = None,
    dark: np.ndarray | None = None,
    flat: np.ndarray | None = None,
    littrow: np.ndarray | None = None,
    persistence_mask: np.ndarray | None = None,
    persistence_model: np.ndarray | None = None,
    saturation_level: float = 65_000.0,
    fix_cosmic_rays: bool = True,
    detect_cosmic_rays: bool = True,
    fix_saturation: bool = True,
    fix_three_read_saturation: bool = False,
    iterative_cosmic_rays: bool = False,
    nfowler: int = 10,
    up_the_ramp: bool = False,
    output_electrons: bool = False,
    use_reference: bool = True,
    q3fix: bool = False,
    return_cube: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> ProcessResult:
    """Reduce an already-loaded APOGEE ramp array to calibrated 2-D images.

    Arrays use ``(nread, ny, nx)`` ordering. Most callers should use
    :func:`process_cube` with a decompressed FITS file or :func:`process_file`
    with a FITS or APZ file. ``linearity_mode="idl"`` reproduces the legacy
    IDL first-read-only behavior; use ``"all"`` for the intended correction
    of every read or ``"none"`` to disable it.

    The reduction includes bad-read handling, reference correction, detector
    calibrations, cosmic-ray and saturation treatment, ramp sampling,
    persistence subtraction, and flat-fielding. With ``verbose=True``, major
    stages are timestamped. The returned FITS header records the UTC processing
    start in ``AP3DTIME`` and elapsed seconds in ``AP3DSEC``.
    """

    processing_started = perf_counter()
    processing_timestamp = datetime.now(timezone.utc)
    raw = np.asarray(raw_cube)
    if raw.ndim != 3 or raw.shape[1] != 2048 or raw.shape[2] not in (2048, 2560):
        raise ValueError("raw_cube must have shape (nread, 2048, 2048|2560)")
    nread = raw.shape[0]
    if nread < 2:
        raise ValueError("At least two reads are required")
    hdr = fits.Header() if header is None else header.copy()
    if verbose:
        _log(
            f"Processing ramp: shape={raw.shape}, "
            f"chip={hdr.get('CHIP', 'unknown')}, nread={nread}"
        )

    pre_bad = _detect_bad_reads(raw, debug=debug)
    if verbose:
        rejected = np.flatnonzero(pre_bad) + 1
        _log(
            "Initial reference RMS reads rejected: "
            + (", ".join(map(str, rejected)) if rejected.size else "none")
        )
    if raw.shape[2] == 2560:
        if verbose:
            _log("Applying reference-output and reference-pixel correction")
        cube, mask, ref_bad, _ = reference_correct(raw,hdr,keep_reference=use_reference,
                                                   q3fix=q3fix,verbose=verbose,debug=debug)
        bad_reads = pre_bad | ref_bad
    else:
        cube = raw.astype(np.float32)
        mask = np.zeros((2048, 2048), dtype=np.uint16)
        ref_bad = np.zeros(nread, dtype=bool)
        bad_reads = pre_bad
    if verbose:
        rejected = np.flatnonzero(bad_reads) + 1
        _log(
            "Combined reads rejected: "
            + (", ".join(map(str, rejected)) if rejected.size else "none")
        )
    if (~bad_reads).sum() < 2:
        raise ValueError(
            "Not enough good reads after reference checks: "
            f"{(~bad_reads).sum()} of {nread} remain; "
            f"initial RMS rejected reads "
            f"{(np.flatnonzero(pre_bad) + 1).tolist()}, "
            f"reference-pattern check rejected reads "
            f"{(np.flatnonzero(ref_bad) + 1).tolist()}. "
            "Run with verbose=True, debug=True for per-read statistics."
        )
    cube = _interpolate_bad_reads(cube, bad_reads)
    good_reads = np.flatnonzero(~bad_reads)
    if verbose and bad_reads.any():
        _log(f"Interpolated {bad_reads.sum()} rejected read(s)")

    science_nx = 2048
    if bpm is not None:
        b = np.asarray(bpm)
        bad = b > 0
        mask[bad] |= b[bad].astype(mask.dtype)
        bad_y, bad_x = np.nonzero(bad)
        # The working cube can still contain the 512-column reference output.
        # BPM coordinates apply only to the first 2048 science columns.
        cube[:, bad_y, bad_x] = 0
    if littrow is not None:
        mask[np.asarray(littrow) == 1] |= PIXMASK.getval("LITTROW_GHOST")
    if persistence_mask is not None:
        p = np.asarray(persistence_mask)
        mask[(p & 1) != 0] |= PIXMASK.getval("PERSIST_HIGH")
        mask[(p & 2) != 0] |= PIXMASK.getval("PERSIST_MED")
        mask[(p & 4) != 0] |= PIXMASK.getval("PERSIST_LOW")

    linearity_mode = str(linearity_mode).lower()
    if linearity_mode not in {"idl", "all", "none"}:
        raise ValueError("linearity_mode must be 'idl', 'all', or 'none'")
    if linearity is not None and linearity_mode != "none":
        if verbose:
            _log(
                "Applying detector linearity correction "
                f"(mode={linearity_mode})"
            )
        science = _apply_linearity(cube[:, :, :science_nx],
                                   linearity,mode=linearity_mode,)
        cube[:, :, :science_nx] = science
    elif linearity is not None and verbose:
        _log("Skipping detector linearity correction (mode=none)")
    if dark is not None:
        if verbose:
            _log("Subtracting dark ramp")
        dark_cube = np.asarray(dark, dtype=np.float32)
        if dark_cube.shape[0] < nread or dark_cube.shape[1:] != (2048, 2048):
            raise ValueError("dark must have shape (>=nread, 2048, 2048)")
        cube[:, :, :science_nx] -= dark_cube[:nread]
    else:
        dark_cube = None

    ny, nx = cube.shape[1:]
    sat_info = np.zeros((ny, nx, 3), dtype=np.int32)
    variability = np.zeros((ny, nx), dtype=np.float32)
    sat_error = np.zeros((ny, nx), dtype=np.float32)
    median_dcounts = np.zeros((ny, nx), dtype=np.float32)
    cosmic_rays: list[CosmicRay] = []
    if verbose:
        _log("Scanning detector rows for saturation and cosmic rays")

    for y in range(ny):
        if verbose and (y == 0 or (y + 1) % 256 == 0 or y == ny - 1):
            _log(f"  row {y + 1:4d}/{ny}")
        ramp = cube[:, y, :].T.astype(np.float32)
        saturated = ramp > saturation_level
        sat_pixels = np.flatnonzero(saturated.any(axis=1))
        for x in sat_pixels:
            first = int(np.flatnonzero(saturated[x])[0])
            ramp[x, first:] = np.nan
            sat_info[y, x] = (1, first, nread - first)
            if x < science_nx:
                mask[y, x] |= PIXMASK.getval("SATPIX")

        dcounts = np.diff(ramp, axis=1)
        if detect_cosmic_rays and nread > 2:
            dcounts, med, row_var, events = detect_and_fix_cosmic_rays(
                dcounts,sat_info[y],
                noise=float(np.nanmedian(readnoise) if readnoise is not None else 12.0)
                * np.sqrt(2.0),fix=fix_cosmic_rays)
            variability[y] = row_var
            for event in events:
                event.y = y
                cosmic_rays.append(event)
                if event.x < science_nx:
                    mask[y, event.x] |= PIXMASK.getval("CRPIX")
        else:
            med = np.nanmedian(dcounts, axis=1)

        ngood = np.isfinite(dcounts).sum(axis=1)
        threshold = 1 if fix_three_read_saturation and nread == 3 else 2
        unfixable = (sat_info[y, :, 0] == 1) & (ngood < threshold)
        dcounts[unfixable] = 0.0
        unfixable_science = np.flatnonzero(unfixable)
        unfixable_science = unfixable_science[unfixable_science < science_nx]
        mask[y, unfixable_science] |= PIXMASK.getval("UNFIXABLE")
        for x in np.flatnonzero((sat_info[y, :, 0] == 1) & ~unfixable):
            start = max(sat_info[y, x, 1] - 1, 0)
            if fix_saturation:
                dcounts[x, start:] = med[x]
                sigma = variability[y, x] * max(med[x], 1e-4)
                sat_error[y, x] = sigma * sat_info[y, x, 2]
            else:
                dcounts[x, start:] = 0.0

        first = np.nan_to_num(ramp[:, 0], nan=0.0)
        science_first = first[:science_nx]
        science_first[(mask[y] & PIXMASK.getval("UNFIXABLE")) != 0] = 0
        fixed = np.empty_like(ramp)
        fixed[:, 0] = first
        fixed[:, 1:] = first[:, None] + np.cumsum(dcounts, axis=1)
        # AP3DPROC stores ROUND(slice_fixed) back into its LONG cube before
        # fitting the ramp. This affects every pixel, not only CR/saturation.
        cube[:, y, :] = np.rint(fixed.T)
        median_dcounts[y] = np.nanmedian(dcounts, axis=1)

    # The IDL neighbor iteration contains a likely diagonal-only condition
    # ``j ne ix and k ne iy``.  Do not silently reproduce that ambiguity.
    if iterative_cosmic_rays:
        raise NotImplementedError(
            "Iterative neighboring-pixel CR detection should be ported only "
            "after deciding whether IDL's diagonal-only neighbor test is intended"
        )

    gain_image = (
        np.ones((ny, science_nx), dtype=np.float32)
        if gain is None
        else _expand_quadrants(gain, (ny, science_nx))
        if np.asarray(gain).size == 4
        else np.broadcast_to(gain, (ny, science_nx)).astype(np.float32)
    )
    noise_image = (
        np.full((ny, science_nx), 12.0, dtype=np.float32)
        if readnoise is None
        else _expand_quadrants(readnoise, (ny, science_nx))
        if np.asarray(readnoise).size == 4
        else np.broadcast_to(readnoise, (ny, science_nx)).astype(np.float32)
    )

    if not up_the_ramp:
        image, sample_noise, nf = fowler_sampling(cube,good_reads,
                                                  noise_image,nfowler=nfowler)
        if verbose:
            _log(f"Collapsing ramp with Fowler sampling (Nfowler={nf})")
    else:
        if verbose:
            _log(
                f"Collapsing ramp with up-the-ramp sampling "
                f"({good_reads.size} good reads)"
            )
        image, sample_noise = up_the_ramp_sampling(cube,good_reads,noise_image,
                                                gain_image,science_nx=science_nx)

    if use_reference and image.shape[1] == 2560:
        ref = image[:, 2048:].copy()
        # IDL uses MEDIAN(ref, DIM=1) without /EVEN. Because the reference
        # output has 512 columns, it selects the upper middle value rather
        # than averaging the central pair as np.median would.
        ref_profile = utils.idl_median(ref, axis=1).astype(np.float32)
        smooth_profile = utils.idl_median_filter_1d(ref_profile[None, :], 7,
                                                    edge_copy=False)[0]
        ref -= smooth_profile[:, None]
        science = image[:, :2048].copy()
        _reference_subtract_image(science, ref)
        image = science
    else:
        image = image[:, :science_nx]

    pmodel = None if persistence_model is None else np.asarray(
        persistence_model, dtype=np.float32
    )
    if pmodel is not None:
        if verbose:
            _log("Subtracting supplied persistence model")
        if pmodel.shape != image.shape:
            raise ValueError("persistence_model must match the 2-D science image")
        image -= pmodel

    variance = np.zeros_like(image, dtype=np.float32)
    if not up_the_ramp:
        variance += np.maximum(image + (pmodel if pmodel is not None else 0), 0) / gain_image
    if dark_cube is not None:
        variance += np.maximum(dark_cube[nread - 1], 0) / gain_image
    variance += sample_noise**2
    if fix_saturation:
        variance += sat_error[:, :science_nx]
    else:
        variance[sat_info[:, :science_nx, 0] != 0] = BAD_VARIANCE
    variance[(mask & PIXMASK.getval("UNFIXABLE")) != 0] = BAD_VARIANCE
    if fix_cosmic_rays:
        for event in cosmic_rays:
            if event.x < science_nx:
                variance[event.y, event.x] += event.fix_error
    else:
        variance[(mask & PIXMASK.getval("CRPIX")) != 0] = BAD_VARIANCE
    variance[(mask & PIXMASK.getval("BADPIX")) != 0] = BAD_VARIANCE

    if flat is not None:
        if verbose:
            _log("Applying flat-field correction")
        flat_image = np.asarray(flat, dtype=np.float32)
        if flat_image.shape != image.shape:
            raise ValueError("flat must match the 2-D science image")
        image /= flat_image
        variance /= flat_image**2
    if output_electrons:
        image *= gain_image
        variance *= gain_image**2

    finite_var = variability[:, :science_nx][(sat_info[:, :science_nx, 0] == 0)
                                             & ((mask & PIXMASK.getval("CRPIX")) == 0)
                                             & (median_dcounts[:, :science_nx] > 40)]
    if finite_var.size == 0:
        finite_var = variability[:, :science_nx][(sat_info[:, :science_nx, 0] == 0)
                                                 & (median_dcounts[:, :science_nx] > 20)]
    global_variability = float(np.median(finite_var)) if finite_var.size else -1.0
    error = np.maximum(np.sqrt(variance), 1.0)
    _update_header(hdr,nread=nread,gain=float(np.median(gain_image)),
        # AP3DPROC uses MEDIAN([sample_noise]) without /EVEN. The helper also
        # ignores the minority of non-finite per-pixel noise estimates.
        readnoise=float(utils.idl_median(sample_noise[np.isfinite(sample_noise)])),
        up_the_ramp=up_the_ramp,
        nfowler=None if up_the_ramp else nf,
        global_variability=global_variability,
        output_electrons=output_electrons)
    if verbose:
        nsat = int(np.count_nonzero(mask & PIXMASK.getval("SATPIX")))
        nunfix = int(np.count_nonzero(mask & PIXMASK.getval("UNFIXABLE")))
        nbad = int(np.count_nonzero(mask & PIXMASK.getval("BADPIX")))
        _log(
            f"Reduction summary: bad={nbad}, CR={len(cosmic_rays)}, "
            f"saturated={nsat}, unfixable={nunfix}, "
            f"global variability={global_variability:.4f}"
        )
    processing_seconds = perf_counter() - processing_started
    hdr["AP3DTIME"] = (
        processing_timestamp.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "UTC start of Python AP3D processing")
    hdr["AP3DSEC"] = (processing_seconds,
                      "Python AP3D array processing time (seconds)")
    if verbose:
        _log(f"Completed array processing in {processing_seconds:.2f} s",
            processing_started)
    return ProcessResult(flux=image.astype(np.float32),error=error.astype(np.float32),
                         mask=mask,header=hdr,cosmic_rays=cosmic_rays,
                         saturation=sat_info[:, :science_nx],
                         fixed_cube=cube if return_cube else None,
                         persistence_model=pmodel,read_mask=bad_reads,
                         global_variability=global_variability)


def _update_header(
    header: fits.Header,
    *,
    nread: int,
    gain: float,
    readnoise: float,
    up_the_ramp: bool,
    nfowler: int | None,
    global_variability: float,
    output_electrons: bool,
) -> None:
    """Record sampling, noise, timing, and output-unit metadata in-place."""

    header["AP3DVER"] = (AP3D_VERSION, "Python AP3D implementation version")
    header["GAIN"] = (gain, "Median gain in electron/ADU")
    header["RDNOISE"] = (readnoise, "Median sampled read noise")
    header.add_history("AP3D Python 3-D to 2-D processing")
    header.add_history("Up-the-ramp sampling" if up_the_ramp
                       else f"Fowler sampling, Nfowler={nfowler}")
    header.add_history(
        f"AP3D: Global fractional variability = {global_variability:.3f}"
    )
    if header.get("NFRAMES") != nread:
        header["EXPTIME"] = nread * 10.647
    if "DATE-OBS" in header and "EXPTIME" in header:
        mid = Time(header["DATE-OBS"], format="isot", scale="utc") + TimeDelta(
            0.5 * float(header["EXPTIME"]), format="sec"
        )
        header["UT-MID"] = (mid.isot, "Date at midpoint of exposure")
        header["JD-MID"] = (mid.jd, "JD at midpoint of exposure")
    header["BUNIT"] = "electron" if output_electrons else "ADU"
    for key in ("CHECKSUM", "DATASUM"):
        if key in header:
            del header[key]


def read_ramp(
    filename: str | Path,
    max_read: int | None = None,
    *,
    verbose: bool = False,
) -> tuple[np.ndarray, fits.Header]:
    """Read a decompressed APOGEE 3-D FITS ramp."""

    filename = Path(filename)
    if filename.suffix.lower() == ".apz":
        raise ValueError(
            "read_ramp() requires a decompressed FITS file; "
            "use process_file() for APZ inputs"
        )

    # Raw APOGEE reads use FITS BZERO/BSCALE to represent unsigned integers.
    # Astropy cannot apply that scaling with strict memory mapping enabled.
    with fits.open(filename, memmap=False, uint=True) as hdul:
        header = hdul[0].header.copy()
        if hdul[0].data is not None and hdul[0].data.ndim == 3:
            cube = np.asarray(hdul[0].data)
        else:
            image_hdus = [hdu for hdu in hdul[1:] if hdu.data is not None]
            images = [hdu.data for hdu in image_hdus]
            if max_read is not None:
                images = images[:max_read]
            if len(images) < 2:
                raise ValueError("Ramp file contains fewer than two image reads")
            cube = np.stack(images)
            # APZ primary headers can omit detector metadata that is retained
            # in the compressed-image extensions.
            if image_hdus:
                header.extend(image_hdus[0].header, update=True)
    if max_read is not None:
        cube = cube[:max_read]
    return cube, header


def load_raw_ramp(
    filename,
    *,
    max_read=None,
    temporary_directory=None,
    keep_temporary=False,
    unlock=False,
    verbose=False,
):
    """Load an APOGEE raw ramp, decompressing APZ input when necessary.

    Parameters
    ----------
    filename : str or Path
        Raw FITS or APZ ramp.
    max_read : int, optional
        Maximum number of reads to load.
    temporary_directory : str or Path, optional
        Directory used for APZ decompression.
    keep_temporary : bool, optional
        Keep the unpacked fits file. Default is False.
    unlock : bool, optional
        Clear an existing decompression lock.
    verbose : bool, optional
        Print decompression information.

    Returns
    -------
    cube : ndarray
        Raw ramp with shape ``(nread, ny, nx)``.
    header : fits.Header
        Raw ramp header.
    """
    input_file = Path(filename)
    temporary_file = None
    created_temporary = False

    try:
        if input_file.suffix.lower() == ".apz":
            if temporary_directory is None:
                temporary_directory = Path(utils.localdir()) / "ap3d"
            else:
                temporary_directory = Path(temporary_directory)

            temporary_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_file = (
                temporary_directory
                / f"{input_file.stem}.fits"
            )

            if not temporary_file.exists():
                created_temporary = True
                apzip.unzip(
                    str(input_file),
                    fitsdir=str(temporary_directory),
                    unlock=unlock,
                    silent=not verbose,
                )

            if not temporary_file.exists():
                raise RuntimeError(
                    "APZ decompression did not create "
                    f"{temporary_file}"
                )

            ramp_file = temporary_file

        else:
            ramp_file = input_file

        return read_ramp(
            ramp_file,
            max_read=max_read,
            verbose=verbose,
        )

    finally:
        if (created_temporary and temporary_file is not None
            and temporary_file.exists() and not keep_temporary):
            temporary_file.unlink()

            
def process_cube(
    filename: str | Path,
    *,
    max_read: int | None = None,
    verbose: bool = False,
    debug: bool = False,
    **options: Any,
) -> ProcessResult:
    """Process one decompressed APOGEE 3-D FITS ramp.

    APZ decompression intentionally belongs to :func:`process_file`, the
    orchestration layer. Calibration arrays and reduction settings are passed
    through ``options`` to :func:`process_array`. With ``verbose=True``, ramp
    loading and total elapsed time are reported with UTC timestamps.
    """

    started = perf_counter()
    filename = Path(filename)
    if filename.suffix.lower() == ".apz":
        raise ValueError("process_cube() requires a decompressed FITS file; "
                         "use process_file() for APZ inputs")
    if verbose:
        _log(f"Reading decompressed ramp: {filename}", started)
    cube, header = read_ramp(filename, max_read=max_read, verbose=verbose)
    if verbose:
        _log(f"Loaded ramp with shape {cube.shape} and dtype {cube.dtype}",
            started)
    result = process_array(cube,header,verbose=verbose,debug=debug,**options)
    if verbose:
        _log("Finished decompressed-ramp processing", started)
    return result


def read_calibrations(
    *,
    detector: str | Path | None = None,
    bpm: str | Path | None = None,
    dark: str | Path | None = None,
    flat: str | Path | None = None,
    littrow: str | Path | None = None,
    persistence_mask: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Read the calibration HDUs consumed by :func:`process_cube`."""

    result: dict[str, np.ndarray] = {}
    if detector is not None:
        with fits.open(detector, memmap=True) as hdul:
            result["readnoise"] = np.asarray(hdul[1].data)
            result["gain"] = np.asarray(hdul[2].data)
            linearity = np.asarray(hdul[3].data)
            # FITS reverses the IDL (x,y,coefficient) axis order.
            if linearity.ndim == 3 and linearity.shape[0] == 3:
                linearity = np.moveaxis(linearity, 0, -1)
            # Per-output files are IDL (output, coefficient), which Astropy
            # reads as (coefficient, output).
            elif (
                linearity.ndim == 2
                and linearity.shape[0] != 4
                and linearity.shape[1] == 4
            ):
                linearity = linearity.T
            result["linearity"] = linearity
    for name, filename in (
        ("bpm", bpm),
        ("flat", flat),
        ("littrow", littrow),
        ("persistence_mask", persistence_mask),
    ):
        if filename is not None:
            result[name] = fits.getdata(filename)
    if dark is not None:
        with fits.open(dark, memmap=True) as hdul:
            # Normal apDark products store the complete dark ramp as a
            # 3-D image (often in extension 1) and can have additional HDUs
            # with unrelated dimensions.  Use the first 3-D image rather
            # than attempting to stack every nonempty extension.
            dark_cube = next((np.asarray(hdu.data)
                              for hdu in hdul
                              if hdu.data is not None and hdu.data.ndim == 3),None)
            if dark_cube is None:
                # Older products can store one 2-D read per extension.
                reads = [
                    np.asarray(hdu.data)
                    for hdu in hdul[1:]
                    if hdu.data is not None and hdu.data.ndim == 2
                ]
                if len(reads) < 2:
                    raise ValueError(
                        f"Dark calibration {dark} contains neither a 3-D "
                        "ramp nor at least two 2-D read images"
                    )
                shape = reads[0].shape
                matching_reads = [read for read in reads if read.shape == shape]
                if len(matching_reads) != len(reads):
                    raise ValueError(
                        f"Dark calibration {dark} has 2-D image extensions "
                        "with inconsistent shapes"
                    )
                dark_cube = np.stack(matching_reads)
            result["dark"] = dark_cube
    return result


def _add_provenance_header(
    result: ProcessResult,
    *,
    output: str | Path,
    detector: str | Path | None,
    bpm: str | Path | None,
    dark: str | Path | None,
    flat: str | Path | None,
    littrow: str | Path | None,
    persistence_mask: str | Path | None,
    up_the_ramp: bool,
    nfowler: int,
    fix_cosmic_rays: bool,
    fix_saturation: bool,
) -> None:
    """Add IDL-compatible software, calibration, and processing provenance."""

    header = result.header
    reduction_version = utils.reduction_version()
    header["V_APRED"] = (utils.software_version(), "apogee software version")
    header["APRED"] = (reduction_version, "apogee reduction version")
    header["LONGSTRN"] = ("OGIP 1.0",
                          "The OGIP long string convention may be used.")

    calibrations = (("BPMFILE", bpm, "bpm file", "BAD PIXEL MASK"),
                    ("DETFILE", detector, "det file", "DETECTOR"),
                    ("DARKFILE", dark, "dark file", "Dark Current Correction"),
                    ("FLATFILE", flat, "flat file", "Flat Field Correction"),
                    ("LITTROW", littrow, "littrow file", "Littrow ghost mask"),
                    ("PERSIST",persistence_mask,"persist mask file",
                     "Persistence mask"))
    for keyword, filename, comment, label in calibrations:
        if filename is not None:
            value = str(filename)
            header[keyword] = (value, comment)
            header.add_history(f'AP3D: {label} file="{value}"')

    unit = "electrons" if header.get("BUNIT") == "electron" else "ADU"
    header.add_history("AP3D: "
                       + datetime.now().astimezone().strftime("%a %b %d %H:%M:%S %Y"))
    header.add_history(f"AP3D: {getpass.getuser()} on {socket.gethostname()}")
    header.add_history(f"AP3D: Python {platform.python_version()} "
                       f"{platform.system().lower()} {platform.machine()}")
    header.add_history(
        f"AP3D: APOGEE Reduction Pipeline Version: {reduction_version}")
    header.add_history(f"AP3D: Output File: {output}")
    header.add_history(f"AP3D: HDU1 - image ({unit})")
    header.add_history(f"AP3D: HDU2 - error ({unit})")
    header.add_history("AP3D: HDU3 - flag mask")
    header.add_history("AP3D:        1 - bad pixels")
    header.add_history("AP3D:        2 - cosmic ray")
    header.add_history("AP3D:        4 - saturated")
    header.add_history("AP3D:        8 - unfixable")
    if result.persistence_model is not None:
        header.add_history("AP3D: HDU4 - persistence correction (ADU)")

    nbad = int(np.count_nonzero(result.mask & PIXMASK.getval("BADPIX")))
    ncr = int(np.count_nonzero(result.mask & PIXMASK.getval("CRPIX")))
    nsat = int(np.count_nonzero(result.mask & PIXMASK.getval("SATPIX")))
    nunfixable = int(np.count_nonzero(result.mask & PIXMASK.getval("UNFIXABLE")))
    nfixed_saturation = nsat - nunfixable if fix_saturation else 0
    header.add_history(f"AP3D: {nbad} pixels are bad")
    header.add_history(f"AP3D: {ncr} pixels have cosmic rays")
    if fix_cosmic_rays:
        header.add_history("AP3D: Cosmic Rays FIXED")
    header.add_history(f"AP3D: {nsat} pixels are saturated")
    if fix_saturation:
        header.add_history(f"AP3D: {nfixed_saturation} saturated pixels FIXED")
    header.add_history(f"AP3D: {nunfixable} pixels are unfixable")
    if up_the_ramp:
        header.add_history("AP3D: UP-THE-RAMP Sampling")
    else:
        header.add_history(f"AP3D: Fowler Sampling, Nfowler={nfowler}")


def write_ap2d(
    filename: str | Path,
    result: ProcessResult,
    *,
    overwrite: bool = False,
    integer_output: bool = False,
) -> None:
    """Write an APOGEE-style ap2D product.

    Nonfinite flux values are written as zero to match the operational IDL
    product convention. Their validity remains encoded in the pixel mask.
    Nonfinite uncertainties are written as ``1e10`` to reproduce the IDL
    ``errout`` convention. This differs from an ordinary flagged pixel, whose
    finite variance is ``BAD_VARIANCE`` and whose error is therefore about
    ``1e4``.
    """

    flux = np.nan_to_num(result.flux,nan=0.0,posinf=0.0,neginf=0.0,)
    error = np.nan_to_num(result.error,nan=NONFINITE_ERROR,
                          posinf=NONFINITE_ERROR,neginf=NONFINITE_ERROR)
    if integer_output:
        flux = np.rint(flux).astype(np.int32)
        error = np.rint(error).astype(np.int32)
    unit = result.header.get("BUNIT", "ADU")
    hdus: list[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU] = [
        fits.PrimaryHDU(header=result.header),
        fits.ImageHDU(flux, name="FLUX"),
        fits.ImageHDU(error, name="ERROR"),
        fits.ImageHDU(result.mask.astype(np.uint16), name="MASK"),
    ]
    hdus[1].header["BUNIT"] = (unit, "Flux unit")
    hdus[2].header["BUNIT"] = (unit, "Uncertainty unit")
    hdus[3].header["BUNIT"] = ("bitwise", "Pixel flag mask")
    hdus[3].header.add_history("Explanation of BITWISE flag mask")
    hdus[3].header.add_history(" 1 - bad pixels")
    hdus[3].header.add_history(" 2 - cosmic ray")
    hdus[3].header.add_history(" 4 - saturated")
    hdus[3].header.add_history(" 8 - unfixable")
    if result.persistence_model is not None:
        hdus.append(
            fits.ImageHDU(result.persistence_model, name="PERSIST CORRECTION"))
        hdus[-1].header["BUNIT"] = ("ADU", "Persistence correction")
    fits.HDUList(hdus).writeto(filename, overwrite=overwrite, checksum=True)


def process_file(
    filename: str | Path,
    output: str | Path,
    *,
    detector: str | Path | None = None,
    bpm: str | Path | None = None,
    dark: str | Path | None = None,
    flat: str | Path | None = None,
    littrow: str | Path | None = None,
    persistence_mask: str | Path | None = None,
    overwrite: bool = False,
    max_read: int | None = None,
    verbose: bool = False,
    debug: bool = False,
    **options: Any,
) -> ProcessResult:
    """Reduce a raw APOGEE ramp and write an ``ap2D`` FITS product.

    This orchestration entry point loads calibrations, decompresses APZ input
    with :mod:`apogee_drp.utils.apzip`, reduces the resulting 3-D FITS ramp,
    writes the multi-extension product, and removes the temporary FITS file.
    Extra reduction settings, including ``linearity_mode``, are forwarded to
    :func:`process_array` through ``options``.

    With ``verbose=True``, messages contain UTC timestamps and cumulative
    elapsed times. The final time includes calibration I/O, APZ decompression,
    numerical processing, and output writing.
    """

    started = perf_counter()
    input_file = Path(filename)
    if verbose:
        _log(f"Input raw ramp: {input_file}", started)
        for label, value in (
            ("detector", detector),
            ("BPM", bpm),
            ("dark", dark),
            ("flat", flat),
            ("Littrow", littrow),
            ("persistence mask", persistence_mask),
        ):
            if value is not None:
                _log(f"Using {label}: {value}", started)
    calibrations = read_calibrations(detector=detector,bpm=bpm,
                                     dark=dark,flat=flat,littrow=littrow,
                                     persistence_mask=persistence_mask)
    if verbose:
        shapes = ", ".join(
            f"{name}={np.shape(value)}" for name, value in calibrations.items()
        )
        _log(f"Calibration shapes: {shapes}", started)

    temporary_file = None
    try:
        cube_file = input_file
        if input_file.suffix.lower() == ".apz":
            if verbose:
                _log("Starting APZ decompression", started)

            temporary_directory = Path(utils.localdir())
            temporary_directory.mkdir(parents=True, exist_ok=True)
            cube_file = temporary_directory / f"{input_file.stem}.fits"
            temporary_file = cube_file

            apzip.unzip(str(input_file), clobber=True, delete=False, silent=not verbose,
                        fitsdir=str(temporary_directory))

            if not cube_file.exists():
                raise RuntimeError(f"APZ decompression did not create expected file {cube_file}")

            if verbose:
                _log("Finished APZ decompression", started)

        result = process_cube(cube_file, max_read=max_read, **calibrations,
                              verbose=verbose, debug=debug, **options)

    finally:
        if temporary_file is not None and temporary_file.exists():
            temporary_file.unlink()

    _add_provenance_header(result,output=output,detector=detector,bpm=bpm,
                           dark=dark,flat=flat,littrow=littrow,
                           persistence_mask=persistence_mask,
                           up_the_ramp=bool(options.get("up_the_ramp", False)),
                           nfowler=int(options.get("nfowler", 10)),
                           fix_cosmic_rays=bool(options.get("fix_cosmic_rays", True)),
                           fix_saturation=bool(options.get("fix_saturation", True)))
    if verbose:
        _log(f"Writing ap2D product: {output}", started)
    write_ap2d(output, result, overwrite=overwrite)
    if verbose:
        _log(f"Finished processing {input_file.name}", started)
    return result


def _plan_scalar(value: Any, default: Any = None) -> Any:
    """Return a scalar plan value, decoding NumPy byte strings when needed."""

    if value is None:
        return default
    array = np.asarray(value)
    if array.size == 0:
        return default
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, bytes):
        scalar = scalar.decode().strip()
    return scalar


def _record_value(record: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a mapping or a NumPy structured record."""

    if isinstance(record, Mapping):
        return record.get(name, default)
    names = getattr(getattr(record, "dtype", None), "names", None)
    if names and name in names:
        return record[name]
    return getattr(record, name, default)


def _calibration_id(plan_data: Mapping[str, Any], name: str) -> Any | None:
    """Normalize a plan calibration ID, returning ``None`` when disabled."""

    value = _plan_scalar(plan_data.get(name))
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in ("", "none", "null", "0"):
            return None
    try:
        if int(value) == 0:
            return None
    except (TypeError, ValueError):
        pass
    return value


def _plan_bool(value: Any, default: bool = False) -> bool:
    """Interpret plan booleans stored as numbers, strings, or NumPy scalars."""

    scalar = _plan_scalar(value, default)
    if isinstance(scalar, str):
        normalized = scalar.strip().lower()
        if normalized in ("", "0", "false", "f", "no", "n", "off", "none"):
            return False
        if normalized in ("1", "true", "t", "yes", "y", "on"):
            return True
    return bool(scalar)


def _chip_filename(base: str | Path, root: str, chip: str) -> str:
    """Insert an APOGEE chip tag into a filename returned by ``ApLoad``."""

    path = Path(base)
    marker = f"{root}-"
    if marker not in path.name:
        raise ValueError(
            f"Cannot insert chip {chip!r}: {path.name!r} lacks {marker!r}"
        )
    return str(path.with_name(path.name.replace(marker, f"{root}-{chip}-", 1)))


def _flavor_options(flavor: str, *, single_plate: bool = False) -> dict[str, Any]:
    """Return the legacy ``ap3d.pro`` processing choices for an exposure."""

    flavor = str(flavor).strip().lower()
    common = {"fix_saturation": True,
              "fix_three_read_saturation": False,
              "iterative_cosmic_rays": False}
    if flavor == "psf":
        return dict(common, **{
            "detect_cosmic_rays": False,
            "fix_cosmic_rays": False,
            "up_the_ramp": False,
            "nfowler": 1,
        })
    if flavor in ("lamp", "wave"):
        return dict(common, **{
            "detect_cosmic_rays": True,
            "fix_cosmic_rays": True,
            "up_the_ramp": False,
            "nfowler": 1,
        })
    if flavor == "object":
        return dict(common, **{
            "detect_cosmic_rays": not single_plate,
            "fix_cosmic_rays": True,
            "up_the_ramp": True,
            "nfowler": 0,
        })
    if flavor == "flux":
        return dict(common, **{
            "detect_cosmic_rays": False,
            "fix_cosmic_rays": False,
            "up_the_ramp": False,
            "nfowler": 1,
        })
    if flavor == "dark":
        raise ValueError("Dark ramps must be processed by the dark-calibration code")
    raise ValueError(f"Unsupported AP3D exposure flavor {flavor!r}")


def ap3d(
    planfiles: str | Path | Sequence[str | Path],
    *,
    verbose: bool = False,
    debug: bool = False,
    clobber: bool = False,
    calclobber: bool = False,
    unlock: bool = False,
    make_calibrations: bool = False,
    continue_on_error: bool = False,
    plan_loader: Callable[..., Mapping[str, Any]] | None = None,
    load_factory: Callable[..., Any] | None = None,
    calibration_builder: Callable[..., Any] | None = None,
    **process_options: Any,
) -> list[PlanProcessRecord]:
    """Process all exposures and detector chips in APOGEE plan files.

    This is the Python orchestration counterpart of the IDL ``ap3d.pro``.
    It loads each plan, constructs APOGEE filenames with
    :class:`~apogee_drp.utils.apload.ApLoad`, selects the legacy reduction
    options for each exposure flavor, applies chip-specific calibrations, and
    calls :func:`process_file`.

    Calibration creation is optional because it belongs to the separate
    ``makecal`` subsystem. Set ``make_calibrations=True`` to invoke it before
    processing. The injectable loader/factory/builder arguments support unit
    tests and alternate orchestration environments.

    Parameters
    ----------
    planfiles
        One plan filename or a sequence of ``.par``/``.yaml`` plan files.
    clobber
        Replace existing ap2D products.
    calclobber
        Allow the calibration builder to replace existing calibrations.
    unlock
        Forward the unlock request to calibration creation.
    make_calibrations
        Build required detector, BPM, dark, flat, Littrow, and persistence
        products before reducing exposures.
    continue_on_error
        Record failed chips and continue. By default, the first failure is
        raised immediately.
    process_options
        Explicit numerical options forwarded to :func:`process_file`; these
        override the flavor defaults and plan settings.

    Returns
    -------
    records
        One lightweight :class:`PlanProcessRecord` per processed, skipped, or
        failed chip.
    """

    if isinstance(planfiles, (str, Path)):
        plan_paths = [Path(planfiles)]
    else:
        plan_paths = [Path(item) for item in planfiles]
    if not plan_paths:
        raise ValueError("At least one plan file is required")

    if plan_loader is None or load_factory is None:
        from apogee_drp.utils import apload as _apload
        from apogee_drp.utils import plan as _plan

        if plan_loader is None:
            plan_loader = _plan.load
        if load_factory is None:
            load_factory = _apload.ApLoad
    if make_calibrations and calibration_builder is None:
        from apogee_drp.apred.cal.makecal import makecal

        calibration_builder = makecal

    records: list[PlanProcessRecord] = []
    overall_started = perf_counter()
    if verbose:
        _log(f"Running AP3D wrapper for {len(plan_paths)} plan file(s)")

    for plan_index, plan_path in enumerate(plan_paths):
        plan_started = perf_counter()
        if verbose:
            _log(
                f"Plan {plan_index + 1}/{len(plan_paths)}: {plan_path}",
                overall_started,
            )
        plan_data = plan_loader(str(plan_path), np=True, verbose=verbose)
        if plan_data is None:
            raise RuntimeError(f"Plan loader returned no data for {plan_path}")
        apred = str(_plan_scalar(plan_data.get("apred_vers"), "daily"))
        telescope = str(_plan_scalar(plan_data.get("telescope"), "apo25m"))
        load = load_factory(apred=apred, telescope=telescope)
        exposures = plan_data.get("APEXP")
        if exposures is None or len(exposures) == 0:
            if verbose:
                _log(f"No exposures in {plan_path}", plan_started)
            continue

        cal_ids = {
            "detector": _calibration_id(plan_data, "detid"),
            "bpm": _calibration_id(plan_data, "bpmid"),
            "dark": _calibration_id(plan_data, "darkid"),
            "flat": _calibration_id(plan_data, "flatid"),
            "littrow": _calibration_id(plan_data, "littrowid"),
            "persistence_mask": _calibration_id(plan_data, "persistid"),
        }
        if make_calibrations:
            cal_types = {"detector": "det","bpm": "bpm","dark": "dark",
                         "flat": "flat","littrow": "littrow",
                         "persistence_mask": "persist"}
            for name, cal_id in cal_ids.items():
                if cal_id is not None:
                    calibration_builder(cal_id,cal_types[name],
                        load=load,clobber=calclobber,
                        unlock=unlock,verbose=verbose)

        cal_roots = {"detector": "Detector","bpm": "BPM","dark": "Dark",
                     "flat": "Flat","littrow": "Littrow",
                     "persistence_mask": "Persist"}
        cal_bases = {
            name: (
                None
                if cal_id is None
                else load.filename(root, num=cal_id, chips=True)
            )
            for (name, cal_id), root in zip(cal_ids.items(), cal_roots.values())
        }

        plan_plate_type = str(_plan_scalar(plan_data.get("platetype"), "")).strip().lower()
        mjd = int(_plan_scalar(plan_data.get("mjd"), 0))
        plan_reference = _plan_bool(plan_data.get("usereference"), True)
        plan_max_read = _plan_scalar(plan_data.get("maxread"))
        plan_q3fix = _plan_bool(plan_data.get("q3fix"), False)

        # Loop over exposures for this plan file
        for exposure_index, exposure in enumerate(exposures):
            exposure_started = perf_counter()
            number = int(_plan_scalar(_record_value(exposure, "name")))
            flavor = str(
                _plan_scalar(_record_value(exposure, "flavor"), "")
            ).strip().lower()
            settings = _flavor_options(flavor, single_plate=plan_plate_type == "single")
            settings.update({"linearity_mode": "idl",
                             "use_reference": plan_reference})
            if plan_max_read is not None:
                settings["max_read"] = int(plan_max_read)
            settings.update(process_options)

            raw_base = load.filename("R", num=number, mjd=mjd, chips=True)
            output_base = load.filename("2D", num=number, mjd=mjd, chips=True)
            if verbose:
                _log(f"Exposure {exposure_index + 1}/{len(exposures)}: "
                    f"{number:08d} ({flavor})",plan_started)

            # Loop over the three chips
            for chip in ("a", "b", "c"):
                raw_file = _chip_filename(raw_base, "R", chip)
                output_file = _chip_filename(output_base, "2D", chip)
                chip_started = perf_counter()
                # skipping chip
                if Path(output_file).exists() and not clobber:
                    chip_elapsed = perf_counter() - chip_started
                    records.append(PlanProcessRecord(str(plan_path), number, flavor,
                                                     chip,raw_file, output_file,
                                                     "skipped", chip_elapsed))
                    if verbose:
                        _log(f"Skipped chip {chip} in {chip_elapsed:.1f} s")
                    continue
                calibration_files = {}
                for name, base in cal_bases.items():
                    if base is None or (name == "littrow" and chip != "b"):
                        calibration_files[name] = None
                    else:
                        calibration_files[name] = _chip_filename(
                            base, cal_roots[name], chip
                        )
                missing = [
                    path
                    for path in calibration_files.values()
                    if path is not None and not Path(path).exists()
                ]
                if not Path(raw_file).exists():
                    missing.insert(0, raw_file)
                if missing:
                    error = "Required input file(s) missing: " + ", ".join(missing)
                    chip_elapsed = perf_counter() - chip_started
                    if verbose:
                        _log(f"Failed chip {chip} after {chip_elapsed:.1f} s")
                    if not continue_on_error:
                        raise FileNotFoundError(error)
                    records.append(PlanProcessRecord(str(plan_path), number, flavor,
                                                     chip,raw_file, output_file,
                                                     "failed", chip_elapsed, error))
                    continue

                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                chip_settings = settings.copy()
                chip_settings["q3fix"] = bool(
                    chip == "c"
                    and (plan_q3fix or 56_930 < mjd < 57_600)
                )
                max_read = chip_settings.pop("max_read", None)
                try:
                    process_file(raw_file,output_file,overwrite=clobber,
                                 max_read=max_read,verbose=verbose,
                                 debug=debug,**calibration_files,**chip_settings)
                    if verbose:
                        _log(" ")
                except Exception as exc:
                    chip_elapsed = perf_counter() - chip_started
                    if verbose:
                        _log(f"Failed chip {chip} after {chip_elapsed:.1f} s")
                    if not continue_on_error:
                        raise
                    records.append(PlanProcessRecord(str(plan_path), number, flavor,
                                                     chip, raw_file, output_file,
                                                     "failed", chip_elapsed, str(exc)))
                else:
                    chip_elapsed = perf_counter() - chip_started
                    records.append(PlanProcessRecord(str(plan_path), number, flavor,
                                                     chip, raw_file, output_file,
                                                     "processed", chip_elapsed))
                    if verbose:
                        _log(f"Finished chip {chip} in {chip_elapsed:.1f} s")
                        
            # exposure elapsed time
            exposure_elapsed = perf_counter() - exposure_started
            if verbose:
                _log(f"Finished exposure {number:08d} in {exposure_elapsed:.1f} s")
                
        # plan file elapsed time
        if verbose:
            plan_elapsed = perf_counter() - plan_started
            _log(f"Finished plan {plan_path} in {plan_elapsed:.1f} s")

    if verbose:
        nprocessed = sum(item.status == "processed" for item in records)
        nskipped = sum(item.status == "skipped" for item in records)
        nfailed = sum(item.status == "failed" for item in records)
        _log(f"AP3D wrapper finished: processed={nprocessed}, "
             f"skipped={nskipped}, failed={nfailed}",overall_started)
    return records
