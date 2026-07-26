"""APOGEE raw 3-D ramp to calibrated 2-D image processing.

This is a Python translation of the core algorithms in ``ap3dproc.pro`` and
``aprefcorr.pro``.  Arrays use the normal NumPy/FITS ordering
``(nread, ny, nx)``.  The returned 2-D arrays use ``(ny, nx)``.

The high-level IDL ``ap3d.pro`` routine also creates calibrations, interprets
APOGEE plan files, constructs filenames, and manages pipeline locks.  Those
survey-infrastructure tasks deliberately remain outside this numerical module.
Use :func:`process_file` once those filenames have been resolved.

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
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta
from apogee_drp.utils import apzip
from apogee_drp.utils.bitmask import PixelBitMask
from dlnpyutils import utils as dln
from scipy.ndimage import median_filter

__all__ = [
    "PIXMASK",
    "CosmicRay",
    "ProcessResult",
    "reference_correct",
    "detect_and_fix_cosmic_rays",
    "fowler_sampling",
    "up_the_ramp_sampling",
    "process_array",
    "process_cube",
    "read_ramp",
    "read_calibrations",
    "write_ap2d",
    "process_file",
]


PIXMASK = PixelBitMask()


def _bit(name: str) -> int:
    """Return a named APOGEE pixel-mask value."""

    return int(PIXMASK.getval(name))


BAD_VARIANCE = np.float32(99_999_999.0)


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
    """Calibrated 2-D products and diagnostics returned by the reduction.

    ``flux``, ``error``, and ``mask`` have shape ``(2048, 2048)``.  The
    optional arrays retain intermediate information useful for diagnostics
    when requested by :func:`process_array`.
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


def _rolling_nanmedian(a: np.ndarray, width: int) -> np.ndarray:
    """Median-filter the last axis, ignoring NaNs and copying edge behavior."""

    width = max(1, int(width))
    if width == 1:
        return a.copy()
    left = width // 2
    right = width - left - 1
    padded = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(left, right)], mode="edge")
    # Avoid np.lib.stride_tricks.sliding_window_view, which is unavailable in
    # the NumPy version used by some APOGEE DRP environments.  The read axis is
    # short, so looping over its positions adds negligible overhead.
    result = np.empty_like(a, dtype=np.result_type(a.dtype, np.float32))
    for index in range(a.shape[-1]):
        result[..., index] = np.nanmedian(
            padded[..., index : index + width], axis=-1
        )
    return result


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


def _median_filter_2d(a: np.ndarray, size: int) -> np.ndarray:
    """Apply the closest SciPy equivalent of the IDL 2-D median filter."""

    # IDL MEDIAN(ref, n) sets perimeter behavior differently; nearest is the
    # closest useful scipy equivalent and avoids manufacturing edge zeros.
    return median_filter(a, size=(size, size), mode="nearest")


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
            print(
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
        print(
            "Reference-pattern reads rejected: "
            + (", ".join(map(str, rejected)) if rejected.size else "none")
        )
    np.divide(mean_ref, nref, out=mean_ref, where=nref > 0)

    out = np.empty((nread, 2048, 2048), dtype=np.float32)
    refout = (
        np.empty((nread, 2048, 512), dtype=np.float32)
        if keep_reference
        else None
    )
    cds_ref = data[1, :, :2048].astype(np.float64) if cds and nread > 1 else 0.0
    yfrac = np.arange(2048, dtype=np.float64) / 2048.0
    xfrac = np.arange(2048, dtype=np.float64) / 2048.0
    last_good = nread - 1
    nsat0 = 0

    for i in range(nread):
        red = data[i, :, :2048].astype(np.float64)
        saturated = red > saturation
        nsat = int(saturated.sum())
        if i == 0:
            nsat0 = nsat
        if nsat > nsat0 + 2000 and last_good == nread - 1:
            last_good = i - 1
        mask[red == 0] |= _bit("BADPIX")
        mask[saturated] |= _bit("SATPIX")
        red[saturated] = 65_535

        if read_mask[i]:
            out[i] = np.nan
            if refout is not None:
                refout[i] = np.nan
            continue

        if cds:
            red -= cds_ref
        ref = data[i, :, 2048:2560].astype(np.float64)
        if indiv == 1:
            correction = ref
        elif indiv > 1:
            correction = _median_filter_2d(ref, indiv)
        elif indiv < 0:
            correction = mean_ref
        else:
            correction = None
        if correction is not None:
            _reference_subtract_image(red, correction)
            ref -= correction

        if vertical:
            for q in range(4):
                sl = slice(q * 512, (q + 1) * 512)
                rlo = np.nanmean(red[2:4, sl])
                rhi = np.nanmean(red[2045:2047, sl])
                red[:, sl] -= (
                    rlo * (1.0 - yfrac[:, None]) + rhi * yfrac[:, None]
                )

        if horizontal:
            clo = np.nanmean(red[:, 1:4], axis=1)
            chi = np.nanmean(red[:, 2044:2047], axis=1)
            slo = median_filter(clo, size=7, mode="nearest")
            shi = median_filter(chi, size=7, mode="nearest")
            if noflip:
                red -= (
                    slo[:, None] * (1.0 - xfrac[None, :])
                    + shi[:, None] * xfrac[None, :]
                )
            else:
                bias = np.minimum(slo, shi)[:, None] * np.ones((1, 2048))
                bias[:, 512:1024] = bias[:, 512:1024][:, ::-1]
                bias[:, 1536:2048] = bias[:, 1536:2048][:, ::-1]
                red -= bias

        if q3fix:
            offset = 0.5 * (
                np.median(red[:, 923:1024], axis=1)
                - np.median(red[:, 1024:1125], axis=1)
                + np.median(red[:, 1536:1637], axis=1)
                - np.median(red[:, 1435:1536], axis=1)
            )
            offset = median_filter(offset, size=7, mode="nearest")
            red[:, 1024:1536] += offset[:, None]

        red[saturated] = 65_535
        out[i] = red
        if refout is not None:
            refout[i] = ref

    mask[:4, :] |= _bit("BADPIX")
    mask[-4:, :] |= _bit("BADPIX")
    mask[:, :4] |= _bit("BADPIX")
    mask[:, -4:] |= _bit("BADPIX")
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
        (
            np.median(first[:, :4, :2048], axis=0).ravel(),
            np.median(first[:, :, :4], axis=0).ravel(),
            np.median(first[:, :, 2044:2048], axis=0).ravel(),
            np.median(first[:, -4:, :2048], axis=0).ravel(),
        )
    )
    edge_rms = np.empty(nread)
    ref_rms = np.empty(nread) if nx == 2560 else None
    if nx == 2560:
        ref0 = np.median(first[:, :, 2048:], axis=0)
    for i in range(nread):
        edge = np.concatenate(
            (
                cube[i, :4, :2048].ravel(),
                cube[i, :, :4].ravel(),
                cube[i, :, 2044:2048].ravel(),
                cube[i, -4:, :2048].ravel(),
            )
        )
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
        print(f"Initial bad-read diagnostic ({source}; sigma={sigma:.3f}):")
        for i, (value, baseline, bad) in enumerate(
            zip(series, local, rejected)
        ):
            print(
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
        result[index] = result[lo] + (index - lo) * (
            result[hi] - result[lo]
        ) / (hi - lo)
    return result


def _apply_linearity(cube: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Convert observed to corrected counts using detector polynomials."""

    c = np.asarray(coefficients)
    result = cube.astype(np.float32, copy=True)
    if c.shape == (*cube.shape[1:], 3):
        return c[..., 0] + c[..., 1] * result + c[..., 2] * result**2
    if c.ndim == 2 and c.shape[0] != 4 and c.shape[1] == 4:
        c = c.T
    if c.ndim == 2 and c.shape[0] == 4:
        for q in range(4):
            sl = slice(q * 512, (q + 1) * 512)
            x = result[:, :, sl]
            # The operational IDL APLINCORR may contain more than 3 terms.
            poly = np.zeros_like(x)
            for coefficient in c[q, ::-1]:
                poly = poly * x + coefficient
            result[:, :, sl] = poly
        return result
    raise ValueError(
        "Linearity coefficients must be (ny,nx,ncoef), (4,ncoef), "
        f"or (ncoef,4); received {c.shape}"
    )


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
    variability = dln.mad(
        dc - median[:, None], axis=1, zero=True
    ) / np.maximum(median, 0.001)
    variability[~np.isfinite(variability)] = 0
    variability[two] = 0.5
    sigma = np.maximum(dln.mad(dc - model, axis=1, zero=True), noise)
    sigma[~np.isfinite(sigma)] = noise
    sigma[two] = np.maximum(0.3 * median[two], noise)
    nsigma = (dc - model) / sigma[:, None]
    candidates = np.argwhere(
        (nsigma > max(sigma_threshold, 3.0))
        & (dc > noise * max(sigma_threshold, 3.0))
    )
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
            CosmicRay(
                x=int(pixel),
                read=read,
                counts=float(dc[pixel, diff_index] - model[pixel, diff_index]),
                nsigma=float(nsigma[pixel, diff_index]),
                global_sigma=float(sigma[pixel]),
                fixed=fix,
                local_sigma=local_sigma,
                fix_error=fix_error if fix else 0.0,
            )
        )
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
    sample_noise = np.asarray(readnoise, dtype=np.float32) * np.sqrt(
        2.0 / nfowler_used
    )
    return image, sample_noise, nfowler_used


def up_the_ramp_sampling(
    cube: np.ndarray,
    good_reads: np.ndarray,
    readnoise: np.ndarray | float,
    gain: np.ndarray | float,
    *,
    science_nx: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a corrected ramp with an unweighted linear fit.

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

    time = reads.astype(np.float32)
    samples = data[reads]
    centered_time = time - time.mean()
    denominator = np.sum(centered_time**2)
    if denominator <= 0:
        raise ValueError("Good reads do not span more than one read index")
    slope = np.sum(
        centered_time[:, None, None] * samples, axis=0
    ) / denominator
    image = slope * (reads.size - 1)

    n_good = float(reads.size)
    n_total = float(data.shape[0])
    gain_image = np.asarray(gain, dtype=np.float32)
    noise_image = np.asarray(readnoise, dtype=np.float32)
    signal = np.maximum(image[:, :science_nx], 0)
    sample_noise = np.sqrt(
        12.0
        * (n_good - 1.0)
        / (n_total * (n_good + 1.0))
        * noise_image**2
        + 6.0
        * (n_good**2 + 1.0)
        / (5.0 * n_good * (n_good + 1.0))
        * signal
        * gain_image
    ) / gain_image
    return image, sample_noise


def process_array(
    raw_cube: np.ndarray,
    header: fits.Header | None = None,
    *,
    gain: np.ndarray | float | None = None,
    readnoise: np.ndarray | float | None = None,
    linearity: np.ndarray | None = None,
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
    with a FITS or APZ file.
    """

    raw = np.asarray(raw_cube)
    if raw.ndim != 3 or raw.shape[1] != 2048 or raw.shape[2] not in (2048, 2560):
        raise ValueError("raw_cube must have shape (nread, 2048, 2048|2560)")
    nread = raw.shape[0]
    if nread < 2:
        raise ValueError("At least two reads are required")
    hdr = fits.Header() if header is None else header.copy()
    if verbose:
        print(
            f"Processing ramp: shape={raw.shape}, "
            f"chip={hdr.get('CHIP', 'unknown')}, nread={nread}"
        )

    pre_bad = _detect_bad_reads(raw, debug=debug)
    if verbose:
        rejected = np.flatnonzero(pre_bad) + 1
        print(
            "Initial reference RMS reads rejected: "
            + (", ".join(map(str, rejected)) if rejected.size else "none")
        )
    if raw.shape[2] == 2560:
        if verbose:
            print("Applying reference-output and reference-pixel correction")
        cube, mask, ref_bad, _ = reference_correct(
            raw,
            hdr,
            keep_reference=use_reference,
            q3fix=q3fix,
            verbose=verbose,
            debug=debug,
        )
        bad_reads = pre_bad | ref_bad
    else:
        cube = raw.astype(np.float32)
        mask = np.zeros((2048, 2048), dtype=np.uint16)
        ref_bad = np.zeros(nread, dtype=bool)
        bad_reads = pre_bad
    if verbose:
        rejected = np.flatnonzero(bad_reads) + 1
        print(
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
        print(f"Interpolated {bad_reads.sum()} rejected read(s)")

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
        mask[np.asarray(littrow) == 1] |= _bit("LITTROW_GHOST")
    if persistence_mask is not None:
        p = np.asarray(persistence_mask)
        mask[(p & 1) != 0] |= _bit("PERSIST_HIGH")
        mask[(p & 2) != 0] |= _bit("PERSIST_MED")
        mask[(p & 4) != 0] |= _bit("PERSIST_LOW")

    if linearity is not None:
        if verbose:
            print("Applying detector linearity correction")
        science = _apply_linearity(cube[:, :, :science_nx], linearity)
        cube[:, :, :science_nx] = science
    if dark is not None:
        if verbose:
            print("Subtracting dark ramp")
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
        print("Scanning detector rows for saturation and cosmic rays")

    for y in range(ny):
        if verbose and (y == 0 or (y + 1) % 256 == 0 or y == ny - 1):
            print(f"  row {y + 1:4d}/{ny}")
        ramp = cube[:, y, :].T.astype(np.float32)
        saturated = ramp > saturation_level
        sat_pixels = np.flatnonzero(saturated.any(axis=1))
        for x in sat_pixels:
            first = int(np.flatnonzero(saturated[x])[0])
            ramp[x, first:] = np.nan
            sat_info[y, x] = (1, first, nread - first)
            if x < science_nx:
                mask[y, x] |= _bit("SATPIX")

        dcounts = np.diff(ramp, axis=1)
        if detect_cosmic_rays and nread > 2:
            dcounts, med, row_var, events = detect_and_fix_cosmic_rays(
                dcounts,
                sat_info[y],
                noise=float(np.nanmedian(readnoise) if readnoise is not None else 12.0)
                * np.sqrt(2.0),
                fix=fix_cosmic_rays,
            )
            variability[y] = row_var
            for event in events:
                event.y = y
                cosmic_rays.append(event)
                if event.x < science_nx:
                    mask[y, event.x] |= _bit("CRPIX")
        else:
            med = np.nanmedian(dcounts, axis=1)

        ngood = np.isfinite(dcounts).sum(axis=1)
        threshold = 1 if fix_three_read_saturation and nread == 3 else 2
        unfixable = (sat_info[y, :, 0] == 1) & (ngood < threshold)
        dcounts[unfixable] = 0.0
        unfixable_science = np.flatnonzero(unfixable)
        unfixable_science = unfixable_science[unfixable_science < science_nx]
        mask[y, unfixable_science] |= _bit("UNFIXABLE")
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
        science_first[(mask[y] & _bit("UNFIXABLE")) != 0] = 0
        fixed = np.empty_like(ramp)
        fixed[:, 0] = first
        fixed[:, 1:] = first[:, None] + np.cumsum(dcounts, axis=1)
        cube[:, y, :] = fixed.T
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
        image, sample_noise, nf = fowler_sampling(
            cube,
            good_reads,
            noise_image,
            nfowler=nfowler,
        )
        if verbose:
            print(f"Collapsing ramp with Fowler sampling (Nfowler={nf})")
    else:
        if verbose:
            print(
                f"Collapsing ramp with up-the-ramp sampling "
                f"({good_reads.size} good reads)"
            )
        image, sample_noise = up_the_ramp_sampling(
            cube,
            good_reads,
            noise_image,
            gain_image,
            science_nx=science_nx,
        )

    if use_reference and image.shape[1] == 2560:
        ref = image[:, 2048:].copy()
        ref -= median_filter(np.median(ref, axis=1), size=7, mode="nearest")[:, None]
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
            print("Subtracting supplied persistence model")
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
    variance[(mask & _bit("UNFIXABLE")) != 0] = BAD_VARIANCE
    if fix_cosmic_rays:
        for event in cosmic_rays:
            if event.x < science_nx:
                variance[event.y, event.x] += event.fix_error
    else:
        variance[(mask & _bit("CRPIX")) != 0] = BAD_VARIANCE
    variance[(mask & _bit("BADPIX")) != 0] = BAD_VARIANCE

    if flat is not None:
        if verbose:
            print("Applying flat-field correction")
        flat_image = np.asarray(flat, dtype=np.float32)
        if flat_image.shape != image.shape:
            raise ValueError("flat must match the 2-D science image")
        image /= flat_image
        variance /= flat_image**2
    if output_electrons:
        image *= gain_image
        variance *= gain_image**2

    finite_var = variability[:, :science_nx][
        (sat_info[:, :science_nx, 0] == 0)
        & ((mask & _bit("CRPIX")) == 0)
        & (median_dcounts[:, :science_nx] > 40)
    ]
    if finite_var.size == 0:
        finite_var = variability[:, :science_nx][
            (sat_info[:, :science_nx, 0] == 0)
            & (median_dcounts[:, :science_nx] > 20)
        ]
    global_variability = float(np.median(finite_var)) if finite_var.size else -1.0
    error = np.maximum(np.sqrt(variance), 1.0)
    _update_header(
        hdr,
        nread=nread,
        gain=float(np.median(gain_image)),
        readnoise=float(np.median(sample_noise)),
        up_the_ramp=up_the_ramp,
        nfowler=None if up_the_ramp else nf,
        global_variability=global_variability,
        output_electrons=output_electrons,
    )
    if verbose:
        nsat = int(np.count_nonzero(mask & _bit("SATPIX")))
        nunfix = int(np.count_nonzero(mask & _bit("UNFIXABLE")))
        nbad = int(np.count_nonzero(mask & _bit("BADPIX")))
        print(
            f"Reduction summary: bad={nbad}, CR={len(cosmic_rays)}, "
            f"saturated={nsat}, unfixable={nunfix}, "
            f"global variability={global_variability:.4f}"
        )
    return ProcessResult(
        flux=image.astype(np.float32),
        error=error.astype(np.float32),
        mask=mask,
        header=hdr,
        cosmic_rays=cosmic_rays,
        saturation=sat_info[:, :science_nx],
        fixed_cube=cube if return_cube else None,
        persistence_model=pmodel,
        read_mask=bad_reads,
        global_variability=global_variability,
    )


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

    header["GAIN"] = (gain, "Median gain in electron/ADU")
    header["RDNOISE"] = (readnoise, "Median sampled read noise")
    header.add_history("AP3D Python 3-D to 2-D processing")
    header.add_history(
        "Up-the-ramp sampling"
        if up_the_ramp
        else f"Fowler sampling, Nfowler={nfowler}"
    )
    header.add_history(f"Global fractional variability = {global_variability:.3f}")
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
    through ``options`` to :func:`process_array`.
    """

    filename = Path(filename)
    if filename.suffix.lower() == ".apz":
        raise ValueError(
            "process_cube() requires a decompressed FITS file; "
            "use process_file() for APZ inputs"
        )
    if verbose:
        print(f"Reading decompressed ramp: {filename}")
    cube, header = read_ramp(
        filename, max_read=max_read, verbose=verbose
    )
    if verbose:
        print(f"Loaded ramp with shape {cube.shape} and dtype {cube.dtype}")
    return process_array(
        cube,
        header,
        verbose=verbose,
        debug=debug,
        **options,
    )


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
            dark_cube = next(
                (
                    np.asarray(hdu.data)
                    for hdu in hdul
                    if hdu.data is not None and hdu.data.ndim == 3
                ),
                None,
            )
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


def write_ap2d(
    filename: str | Path,
    result: ProcessResult,
    *,
    overwrite: bool = False,
    integer_output: bool = False,
) -> None:
    """Write an APOGEE-style ap2D product."""

    flux = np.nan_to_num(result.flux)
    error = np.nan_to_num(result.error, nan=np.sqrt(BAD_VARIANCE))
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
    if result.persistence_model is not None:
        hdus.append(
            fits.ImageHDU(result.persistence_model, name="PERSIST CORRECTION")
        )
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
    """Orchestrate APZ decoding, calibration loading, reduction, and writing."""

    input_file = Path(filename)
    if verbose:
        print(f"Input raw ramp: {input_file}")
        for label, value in (
            ("detector", detector),
            ("BPM", bpm),
            ("dark", dark),
            ("flat", flat),
            ("Littrow", littrow),
            ("persistence mask", persistence_mask),
        ):
            if value is not None:
                print(f"Using {label}: {value}")
    calibrations = read_calibrations(
        detector=detector,
        bpm=bpm,
        dark=dark,
        flat=flat,
        littrow=littrow,
        persistence_mask=persistence_mask,
    )
    if verbose:
        shapes = ", ".join(
            f"{name}={np.shape(value)}" for name, value in calibrations.items()
        )
        print(f"Calibration shapes: {shapes}")

    temporary_directory = None
    try:
        cube_file = input_file
        if input_file.suffix.lower() == ".apz":
            temporary_directory = tempfile.TemporaryDirectory(prefix="ap3d-")
            apzip.unzip(
                str(input_file),
                clobber=True,
                delete=False,
                silent=not verbose,
                fitsdir=temporary_directory.name,
            )
            cube_file = (
                Path(temporary_directory.name) / f"{input_file.stem}.fits"
            )
            if not cube_file.exists():
                raise RuntimeError(
                    "APZ decompression did not create expected file "
                    f"{cube_file}"
                )

        result = process_cube(
            cube_file,
            max_read=max_read,
            **calibrations,
            verbose=verbose,
            debug=debug,
            **options,
        )
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()

    if verbose:
        print(f"Writing ap2D product: {output}")
    write_ap2d(output, result, overwrite=overwrite)
    if verbose:
        print("Finished")
    return result
