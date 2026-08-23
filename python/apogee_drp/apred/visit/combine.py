"""Numerical core of APOGEE dither combination.

The routines here translate the arithmetic in ``sincinterlaced.pro`` and the
pair/final-combination portions of ``apdithercomb.pro``.  They are kept as pure
array functions so IDL/Python intermediate products can be compared before the
full frame orchestration is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from .io import BADERR
from .dither import DitherPair, dither_pairs
from .models import ChipFrame, VisitFrame


@dataclass
class CombinedSpectrum:
    flux: np.ndarray
    error: np.ndarray
    mask: np.ndarray
    scale_sum: np.ndarray


SPECTRAL_FIELDS = (
    "flux",
    "err",
    "mask",
    "wavelength",
    "sky",
    "skyerr",
    "telluric",
    "telluricerr",
)


def _idl_convol(
    values: np.ndarray, kernel: np.ndarray, *, ignore_nan: bool = True
) -> np.ndarray:
    """IDL ``CONVOL(...,/CENTER,/EDGE_TRUNCATE,/NAN)`` for one dimension."""

    values = np.asarray(values, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    half = len(kernel) // 2
    output = np.zeros(values.size, dtype=np.float64)
    for index in range(values.size):
        lo = max(0, index - half)
        hi = min(values.size, index + half + 1)
        klo = half - (index - lo)
        khi = klo + (hi - lo)
        sample = values[lo:hi]
        weights = kernel[klo:khi]
        if ignore_nan:
            good = np.isfinite(sample)
            output[index] = np.sum(sample[good] * weights[good])
        else:
            output[index] = np.sum(sample * weights)
    return output


def _safe_sinc(x: np.ndarray, frequency: float) -> np.ndarray:
    argument = frequency * np.pi * x
    # np.sinc(z) is sin(pi*z)/(pi*z).
    return np.exp(-((x / 3.25) ** 2)) * np.sinc(frequency * x)


def sinc_interlaced(
    arr1: Any,
    arr2: Any,
    shift: float,
    outshift: float,
    *,
    err1: Any | None = None,
    err2: Any | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Interlaced sinc interpolation from ``sincinterlaced.pro``.

    The second returned array is a variance, matching the IDL ``ERROUT``
    keyword—not a standard deviation.
    """

    first = np.asarray(arr1)
    second = np.asarray(arr2)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("arr1 and arr2 must be equal-length 1-D arrays")
    shift = float(shift)
    outshift = float(outshift)
    have_errors = err1 is not None and err2 is not None

    # Preserve the exact shortcut branches and their input dtypes.
    if outshift == shift:
        if shift > 0:
            variance = np.asarray(err1) ** 2 if err1 is not None else None
            return first.copy(), variance
        variance = np.asarray(err2) ** 2 if err2 is not None else None
        return second.copy(), variance
    if outshift == 0.0:
        if shift > 0:
            variance = np.asarray(err2) ** 2 if err2 is not None else None
            return second.copy(), variance
        variance = np.asarray(err1) ** 2 if err1 is not None else None
        return first.copy(), variance

    if shift > 0:
        left, right = second, first
        if have_errors:
            leftvar = np.asarray(err2, dtype=np.float64) ** 2
            rightvar = np.asarray(err1, dtype=np.float64) ** 2
    else:
        left, right = first, second
        if have_errors:
            leftvar = np.asarray(err1, dtype=np.float64) ** 2
            rightvar = np.asarray(err2, dtype=np.float64) ** 2

    ksize = 21
    # IDL integer division: KSIZE/2 is 10.
    x1 = np.arange(ksize, dtype=np.float64) - (ksize // 2) - outshift
    sincx1 = _safe_sinc(x1, 1.0)
    sinc2x1 = _safe_sinc(x1, 2.0)
    cot_factor = np.pi / np.tan((1.0 - shift) * np.pi)
    afunc = sinc2x1 - cot_factor * x1 * sincx1**2
    apart = _idl_convol(left, afunc)

    x2 = np.arange(ksize, dtype=np.float64) - (ksize // 2)
    x2 -= outshift - shift
    x2 = -x2
    sincx2 = _safe_sinc(x2, 1.0)
    sinc2x2 = _safe_sinc(x2, 2.0)
    bfunc = sinc2x2 - cot_factor * x2 * sincx2**2
    bpart = _idl_convol(right, bfunc)
    output = apart + bpart

    variance = None
    if have_errors:
        variance = _idl_convol(leftvar, afunc**2)
        variance += _idl_convol(rightvar, bfunc**2)
    return output, variance


def interlace_pair(
    flux1: Any,
    flux2: Any,
    error1: Any,
    error2: Any,
    *,
    shift: float,
    absolute_shift: float = 0.0,
    scale1: Any | None = None,
    scale2: Any | None = None,
    npad: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sinc-interlace one dither pair onto a half-pixel grid.

    Inputs are one-dimensional spectra.  `flux1` is the left spectrum and
    `flux2` the right spectrum, matching the naming inside APDITHERCOMB.
    """

    f1 = np.asarray(flux1)
    f2 = np.asarray(flux2)
    e1 = np.asarray(error1)
    e2 = np.asarray(error2)
    if not (f1.shape == f2.shape == e1.shape == e2.shape) or f1.ndim != 1:
        raise ValueError("flux and error inputs must be equal-length 1-D arrays")
    npix = f1.size

    if npad:
        def pad_edge(array: np.ndarray) -> np.ndarray:
            return np.pad(array, npad, mode="edge")

        pf1, pf2, pe1, pe2 = map(pad_edge, (f1, f2, e1, e2))
    else:
        pf1, pf2, pe1, pe2 = f1, f2, e1, e2

    # IDL calls SINCINTERLACED(spec2,spec1,...,err1=err2,err2=err1).
    left, leftvar = sinc_interlaced(
        pf2,
        pf1,
        shift,
        -absolute_shift,
        err1=pe2,
        err2=pe1,
    )
    right, rightvar = sinc_interlaced(
        pf2,
        pf1,
        shift,
        0.5 - absolute_shift,
        err1=pe2,
        err2=pe1,
    )
    section = slice(npad, npad + npix)
    flux = np.empty(2 * npix, dtype=np.float64)
    variance = np.empty(2 * npix, dtype=np.float64)
    flux[0::2] = left[section]
    flux[1::2] = right[section]
    variance[0::2] = leftvar[section]
    variance[1::2] = rightvar[section]
    error = np.sqrt(variance)

    if scale1 is not None and scale2 is not None:
        scales = 0.5 * (
            np.asarray(scale1, dtype=np.float64)
            + np.asarray(scale2, dtype=np.float64)
        )
        x = np.arange(2 * npix, dtype=np.float64) / 2.0
        # IDL INTERPOLATE uses linear interpolation and clamps the last half
        # pixel to the final input sample.
        rescale = np.interp(x, np.arange(npix), scales)
        flux *= rescale
        error *= rescale
    return flux, error


def combine_spectra(
    flux: Any,
    error: Any,
    mask: Any,
    *,
    scales: Any | None = None,
    global_weight: bool = False,
    bad_mask: int = 0,
) -> CombinedSpectrum:
    """Combine fully sampled dither pairs using the IDL weighting equations.

    Arrays have shape ``(nspec, npix)``.  Flux and errors are normalized by
    `scales`, combined, then multiplied by the *sum* of the continua because
    APDITHERCOMB sums exposure flux rather than returning a mean exposure.
    """

    values = np.asarray(flux)
    errors = np.asarray(error)
    masks = np.asarray(mask)
    if values.ndim != 2 or errors.shape != values.shape or masks.shape != values.shape:
        raise ValueError("flux, error, and mask must have shape (nspec, npix)")
    nspec, npix = values.shape
    work_dtype = values.dtype if values.dtype.kind == "f" else np.dtype("f4")
    if scales is None:
        continuum = np.ones_like(values, dtype=work_dtype)
    else:
        continuum = np.asarray(scales, dtype=work_dtype)
        if continuum.shape != values.shape:
            raise ValueError("scales must match flux shape")

    # Preserve operation ordering while working in the input floating dtype.
    normalized_flux = values / continuum
    normalized_error = errors / continuum
    invalid = (
        ~np.isfinite(normalized_flux)
        | ~np.isfinite(normalized_error)
        | (normalized_error <= 0)
    )
    if bad_mask:
        invalid |= (masks.astype(np.int64) & int(bad_mask)) != 0

    work_error = normalized_error.copy()
    work_error[invalid] = np.nan
    work_flux = normalized_flux.copy()
    work_flux[invalid] = np.nan
    if global_weight:
        median_error = np.nanmedian(work_error, axis=1)
        weights = np.broadcast_to(
            (1.0 / median_error**2)[:, None], values.shape
        ).copy()
        weights[invalid] = np.nan
    else:
        weights = 1.0 / work_error**2

    numerator = np.nansum(work_flux * weights, axis=0)
    denominator = np.nansum(weights, axis=0)
    combined = numerator / denominator
    combined_error = np.sqrt(1.0 / denominator)
    combined_mask = np.bitwise_or.reduce(masks.astype(np.int64), axis=0)
    bad = ~np.isfinite(combined) | ~np.isfinite(combined_error)
    combined[bad] = 0.0
    combined_error[bad] = BADERR
    combined_mask[bad] |= 1

    scale_sum = np.sum(continuum, axis=0)
    combined *= scale_sum
    combined_error *= scale_sum
    return CombinedSpectrum(combined, combined_error, combined_mask, scale_sum)


def convert_wcoef_to_half_pixel(wcoef: Any) -> np.ndarray:
    """Convert PIX2WAVE coefficients from native to dither-combined pixels."""

    output = np.asarray(wcoef).copy()
    output[..., 0] *= 2
    output[..., 2] *= 2
    output[..., 3] *= 2
    output[..., 5] *= 2
    for index in range(6, output.shape[-1]):
        output[..., index] /= 2 ** (index - 6)
    return output


def convert_lsf_to_half_pixel(lsfcoef: Any) -> np.ndarray:
    """Convert Gauss-Hermite LSF coefficients to half-pixel sampling.

    This implements the core GH component of the IDL conversion.  Wing
    parameters are also converted when their encoded layout is present.
    """

    output = np.asarray(lsfcoef).copy()
    flat = output.reshape(-1, output.shape[-1])
    for pars in flat:
        if pars[0] != 1:
            continue
        horder = int(pars[2])
        porder = pars[3 : horder + 4].astype(int)
        ngh = int(np.sum(porder + 1))
        start = horder + 4
        coefficients = pars[start : start + ngh].copy()
        offsets = np.concatenate(([0], np.cumsum(porder + 1)))
        for component in range(horder + 1):
            lo, hi = offsets[component], offsets[component + 1]
            for power in range(hi - lo):
                coefficients[lo + power] /= 2**power
            if component == 0:  # sigma
                coefficients[lo:hi] *= 2
        pars[0] *= 2
        pars[1] *= 2
        pars[start : start + ngh] = coefficients

        wing_start = start + ngh
        if wing_start >= pars.size:
            continue
        nwing = int(pars[wing_start + 1])
        orders = pars[wing_start + 2 : wing_start + 2 + nwing].astype(int)
        ncoef = int(np.sum(orders + 1))
        coef_start = wing_start + 2 + nwing
        if coef_start + ncoef > pars.size:
            continue
        coefficients = pars[coef_start : coef_start + ncoef].copy()
        offsets = np.concatenate(([0], np.cumsum(orders + 1)))
        for component in range(nwing):
            lo, hi = offsets[component], offsets[component + 1]
            for power in range(hi - lo):
                coefficients[lo + power] /= 2**power
            if component == 1:
                coefficients[lo:hi] *= 2
        pars[coef_start : coef_start + ncoef] = coefficients
    return output


def pix2wave(pixels: Any, coefficients: Any) -> np.ndarray:
    """Evaluate the APOGEE ``PIX2WAVE.PRO`` wavelength model."""

    x = np.asarray(pixels, dtype=np.float64)
    pars = np.asarray(coefficients, dtype=np.float64)
    xb = x + pars[0]
    sine = pars[1] * (
        np.sin((xb + pars[2]) / pars[3] * np.pi / 180.0) + pars[4]
    )
    scaled = (xb + pars[5]) / 3000.0
    polynomial = np.polynomial.polynomial.polyval(scaled, pars[6:])
    return sine + polynomial


def estimate_continuum(
    flux: Any,
    mask: Any | None = None,
    *,
    bad_mask: int = 0,
    median_width: int = 501,
    smooth_width: int = 100,
) -> np.ndarray:
    """Approximate IDL ``SMOOTH(MEDFILT1D(...,501,EDGE=2),100,/NAN)``.

    Masked samples are linearly filled before the edge-replicated median
    filter.  This is deterministic and close to IDL; golden-data comparison
    will determine whether IDL's exact NaN-window behavior needs a custom
    implementation.
    """

    values = np.asarray(flux)
    dtype = values.dtype if values.dtype.kind == "f" else np.dtype("f4")
    work = np.asarray(values, dtype=dtype).copy()
    bad = ~np.isfinite(work)
    if mask is not None and bad_mask:
        bad |= (np.asarray(mask).astype(np.int64) & int(bad_mask)) != 0
    if np.all(bad):
        return np.ones_like(work)
    if np.any(bad):
        good = np.flatnonzero(~bad)
        work[bad] = np.interp(np.flatnonzero(bad), good, work[good])
    # IDL requires odd MEDFILT width.  Reduce it for short synthetic spectra.
    width = min(int(median_width), work.size if work.size % 2 else work.size - 1)
    width = max(width, 1)
    continuum = median_filter(work, size=width, mode="nearest")
    swidth = min(max(int(smooth_width), 1), work.size)
    continuum = uniform_filter1d(continuum, size=swidth, mode="nearest")
    if np.any(continuum < 0):
        continuum = np.ones_like(continuum)
    return np.asarray(continuum, dtype=dtype)


def _shift_chipfit(frame: VisitFrame) -> np.ndarray:
    value = frame.shift["chipfit"] if isinstance(frame.shift, dict) else frame.shift.chipfit
    array = np.asarray(value)
    return array[0] if array.ndim > 1 and array.shape[0] == 1 else array


def _empty_like_pair(frame: VisitFrame, npix: int) -> VisitFrame:
    output = copy.deepcopy(frame)
    for target in output:
        for name in SPECTRAL_FIELDS:
            value = getattr(target, name)
            if value is not None:
                array = np.asarray(value)
                setattr(target, name, np.zeros((array.shape[0], npix), dtype=array.dtype))
    return output


def _interlace_extra(
    first: np.ndarray, second: np.ndarray, absolute_shift: float
) -> np.ndarray:
    """IDL's simple [frame2,frame1] interlace for sky/telluric arrays."""

    output = np.empty(first.size * 2, dtype=np.result_type(first, second))
    output[0::2] = second
    output[1::2] = first
    if abs(absolute_shift * 2) > 0.5:
        amount = int(np.ceil(abs(absolute_shift * 2)))
        shifted = np.full(output.shape, np.nan, dtype=np.float64)
        if absolute_shift > 0:
            shifted[amount:] = output[:-amount]
        else:
            shifted[:-amount] = output[amount:]
        output = shifted.astype(output.dtype, copy=False)
    return output


def _interlace_mask(
    mask1: np.ndarray,
    mask2: np.ndarray,
    *,
    shift: float,
    absolute_shift: float,
    npad: int,
    thresholds: np.ndarray,
) -> np.ndarray:
    npix = mask1.size
    output = np.zeros(2 * npix, dtype=np.uint16)
    unsigned1 = np.asarray(mask1, dtype=np.int16).view(np.uint16)
    unsigned2 = np.asarray(mask2, dtype=np.int16).view(np.uint16)
    for bit, threshold in enumerate(thresholds):
        value = np.uint16(1 << bit)
        first = ((unsigned1 & value) != 0).astype(np.float32)
        second = ((unsigned2 & value) != 0).astype(np.float32)
        if not np.any(first) and not np.any(second):
            continue
        if npad:
            first = np.pad(first, npad, constant_values=1)
            second = np.pad(second, npad, constant_values=1)
        left, _ = sinc_interlaced(
            second, first, shift, -absolute_shift
        )
        right, _ = sinc_interlaced(
            second, first, shift, 0.5 - absolute_shift
        )
        section = slice(npad, npad + npix)
        contribution = np.empty(2 * npix, dtype=np.float64)
        contribution[0::2] = left[section]
        contribution[1::2] = right[section]
        output[np.abs(contribution) > threshold] |= value
    return output.view(np.int16)


def interlace_frame_pair(
    frames: list[VisitFrame],
    pair: DitherPair,
    *,
    reference_index: int,
    fiber_types: Any | None = None,
    no_scale: bool = False,
    median_scale: bool = True,
    new_error: bool = True,
    npad: int = 50,
    average_wave: bool = True,
    bad_mask: int = 0x40FF,
    mask_thresholds: Any | None = None,
) -> VisitFrame:
    """Interlace all chips and fibers for one APDITHERCOMB pair."""

    frame1 = frames[int(pair.index[0])]
    frame2 = frames[int(pair.index[1])]
    reference = frames[int(reference_index)]
    source = frame1.chipa
    nfiber, npix = source.flux.shape
    output = _empty_like_pair(frame1, 2 * npix)
    output.metadata["filename1"] = source.filename
    output.metadata["filename2"] = frame2.chipa.filename
    thresholds = (
        np.full(16, 0.1, dtype=np.float64)
        if mask_thresholds is None
        else np.asarray(mask_thresholds, dtype=np.float64)
    )
    types = (
        np.full(nfiber, "", dtype="U1")
        if fiber_types is None
        else np.asarray(fiber_types).astype(str)
    )
    fit0 = _shift_chipfit(reference)
    fit1 = _shift_chipfit(frame1)
    fit2 = _shift_chipfit(frame2)

    for ifiber in range(nfiber):
        for ichip in range(3):
            chip1 = frame1.chip(ichip)
            chip2 = frame2.chip(ichip)
            outchip = output.chip(ichip)
            absolute_shift = (
                fit0[0] * ifiber + fit0[ichip + 1]
                - fit1[0] * ifiber
                - fit1[ichip + 1]
            )
            relative_shift = (
                fit1[0] * ifiber + fit1[ichip + 1]
                - fit2[0] * ifiber
                - fit2[ichip + 1]
            )
            f1, f2 = chip1.flux[ifiber].copy(), chip2.flux[ifiber].copy()
            e1, e2 = chip1.err[ifiber].copy(), chip2.err[ifiber].copy()
            m1, m2 = chip1.mask[ifiber], chip2.mask[ifiber]
            scale1 = scale2 = None
            if types[ifiber].strip() != "SKY" and not no_scale:
                scale1 = estimate_continuum(f1, m1, bad_mask=bad_mask)
                scale2 = estimate_continuum(f2, m2, bad_mask=bad_mask)
                bad1 = (m1.astype(np.int64) & bad_mask) != 0
                bad2 = (m2.astype(np.int64) & bad_mask) != 0
                if median_scale:
                    f1[bad1] = scale1[bad1]
                    f2[bad2] = scale2[bad2]
                f1 /= scale1
                e1 /= scale1
                f2 /= scale2
                e2 /= scale2
            combined_flux, combined_error = interlace_pair(
                f1,
                f2,
                e1,
                e2,
                shift=float(relative_shift),
                absolute_shift=float(absolute_shift),
                scale1=scale1,
                scale2=scale2,
                npad=npad,
            )
            outchip.flux[ifiber] = combined_flux
            outchip.err[ifiber] = combined_error
            outchip.mask[ifiber] = _interlace_mask(
                m1,
                m2,
                shift=float(relative_shift),
                absolute_shift=float(absolute_shift),
                npad=npad,
                thresholds=thresholds,
            )
            bad = (outchip.mask[ifiber].astype(np.int64) & bad_mask) != 0
            outchip.err[ifiber, bad] *= 10
            negative = outchip.flux[ifiber] < -5 * outchip.err[ifiber]
            if np.any(negative):
                unsigned = outchip.mask[ifiber].view(np.uint16)
                unsigned[negative] |= np.uint16(1 << 15)

            wcoef1, wcoef2 = chip1.wcoef[ifiber], chip2.wcoef[ifiber]
            y = np.arange(npix, dtype=np.float64)
            wave1 = np.empty(2 * npix, dtype=np.float64)
            wave2 = np.empty(2 * npix, dtype=np.float64)
            wave1[0::2] = pix2wave(y - absolute_shift, wcoef1)
            wave1[1::2] = pix2wave(y + 0.5 - absolute_shift, wcoef1)
            wave2[0::2] = pix2wave(
                y - absolute_shift - relative_shift, wcoef2
            )
            wave2[1::2] = pix2wave(
                y + 0.5 - absolute_shift - relative_shift, wcoef2
            )
            wave = 0.5 * (wave1 + wave2) if average_wave else wave1
            outchip.wavelength[ifiber] = wave
            combined_wcoef = wcoef1.copy()
            if average_wave:
                newy = np.empty(2 * npix, dtype=np.float64)
                newy[0::2] = y - absolute_shift
                newy[1::2] = y + 0.5 - absolute_shift
                combined_wcoef[6:10] = np.polynomial.polynomial.polyfit(
                    (newy + wcoef1[0]) / 3000.0, wave, 3
                )
            outchip.wcoef[ifiber] = combined_wcoef
            for field in ("sky", "skyerr", "telluric", "telluricerr"):
                getattr(outchip, field)[ifiber] = _interlace_extra(
                    getattr(chip1, field)[ifiber], getattr(chip2, field)[ifiber],
                    float(absolute_shift),
                )

    for ichip in range(3):
        chip = output.chip(ichip)
        chip.wcoef = convert_wcoef_to_half_pixel(chip.wcoef)
        chip.lsfcoef = convert_lsf_to_half_pixel(chip.lsfcoef)
    return output


def combine_pair_frames(
    pair_frames: list[VisitFrame],
    *,
    fiber_types: Any | None = None,
    no_scale: bool = False,
    median_scale: bool = True,
    global_weight: bool = False,
    average_wave: bool = True,
    bad_mask: int = 0x40FF,
) -> VisitFrame:
    """Combine all fully sampled pair frames into the visit frame."""

    if not pair_frames:
        raise ValueError("no pair frames")
    if len(pair_frames) == 1:
        return copy.deepcopy(pair_frames[0])
    first = pair_frames[0]
    nfiber, npix = first.chipa.flux.shape
    output = _empty_like_pair(first, npix)
    types = (
        np.full(nfiber, "", dtype="U1")
        if fiber_types is None
        else np.asarray(fiber_types).astype(str)
    )
    for ifiber in range(nfiber):
        for ichip in range(3):
            chips = [frame.chip(ichip) for frame in pair_frames]
            flux = np.asarray([chip.flux[ifiber] for chip in chips])
            error = np.asarray([chip.err[ifiber] for chip in chips])
            mask = np.asarray([chip.mask[ifiber] for chip in chips])
            scales = np.ones_like(flux)
            if types[ifiber].strip() != "SKY" and not no_scale:
                for index in range(len(chips)):
                    scales[index] = estimate_continuum(
                        flux[index], mask[index], bad_mask=bad_mask
                    )
            result = combine_spectra(
                flux,
                error,
                mask,
                scales=scales,
                global_weight=global_weight,
            )
            outchip = output.chip(ichip)
            outchip.flux[ifiber] = result.flux
            outchip.err[ifiber] = result.error
            outchip.mask[ifiber] = result.mask
            for field in ("sky", "skyerr", "telluricerr"):
                getattr(outchip, field)[ifiber] = np.sum(
                    [getattr(chip, field)[ifiber] for chip in chips], axis=0)
            outchip.telluric[ifiber] = np.mean(
                [chip.telluric[ifiber] for chip in chips], axis=0)
            waves = np.asarray([chip.wavelength[ifiber] for chip in chips])
            outchip.wavelength[ifiber] = (
                np.mean(waves, axis=0) if average_wave else waves[0]
            )
            wcoef = chips[0].wcoef[ifiber].copy()
            if average_wave:
                wcoef[6:10] = np.polynomial.polynomial.polyfit(
                    (np.arange(npix) + wcoef[0]) / 3000.0,
                    outchip.wavelength[ifiber],
                    3,
                )
            outchip.wcoef[ifiber] = wcoef
    return output


def dither_combine(
    frames: list[VisitFrame],
    shifts: Any,
    *,
    fiber_types: Any | None = None,
    no_dither: bool = False,
    no_scale: bool = False,
    median_scale: bool = True,
    new_error: bool = True,
    npad: int = 50,
    global_weight: bool = False,
    average_wave: bool = True,
    bad_mask: int = 0x40FF,
) -> tuple[VisitFrame, list[DitherPair] | None]:
    """High-level native Python equivalent of ``APDITHERCOMB``."""

    if len(frames) != len(shifts):
        raise ValueError("frames and shifts must have the same length")
    if len(frames) == 1:
        return copy.deepcopy(frames[0]), None
    if no_dither:
        output = combine_pair_frames(
            frames,
            fiber_types=fiber_types,
            no_scale=no_scale,
            median_scale=median_scale,
            global_weight=global_weight,
            average_wave=average_wave,
            bad_mask=bad_mask,
        )
        return output, None
    pairs = dither_pairs(shifts, snsort=True)
    reference_index = int(pairs[0].index[0])
    pair_frames = [
        interlace_frame_pair(
            frames,
            pair,
            reference_index=reference_index,
            fiber_types=fiber_types,
            no_scale=no_scale,
            median_scale=median_scale,
            new_error=new_error,
            npad=npad,
            average_wave=average_wave,
            bad_mask=bad_mask,
        )
        for pair in pairs
    ]
    output = combine_pair_frames(
        pair_frames,
        fiber_types=fiber_types,
        no_scale=no_scale,
        median_scale=median_scale,
        global_weight=global_weight,
        average_wave=average_wave,
        bad_mask=bad_mask,
    )
    return output, pairs
