"""Build APOGEE relative-flux and blackbody-response calibrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import warnings

import numpy as np
from astropy.io import fits
from scipy.ndimage import uniform_filter1d

from ...utils import apload
from ...utils.bitmask import PixelBitMask
from .utils import product_build_lock

CHIPS = ("a", "b", "c")
__all__ = [
    "build_flux", "build_response", "make_flux_calibrations",
    "make_reference_spectra", "planck",
]


def _robust_polyfit(x, y, degree):
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() <= degree:
        raise ValueError("too few finite points for reference-spectrum fit")
    for _ in range(5):
        coefficients = np.polynomial.polynomial.polyfit(x[good], y[good], degree)
        residual = y - np.polynomial.polynomial.polyval(x, coefficients)
        center = np.nanmedian(residual[good])
        scatter = np.nanmedian(np.abs(residual[good] - center))
        if not np.isfinite(scatter) or scatter == 0:
            break
        keep = good & (np.abs(residual - center) <= 5 * scatter)
        if keep.sum() <= degree or np.array_equal(keep, good):
            break
        good = keep
    return np.polynomial.polynomial.polyfit(x[good], y[good], degree)


def _running_nanmedian(values, width):
    """NaN-aware, edge-replicated running median along the pixel axis."""
    data = np.asarray(values, float)
    if width <= 0 or int(width) != width:
        raise ValueError("median width must be a positive integer")
    width = int(width)
    before, after = width // 2, width - 1 - width // 2
    padded = np.pad(data, ((before, after), (0, 0)), mode="edge")
    # sliding_window_view requires NumPy >= 1.20, while the APOGEE
    # production environment still supports older NumPy releases. Construct
    # the same read-only view with the long-standing as_strided API.
    windows = np.lib.stride_tricks.as_strided(
        padded,
        shape=(data.shape[0], data.shape[1], width),
        strides=(padded.strides[0], padded.strides[1], padded.strides[0]),
        writeable=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(windows, axis=-1)


def _fill_nonfinite(values):
    result = np.asarray(values, float).copy()
    x = np.arange(result.shape[0])
    for fiber in range(result.shape[1]):
        good = np.isfinite(result[:, fiber])
        if good.any():
            result[:, fiber] = np.interp(
                x, x[good], result[good, fiber])
    return result


def planck(wavelength, temperature):
    """Return a Planck spectrum per unit wavelength, normalized arbitrarily."""
    wavelength = np.asarray(wavelength, float)
    if temperature <= 0 or np.any(wavelength <= 0):
        raise ValueError("wavelength and temperature must be positive")
    meters = wavelength * 1e-10
    exponent = 1.438776877e-2 / (meters * float(temperature))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        spectrum = 1.0 / (meters ** 5 * np.expm1(exponent))
    return spectrum


def make_reference_spectra(fluxes, masks=None, *, telescope="apo25m",
                           exptype="UNKNOWN", wavelengths=None, bbtemp=None,
                           absolute=False, bad_pixel_bits=None,
                           median_width=41):
    """Construct the original and fitted lamp reference spectra."""
    arrays = [np.asarray(flux, float) for flux in fluxes]
    if len(arrays) != 3 or any(array.ndim != 2 for array in arrays):
        raise ValueError("fluxes must contain three [pixel, fiber] arrays")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("all flux arrays must have the same shape")
    npix, _ = arrays[0].shape
    if str(exptype).strip().upper() == "BLACKBODY":
        if wavelengths is None or len(wavelengths) != 3:
            raise ValueError("BLACKBODY calibration requires three wavelength arrays")
        if bbtemp is None:
            raise ValueError("BLACKBODY calibration requires bbtemp")
        original = np.column_stack([
            planck(np.asarray(wave, float), bbtemp) for wave in wavelengths])
        if original.shape != (npix, 3):
            raise ValueError("wavelength arrays must have one value per pixel")
        fitted = original.copy()
        if not absolute:
            fitted /= fitted[npix // 2, 1]
        return fitted, original

    bits = PixelBitMask().badval() if bad_pixel_bits is None else int(bad_pixel_bits)
    if masks is None:
        masks = [None] * 3
    if len(masks) != 3:
        raise ValueError("masks must contain three arrays")
    original = np.empty((npix, 3), float)
    for chip, (flux, mask) in enumerate(zip(arrays, masks)):
        work = flux.copy()
        if mask is not None:
            if np.shape(mask) != flux.shape:
                raise ValueError("mask and flux shapes must match")
            work[(np.asarray(mask, dtype=np.uint64) & bits) != 0] = np.nan
        work[:, ~np.isfinite(np.nanmedian(work, axis=0))] = np.nan
        smooth = _running_nanmedian(work, median_width)
        original[:, chip] = np.nanmedian(smooth, axis=1)
        original[:4, chip] = np.nan
        original[-4:, chip] = np.nan

    offsets = np.array([-2048 - 150, 0, 2048 + 150], float)
    x = np.arange(npix, dtype=float)[:, None] - (npix - 1) / 2 + offsets
    good = np.isfinite(original)
    if telescope == "lco25m":
        flat_pixel = np.arange(3 * npix).reshape(npix, 3, order="F")
        good &= (flat_pixel < 700) | (flat_pixel > 1900)
    coefficients = _robust_polyfit(x[good], original[good], 4)
    fitted = np.polynomial.polynomial.polyval(x, coefficients)
    if telescope == "lco25m":
        pixel = np.arange(npix, dtype=float)
        outside = (pixel < 700) | (pixel > 1900)
        redfit = _robust_polyfit(pixel[outside], original[outside, 0], 2)
        fitted[:, 0] *= original[:, 0] / np.polynomial.polynomial.polyval(
            pixel, redfit)
    return fitted, original


def _repair_fibers(ratio, *, mjd, telescope):
    medians = np.nanmedian(ratio, axis=0)
    broken = np.flatnonzero(~np.isfinite(medians))
    targets = list(broken)
    if int(mjd) >= 59556:
        targets = list((87, 218) if telescope == "lco25m" else (75, 225)) + targets
    good = np.flatnonzero(np.isfinite(medians))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        global_ratio = np.nanmedian(ratio, axis=1)
    for target in sorted(set(t for t in targets if 0 <= t < ratio.shape[1])):
        candidates = good[good != target]
        nearest = candidates[np.argsort(np.abs(candidates - target))[:2]]
        if nearest.size == 2 and abs(int(nearest[0]) - target) < 10:
            ratio[:, target] = np.nanmean(ratio[:, nearest], axis=1)
        else:
            ratio[:, target] = global_ratio


def make_flux_calibrations(fluxes, masks, reference, *, mjd,
                           telescope="apo25m", bad_pixel_bits=None,
                           littrow_bit=None, median_width=51,
                           fill_width=201, smooth_width=100):
    """Calculate relative-flux images and normalized fiber throughputs."""
    if len(fluxes) != 3 or len(masks) != 3:
        raise ValueError("fluxes and masks must each contain three arrays")
    bits = PixelBitMask().badval() if bad_pixel_bits is None else int(bad_pixel_bits)
    ghost = (PixelBitMask().getval("LITTROW_GHOST") if littrow_bit is None
             else int(littrow_bit))
    results, throughputs = [], []
    for chip, (flux, mask) in enumerate(zip(fluxes, masks)):
        flux, mask = np.asarray(flux, float), np.asarray(mask)
        if flux.ndim != 2 or mask.shape != flux.shape:
            raise ValueError("flux and mask arrays must have matching 2-D shapes")
        if np.shape(reference) != (flux.shape[0], 3):
            raise ValueError("reference must have shape [pixel, 3]")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = flux / np.asarray(reference)[:, chip, None]
        ratio[ratio < 1e-3] = np.nan
        ratio[:4] = np.nan
        ratio[-4:] = np.nan
        ratio[(mask.astype(np.uint64) & bits) != 0] = np.nan
        _repair_fibers(ratio, mjd=mjd, telescope=telescope)
        pixel = np.arange(flux.shape[0], dtype=float)
        for fiber in range(flux.shape[1]):
            affected = np.flatnonzero((mask[:, fiber].astype(np.uint64) & ghost) != 0)
            if affected.size:
                ratio[affected, fiber] = np.nan
                lo, hi = max(0, affected[0] - 50), min(flux.shape[0], affected[-1] + 51)
                good = np.isfinite(ratio[lo:hi, fiber])
                if good.sum() > 2:
                    coefficients = _robust_polyfit(
                        pixel[lo:hi][good], ratio[lo:hi, fiber][good], 2)
                    ratio[affected, fiber] = np.polynomial.polynomial.polyval(
                        pixel[affected], coefficients)
        smoothed = _running_nanmedian(ratio, median_width)
        fallback = _running_nanmedian(smoothed, fill_width)
        missing = ~np.isfinite(smoothed)
        smoothed[missing] = fallback[missing]
        smoothed = _fill_nonfinite(smoothed)
        smoothed = uniform_filter1d(
            smoothed, size=int(smooth_width), axis=0, mode="nearest")
        throughput = np.nanmedian(smoothed, axis=0)
        throughput /= np.nanmedian(throughput)
        results.append(smoothed.astype(np.float32))
        throughputs.append(throughput.astype(np.float32))
    return results, throughputs


def _load_1d(load, number, chip):
    filename = load.filename("1D", num=int(number), chip=chip)
    with fits.open(filename) as hdus:
        return {
            "header": hdus[0].header.copy(),
            "flux": np.asarray(hdus[1].data).T,
            "mask": np.asarray(hdus[3].data).T,
        }


def _process(load, frames, **kwargs):
    from ..process import process
    return process(frames, load=load, fluxid=0, nocr=True, nfs=1,
                   doproc=True, **kwargs)


def _write_flux(filename, calibration, throughput, reference, original,
                header, *, apred, exptype):
    primary = fits.PrimaryHDU(header=header.copy())
    primary.header["OBSTYPE"] = "FLUXCORR"
    primary.header["APRED"] = str(apred)
    primary.header["LAMPTYPE"] = str(exptype)
    flux_hdu = fits.ImageHDU(calibration.T, name="RELATIVE FLUX")
    flux_hdu.header["BUNIT"] = "Relative Flux"
    thru_hdu = fits.ImageHDU(throughput, name="THROUGHPUT")
    thru_hdu.header["BUNIT"] = "Throughput"
    ref_hdu = fits.ImageHDU(reference, name="REFERENCE SPECTRUM")
    med_hdu = fits.ImageHDU(original, name="MEDIAN REFERENCE SPECTRUM")
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, flux_hdu, thru_hdu, ref_hdu, med_hdu]).writeto(
        filename, overwrite=True)


def build_flux(ims: Sequence[int] | int, *, apred="daily", telescope="apo25m",
               cmjd=None, darkid=None, flatid=None, psfid=None, modelpsf=None,
               waveid=None, littrowid=None, persistid=None, clobber=False,
               onedclobber=False, bbtemp=None, plate=None, plugid=None,
               holtz=False, temp=None, unlock=False, verbose=False):
    """Reduce lamp exposure(s) and build three relative-flux calibrations."""
    del plate, plugid
    if holtz:
        raise NotImplementedError("the obsolete holtz algorithm is not supported")
    frames = [int(value) for value in np.atleast_1d(ims)]
    if not frames:
        raise ValueError("ims must contain at least one exposure")
    fluxid = frames[0]
    load = apload.ApLoad(apred=apred, telescope=telescope)
    with product_build_lock(
        load, "flux", fluxid, clobber=clobber, unlock=unlock,
        verbose=verbose,
    ) as (build, outputs):
        if not build:
            if temp is not None:
                build_response(
                    fluxid, waveid=waveid, temp=temp, load=load,
                    clobber=clobber, unlock=unlock, verbose=verbose)
            return
        if len(outputs) != len(CHIPS):
            raise RuntimeError(
                f"Flux product {fluxid} resolved to {len(outputs)} files; "
                f"expected {len(CHIPS)}")
        for chip in CHIPS:
            path = Path(load.filename("1D", num=fluxid, chip=chip))
            if path.exists() or path.is_symlink():
                path.unlink()
        _process(
            load, frames, cmjd=cmjd, darkid=darkid, flatid=flatid,
            psfid=psfid, modelpsf=modelpsf, waveid=waveid,
            littrowid=littrowid, persistid=persistid, clobber=clobber,
            onedclobber=onedclobber, unlock=unlock, verbose=verbose)
        chips = [_load_1d(load, fluxid, chip) for chip in CHIPS]
        fluxes = [chip["flux"] for chip in chips]
        masks = [chip["mask"] for chip in chips]
        exptype = chips[0]["header"].get("EXPTYPE", "UNKNOWN")
        wavelengths = None
        if str(exptype).strip().upper() == "BLACKBODY":
            if waveid is None:
                raise ValueError("BLACKBODY calibration requires waveid")
            wavelengths = [_load_wavelength(load, waveid, chip) for chip in CHIPS]
        reference, original = make_reference_spectra(
            fluxes, masks, telescope=telescope, exptype=exptype,
            wavelengths=wavelengths, bbtemp=bbtemp)
        calibrations, throughputs = make_flux_calibrations(
            fluxes, masks, reference, mjd=int(load.cmjd(fluxid)),
            telescope=telescope)
        for index, filename in enumerate(outputs):
            _write_flux(filename, calibrations[index], throughputs[index],
                        reference[:, index], original[:, index],
                        chips[0]["header"], apred=apred, exptype=exptype)
    if temp is not None:
        build_response(fluxid, waveid=waveid, temp=temp, load=load,
                       clobber=clobber, unlock=unlock, verbose=verbose)


def _load_wavelength(load, waveid, chip):
    filename = load.filename("Wave", num=waveid, chip=chip)
    data = np.asarray(fits.getdata(filename, 2), float)
    if data.ndim == 1:
        return data
    if data.shape[0] < data.shape[1]:
        return data[data.shape[0] // 2]
    return data[:, data.shape[1] // 2]


def build_response(number, *, waveid, temp, load=None, apred="daily",
                   telescope="apo25m", clobber=False, unlock=False,
                   verbose=False):
    """Build chip-level blackbody response vectors from Flux products."""
    if waveid is None or temp is None:
        raise ValueError("response calibration requires waveid and temp")
    if load is None:
        load = apload.ApLoad(apred=apred, telescope=telescope)
    else:
        apred = load.apred
    with product_build_lock(
        load, "response", number, clobber=clobber, unlock=unlock,
        verbose=verbose,
    ) as (build, outputs):
        if not build:
            return
        if len(outputs) != len(CHIPS):
            raise RuntimeError(
                f"Response product {number} resolved to {len(outputs)} "
                f"files; expected {len(CHIPS)}")
        flux_status = load.product_status("flux", number)
        wave_status = load.product_status("wave", waveid)
        missing = [
            filename
            for filename, complete in {**flux_status, **wave_status}.items()
            if not complete
        ]
        if missing:
            raise FileNotFoundError(
                "Missing Response dependency files: " + ", ".join(missing))
        flux_files = load.product_files("flux", number)
        references = [np.asarray(fits.getdata(name, 3), float)
                      for name in flux_files]
        waves = [_load_wavelength(load, waveid, chip) for chip in CHIPS]
        center = len(references[1]) // 2
        if (any(reference.ndim != 1 for reference in references) or
                any(wave.shape != reference.shape
                    for wave, reference in zip(waves, references))):
            raise ValueError(
                "Flux reference spectra and Wave arrays must be matching 1-D arrays")
        if not np.isfinite(references[1][center]) or references[1][center] == 0:
            raise ValueError("central Flux reference value must be finite and nonzero")
        normalization = references[1][center] / planck(waves[1][center], temp)
        for filename, reference, wave in zip(outputs, references, waves):
            with np.errstate(divide="ignore", invalid="ignore"):
                response = planck(wave, temp) * normalization / reference
            if not np.all(np.isfinite(response)):
                raise ValueError("Response contains nonfinite values")
            header = fits.Header({"APRED": str(apred), "BBTEMP": float(temp),
                                  "WAVEID": str(waveid)})
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            fits.PrimaryHDU(response.astype(np.float32), header).writeto(
                filename, overwrite=True)

