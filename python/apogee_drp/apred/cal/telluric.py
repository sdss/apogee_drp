"""Build LSF-convolved APOGEE telluric-model calibrations."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from ...utils import apload
from .utils import product_build_lock

CHIPS = ("a", "b", "c")
SPECIES = ("CH4", "CO2", "H2O")

__all__ = [
    "build_telluric", "convolve_telluric_models", "load_telluric_models",
    "oversampled_wavelength", "parse_telluric_id",
]


def parse_telluric_id(tellid):
    """Return integer ``(waveid, lsfid)`` from ``WAVEID-LSFID``."""
    name = str(tellid).strip()
    parts = name.split("-")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(
            f"Telluric ID must be '<waveid>-<lsfid>', got {tellid!r}")
    waveid, lsfid = map(int, parts)
    if waveid <= 0 or lsfid <= 0:
        raise ValueError("waveid and lsfid must be positive")
    return waveid, lsfid


def _default_model_directory():
    environment = os.environ.get("APOGEE_DRP_DIR")
    if environment:
        candidate = Path(environment) / "data" / "telluric"
        if candidate.is_dir():
            return candidate
    return Path(__file__).resolve().parents[4] / "data" / "telluric"


def load_telluric_models(model_directory=None):
    """Load CH4, CO2, and H2O grids as ``[species,air,scale,wave]``."""
    directory = (_default_model_directory() if model_directory is None
                 else Path(model_directory))
    grids, wavelength, metadata = [], None, None
    for species in SPECIES:
        filename = directory / f"{species}.fits"
        if not filename.is_file():
            raise FileNotFoundError(f"Telluric model not found: {filename}")
        data, header = fits.getdata(filename, 0, header=True)
        data = np.asarray(data, dtype=float)
        if data.ndim != 3:
            raise ValueError(f"{filename} must contain a 3-D model grid")
        nscale, nair, nwave = data.shape
        wave = (float(header["CRVAL1"]) +
                np.arange(nwave, dtype=float) * float(header["CDELT1"]))
        if wavelength is not None and not np.array_equal(wave, wavelength):
            raise ValueError("telluric species use inconsistent wavelength grids")
        wavelength = wave
        current = {
            "air0": float(header.get("CRVAL2", 1.0)),
            "dair": float(header.get("CDELT2", 1.0)),
            "scale0": float(header.get("CRVAL3", 1.0)),
            "dscale": float(header.get("CDELT3", 1.0)),
            "nair": nair, "nscale": nscale,
        }
        if metadata is not None and current != metadata:
            raise ValueError("telluric species use inconsistent air/scale grids")
        metadata = current
        grids.append(data.transpose(1, 0, 2))
    return wavelength, np.asarray(grids), metadata


def oversampled_wavelength(pixel_wavelength, *, oversample=2, extend=20):
    """Evaluate a detector wavelength solution on the IDL half-pixel grid."""
    values = np.asarray(pixel_wavelength, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("pixel_wavelength must be a finite one-dimensional array")
    if int(oversample) != oversample or int(oversample) <= 0:
        raise ValueError("oversample must be a positive integer")
    if int(extend) != extend or int(extend) < 0:
        raise ValueError("extend must be a nonnegative integer")
    oversample, extend = int(oversample), int(extend)
    fine_pixel = np.arange(
        oversample * (values.size + 2 * extend), dtype=float) / oversample - extend
    interpolation = interp1d(
        np.arange(values.size, dtype=float), values, kind="linear",
        bounds_error=False, fill_value="extrapolate", assume_sorted=True)
    return fine_pixel, np.asarray(interpolation(fine_pixel), float)


def _lsf_sigmas(lsf_array):
    array = np.asarray(lsf_array, dtype=float)
    if array.ndim != 3:
        raise ValueError("LSF array must have [offset,fiber,pixel] dimensions")
    weights = np.maximum(array, 0)
    total = weights.sum(axis=0)
    if np.any(total <= 0):
        raise ValueError("LSF array contains an unnormalized fiber/pixel")
    offsets = np.arange(array.shape[0], dtype=float) - array.shape[0] // 2
    variance = ((weights * offsets[:, None, None] ** 2).sum(axis=0) / total)
    sigma = np.sqrt(np.maximum(variance, 0))
    result = np.nanmedian(sigma, axis=1)
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("could not determine positive LSF widths")
    return result


def convolve_telluric_models(model_wave, models, target_wave, lsf_array,
                              *, oversample=2):
    """Interpolate and LSF-convolve all telluric models for every fiber.

    Returns an array with dimensions ``[air,scale,species,fiber,wavelength]``.
    The current LSF products vary slowly with detector position; the median
    second-moment width for each fiber is used, matching their Gaussian model.
    """
    model_wave = np.asarray(model_wave, float)
    models = np.asarray(models, float)
    target_wave = np.asarray(target_wave, float)
    if model_wave.ndim != 1 or target_wave.ndim != 1:
        raise ValueError("model_wave and target_wave must be one-dimensional")
    if models.ndim != 4 or models.shape[0] != 3 or models.shape[-1] != len(model_wave):
        raise ValueError("models must have [3,air,scale,wavelength] dimensions")
    order = np.argsort(model_wave)
    interpolated = np.empty(models.shape[:-1] + (len(target_wave),), float)
    for species in range(3):
        for air in range(models.shape[1]):
            for scale in range(models.shape[2]):
                interpolated[species, air, scale] = np.interp(
                    target_wave, model_wave[order], models[species, air, scale, order],
                    left=models[species, air, scale, order[0]],
                    right=models[species, air, scale, order[-1]])
    sigmas = _lsf_sigmas(lsf_array) * float(oversample)
    output = np.empty((models.shape[1], models.shape[2], 3,
                       len(sigmas), len(target_wave)), dtype=np.float32)
    for fiber, sigma in enumerate(sigmas):
        for species in range(3):
            for air in range(models.shape[1]):
                for scale in range(models.shape[2]):
                    output[air, scale, species, fiber] = gaussian_filter1d(
                        interpolated[species, air, scale], sigma=sigma,
                        mode="nearest", truncate=7.0)
    return output


def _load_wave_grid(filename):
    with fits.open(filename) as hdus:
        if len(hdus) <= 2 or hdus[2].data is None:
            raise ValueError(f"Wave file has no wavelength extension: {filename}")
        wavelength = np.asarray(hdus[2].data, float)
    if wavelength.ndim == 1:
        return wavelength[None, :]
    if wavelength.ndim != 2:
        raise ValueError(f"Wave grid must be 2-D: {filename}")
    return wavelength if wavelength.shape[0] <= wavelength.shape[1] else wavelength.T


def _load_lsf_array(filename):
    with fits.open(filename) as hdus:
        if len(hdus) > 1 and hdus[1].data is not None:
            return np.asarray(hdus[1].data, float)
        raise ValueError(f"LSF file has no LSF ARRAY extension: {filename}")


def _write_telluric(filename, wavelength, convolved, metadata, *, apred,
                    waveid, lsfid):
    nfiber = convolved.shape[3]
    wave_image = np.broadcast_to(wavelength, (nfiber, len(wavelength))).astype(np.float64)
    primary = fits.PrimaryHDU(wave_image)
    primary.header["EXTNAME"] = "WAVELENGTH"
    primary.header["AIR0"] = metadata["air0"]
    primary.header["DAIR"] = metadata["dair"]
    primary.header["NSPECIES"] = 3
    primary.header["NSCALE"] = metadata["nscale"]
    primary.header["APRED"] = str(apred)
    primary.header["WAVEID"] = str(waveid)
    primary.header["LSFID"] = str(lsfid)
    hdus = [primary]
    for air in range(metadata["nair"]):
        image = fits.ImageHDU(convolved[air], name="TELLURIC")
        image.header["AIRMASS"] = metadata["air0"] + air * metadata["dair"]
        image.header["SPECIES1"] = "CH4"
        image.header["SPECIES2"] = "CO2"
        image.header["SPECIES3"] = "H2O"
        image.header["CRVAL4"] = metadata["scale0"]
        image.header["CDELT4"] = metadata["dscale"]
        hdus.append(image)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(filename, overwrite=True)


def build_telluric(tellid, *, apred="daily", telescope="apo25m",
                   model_directory=None, oversample=2, extend=20,
                   clobber=False, nowait=False, unlock=False, verbose=False):
    """Build three ``Telluric-WAVEID-LSFID`` calibration files."""
    waveid, lsfid = parse_telluric_id(tellid)
    name = f"{waveid}-{lsfid}"
    load = apload.ApLoad(apred=apred, telescope=telescope)
    with product_build_lock(
        load, "telluric", name, clobber=clobber, unlock=unlock,
        waittime=(0 if nowait else 10), verbose=verbose,
    ) as (build, outputs):
        if not build:
            return
        if len(outputs) != len(CHIPS):
            raise RuntimeError(
                f"Telluric product {name} resolved to {len(outputs)} files; "
                f"expected {len(CHIPS)}")
        wave_status = load.product_status("wave", waveid)
        lsf_status = load.product_status("lsf", lsfid)
        missing = [filename for filename, complete in
                   {**wave_status, **lsf_status}.items() if not complete]
        if missing:
            raise FileNotFoundError(
                "Missing Telluric dependency files: " + ", ".join(missing))
        wavefiles = load.product_files("wave", waveid)
        lsffiles = load.product_files("lsf", lsfid)[:len(CHIPS)]
        model_wave, models, metadata = load_telluric_models(model_directory)
        for chip, wavefile, lsffile, output in zip(
                CHIPS, wavefiles, lsffiles, outputs):
            wave_grid = _load_wave_grid(wavefile)
            center_wave = wave_grid[wave_grid.shape[0] // 2]
            _, fine_wave = oversampled_wavelength(
                center_wave, oversample=oversample, extend=extend)
            lsf_array = _load_lsf_array(lsffile)
            convolved = convolve_telluric_models(
                model_wave, models, fine_wave, lsf_array,
                oversample=oversample)
            _write_telluric(
                output, fine_wave, convolved, metadata, apred=apred,
                waveid=waveid, lsfid=lsfid)
            if verbose:
                print(f" writing Telluric chip {chip}: {output}")
        if not load.product_exists("telluric", name):
            raise RuntimeError(f"Telluric {name} did not create all chip files")
        directory = Path(load.filename("Telluric", num=0, directory=True))
        prefix = getattr(
            load, "prefix", "ap" if "apo" in load.telescope else "as")
        (directory / f"{prefix}Telluric-{name}.dat").touch()
