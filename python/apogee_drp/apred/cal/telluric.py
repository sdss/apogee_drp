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
    "build_telluric", "convolve_gauss_hermite_models",
    "convolve_telluric_models", "load_telluric_models",
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
    if model_wave.ndim != 1 or target_wave.ndim not in (1, 2):
        raise ValueError(
            "model_wave must be one-dimensional and target_wave one- or "
            "two-dimensional")
    if models.ndim != 4 or models.shape[0] != 3 or models.shape[-1] != len(model_wave):
        raise ValueError("models must have [3,air,scale,wavelength] dimensions")
    sigmas = _lsf_sigmas(lsf_array) * float(oversample)
    if target_wave.ndim == 1:
        target_wave = np.broadcast_to(target_wave, (len(sigmas), target_wave.size))
    if target_wave.shape[0] != len(sigmas):
        raise ValueError("target_wave and LSF must contain the same fibers")
    output = np.empty((models.shape[1], models.shape[2], 3,
                       len(sigmas), target_wave.shape[1]), dtype=np.float32)
    for fiber, sigma in enumerate(sigmas):
        interpolated = _interpolate_models(
            model_wave, models, target_wave[fiber])
        for species in range(3):
            for air in range(models.shape[1]):
                for scale in range(models.shape[2]):
                    output[air, scale, species, fiber] = gaussian_filter1d(
                        interpolated[species, air, scale], sigma=sigma,
                        mode="nearest", truncate=7.0)
    return output


def _validate_model_grids(model_wave, models, target_wave):
    model_wave = np.asarray(model_wave, dtype=float)
    models = np.asarray(models, dtype=float)
    target_wave = np.asarray(target_wave, dtype=float)
    if model_wave.ndim != 1 or target_wave.ndim != 1:
        raise ValueError("model_wave and target_wave must be one-dimensional")
    if (models.ndim != 4 or models.shape[0] != len(SPECIES) or
            models.shape[-1] != model_wave.size):
        raise ValueError(
            "models must have [3,air,scale,wavelength] dimensions")
    if (model_wave.size < 2 or target_wave.size < 2 or
            not np.all(np.isfinite(model_wave)) or
            not np.all(np.isfinite(target_wave))):
        raise ValueError("wavelength grids must be finite and nonempty")
    return model_wave, models, target_wave


def _interpolate_models(model_wave, models, target_wave):
    """Interpolate every atmosphere model onto one detector wavelength grid."""
    model_wave, models, target_wave = _validate_model_grids(
        model_wave, models, target_wave)
    order = np.argsort(model_wave)
    source_wave = model_wave[order]
    flat = models.reshape(-1, model_wave.size)
    result = np.empty((flat.shape[0], target_wave.size), dtype=float)
    for index, spectrum in enumerate(flat):
        ordered = spectrum[order]
        result[index] = np.interp(
            target_wave, source_wave, ordered,
            left=ordered[0], right=ordered[-1])
    return result.reshape(models.shape[:-1] + (target_wave.size,))


def _normalize_lsf_parameters(parameters):
    """Return LSF parameters as ``[fiber,parameter]``."""
    array = np.asarray(parameters, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError("LSF parameters must be one- or two-dimensional")

    def valid_vector(vector):
        try:
            horder = int(vector[2])
            if horder < 0 or horder > 20 or vector[2] != horder:
                return False
            porder = vector[3:4 + horder].astype(int)
            if len(porder) != horder + 1 or np.any(porder < 0):
                return False
            wing = 4 + horder + int(np.sum(porder + 1))
            nwpar = int(vector[wing + 1])
            wporder = vector[wing + 2:wing + 2 + nwpar].astype(int)
            expected = wing + 2 + nwpar + int(np.sum(wporder + 1))
            return nwpar >= 0 and expected == len(vector)
        except (IndexError, TypeError, ValueError, OverflowError):
            return False

    row_valid = valid_vector(array[0])
    column_valid = valid_vector(array[:, 0])
    if column_valid and not row_valid:
        array = array.T
    elif not row_valid:
        raise ValueError("could not identify LSF parameter-vector orientation")
    if array.shape[1] < 10:
        raise ValueError("LSF parameter vectors are incomplete")
    if not np.all(np.isfinite(array)):
        raise ValueError("LSF parameters must be finite")
    return array


def _apply_position_dependent_lsf(spectra, profiles):
    """Convolve spectra with profiles varying at every output sample."""
    spectra = np.asarray(spectra, dtype=float)
    profiles = np.asarray(profiles, dtype=float)
    if spectra.ndim != 2:
        raise ValueError("spectra must have [model,wavelength] dimensions")
    if profiles.ndim != 2 or profiles.shape[0] != spectra.shape[1]:
        raise ValueError("profiles must have [wavelength,offset] dimensions")

    half = profiles.shape[1] // 2
    padded = np.pad(spectra, ((0, 0), (half, half)), mode="edge")
    output = np.zeros_like(spectra)
    for index in range(profiles.shape[1]):
        output += padded[:, index:index + spectra.shape[1]] * profiles[:, index]
    return output


def convolve_gauss_hermite_models(
        model_wave, models, target_wave, fine_pixel, lsf_parameters, *,
        oversample=2, kernel_half_width=14):
    """Convolve telluric grids with the full APOGEE Gauss-Hermite LSF.

    The Doppler evaluator retains the Hermite terms, extended wings, and
    detector-position dependence that are lost when an LSF is replaced by a
    single Gaussian width.  The result has dimensions
    ``[air,scale,species,fiber,wavelength]``.
    """
    from .lsf import evaluate_gauss_hermite_lsf

    model_wave = np.asarray(model_wave, dtype=float)
    models = np.asarray(models, dtype=float)
    target_wave = np.asarray(target_wave, dtype=float)
    if target_wave.ndim == 1:
        target_wave = target_wave[None, :]
    if target_wave.ndim != 2:
        raise ValueError("target_wave must be one- or two-dimensional")
    _validate_model_grids(model_wave, models, target_wave[0])
    fine_pixel = np.asarray(fine_pixel, dtype=float)
    if fine_pixel.ndim == 1:
        fine_pixel = np.broadcast_to(fine_pixel, target_wave.shape)
    if (fine_pixel.shape != target_wave.shape or
            not np.all(np.isfinite(fine_pixel)) or
            not np.all(np.isfinite(target_wave))):
        raise ValueError("fine_pixel must match target_wave")
    if int(oversample) != oversample or int(oversample) <= 0:
        raise ValueError("oversample must be a positive integer")
    if int(kernel_half_width) != kernel_half_width or kernel_half_width <= 0:
        raise ValueError("kernel_half_width must be a positive integer")
    oversample = int(oversample)
    parameters = _normalize_lsf_parameters(lsf_parameters)
    if target_wave.shape[0] == 1 and parameters.shape[0] > 1:
        target_wave = np.broadcast_to(
            target_wave, (parameters.shape[0], target_wave.shape[1]))
        fine_pixel = np.broadcast_to(fine_pixel, target_wave.shape)
    if target_wave.shape[0] != parameters.shape[0]:
        raise ValueError(
            "target_wave and LSF parameters must contain the same fibers")

    half = int(kernel_half_width) * oversample
    sample_offsets = np.arange(-half, half + 1, dtype=float)
    detector_offsets = sample_offsets / oversample
    output = np.empty(
        models.shape[1:3] + (len(SPECIES), parameters.shape[0],
                             target_wave.shape[1]),
        dtype=np.float32)
    for fiber, vector in enumerate(parameters):
        interpolated = _interpolate_models(
            model_wave, models, target_wave[fiber])
        flat = interpolated.reshape(-1, target_wave.shape[1])
        profiles = evaluate_gauss_hermite_lsf(
            vector, fine_pixel[fiber], detector_offsets,
            positive=True, normalize=True)
        convolved = _apply_position_dependent_lsf(flat, profiles)
        shaped = convolved.reshape(
            len(SPECIES), models.shape[1], models.shape[2],
            target_wave.shape[1])
        output[:, :, :, fiber] = shaped.transpose(1, 2, 0, 3)
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


def _load_lsf(filename):
    """Load the parameter vectors, evaluated array, and fitting method."""
    with fits.open(filename) as hdus:
        if hdus[0].data is None:
            raise ValueError(f"LSF file has no parameter array: {filename}")
        method = str(hdus[0].header.get("LSFMETH", "")).strip().upper()
        array = (None if len(hdus) <= 1 or hdus[1].data is None else
                 np.asarray(hdus[1].data, dtype=float))
        raw_parameters = np.asarray(hdus[0].data, dtype=float)
        parameters = (_normalize_lsf_parameters(raw_parameters)
                      if method in {"GAUSS-HERMITE", "GAUSS_HERMITE", "GH"}
                      else raw_parameters)
    return parameters, array, method


def _write_telluric(filename, wavelength, convolved, metadata, *, apred,
                    waveid, lsfid, lsf_method):
    nfiber = convolved.shape[3]
    wavelength = np.asarray(wavelength, dtype=float)
    if wavelength.ndim == 1:
        wavelength = np.broadcast_to(wavelength, (nfiber, wavelength.size))
    if wavelength.shape != (nfiber, convolved.shape[-1]):
        raise ValueError("wavelength must match convolved fiber/pixel dimensions")
    wave_image = wavelength.astype(np.float64)
    primary = fits.PrimaryHDU(wave_image)
    primary.header["EXTNAME"] = "WAVELENGTH"
    primary.header["AIR0"] = metadata["air0"]
    primary.header["DAIR"] = metadata["dair"]
    primary.header["NSPECIES"] = 3
    primary.header["NSCALE"] = metadata["nscale"]
    primary.header["APRED"] = str(apred)
    primary.header["WAVEID"] = str(waveid)
    primary.header["LSFID"] = str(lsfid)
    primary.header["LSFMETH"] = str(lsf_method)
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
                   full=True, kernel_half_width=14,
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
            lsf_parameters, lsf_array, lsf_method = _load_lsf(lsffile)
            use_full = bool(full) and lsf_method in {
                "GAUSS-HERMITE", "GAUSS_HERMITE", "GH"}
            nfiber = (lsf_parameters.shape[0] if use_full else
                      np.asarray(lsf_array).shape[1])
            if wave_grid.shape[0] != nfiber:
                raise ValueError(
                    f"Wave and LSF fiber counts differ for chip {chip}: "
                    f"{wave_grid.shape[0]} != {nfiber}")
            sampled = [oversampled_wavelength(
                wave_grid[fiber], oversample=oversample, extend=extend)
                for fiber in range(nfiber)]
            fine_pixel = np.asarray([item[0] for item in sampled])
            fine_wave = np.asarray([item[1] for item in sampled])
            if use_full:
                convolved = convolve_gauss_hermite_models(
                    model_wave, models, fine_wave, fine_pixel,
                    lsf_parameters, oversample=oversample,
                    kernel_half_width=kernel_half_width)
                convolution_method = "GAUSS-HERMITE"
            else:
                if lsf_array is None:
                    raise ValueError(f"LSF file has no LSF ARRAY: {lsffile}")
                convolved = convolve_telluric_models(
                    model_wave, models, fine_wave, lsf_array,
                    oversample=oversample)
                convolution_method = "GAUSSIAN"
            _write_telluric(
                output, fine_wave, convolved, metadata, apred=apred,
                waveid=waveid, lsfid=lsfid,
                lsf_method=convolution_method)
            if verbose:
                print(f" writing Telluric chip {chip}: {output}")
        if not load.product_exists("telluric", name):
            raise RuntimeError(f"Telluric {name} did not create all chip files")
        directory = Path(load.filename("Telluric", num=0, directory=True))
        prefix = getattr(
            load, "prefix", "ap" if "apo" in load.telescope else "as")
        (directory / f"{prefix}Telluric-{name}.dat").touch()
