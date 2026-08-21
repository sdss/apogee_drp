"""Tests for the numbered Telluric calibration implementation."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from apogee_drp.apred.cal import makecal
from apogee_drp.apred.cal import telluric
from apogee_drp.apred.cal import utils as cal_utils


class FakeLoad:
    apred = "daily"
    telescope = "apo25m"
    prefix = "ap"

    def __init__(self, root):
        self.root = Path(root)
        self.delete_calls = []

    def filename(self, kind, num=None, chip=None, directory=False, **kwargs):
        root = self.root / kind.lower()
        if directory:
            return str(root)
        infix = f"-{chip}" if chip is not None else ""
        return str(root / f"ap{kind}{infix}-{num}.fits")

    def product_files(self, product, name):
        if product == "telluric":
            return [str(self.root / "telluric" /
                        f"apTelluric-{chip}-{name}.fits") for chip in "abc"]
        if product == "wave":
            return [self.filename("Wave", num=name, chip=chip)
                    for chip in "abc"]
        if product == "lsf":
            files = [self.filename("LSF", num=name, chip=chip)
                     for chip in "abc"]
            files.append(str(self.root / "lsf" /
                             f"apLSF-{name}-diagnostics.fits"))
            return files
        raise AssertionError(product)

    def product_status(self, product, name):
        return {filename: Path(filename).is_file() and
                Path(filename).stat().st_size > 0
                for filename in self.product_files(product, name)}

    def product_exists(self, product, name):
        return all(self.product_status(product, name).values())

    def product_delete(self, product, name, **kwargs):
        self.delete_calls.append((product, name, kwargs))
        for filename in self.product_files(product, name):
            path = Path(filename)
            if path.exists() or path.is_symlink():
                path.unlink()


@pytest.mark.parametrize(
    "value, expected", [("12-19601", (12, 19601)),
                         (" 12345678-99 ", (12345678, 99))])
def test_parse_telluric_id(value, expected):
    assert telluric.parse_telluric_id(value) == expected


@pytest.mark.parametrize("value", ["12", "12-13-14", "a-13", "0-2", "1--2"])
def test_parse_telluric_id_rejects_invalid_names(value):
    with pytest.raises(ValueError, match="Telluric ID|positive"):
        telluric.parse_telluric_id(value)


def test_product_files_preserve_compound_id(tmp_path):
    files = FakeLoad(tmp_path).product_files("telluric", "12-19601")
    assert [Path(filename).name for filename in files] == [
        "apTelluric-a-12-19601.fits", "apTelluric-b-12-19601.fits",
        "apTelluric-c-12-19601.fits"]


def test_obsolete_product_filename_and_load_helpers_are_removed():
    assert not hasattr(telluric, "product_files")
    assert not hasattr(telluric, "_input_files")
    assert not hasattr(telluric, "_telluric_directory")
    assert not hasattr(telluric, "_make_load")


def test_load_repository_models():
    wave, models, metadata = telluric.load_telluric_models()
    assert models.shape == (3, 7, 4, 19500)
    assert wave[0] == 15100
    assert wave[1] - wave[0] == pytest.approx(0.1)
    assert metadata == {
        "air0": 1.0, "dair": 0.25, "scale0": 0.5, "dscale": 0.5,
        "nair": 7, "nscale": 4}


def test_load_models_rejects_missing_species(tmp_path):
    with pytest.raises(FileNotFoundError, match="CH4"):
        telluric.load_telluric_models(tmp_path)


def test_load_models_rejects_non_grid(tmp_path):
    for species in telluric.SPECIES:
        fits.PrimaryHDU(np.zeros(10)).writeto(tmp_path / f"{species}.fits")
    with pytest.raises(ValueError, match="3-D"):
        telluric.load_telluric_models(tmp_path)


def test_oversampled_wavelength_matches_idl_grid():
    pixel, wave = telluric.oversampled_wavelength(
        np.arange(5.0) + 15000, oversample=2, extend=2)
    np.testing.assert_allclose(pixel[:5], [-2, -1.5, -1, -0.5, 0])
    assert len(pixel) == 18
    np.testing.assert_allclose(wave, pixel + 15000)


@pytest.mark.parametrize("kwargs", [
    {"oversample": 0}, {"oversample": 1.5}, {"extend": -1}])
def test_oversampled_wavelength_validates_options(kwargs):
    with pytest.raises(ValueError):
        telluric.oversampled_wavelength([1, 2], **kwargs)


def gaussian_lsf(nfiber=2, npix=10, width=7, sigma=1.0):
    offset = np.arange(width) - width // 2
    profile = np.exp(-0.5 * (offset / sigma) ** 2)
    profile /= profile.sum()
    return np.broadcast_to(profile[:, None, None],
                           (width, nfiber, npix)).copy()


def test_lsf_sigmas_recover_gaussian_width():
    sigma = telluric._lsf_sigmas(gaussian_lsf(sigma=1.2))
    np.testing.assert_allclose(sigma, 1.2, atol=0.04)


def test_lsf_sigmas_reject_invalid_array():
    with pytest.raises(ValueError, match="dimensions"):
        telluric._lsf_sigmas(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="unnormalized"):
        telluric._lsf_sigmas(np.zeros((3, 2, 4)))


def test_convolution_shape_and_constant_preservation():
    wave = np.linspace(15000, 15100, 100)
    models = np.ones((3, 2, 2, 100))
    target = np.linspace(15010, 15090, 40)
    result = telluric.convolve_telluric_models(
        wave, models, target, gaussian_lsf(nfiber=3), oversample=2)
    assert result.shape == (2, 2, 3, 3, 40)
    np.testing.assert_allclose(result, 1)


def test_convolution_accepts_distinct_fiber_wavelength_grids():
    wave = np.linspace(15000, 15100, 100)
    models = np.ones((3, 1, 1, wave.size))
    target = np.vstack((wave[10:30], wave[11:31]))
    result = telluric.convolve_telluric_models(
        wave, models, target, gaussian_lsf(nfiber=2), oversample=1)
    assert result.shape == (1, 1, 3, 2, 20)
    np.testing.assert_allclose(result, 1)


def test_convolution_smooths_narrow_absorption():
    wave = np.linspace(15000, 15100, 101)
    models = np.ones((3, 1, 1, 101))
    models[:, :, :, 50] = 0
    narrow = telluric.convolve_telluric_models(
        wave, models, wave, gaussian_lsf(sigma=0.5), oversample=1)
    broad = telluric.convolve_telluric_models(
        wave, models, wave, gaussian_lsf(sigma=1.5), oversample=1)
    assert narrow[0, 0, 0, 0, 50] < broad[0, 0, 0, 0, 50]


def test_convolution_does_not_modify_inputs():
    wave = np.linspace(1, 2, 20)
    models = np.ones((3, 1, 1, 20))
    lsf = gaussian_lsf(nfiber=1, npix=5)
    original = models.copy(), lsf.copy()
    telluric.convolve_telluric_models(wave, models, wave, lsf)
    np.testing.assert_array_equal(models, original[0])
    np.testing.assert_array_equal(lsf, original[1])


def gh_parameters(nfiber=2):
    # binsize, xoffset, Horder, Porder[0:6], GH coefficients,
    # Wproftype, nWpar, WPorder[0:2], wing coefficients.
    vector = np.array([
        1, -1024, 5, 0, 0, 0, 0, 0, 0,
        1.1, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0.03, 3.5,
    ], dtype=float)
    return np.broadcast_to(vector, (nfiber, vector.size)).copy()


def test_normalize_lsf_parameters_accepts_fits_orientation():
    parameters = gh_parameters(3)
    np.testing.assert_array_equal(
        telluric._normalize_lsf_parameters(parameters.T), parameters)


def test_position_dependent_convolution_uses_each_profile():
    spectra = np.zeros((1, 7))
    spectra[0, 3] = 1
    profiles = np.zeros((7, 3))
    profiles[:, 1] = 1
    profiles[2] = [0, 0, 1]
    result = telluric._apply_position_dependent_lsf(spectra, profiles)
    assert result[0, 3] == 1
    assert result[0, 2] == 1


def test_gauss_hermite_convolution_uses_full_evaluator(monkeypatch):
    from apogee_drp.apred.cal import lsf

    calls = []

    def evaluator(parameters, centers, offsets, **kwargs):
        calls.append((parameters.copy(), centers.copy(), offsets.copy(), kwargs))
        profile = np.exp(-0.5 * (offsets / 1.2) ** 2)
        profile /= profile.sum()
        return np.broadcast_to(profile, (len(centers), len(offsets))).copy()

    monkeypatch.setattr(lsf, "evaluate_gauss_hermite_lsf", evaluator)
    wave = np.linspace(15000, 15100, 31)
    models = np.ones((3, 1, 1, wave.size))
    result = telluric.convolve_gauss_hermite_models(
        wave, models, wave, np.arange(wave.size) / 2,
        gh_parameters(2), oversample=2, kernel_half_width=4)
    assert result.shape == (1, 1, 3, 2, wave.size)
    np.testing.assert_allclose(result, 1, atol=1e-6)
    assert len(calls) == 2
    np.testing.assert_allclose(calls[0][2], np.arange(-8, 9) / 2)
    assert calls[0][3] == {"positive": True, "normalize": True}


def test_write_telluric_has_idl_compatible_layout(tmp_path):
    filename = tmp_path / "tell.fits"
    metadata = {"air0": 1.0, "dair": 0.25, "scale0": 0.5,
                "dscale": 0.5, "nair": 2, "nscale": 2}
    data = np.ones((2, 2, 3, 4, 10), dtype=np.float32)
    telluric._write_telluric(
        filename, np.arange(10), data, metadata, apred="daily",
        waveid=12, lsfid=34, lsf_method="GAUSS-HERMITE")
    with fits.open(filename) as hdus:
        assert len(hdus) == 3
        assert hdus[0].data.shape == (4, 10)
        assert hdus[0].header["NSPECIES"] == 3
        assert hdus[0].header["LSFMETH"] == "GAUSS-HERMITE"
        assert hdus[1].data.shape == (2, 3, 4, 10)
        assert hdus[1].header["AIRMASS"] == 1
        assert hdus[2].header["AIRMASS"] == 1.25
        assert hdus[1].header["SPECIES3"] == "H2O"


def patch_builder(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    monkeypatch.setattr(telluric.apload, "ApLoad", lambda **kwargs: load)
    locks = []
    monkeypatch.setattr(
        cal_utils.lock, "lock",
        lambda *args, **kwargs: locks.append((args, kwargs)))
    for product, number in (("wave", 12), ("lsf", 34)):
        for filename in load.product_files(product, number):
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            Path(filename).write_bytes(b"dependency")
    monkeypatch.setattr(
        telluric, "load_telluric_models",
        lambda directory=None: (
            np.linspace(15000, 15100, 20), np.ones((3, 2, 2, 20)),
            {"air0": 1.0, "dair": 0.25, "scale0": 0.5,
             "dscale": 0.5, "nair": 2, "nscale": 2}))
    monkeypatch.setattr(
        telluric, "_load_wave_grid",
        lambda filename: np.broadcast_to(np.linspace(15010, 15090, 8), (4, 8)))
    monkeypatch.setattr(
        telluric, "_load_lsf",
        lambda filename: (
            gh_parameters(4), gaussian_lsf(nfiber=4, npix=8), "GAUSSIAN"))
    monkeypatch.setattr(
        telluric, "convolve_telluric_models",
        lambda model_wave, models, target_wave, lsf_array, **kwargs:
        np.ones((2, 2, 3, 4, target_wave.shape[-1]), dtype=np.float32))
    return load, locks


def test_build_telluric_dispatches_full_gh_convolution(
        monkeypatch, tmp_path):
    load, _ = patch_builder(monkeypatch, tmp_path)
    monkeypatch.setattr(
        telluric, "_load_lsf",
        lambda filename: (
            gh_parameters(4), gaussian_lsf(nfiber=4, npix=8),
            "GAUSS-HERMITE"))
    calls = []

    def convolve(model_wave, models, target_wave, fine_pixel, parameters,
                 **kwargs):
        calls.append((fine_pixel.copy(), parameters.copy(), kwargs))
        return np.ones((2, 2, 3, 4, target_wave.shape[-1]), dtype=np.float32)

    monkeypatch.setattr(telluric, "convolve_gauss_hermite_models", convolve)
    telluric.build_telluric(
        "12-34", oversample=2, kernel_half_width=9)
    assert len(calls) == 3
    assert calls[0][1].shape == (4, gh_parameters().shape[1])
    assert calls[0][2] == {"oversample": 2, "kernel_half_width": 9}
    with fits.open(load.product_files("telluric", "12-34")[0]) as hdus:
        assert hdus[0].header["LSFMETH"] == "GAUSS-HERMITE"


def test_build_telluric_writes_three_chips(monkeypatch, tmp_path, capsys):
    load, locks = patch_builder(monkeypatch, tmp_path)
    assert telluric.build_telluric("12-34", verbose=True) is None
    outputs = load.product_files("telluric", "12-34")
    assert all(Path(filename).stat().st_size > 0 for filename in outputs)
    assert (tmp_path / "telluric" / "apTelluric-12-34.dat").is_file()
    assert "writing Telluric chip c" in capsys.readouterr().out
    assert locks[-1][1] == {"clear": True}


def test_build_telluric_existing_short_circuit(monkeypatch, tmp_path, capsys):
    load, locks = patch_builder(monkeypatch, tmp_path)
    outputs = load.product_files("telluric", "12-34")
    for filename in outputs:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"existing")
    assert telluric.build_telluric("12-34", verbose=True) is None
    assert "telluric product 12-34 already exists" in capsys.readouterr().out
    assert not locks


def test_build_telluric_nowait_uses_zero_wait(monkeypatch, tmp_path):
    _, locks = patch_builder(monkeypatch, tmp_path)
    telluric.build_telluric("12-34", nowait=True)
    assert locks[0][1]["waittime"] == 0


def test_build_telluric_reports_missing_dependencies(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    monkeypatch.setattr(telluric.apload, "ApLoad", lambda **kwargs: load)
    lock_calls = []
    monkeypatch.setattr(
        cal_utils.lock, "lock",
        lambda *args, **kwargs: lock_calls.append((args, kwargs)))
    with pytest.raises(FileNotFoundError, match="Missing Telluric dependency"):
        telluric.build_telluric("12-34")
    assert lock_calls[-1][1] == {"clear": True}


def test_build_telluric_clears_lock_on_failure(monkeypatch, tmp_path):
    _, locks = patch_builder(monkeypatch, tmp_path)
    monkeypatch.setattr(
        telluric, "load_telluric_models",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        telluric.build_telluric("12-34")
    assert locks[-1][1] == {"clear": True}


def test_makecal_telluric_dispatches_current_builder(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        makecal, "build_telluric",
        lambda *args, **kwargs: calls.append((args, kwargs)))
    context = makecal.CalibrationContext(
        load=FakeLoad(tmp_path), calfile="cal.par", allcaldict={},
        clobber=True, unlock=True, verbose=True)
    makecal.telluric("12-34", context)
    assert calls == [(('12-34',), {
        "apred": "daily", "telescope": "apo25m", "clobber": True,
        "unlock": True, "verbose": True})]
