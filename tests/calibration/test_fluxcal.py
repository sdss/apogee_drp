"""Tests for the numbered flux-calibration implementation."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from apogee_drp.apred.cal import fluxcal
from apogee_drp.apred.cal import makecal
from apogee_drp.apred.cal import utils as cal_utils


class FakeLoad:
    apred = "daily"
    telescope = "apo25m"
    instrument = "apogee-n"

    def __init__(self, root):
        self.root = Path(root)
        self.delete_calls = []
        self.spectra = None
        self.waves = None

    def cmjd(self, number):
        return "60000"

    def filename(self, kind, num=None, chip=None, directory=False, **kwargs):
        root = self.root / ("exp" if kind == "1D" else "cal")
        if directory:
            return str(root)
        infix = f"-{chip}" if chip is not None else ""
        return str(root / f"ap{kind}{infix}-{int(num):08d}.fits")

    def product_files(self, product, name):
        kind = {"flux": "Flux", "response": "Response", "wave": "Wave"}[product]
        return [self.filename(kind, num=name, chip=chip) for chip in "abc"]

    def product_status(self, product, name):
        return {
            filename: Path(filename).is_file() and Path(filename).stat().st_size > 0
            for filename in self.product_files(product, name)
        }

    def product_exists(self, product, name):
        return all(self.product_status(product, name).values())

    def product_delete(self, product, name, **kwargs):
        self.delete_calls.append((product, name, kwargs))
        for filename in self.product_files(product, name):
            path = Path(filename)
            if path.exists() or path.is_symlink():
                path.unlink()

    def spectrum(self, number, chip=None):
        if self.spectra is None:
            raise RuntimeError("synthetic spectra have not been configured")
        return self.spectra if chip is None else self.spectra[chip]

    def wave(self, number, chip=None):
        if self.waves is None:
            raise RuntimeError("synthetic wavelengths have not been configured")
        return self.waves if chip is None else self.waves[chip]


def test_product_files_are_three_chips(tmp_path):
    names = FakeLoad(tmp_path).product_files("flux", 123)
    assert [Path(name).name for name in names] == [
        "apFlux-a-00000123.fits", "apFlux-b-00000123.fits",
        "apFlux-c-00000123.fits"]


def test_obsolete_product_filename_and_load_helpers_are_removed():
    assert not hasattr(fluxcal, "product_files")
    assert not hasattr(fluxcal, "_chip_filename")
    assert not hasattr(fluxcal, "_make_load")
    assert not hasattr(fluxcal, "_process")
    assert not hasattr(fluxcal, "_load_1d")
    assert not hasattr(fluxcal, "_load_wavelength")


def test_planck_shape_and_temperature_dependence():
    wave = np.array([12000.0, 16000.0])
    cool = fluxcal.planck(wave, 3000)
    hot = fluxcal.planck(wave, 6000)
    assert cool.shape == wave.shape
    assert np.all(np.isfinite(cool))
    assert np.all(hot > cool)


@pytest.mark.parametrize("wave, temperature", [([-1], 3000), ([1], 0)])
def test_planck_rejects_nonphysical_values(wave, temperature):
    with pytest.raises(ValueError, match="positive"):
        fluxcal.planck(wave, temperature)


def simple_frames(npix=64, nfiber=8):
    pixel = np.arange(npix, dtype=float)
    spectrum = 100 + 0.1 * pixel
    fluxes = [spectrum[:, None] * np.ones((1, nfiber)) for _ in range(3)]
    masks = [np.zeros((npix, nfiber), dtype=np.int16) for _ in range(3)]
    return fluxes, masks


def test_reference_spectrum_follows_smooth_lamp():
    fluxes, masks = simple_frames()
    fitted, original = fluxcal.make_reference_spectra(
        fluxes, masks, median_width=5)
    assert fitted.shape == original.shape == (64, 3)
    np.testing.assert_allclose(fitted[10:-10, 0], fluxes[0][10:-10, 0], rtol=2e-4)


def test_reference_ignores_masked_outlier():
    fluxes, masks = simple_frames()
    fluxes[1][30, 2] = 1e12
    masks[1][30, 2] = 1
    _, original = fluxcal.make_reference_spectra(
        fluxes, masks, median_width=1, bad_pixel_bits=1)
    assert original[30, 1] == pytest.approx(103.0)


def test_blackbody_reference_requires_inputs():
    fluxes, masks = simple_frames()
    with pytest.raises(ValueError, match="wavelength"):
        fluxcal.make_reference_spectra(fluxes, masks, exptype="BLACKBODY")
    waves = [np.linspace(12000, 17000, 64)] * 3
    with pytest.raises(ValueError, match="bbtemp"):
        fluxcal.make_reference_spectra(
            fluxes, masks, exptype="BLACKBODY", wavelengths=waves)


def test_blackbody_reference_is_normalized():
    fluxes, masks = simple_frames()
    waves = [np.linspace(12000, 17000, 64)] * 3
    fitted, original = fluxcal.make_reference_spectra(
        fluxes, masks, exptype=" BLACKBODY ", wavelengths=waves,
        bbtemp=4000)
    assert fitted[32, 1] == pytest.approx(1)
    assert original[32, 1] != pytest.approx(1)


@pytest.mark.parametrize("fluxes", [[np.zeros((4, 2))] * 2,
                                      [np.zeros(4)] * 3])
def test_reference_rejects_invalid_flux_collection(fluxes):
    with pytest.raises(ValueError, match="three"):
        fluxcal.make_reference_spectra(fluxes)


def test_reference_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        fluxcal.make_reference_spectra([
            np.zeros((10, 2)), np.zeros((11, 2)), np.zeros((10, 2))])


def test_flux_calibration_of_identical_fibers_is_unity():
    fluxes, masks = simple_frames()
    reference = np.column_stack([flux[:, 0] for flux in fluxes])
    products, throughput = fluxcal.make_flux_calibrations(
        fluxes, masks, reference, mjd=59000, median_width=3,
        fill_width=5, smooth_width=3)
    for product, thru in zip(products, throughput):
        np.testing.assert_allclose(product[8:-8], 1)
        np.testing.assert_allclose(thru, 1)


def test_flux_calibration_preserves_relative_fiber_throughput():
    fluxes, masks = simple_frames(nfiber=4)
    scale = np.array([0.5, 1.0, 1.5, 2.0])
    fluxes = [flux * scale for flux in fluxes]
    reference = np.column_stack([flux[:, 1] for flux in fluxes])
    _, throughput = fluxcal.make_flux_calibrations(
        fluxes, masks, reference, mjd=59000, median_width=3,
        fill_width=5, smooth_width=3)
    np.testing.assert_allclose(throughput[0], scale / np.median(scale))


def test_littrow_region_is_interpolated():
    fluxes, masks = simple_frames()
    reference = np.column_stack([flux[:, 0] for flux in fluxes])
    fluxes[0][28:34, 3] = 50 * reference[28:34, 0]
    masks[0][28:34, 3] = 1
    products, _ = fluxcal.make_flux_calibrations(
        fluxes, masks, reference, mjd=59000, bad_pixel_bits=0,
        littrow_bit=1, median_width=1, fill_width=3, smooth_width=1)
    np.testing.assert_allclose(products[0][28:34, 3], 1, atol=1e-5)


def test_flux_calibration_does_not_modify_inputs():
    fluxes, masks = simple_frames()
    original = [array.copy() for array in fluxes + masks]
    reference = np.column_stack([flux[:, 0] for flux in fluxes])
    fluxcal.make_flux_calibrations(
        fluxes, masks, reference, mjd=59000, median_width=3,
        fill_width=5, smooth_width=3)
    for actual, expected in zip(fluxes + masks, original):
        np.testing.assert_array_equal(actual, expected)


def test_flux_calibration_validates_inputs():
    with pytest.raises(ValueError, match="three"):
        fluxcal.make_flux_calibrations([], [], np.zeros((3, 3)), mjd=1)
    fluxes, masks = simple_frames()
    with pytest.raises(ValueError, match="reference"):
        fluxcal.make_flux_calibrations(
            fluxes, masks, np.zeros((4, 3)), mjd=1)


def test_central_wavelength_accepts_one_dimensional_array():
    wavelength = np.linspace(12000.0, 17000.0, 20)

    result = fluxcal._central_wavelength(wavelength)

    np.testing.assert_array_equal(result, wavelength)


def test_central_wavelength_selects_middle_row():
    wavelength = np.arange(15.0).reshape(3, 5)

    result = fluxcal._central_wavelength(wavelength)

    np.testing.assert_array_equal(result, wavelength[1])


def test_central_wavelength_selects_middle_column():
    wavelength = np.arange(15.0).reshape(5, 3)

    result = fluxcal._central_wavelength(wavelength)

    np.testing.assert_array_equal(result, wavelength[:, 1])


def test_central_wavelength_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="one or two dimensions"):
        fluxcal._central_wavelength(np.zeros((2, 3, 4)))


def patch_builder(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    monkeypatch.setattr(fluxcal.apload, "ApLoad", lambda **kwargs: load)
    locks, processes = [], []
    monkeypatch.setattr(
        cal_utils.lock, "lock",
        lambda *args, **kwargs: locks.append((args, kwargs)))
    monkeypatch.setattr(
        fluxcal, "process",
        lambda *args, **kwargs: processes.append((args, kwargs)))
    fluxes, masks = simple_frames(npix=32, nfiber=6)
    load.spectra = {
        chip: {
            "header": fits.Header({"EXPTYPE": "DOMEFLAT"}),
            "flux": fluxes[index],
            "err": np.ones_like(fluxes[index]),
            "mask": masks[index],
        }
        for index, chip in enumerate("abc")
    }
    monkeypatch.setattr(
        fluxcal, "make_reference_spectra",
        lambda *args, **kwargs: (np.ones((32, 3)), np.ones((32, 3))))
    monkeypatch.setattr(
        fluxcal, "make_flux_calibrations",
        lambda *args, **kwargs: (
            [np.full((32, 6), i + 1.0) for i in range(3)],
            [np.full(6, i + 1.0) for i in range(3)]))
    return load, locks, processes


def test_build_flux_processes_and_writes_data_model(monkeypatch, tmp_path):
    load, locks, processes = patch_builder(monkeypatch, tmp_path)
    result = fluxcal.build_flux(
        [12, 13], darkid=1, flatid=2, psfid=3, waveid=4,
        littrowid=5, persistid=6, verbose=True)
    assert result is None
    outputs = load.product_files("flux", 12)
    assert processes[0][0][0] == [12, 13]
    assert processes[0][1]["load"] is load
    assert processes[0][1]["darkid"] == 1
    assert processes[0][1]["persistid"] == 6
    for index, filename in enumerate(outputs):
        with fits.open(filename) as hdus:
            assert len(hdus) == 5
            assert hdus[0].header["OBSTYPE"] == "FLUXCORR"
            assert hdus[1].header["EXTNAME"] == "RELATIVE FLUX"
            assert hdus[1].data.shape == (6, 32)
            np.testing.assert_array_equal(hdus[2].data, index + 1)
            assert hdus[3].header["EXTNAME"] == "REFERENCE SPECTRUM"
            assert hdus[4].header["EXTNAME"] == "MEDIAN REFERENCE SPECTRUM"
    assert locks[-1][1] == {"clear": True}


def test_build_flux_existing_short_circuit(monkeypatch, tmp_path, capsys):
    load, locks, processes = patch_builder(monkeypatch, tmp_path)
    outputs = load.product_files("flux", 12)
    for filename in outputs:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"existing")
    assert fluxcal.build_flux(12, verbose=True) is None
    assert not processes
    assert not locks
    assert "flux product 12 already exists" in capsys.readouterr().out


def test_existing_flux_still_builds_requested_response(monkeypatch, tmp_path):
    load, _, _ = patch_builder(monkeypatch, tmp_path)
    for filename in load.product_files("flux", 12):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"existing")
    calls = []
    monkeypatch.setattr(
        fluxcal, "build_response",
        lambda *args, **kwargs: calls.append((args, kwargs)))
    fluxcal.build_flux(12, waveid=99, temp=4000)
    assert calls[0][0] == (12,)
    assert calls[0][1]["load"] is load


def test_build_flux_partial_product_is_rebuilt(monkeypatch, tmp_path):
    load, _, processes = patch_builder(monkeypatch, tmp_path)
    outputs = load.product_files("flux", 12)
    Path(outputs[0]).parent.mkdir(parents=True)
    Path(outputs[0]).write_bytes(b"partial")
    fluxcal.build_flux(12)
    assert processes
    assert all(Path(filename).stat().st_size > 0 for filename in outputs)


def test_build_flux_clears_lock_after_failure(monkeypatch, tmp_path):
    _, locks, _ = patch_builder(monkeypatch, tmp_path)
    monkeypatch.setattr(
        fluxcal, "process",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        fluxcal.build_flux(12)
    assert locks[-1][1] == {"clear": True}


def test_build_flux_rejects_obsolete_holtz_branch():
    with pytest.raises(NotImplementedError, match="holtz"):
        fluxcal.build_flux(12, holtz=True)


def test_build_response_writes_three_vectors(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    monkeypatch.setattr(cal_utils.lock, "lock", lambda *args, **kwargs: None)
    for filename in load.product_files("flux", 12):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList([
            fits.PrimaryHDU(), fits.ImageHDU(), fits.ImageHDU(),
            fits.ImageHDU(np.ones(20)),
        ]).writeto(filename)
    for filename in load.product_files("wave", 99):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"wave")
    load.waves = {
        chip: {
            "header": fits.Header({"CHIP": chip}),
            "wavelength": np.linspace(12000, 17000, 20),
        }
        for chip in "abc"
    }
    result = fluxcal.build_response(12, waveid=99, temp=4000, load=load)
    assert result is None
    outputs = load.product_files("response", 12)
    assert len(outputs) == 3
    for filename in outputs:
        data = fits.getdata(filename)
        assert data.shape == (20,)
        assert np.all(np.isfinite(data))


def test_build_response_existing_short_circuits_without_lock(
        monkeypatch, tmp_path, capsys):
    load = FakeLoad(tmp_path)
    lock_calls = []
    monkeypatch.setattr(
        cal_utils.lock, "lock",
        lambda *args, **kwargs: lock_calls.append((args, kwargs)))
    for filename in load.product_files("response", 12):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"existing")
    assert fluxcal.build_response(
        12, waveid=99, temp=4000, load=load, verbose=True) is None
    assert not lock_calls
    assert "response product 12 already exists" in capsys.readouterr().out


def test_build_response_reports_missing_registry_dependencies(
        monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    lock_calls = []
    monkeypatch.setattr(
        cal_utils.lock, "lock",
        lambda *args, **kwargs: lock_calls.append((args, kwargs)))
    with pytest.raises(
            FileNotFoundError, match="Missing Response dependency") as error:
        fluxcal.build_response(12, waveid=99, temp=4000, load=load)
    assert "apFlux-a-00000012.fits" in str(error.value)
    assert lock_calls[-1][1] == {"clear": True}


def test_response_requires_wave_and_temperature():
    with pytest.raises(ValueError, match="waveid and temp"):
        fluxcal.build_response(12, waveid=None, temp=4000)


def test_makecal_flux_dispatches_current_builder(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        makecal, "build_flux",
        lambda *args, **kwargs: calls.append((args, kwargs)))
    context = makecal.CalibrationContext(
        load=FakeLoad(tmp_path), calfile="cal.par", allcaldict={},
        modelpsf="40-50", verbose=True)
    monkeypatch.setattr(context, "calibrations", lambda mjd: {
        "darkid": 1, "flatid": 2, "waveid": 3,
        "littrowid": 4, "persistid": 5})
    makecal.flux("12", context)
    assert calls[0][0] == ([12],)
    assert calls[0][1]["modelpsf"] == "40-50"
    assert calls[0][1]["persistid"] == 5
    assert calls[0][1]["verbose"] is True


def test_makecal_response_dispatches_current_builder(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        makecal, "build_response",
        lambda *args, **kwargs: calls.append((args, kwargs)))
    context = makecal.CalibrationContext(
        load=FakeLoad(tmp_path), calfile="cal.par", allcaldict={},
        clobber=True, unlock=True, verbose=True)
    monkeypatch.setattr(
        context, "row", lambda *args, **kwargs: {"temp": 4000})
    monkeypatch.setattr(
        context, "calibrations", lambda mjd: {"waveid": 99})
    makecal.response("12", context)
    assert calls[0][0] == (12,)
    assert calls[0][1]["waveid"] == 99
    assert calls[0][1]["temp"] == 4000
    assert calls[0][1]["clobber"] is True
    assert calls[0][1]["unlock"] is True


