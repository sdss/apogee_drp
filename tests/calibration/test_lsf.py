"""Tests for the numbered LSF calibration implementation."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from apogee_drp.apred import process as process_module
from apogee_drp.apred.cal import lsf as lv
from apogee_drp.apred.cal import makecal
from apogee_drp.apred.cal import utils as cal_utils


class FakeLoad:
    apred = "daily"
    telescope = "apo25m"

    def __init__(self, root):
        self.root = Path(root)
        self.delete_calls = []
        self.spectrum_calls = []

    def filename(self, kind, num=None, chip=None, directory=False, **kwargs):
        root = self.root / kind
        if directory:
            return str(root)
        infix = f"-{chip}" if chip is not None else ""
        return str(root / f"ap{kind}{infix}-{int(num):08d}.fits")

    def cmjd(self, number):
        return 60000

    def product_files(self, product, name):
        assert product == "lsf"
        files = [self.filename("LSF", num=name, chip=chip)
                 for chip in "abc"]
        template = Path(self.filename("LSF", num=name))
        files.append(str(template.with_name(
            f"{template.stem}-diagnostics{template.suffix}")))
        return files

    def product_exists(self, product, name):
        return all(Path(filename).is_file() and
                   Path(filename).stat().st_size > 0
                   for filename in self.product_files(product, name))

    def product_delete(self, product, name, **kwargs):
        self.delete_calls.append((product, name, kwargs))
        for filename in self.product_files(product, name):
            path = Path(filename)
            if path.exists() or path.is_symlink():
                path.unlink()

    def spectrum(self, number, chip=None):
        self.spectrum_calls.append((int(number), chip))
        spectra = {}
        chips = ("a", "b", "c") if chip is None else (chip,)
        for current_chip in chips:
            frame = synthetic_frame()
            spectra[current_chip] = {
                "header": frame["header"].copy(),
                "flux": frame["flux"].T.copy(),
                "err": frame["err"].T.copy(),
                "mask": frame["mask"].T.copy(),
            }
        return spectra if chip is None else spectra[chip]


def synthetic_frame(nfiber=4, npix=100, sigma=1.5, dither=0.0):
    x = np.arange(npix)
    flux = np.full((nfiber, npix), 10.0)
    for fiber in range(nfiber):
        slope = 0.002 * fiber
        for center in (20, 50, 80):
            width = sigma + slope * (center - 50)
            flux[fiber] += 100 * np.exp(-0.5 * ((x - center) / width) ** 2)
    return {"flux": flux, "err": np.ones_like(flux),
            "mask": np.zeros_like(flux, dtype=np.uint16),
            "header": fits.Header({"DITHPIX": dither, "LAMPUNE": 1})}


def test_product_files_include_portable_diagnostics(tmp_path):
    files = FakeLoad(tmp_path).product_files("lsf", 123)
    assert len(files) == 4
    assert files[0].endswith("apLSF-a-00000123.fits")
    assert files[2].endswith("apLSF-c-00000123.fits")
    assert files[3].endswith("apLSF-00000123-diagnostics.fits")


def test_obsolete_product_filename_and_load_helpers_are_removed():
    assert not hasattr(lv, "product_files")
    assert not hasattr(lv, "_chip_filename")
    assert not hasattr(lv, "_make_load")
    assert not hasattr(lv, "_load_1d")
    assert not hasattr(lv, "_process_frames")


def test_sanitize_frame_replaces_invalid_values():
    data = synthetic_frame(nfiber=1)
    data["flux"][0, 2] = np.nan
    data["err"][0, 3] = 0
    clean = lv._sanitize_frame(data)
    assert clean["flux"][0, 2] == 0
    assert clean["err"][0, 3] == lv.BADERR
    assert clean["mask"][0, 2] & 1
    assert clean["mask"][0, 3] & 1


def test_sanitize_frame_validates_shapes():
    with pytest.raises(ValueError, match="matching"):
        lv._sanitize_frame({"flux": np.zeros((2, 3)), "err": np.zeros(4)})


def test_combine_frames_adds_flux_and_variance():
    first, second = synthetic_frame(), synthetic_frame()
    result = lv.combine_frames([first, second])
    np.testing.assert_allclose(result["flux"], first["flux"] * 2)
    np.testing.assert_allclose(result["err"], np.sqrt(2))


def test_combine_frames_rejects_different_dithers():
    with pytest.raises(NotImplementedError, match="dither"):
        lv.combine_frames([synthetic_frame(dither=0), synthetic_frame(dither=0.5)])


def test_combine_frames_requires_input():
    with pytest.raises(ValueError, match="at least one"):
        lv.combine_frames([])


def test_remove_continuum_preserves_emission_lines():
    data = synthetic_frame(nfiber=1)
    residual = lv.remove_continuum(data["flux"], width=11)
    assert residual[0, 50] > 80
    assert abs(np.median(residual[0])) < 1e-6


@pytest.mark.parametrize("width", [0, 2, 4])
def test_remove_continuum_validates_width(width):
    with pytest.raises(ValueError, match="odd"):
        lv.remove_continuum(np.ones((1, 10)), width=width)


def test_measure_line_widths_recovers_gaussians():
    data = synthetic_frame(nfiber=1, sigma=1.4)
    residual = lv.remove_continuum(data["flux"], width=11)[0]
    lines = lv.measure_line_widths(residual, data["err"][0])
    np.testing.assert_allclose(lines[:, 0], [20, 50, 80], atol=0.1)
    np.testing.assert_allclose(lines[:, 1], 1.4, atol=0.15)


def test_measure_line_widths_honors_mask():
    data = synthetic_frame(nfiber=1)
    residual = lv.remove_continuum(data["flux"], width=11)[0]
    data["mask"][0, 45:56] = 1
    lines = lv.measure_line_widths(residual, data["err"][0], data["mask"][0])
    assert len(lines) == 2


def test_measure_line_widths_returns_empty_without_errors():
    assert lv.measure_line_widths(np.ones(10), np.zeros(10)).shape == (0, 4)


def test_fit_sigma_model_rejects_outlier():
    x = np.array([10, 30, 50, 70, 90], float)
    sigma = 1.2 + 0.002 * (x - 50)
    sigma[2] = 5
    measurements = np.column_stack((x, sigma, np.ones((5, 2))))
    coefficient, used, xoffset = lv.fit_sigma_model(measurements, npix=101)
    assert not used[2]
    assert xoffset == -50
    assert coefficient[0] == pytest.approx(1.2, abs=0.05)


def test_parameter_vector_historical_layout():
    pars = lv._parameter_vector([1.5, 0.001], -50)
    np.testing.assert_allclose(pars, [1, -50, 0, 1, 1.5, 0.001])


def test_gaussian_lsf_array_is_normalized_and_odd():
    result = lv.gaussian_lsf_array([1, -50, 0, 1, 1.5, 0], 100)
    assert result.shape[0] % 2 == 1
    np.testing.assert_allclose(result.sum(axis=0), 1, atol=1e-6)
    np.testing.assert_allclose(result[:, 10], result[::-1, 10])


def test_pack_ghlsf_parameters_round_trips_through_doppler():
    from doppler.lsf import unpack_ghlsf_params

    parameters = lv.pack_ghlsf_parameters(
        1.0,
        -50.0,
        porder=(1, 0),
        ghcoefs=np.array([[1.5, 0.001], [0.02, 0.0]]),
        wporder=(0, 0),
        wcoefs=np.array([[0.03], [3.5]]),
    )
    unpacked = unpack_ghlsf_params(parameters)

    assert unpacked["binsize"] == 1.0
    assert unpacked["Xoffset"] == -50.0
    assert unpacked["Horder"] == 1
    np.testing.assert_array_equal(unpacked["Porder"], [1, 0])
    np.testing.assert_allclose(
        unpacked["GHcoefs"][:, :2],
        [[1.5, 0.001], [0.02, 0.0]],
    )
    assert unpacked["Wproftype"] == 1
    np.testing.assert_array_equal(unpacked["WPorder"], [0, 0])
    np.testing.assert_allclose(unpacked["Wcoefs"][:, 0], [0.03, 3.5])


def test_pack_ghlsf_parameters_validates_coefficient_count():
    with pytest.raises(ValueError, match="expected"):
        lv.pack_ghlsf_parameters(
            1.0, 0.0, porder=(1, 0), ghcoefs=[1.5],
            wporder=(0, 0), wcoefs=[0.03, 3.5])


def test_pack_ghlsf_parameters_requires_doppler_wing_block():
    with pytest.raises(ValueError, match="two wing parameters"):
        lv.pack_ghlsf_parameters(
            1.0, 0.0, porder=(0,), ghcoefs=[1.5],
            wporder=(), wcoefs=())


def test_evaluate_gauss_hermite_lsf_is_normalized():
    parameters = lv.pack_ghlsf_parameters(
        1.0,
        -50.0,
        porder=(1, 0),
        ghcoefs=np.array([[1.5, 0.001], [0.02, 0.0]]),
        wporder=(0, 0),
        wcoefs=np.array([[0.03], [3.5]]),
    )
    centers = np.array([10.0, 50.0, 90.0])
    offsets = np.arange(-15.0, 16.0)

    profiles = lv.evaluate_gauss_hermite_lsf(
        parameters, centers, offsets, positive=True, normalize=True)

    assert profiles.shape == (3, 31)
    np.testing.assert_allclose(profiles.sum(axis=1), 1.0)
    assert np.all(profiles >= 0)
    assert not np.allclose(profiles[0], profiles[-1])


def test_fit_gauss_hermite_chip_produces_doppler_parameters():
    from doppler.lsf import unpack_ghlsf_params

    data = synthetic_frame(nfiber=2, npix=160)
    pixel = np.arange(160)
    for center in (110, 140):
        data["flux"] += (
            100 * np.exp(-0.5 * ((pixel - center) / 1.5) ** 2)
        )

    parameters, array, diagnostics = lv.fit_gauss_hermite_chip(
        data,
        fibers=[0],
        continuum_width=11,
        porder=(0, 0),
        wporder=(0, 0),
        nlsfpix=31,
        max_nfev=500,
    )

    assert parameters.shape[0] == 2
    assert array.shape == (31, 2, 160)
    assert len(diagnostics) >= 5
    np.testing.assert_allclose(array.sum(axis=0), 1.0, atol=1e-6)
    unpacked = unpack_ghlsf_params(parameters[0])
    assert unpacked["Horder"] == 1
    assert unpacked["GHcoefs"][0, 0] > 0
    assert unpacked["Wcoefs"][0, 0] >= 0


def test_fit_lsf_chip_outputs_compatible_shapes():
    parameters, array, diagnostics = lv.fit_lsf_chip(
        synthetic_frame(), continuum_width=11)
    assert parameters.shape == (4, 6)
    assert array.shape[1:] == (4, 100)
    assert len(diagnostics) == 12
    np.testing.assert_allclose(array.sum(axis=0), 1, atol=1e-6)


def test_fit_lsf_chip_fills_missing_fiber_from_neighbor():
    data = synthetic_frame()
    data["flux"][2] = 0
    parameters, array, _ = lv.fit_lsf_chip(data, continuum_width=11)
    assert np.all(np.isfinite(parameters[2]))
    np.testing.assert_allclose(array[:, 2].sum(axis=0), 1, atol=2e-7)


def test_fit_lsf_chip_validates_fibers():
    with pytest.raises(ValueError, match="out-of-range"):
        lv.fit_lsf_chip(synthetic_frame(), fibers=[9], continuum_width=11)


def test_fit_lsf_chip_interpolates_unselected_fibers():
    parameters, array, diagnostics = lv.fit_lsf_chip(
        synthetic_frame(), fibers=[1, 3], continuum_width=11)
    assert np.all(np.isfinite(parameters))
    np.testing.assert_allclose(array.sum(axis=0), 1, atol=2e-7)
    assert set(diagnostics["fiber"]) == {1, 3}


def patch_builder(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    monkeypatch.setattr(lv.apload, "ApLoad", lambda **kwargs: load)
    lock_calls = []
    monkeypatch.setattr(cal_utils.lock, "lock",
                        lambda *args, **kwargs: lock_calls.append((args, kwargs)))
    process_calls = []
    monkeypatch.setattr(
        process_module,
        "process",
        lambda *args, **kwargs: process_calls.append((args, kwargs)),
    )
    return load, lock_calls, process_calls


def test_build_lsf_writes_three_chips_and_diagnostics(monkeypatch, tmp_path):
    load, lock_calls, process_calls = patch_builder(monkeypatch, tmp_path)
    assert lv.build_lsf(
        [123], 60000, psfid=50, continuum_width=11,
        verbose=True) is None
    outputs = load.product_files("lsf", 123)
    assert all(Path(filename).is_file() for filename in outputs)
    assert process_calls[0][0] == ([123],)
    assert process_calls[0][1]["load"] is load
    assert process_calls[0][1]["waveid"] == 60000
    assert process_calls[0][1]["psfid"] == 50
    assert process_calls[0][1]["fluxid"] is None
    assert process_calls[0][1]["doproc"] is True
    assert process_calls[0][1]["skywave"] is True
    assert load.spectrum_calls == [(123, None)]
    assert fits.getdata(outputs[0], 0).shape == (6, 4)
    assert fits.getdata(outputs[0], 1).shape[1:] == (4, 100)
    assert len(fits.open(outputs[3])) == 4
    assert lock_calls[-1][1] == {"clear": True}


def test_build_lsf_existing_short_circuit(monkeypatch, tmp_path, capsys):
    load, lock_calls, process_calls = patch_builder(monkeypatch, tmp_path)
    outputs = load.product_files("lsf", 123)
    for filename in outputs:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"existing")
    assert lv.build_lsf(123, 60000, psfid=50, verbose=True) is None
    assert not process_calls
    assert not load.spectrum_calls
    assert not lock_calls
    assert "lsf product 123 already exists" in capsys.readouterr().out


def test_build_lsf_partial_product_is_rebuilt(monkeypatch, tmp_path):
    load, _, process_calls = patch_builder(monkeypatch, tmp_path)
    outputs = load.product_files("lsf", 123)
    Path(outputs[0]).parent.mkdir(parents=True)
    Path(outputs[0]).write_bytes(b"partial")
    lv.build_lsf(123, 60000, psfid=50, continuum_width=11)
    assert process_calls
    assert all(Path(filename).is_file() for filename in outputs)


def test_build_lsf_clears_lock_on_failure(monkeypatch, tmp_path):
    _, lock_calls, _ = patch_builder(monkeypatch, tmp_path)
    monkeypatch.setattr(lv, "fit_lsf_chip",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fit")))
    with pytest.raises(RuntimeError, match="fit"):
        lv.build_lsf(123, 60000, psfid=50)
    assert lock_calls[-1][1] == {"clear": True}


def test_build_lsf_nowait_uses_zero_wait(monkeypatch, tmp_path):
    _, lock_calls, _ = patch_builder(monkeypatch, tmp_path)
    lv.build_lsf(
        123, 60000, psfid=50, continuum_width=11, nowait=True)
    assert lock_calls[0][1]["waittime"] == 0


def test_build_lsf_full_uses_gauss_hermite_fitter(monkeypatch, tmp_path):
    load, _, _ = patch_builder(monkeypatch, tmp_path)
    calls = []

    def fit(frame, **kwargs):
        calls.append((frame, kwargs))
        parameters = np.zeros((4, 12), dtype=float)
        array = np.ones((21, 4, 100), dtype=np.float32) / 21
        dtype = [
            ("fiber", np.int16), ("center", np.float32),
            ("sigma", np.float32), ("height", np.float32),
            ("flux", np.float32), ("accepted", bool),
        ]
        return parameters, array, np.empty(0, dtype=dtype)

    monkeypatch.setattr(lv, "fit_gauss_hermite_chip", fit)
    lv.build_lsf(
        123, 60000, psfid=50, full=True,
        porder=(1, 0), wporder=(0, 0))

    assert len(calls) == 3
    assert calls[0][1]["porder"] == (1, 0)
    assert calls[0][1]["wporder"] == (0, 0)
    for filename in load.product_files("lsf", 123)[:3]:
        assert fits.getheader(filename)["LSFMETH"] == "GAUSS-HERMITE"


def test_makecal_lsf_dispatches_current_builder(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(makecal, "build_lsf",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    context = makecal.CalibrationContext(
        load=FakeLoad(tmp_path), calfile="cal.par", allcaldict={}, verbose=True)
    monkeypatch.setattr(context, "frames", lambda *args: [123])
    monkeypatch.setattr(context, "row", lambda *args: {"psfid": 50})
    monkeypatch.setattr(context, "calibrations", lambda mjd: {
        "darkid": 1, "flatid": 2, "fiberid": 3, "multiwaveid": 60000})
    makecal.lsf("123", context)
    assert calls[0][0] == ([123], 60000)
    assert calls[0][1]["psfid"] == 50
    assert calls[0][1]["apred"] == "daily"
