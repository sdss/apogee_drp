"""Tests for the numbered LSF calibration implementation."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from apogee_drp.apred.cal import lsf as lv
from apogee_drp.apred.cal import makecal


class FakeLoad:
    apred = "daily"
    telescope = "apo25m"

    def __init__(self, root):
        self.root = Path(root)

    def filename(self, kind, num=None, chips=False, **kwargs):
        return str(self.root / kind / f"ap{kind}-{int(num):08d}.fits")

    def cmjd(self, number):
        return 60000


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
    files = lv.product_files(FakeLoad(tmp_path), 123, diagnostics=True)
    assert len(files) == 4
    assert files[0].endswith("apLSF-a-00000123.fits")
    assert files[2].endswith("apLSF-c-00000123.fits")
    assert files[3].endswith("apLSF-00000123-diagnostics.fits")


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
    monkeypatch.setattr(lv, "_make_load", lambda **kwargs: load)
    lock_calls = []
    monkeypatch.setattr(lv.lock, "lock",
                        lambda *args, **kwargs: lock_calls.append((args, kwargs)))
    process_calls = []
    monkeypatch.setattr(lv, "_process_frames",
                        lambda *args, **kwargs: process_calls.append((args, kwargs)))
    monkeypatch.setattr(lv, "_load_1d", lambda *args: synthetic_frame())
    return load, lock_calls, process_calls


def test_build_lsf_writes_three_chips_and_diagnostics(monkeypatch, tmp_path):
    load, lock_calls, process_calls = patch_builder(monkeypatch, tmp_path)
    outputs = lv.build_lsf(
        [123], 60000, psfid=50, continuum_width=11, verbose=True)
    assert outputs == lv.product_files(load, 123, diagnostics=True)
    assert all(Path(filename).is_file() for filename in outputs)
    assert process_calls[0][1]["waveid"] == 60000
    assert process_calls[0][1]["psfid"] == 50
    assert fits.getdata(outputs[0], 0).shape == (6, 4)
    assert fits.getdata(outputs[0], 1).shape[1:] == (4, 100)
    assert len(fits.open(outputs[3])) == 4
    assert lock_calls[-1][1] == {"clear": True}


def test_build_lsf_existing_short_circuit(monkeypatch, tmp_path, capsys):
    load, _, process_calls = patch_builder(monkeypatch, tmp_path)
    outputs = lv.product_files(load, 123, diagnostics=True)
    for filename in outputs:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"existing")
    assert lv.build_lsf(123, 60000, psfid=50, verbose=True) == outputs
    assert not process_calls
    assert "already made" in capsys.readouterr().out


def test_build_lsf_partial_product_is_rebuilt(monkeypatch, tmp_path):
    load, _, process_calls = patch_builder(monkeypatch, tmp_path)
    outputs = lv.product_files(load, 123, diagnostics=True)
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


def test_build_lsf_rejects_unvalidated_full_fit(monkeypatch, tmp_path):
    patch_builder(monkeypatch, tmp_path)
    with pytest.raises(NotImplementedError, match="Gauss-Hermite"):
        lv.build_lsf(123, 60000, psfid=50, full=True)


def test_makecal_lsf_dispatches_numbered_builder(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(makecal_v7, "build_lsf",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    context = makecal_v7.CalibrationContext(
        load=FakeLoad(tmp_path), calfile="cal.par", allcaldict={}, verbose=True)
    monkeypatch.setattr(context, "frames", lambda *args: [123])
    monkeypatch.setattr(context, "row", lambda *args: {"psfid": 50})
    monkeypatch.setattr(context, "calibrations", lambda mjd: {
        "darkid": 1, "flatid": 2, "fiberid": 3, "multiwaveid": 60000})
    makecal_v7.lsf("123", context)
    assert calls[0][0] == ([123], 60000)
    assert calls[0][1]["psfid"] == 50
    assert calls[0][1]["apred"] == "daily"
