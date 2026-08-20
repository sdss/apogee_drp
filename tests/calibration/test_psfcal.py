"""Tests for ModelPSF support in the numbered PSF calibration module."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from apogee_drp.apred.cal import makecal
from apogee_drp.apred.cal import psfcal


class FakeLoad:
    apred = "daily"
    telescope = "apo25m"

    def __init__(self, root):
        self.root = Path(root)

    def filename(self, kind, num=None, chips=False, **kwargs):
        return str(self.root / kind.lower() / f"ap{kind}-{num}.fits")


def model_grid(nx=3, ny=4, noffset=31):
    offsets = np.linspace(-5, 5, noffset)
    profile = np.exp(-0.5 * (offsets / 1.2) ** 2)
    profiles = np.broadcast_to(profile, (nx, ny, noffset)).copy()
    x = np.broadcast_to(np.linspace(10, 2000, nx)[:, None], (nx, ny))
    y = np.broadcast_to(np.linspace(10, 2000, ny)[None, :], (nx, ny))
    return profiles, np.stack((x, y)), offsets


def test_modelpsf_product_files_preserve_name(tmp_path):
    files = psfcal.modelpsf_product_files(FakeLoad(tmp_path), "12-34")
    assert [Path(filename).name for filename in files] == [
        "apPSFModel-a-12-34.fits", "apPSFModel-b-12-34.fits",
        "apPSFModel-c-12-34.fits"]


def test_modelpsf_product_files_reject_empty_name(tmp_path):
    with pytest.raises(ValueError, match="cannot be empty"):
        psfcal.modelpsf_product_files(FakeLoad(tmp_path), " ")


def test_validate_model_grid_normalizes_profiles():
    profiles, labels, offsets = model_grid()
    normalized, actual_labels, actual_offsets = psfcal._validate_model_grid(
        profiles, labels, offsets)
    np.testing.assert_allclose(
        np.trapezoid(normalized, actual_offsets, axis=2), 1, rtol=2e-6)
    np.testing.assert_allclose(actual_labels, labels, rtol=1e-7)
    assert normalized.dtype == np.float32


@pytest.mark.parametrize("mutation, message", [
    (lambda p, l, x: (p[:, :, :-1], l, x), "profile length"),
    (lambda p, l, x: (p, l[:, :-1], x), "incompatible"),
    (lambda p, l, x: (np.where(p > 0.9, np.nan, p), l, x), "nonfinite"),
    (lambda p, l, x: (-p, l, x), "negative"),
    (lambda p, l, x: (np.zeros_like(p), l, x), "empty profile"),
])
def test_validate_model_grid_rejects_invalid_data(mutation, message):
    values = mutation(*model_grid())
    with pytest.raises(ValueError, match=message):
        psfcal._validate_model_grid(*values)


def test_write_model_grid_data_model(tmp_path):
    filename = tmp_path / "model.fits"
    profiles, labels, offsets = model_grid()
    psfcal._write_model_grid(
        filename, profiles, labels, offsets, apred="daily",
        sparseid=12, psfid=34, nfbin=5, ncbin=200)
    with fits.open(filename) as hdus:
        assert len(hdus) == 3
        assert hdus[0].header["TYPE"] == "grid"
        assert hdus[0].header["LOG"] is False
        assert hdus[0].header["SPARSEID"] == "12"
        assert hdus[0].header["PSFID"] == "34"
        assert hdus[1].header["EXTNAME"] == "LABELS"
        assert hdus[1].data.shape == labels.shape
        assert hdus[2].header["EXTNAME"] == "X"


def test_written_grid_is_readable_by_psf_class(tmp_path):
    from apogee_drp.apred.psf import PSF
    filename = tmp_path / "model.fits"
    psfcal._write_model_grid(
        filename, *model_grid(), apred="daily", sparseid=12, psfid=34,
        nfbin=5, ncbin=200)
    model = PSF.read(str(filename))
    assert model.kind == "grid"
    assert model.npix == 31
    assert np.all(np.isfinite(model.model([1000, 1000])))


def patch_builder(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    monkeypatch.setattr(psfcal, "_make_load", lambda **kwargs: load)
    locks, grid_calls = [], []
    monkeypatch.setattr(
        psfcal.lock, "lock",
        lambda *args, **kwargs: locks.append((args, kwargs)))
    sparse = Path(load.filename("Sparse", num=12, chips=True))
    sparse.parent.mkdir(parents=True)
    sparse.write_bytes(b"sparse")
    for chip in "abc":
        filename = Path(psfcal._chip_filename(load, "EPSF", 34, chip))
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_bytes(b"epsf")

    def make_grid(epsf, sparse, **kwargs):
        grid_calls.append((epsf, sparse, kwargs))
        return model_grid()

    monkeypatch.setattr(psfcal, "_make_profile_grid", make_grid)
    return load, locks, grid_calls


def test_build_modelpsf_writes_all_chips(monkeypatch, tmp_path, capsys):
    load, locks, grid_calls = patch_builder(monkeypatch, tmp_path)
    outputs = psfcal.build_modelpsf(
        "12-34", sparseid=12, psfid=34, verbose=True)
    assert outputs == psfcal.modelpsf_product_files(load, "12-34")
    assert len(grid_calls) == 3
    assert [Path(call[0]).name for call in grid_calls] == [
        "apEPSF-a-34.fits", "apEPSF-b-34.fits", "apEPSF-c-34.fits"]
    assert all(Path(filename).stat().st_size > 0 for filename in outputs)
    marker = Path(outputs[0].replace("PSFModel-a-", "PSFModel-")).with_suffix(".dat")
    assert marker.is_file()
    assert "writing ModelPSF chip c" in capsys.readouterr().out
    assert locks[-1][1] == {"clear": True}


def test_build_modelpsf_existing_short_circuit(monkeypatch, tmp_path, capsys):
    load, _, grid_calls = patch_builder(monkeypatch, tmp_path)
    outputs = psfcal.modelpsf_product_files(load, "12-34")
    for filename in outputs:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"existing")
    assert psfcal.build_modelpsf(
        "12-34", sparseid=12, psfid=34, verbose=True) == outputs
    assert not grid_calls
    assert "already made" in capsys.readouterr().out


def test_build_modelpsf_partial_product_rebuilds(monkeypatch, tmp_path):
    load, _, grid_calls = patch_builder(monkeypatch, tmp_path)
    outputs = psfcal.modelpsf_product_files(load, "12-34")
    Path(outputs[0]).parent.mkdir(parents=True)
    Path(outputs[0]).write_bytes(b"partial")
    psfcal.build_modelpsf("12-34", sparseid=12, psfid=34)
    assert len(grid_calls) == 3
    assert all(Path(filename).stat().st_size > 0 for filename in outputs)


@pytest.mark.parametrize("kwargs, message", [
    ({"sparseid": None, "psfid": 34}, "sparseid"),
    ({"sparseid": 12, "psfid": None}, "psfid"),
    ({"sparseid": 12, "psfid": 34, "nfbin": 0}, "nfbin"),
])
def test_build_modelpsf_validates_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        psfcal.build_modelpsf("12-34", **kwargs)


def test_build_modelpsf_reports_missing_inputs(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    monkeypatch.setattr(psfcal, "_make_load", lambda **kwargs: load)
    monkeypatch.setattr(psfcal.lock, "lock", lambda *args, **kwargs: None)
    with pytest.raises(FileNotFoundError, match="Missing ModelPSF inputs"):
        psfcal.build_modelpsf("12-34", sparseid=12, psfid=34)


def test_build_modelpsf_clears_lock_on_failure(monkeypatch, tmp_path):
    _, locks, _ = patch_builder(monkeypatch, tmp_path)
    monkeypatch.setattr(
        psfcal, "_make_profile_grid",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        psfcal.build_modelpsf("12-34", sparseid=12, psfid=34)
    assert locks[-1][1] == {"clear": True}


def test_makecal_modelpsf_dispatches_v3_builder(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        makecal_v11, "build_modelpsf",
        lambda *args, **kwargs: calls.append((args, kwargs)))
    context = makecal_v11.CalibrationContext(
        load=FakeLoad(tmp_path), calfile="cal.par", allcaldict={},
        clobber=True, unlock=True, verbose=True)
    monkeypatch.setattr(
        context, "row",
        lambda *args, **kwargs: {"sparse": 12, "psf": 34})
    makecal_v11.modelpsf("12-34", context)
    assert calls == [(('12-34',), {
        "sparseid": 12, "psfid": 34, "apred": "daily",
        "telescope": "apo25m", "clobber": True, "unlock": True,
        "verbose": True})]


def test_legacy_mkmodelpsf_wrapper(monkeypatch):
    from apogee_drp.apred.cal import mkmodelpsf
    calls = []
    monkeypatch.setattr(
        mkmodelpsf, "build_modelpsf",
        lambda *args, **kwargs: calls.append((args, kwargs)) or ["done"])
    assert mkmodelpsf.mkmodelpsf(
        "12-34", sparseid=12, psfid=34, clobber=True) == ["done"]
    assert calls[0][0] == ("12-34",)
    assert calls[0][1]["clobber"] is True
