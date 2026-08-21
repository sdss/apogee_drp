"""Tests for ModelPSF support in the numbered PSF calibration module."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from apogee_drp.apred.cal import makecal
from apogee_drp.apred.cal import psfcal
from apogee_drp.apred.cal import utils as cal_utils


class FakeLoad:
    apred = "daily"
    telescope = "apo25m"

    def __init__(self, root):
        self.root = Path(root)

    def filename(self, kind, num=None, chip=None, **kwargs):
        infix = f"-{chip}" if chip is not None else ""
        return str(self.root / kind.lower() / f"ap{kind}{infix}-{num}.fits")

    def product_files(self, product, name):
        if product == "fiber":
            return [self.filename("Fiber", num=name, chip=chip)
                    for chip in psfcal.CHIPS]
        if product == "sparse":
            return [self.filename("Sparse", num=name)] + [
                self.filename("EPSF", num=name, chip=chip)
                for chip in psfcal.CHIPS]
        if product == "psf":
            return [
                self.filename(kind, num=name, chip=chip)
                for kind in ("PSF", "EPSF", "ETrace")
                for chip in psfcal.CHIPS
            ]
        if product == "modelpsf":
            return [self.filename("PSFModel", num=name, chip=chip)
                    for chip in psfcal.CHIPS]
        raise AssertionError(f"unexpected product {product}")

    def product_exists(self, product, name):
        return all(Path(filename).is_file() and Path(filename).stat().st_size > 0
                   for filename in self.product_files(product, name))

    def product_delete(self, product, name, **kwargs):
        for filename in self.product_files(product, name):
            path = Path(filename)
            if path.exists() or path.is_symlink():
                path.unlink()


def model_grid(nx=3, ny=4, noffset=31):
    offsets = np.linspace(-5, 5, noffset)
    profile = np.exp(-0.5 * (offsets / 1.2) ** 2)
    profiles = np.broadcast_to(profile, (nx, ny, noffset)).copy()
    x = np.broadcast_to(np.linspace(10, 2000, nx)[:, None], (nx, ny))
    y = np.broadcast_to(np.linspace(10, 2000, ny)[None, :], (nx, ny))
    return profiles, np.stack((x, y)), offsets


def test_registry_modelpsf_files_preserve_compound_name(tmp_path):
    files = FakeLoad(tmp_path).product_files("modelpsf", "12-34")
    assert [Path(filename).name for filename in files] == [
        "apPSFModel-a-12-34.fits", "apPSFModel-b-12-34.fits",
        "apPSFModel-c-12-34.fits"]


def test_obsolete_product_filename_helpers_are_removed():
    assert not hasattr(psfcal, "product_files")
    assert not hasattr(psfcal, "modelpsf_product_files")
    assert not hasattr(psfcal, "_chip_filename")


@pytest.mark.parametrize(
    "product,name,builder,args,kwargs",
    [
        ("fiber", 12, psfcal.build_fiber, (12,), {}),
        ("sparse", 12, psfcal.build_sparse, ([12],), {}),
        ("psf", 12, psfcal.build_psf, (12,), {"sparseid": 9}),
    ],
)
def test_existing_registered_products_skip_build(
        monkeypatch, tmp_path, product, name, builder, args, kwargs):
    load = FakeLoad(tmp_path)
    for filename in load.product_files(product, name):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"existing")

    lock_calls = []
    monkeypatch.setattr(psfcal.apload, "ApLoad", lambda **options: load)
    monkeypatch.setattr(
        cal_utils.lock, "lock",
        lambda *call_args, **options:
        lock_calls.append((call_args, options)),
    )
    monkeypatch.setattr(
        psfcal, "_reduce",
        lambda *call_args, **options:
        pytest.fail("an existing product must not be rebuilt"),
    )

    assert builder(*args, **kwargs) is None
    assert lock_calls == []


def test_validate_model_grid_normalizes_profiles():
    profiles, labels, offsets = model_grid()
    normalized, actual_labels, actual_offsets = psfcal._validate_model_grid(
        profiles, labels, offsets)
    np.testing.assert_allclose(
        np.trapz(normalized, actual_offsets, axis=2), 1, rtol=2e-6)
    np.testing.assert_allclose(actual_labels, labels, rtol=1e-7)
    # The grid remains float64 so every Numba gridinterp branch has the same
    # return dtype; query labels are converted to float64 by PSF.gridinterp().
    assert normalized.dtype == np.float64


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
    monkeypatch.setattr(psfcal.apload, "ApLoad", lambda **kwargs: load)
    locks, grid_calls = [], []
    monkeypatch.setattr(
        cal_utils.lock, "lock",
        lambda *args, **kwargs: locks.append((args, kwargs)))
    sparse = Path(load.filename("Sparse", num=12))
    sparse.parent.mkdir(parents=True)
    sparse.write_bytes(b"sparse")
    for chip in "abc":
        filename = Path(load.filename("EPSF", num=34, chip=chip))
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_bytes(b"epsf")

    def make_grid(epsf, sparse, **kwargs):
        grid_calls.append((epsf, sparse, kwargs))
        return model_grid()

    monkeypatch.setattr(psfcal, "_make_profile_grid", make_grid)
    return load, locks, grid_calls


def test_build_modelpsf_writes_all_chips(monkeypatch, tmp_path, capsys):
    load, locks, grid_calls = patch_builder(monkeypatch, tmp_path)
    outputs = load.product_files("modelpsf", "12-34")
    assert psfcal.build_modelpsf(
        "12-34", sparseid=12, psfid=34, verbose=True) is None
    assert len(grid_calls) == 3
    assert [Path(call[0]).name for call in grid_calls] == [
        "apEPSF-a-34.fits", "apEPSF-b-34.fits", "apEPSF-c-34.fits"]
    assert all(Path(filename).stat().st_size > 0 for filename in outputs)
    assert "writing ModelPSF chip c" in capsys.readouterr().out
    assert locks[-1][1] == {"clear": True}


def test_build_modelpsf_existing_short_circuit(monkeypatch, tmp_path, capsys):
    load, _, grid_calls = patch_builder(monkeypatch, tmp_path)
    outputs = load.product_files("modelpsf", "12-34")
    for filename in outputs:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"existing")
    assert psfcal.build_modelpsf(
        "12-34", sparseid=12, psfid=34, verbose=True) is None
    assert not grid_calls
    assert "modelpsf product 12-34 already exists" in capsys.readouterr().out


def test_build_modelpsf_partial_product_rebuilds(monkeypatch, tmp_path):
    load, _, grid_calls = patch_builder(monkeypatch, tmp_path)
    outputs = load.product_files("modelpsf", "12-34")
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
    monkeypatch.setattr(psfcal.apload, "ApLoad", lambda **kwargs: load)
    monkeypatch.setattr(cal_utils.lock, "lock", lambda *args, **kwargs: None)
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


def test_makecal_modelpsf_dispatches_builder(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        makecal, "build_modelpsf",
        lambda *args, **kwargs: calls.append((args, kwargs)))
    context = makecal.CalibrationContext(
        load=FakeLoad(tmp_path), calfile="cal.par", allcaldict={},
        clobber=True, unlock=True, verbose=True)
    monkeypatch.setattr(
        context, "row",
        lambda *args, **kwargs: {"sparse": 12, "psf": 34})
    makecal.modelpsf("12-34", context)
    assert calls == [(('12-34',), {
        "sparseid": 12, "psfid": 34, "apred": "daily",
        "telescope": "apo25m", "clobber": True, "unlock": True,
        "verbose": True})]
