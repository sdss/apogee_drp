"""Tests for the numbered Littrow calibration implementation."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from apogee_drp.apred.cal import littrow as lit
from apogee_drp.apred.cal import makecal


class FakeLoad:
    apred = "daily"
    telescope = "apo25m"

    def __init__(self, root):
        self.root = Path(root)

    def filename(self, kind, num, chips=True):
        directory = self.root / kind
        return str(directory / f"ap{kind}-{int(num):08d}.fits")

    def cmjd(self, number):
        return 60000


def test_chip_filename_inserts_b(tmp_path):
    result = lit._chip_filename(FakeLoad(tmp_path), "Littrow", 123)
    assert result.endswith("apLittrow-b-00000123.fits")


def test_subtract_scattered_light_uses_detector_edges():
    image = np.full((12, 14), 20.0)
    image[1:3, 2:12] = 8
    image[9:11, 2:12] = 12
    corrected, level = lit.subtract_scattered_light(
        image, x_range=(2, 11), bottom_rows=(1, 2), top_rows=(9, 10))
    assert level == 10
    assert corrected[5, 5] == 10
    assert image[5, 5] == 20  # input is not modified


@pytest.mark.parametrize("image", [np.ones(4), np.ones((3, 3, 2))])
def test_subtract_scattered_light_requires_image(image):
    with pytest.raises(ValueError, match="two-dimensional"):
        lit.subtract_scattered_light(image)


def test_subtract_scattered_light_validates_regions():
    with pytest.raises(ValueError, match="outside"):
        lit.subtract_scattered_light(
            np.ones((10, 10)), x_range=(0, 10),
            bottom_rows=(0, 1), top_rows=(8, 9))


def test_subtract_scattered_light_requires_finite_edges():
    with pytest.raises(ValueError, match="finite"):
        lit.subtract_scattered_light(
            np.full((6, 8), np.nan), x_range=(1, 6),
            bottom_rows=(0, 1), top_rows=(4, 5))


def test_fill_nonfinite_nearest():
    values = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
    filled = lit._fill_nonfinite_nearest(values)
    assert np.all(np.isfinite(filled))
    assert filled[0, 0] == 1


def test_fill_nonfinite_rejects_all_bad():
    with pytest.raises(ValueError, match="no finite"):
        lit._fill_nonfinite_nearest(np.full((3, 3), np.nan))


def test_make_littrow_mask_corrects_idl_axis_order():
    flux = np.zeros((20, 30))
    model = np.zeros_like(flux)
    # Signal occupies columns 5:14 and only rows 7:12.
    flux[7:13, 5:15] = 25
    result = lit.make_littrow_mask(
        flux, model, threshold=10, median_width=1,
        search_columns=(5, 14), output_columns=(7, 12))
    assert result.dtype == np.int16
    assert np.all(result[7:13, 7:13] == 1)
    assert not np.any(result[:, :7])
    assert not np.any(result[:, 13:])


def test_make_littrow_mask_applies_threshold_after_median():
    flux = np.zeros((7, 12))
    flux[2:5, 3:9] = 20
    result = lit.make_littrow_mask(
        flux, np.zeros_like(flux), threshold=10, median_width=3,
        search_columns=(3, 8), output_columns=(4, 7))
    assert result[3, 5] == 1
    assert result[0, 5] == 0


def test_make_littrow_mask_ignores_bad_pixel():
    flux = np.zeros((5, 10))
    flux[2, 4] = 100
    mask = np.zeros_like(flux, dtype=np.uint16)
    mask[2, 4] = 1
    result = lit.make_littrow_mask(
        flux, np.zeros_like(flux), mask, bad_pixel_bits=1,
        median_width=1, search_columns=(2, 7), output_columns=(3, 6))
    assert not np.any(result)


@pytest.mark.parametrize(
    "kwargs,match",
    [({"median_width": 0}, "median_width"),
     ({"search_columns": (1, 20)}, "bounds"),
     ({"search_columns": (4, 8), "output_columns": (2, 5)}, "map inside")],
)
def test_make_littrow_mask_validates_options(kwargs, match):
    defaults = {"search_columns": (2, 8), "output_columns": (3, 7)}
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=match):
        lit.make_littrow_mask(np.zeros((8, 12)), np.zeros((8, 12)),
                              **defaults)


def test_make_littrow_mask_validates_shapes():
    with pytest.raises(ValueError, match="matching"):
        lit.make_littrow_mask(np.zeros((2, 3)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="pixel_mask"):
        lit.make_littrow_mask(
            np.zeros((4, 8)), np.zeros((4, 8)), np.zeros((3, 8)),
            search_columns=(1, 6), output_columns=(2, 5))


def test_write_littrow_data_model(tmp_path):
    filename = tmp_path / "mask.fits"
    lit._write_littrow(filename, np.ones((4, 5)), apred="daily",
                       frameid=123, scatter_level=2.5)
    assert fits.getdata(filename).dtype.kind == "i"
    header = fits.getheader(filename)
    assert header["EXTNAME"] == "LITTROW MASK"
    assert header["LITID"] == 123
    assert header["SCATLEV"] == 2.5


def test_move_auxiliary_files(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    source = Path(load.filename("PSF", num=44, chips=True))
    source.parent.mkdir(parents=True)
    source = source.with_name("apPSF-b-00000044.fits")
    source.write_bytes(b"psf")
    destination = tmp_path / "Littrow"
    moved = lit._move_auxiliary_files(load, 44, destination)
    assert moved == [str(destination / source.name)]
    assert not source.exists()
    assert (destination / source.name).read_bytes() == b"psf"


def patch_builder(monkeypatch, tmp_path, *, model_in_return=True):
    load = FakeLoad(tmp_path)
    monkeypatch.setattr(lit, "_make_load", lambda **kwargs: load)
    lock_calls = []
    monkeypatch.setattr(lit.lock, "lock",
                        lambda *args, **kwargs: lock_calls.append((args, kwargs)))
    psf_calls = []
    monkeypatch.setattr(lit, "build_psf",
                        lambda *args, **kwargs: psf_calls.append((args, kwargs)))
    shape = (2048, 2048)
    reduced = {"flux": np.full(shape, 5.0),
               "mask": np.zeros(shape, dtype=np.uint16),
               "header": fits.Header()}
    monkeypatch.setattr(lit, "_load_reduced_frame", lambda filename: reduced)
    model = np.zeros(shape, dtype=np.float32)
    models = {1: model} if model_in_return else None
    monkeypatch.setattr(lit, "_run_empirical_extraction",
                        lambda *args, **kwargs: (None, models))
    monkeypatch.setattr(lit, "subtract_scattered_light",
                        lambda image: (image, 5.0))
    monkeypatch.setattr(lit, "make_littrow_mask",
                        lambda *args, **kwargs: np.zeros(shape, dtype=np.int16))
    monkeypatch.setattr(lit, "_move_auxiliary_files", lambda *args: [])
    return load, lock_calls, psf_calls


def test_build_littrow_workflow(monkeypatch, tmp_path):
    _, lock_calls, psf_calls = patch_builder(monkeypatch, tmp_path)
    output = lit.build_littrow(
        123, darkid=1, flatid=2, bpmid=3, sparseid=4, fiberid=5)
    assert Path(output).is_file()
    assert fits.getdata(output).shape == (2048, 2048)
    assert psf_calls[0][1]["average"] == 200
    assert psf_calls[0][1]["clobber"] is True
    assert psf_calls[0][1]["sparseid"] == 4
    assert lock_calls[-1][1] == {"clear": True}


def test_build_littrow_existing_short_circuit(monkeypatch, tmp_path, capsys):
    load, _, psf_calls = patch_builder(monkeypatch, tmp_path)
    output = Path(lit._chip_filename(load, "Littrow", 123))
    output.parent.mkdir(parents=True)
    output.write_bytes(b"existing")
    assert lit.build_littrow(123, verbose=True) == str(output)
    assert not psf_calls
    assert "already made" in capsys.readouterr().out


def test_build_littrow_clears_lock_after_failure(monkeypatch, tmp_path):
    _, lock_calls, _ = patch_builder(monkeypatch, tmp_path)
    monkeypatch.setattr(lit, "make_littrow_mask",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad")))
    with pytest.raises(RuntimeError, match="bad"):
        lit.build_littrow(123, sparseid=4)
    assert lock_calls[-1][1] == {"clear": True}


def test_makecal_littrow_dispatches_numbered_builder(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(makecal_v7, "build_littrow",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    load = FakeLoad(tmp_path)
    context = makecal_v7.CalibrationContext(
        load=load, calfile="cal.par", allcaldict={}, verbose=True)
    monkeypatch.setattr(context, "calibrations", lambda mjd: {
        "darkid": 1, "flatid": 2, "bpmid": 3,
        "sparseid": 4, "fiberid": 5})
    makecal_v7.littrow("123", context)
    assert calls[0][0] == ("123",)
    assert calls[0][1]["sparseid"] == 4
    assert calls[0][1]["apred"] == "daily"
