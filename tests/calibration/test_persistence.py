"""Tests for the persistence-mask calibration builder."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from apogee_drp.apred.cal import makecal
from apogee_drp.apred.cal import persistence as persist
from apogee_drp.apred.cal import utils as cal_utils


class FakeLoad:
    apred = "daily"
    telescope = "apo25m"

    def __init__(self, root):
        self.root = Path(root)
        self.exists_calls = 0
        self.delete_calls = []

    def cmjd(self, number):
        return "60000"

    def filename(self, kind, num=None, mjd=None, chip=None,
                 directory=False):
        root = self.root / ("exp" if kind == "2D" else "cal")
        if directory:
            return str(root)
        infix = f"-{chip}" if chip is not None else ""
        return str(root / f"ap{kind}{infix}-{int(num):08d}.fits")

    def product_files(self, product, name):
        assert product == "persist"
        return [self.filename("Persist", num=name, chip=chip)
                for chip in "abc"]

    def product_exists(self, product, name):
        self.exists_calls += 1
        return all(
            Path(filename).is_file() and Path(filename).stat().st_size > 0
            for filename in self.product_files(product, name)
        )

    def product_delete(self, product, name, **kwargs):
        self.delete_calls.append((product, name, kwargs))
        deleted = []
        for filename in self.product_files(product, name):
            path = Path(filename)
            if path.exists() or path.is_symlink():
                path.unlink()
                deleted.append(str(path))
        return deleted

    def frame(self, number, chip=None, **kwargs):
        chips = ("a", "b", "c") if chip is None else (chip,)
        frames = {}
        for current_chip in chips:
            filename = self.filename(
                "2D", num=number, chip=current_chip)
            with fits.open(filename, memmap=False) as hdus:
                frames[current_chip] = {
                    "header": hdus[0].header.copy(),
                    "flux": np.asarray(hdus[1].data).copy(),
                    "err": np.asarray(hdus[2].data).copy(),
                    "mask": np.asarray(hdus[3].data).copy(),
                }
        return frames if chip is None else frames[chip]


def test_registry_product_files_are_three_chips(tmp_path):
    files = FakeLoad(tmp_path).product_files("persist", 123)
    assert [Path(name).name for name in files] == [
        "apPersist-a-00000123.fits",
        "apPersist-b-00000123.fits",
        "apPersist-c-00000123.fits",
    ]


def test_obsolete_product_filename_and_load_helpers_are_removed():
    assert not hasattr(persist, "product_files")
    assert not hasattr(persist, "_chip_filename")
    assert not hasattr(persist, "_make_load")
    assert not hasattr(persist, "_load_2d")


def test_mask_severity_matches_idl_threshold_order():
    dark = np.array([[0.0, 0.026, 0.051, 0.101]])
    mask, rate = persist.make_persistence_mask(
        dark, np.ones_like(dark), threshold=0.1, smooth_size=(1, 1))
    np.testing.assert_array_equal(mask, [[0, 4, 2, 1]])
    np.testing.assert_allclose(rate, dark)
    assert mask.dtype == np.int16
    assert rate.dtype == np.float32


def test_threshold_boundaries_are_strict():
    dark = np.array([[0.025, 0.05, 0.1]])
    mask, _ = persist.make_persistence_mask(
        dark, np.ones_like(dark), threshold=0.1, smooth_size=(1, 1))
    np.testing.assert_array_equal(mask, [[0, 4, 2]])


def test_ratio_uses_dark_divided_by_flat():
    mask, rate = persist.make_persistence_mask(
        np.full((2, 2), 2.0), np.full((2, 2), 4.0),
        threshold=1.0, smooth_size=(1, 1))
    np.testing.assert_allclose(rate, 0.5)
    np.testing.assert_array_equal(mask, 4)


def test_zero_and_nonfinite_inputs_are_ignored():
    dark = np.array([[1.0, np.nan, np.inf]])
    flat = np.array([[0.0, 1.0, 1.0]])
    mask, rate = persist.make_persistence_mask(
        dark, flat, smooth_size=(1, 1))
    np.testing.assert_array_equal(rate, 0)
    np.testing.assert_array_equal(mask, 0)


def test_bad_dark_and_flat_pixels_are_ignored():
    dark = np.ones((2, 2))
    dark_mask = np.array([[1, 0], [0, 0]])
    flat_mask = np.array([[0, 1], [0, 0]])
    mask, rate = persist.make_persistence_mask(
        dark, dark, dark_mask, flat_mask, threshold=0.5,
        smooth_size=(1, 1), bad_pixel_bits=1)
    np.testing.assert_array_equal(rate, [[0, 0], [1, 1]])
    np.testing.assert_array_equal(mask, [[0, 0], [1, 1]])


def test_unselected_mask_bits_do_not_reject_pixels():
    input_mask = np.full((2, 2), 2)
    mask, _ = persist.make_persistence_mask(
        np.ones((2, 2)), np.ones((2, 2)), dark_mask=input_mask,
        threshold=0.5, smooth_size=(1, 1), bad_pixel_bits=1)
    np.testing.assert_array_equal(mask, 1)


def test_running_median_suppresses_isolated_pixel():
    dark = np.zeros((5, 5))
    dark[2, 2] = 10
    mask, rate = persist.make_persistence_mask(
        dark, np.ones_like(dark), smooth_size=(3, 3))
    assert rate[2, 2] == 0
    assert not np.any(mask)


def test_inputs_are_not_modified():
    dark = np.ones((3, 3))
    flat = np.ones((3, 3))
    original_dark, original_flat = dark.copy(), flat.copy()
    persist.make_persistence_mask(dark, flat, smooth_size=(1, 1))
    np.testing.assert_array_equal(dark, original_dark)
    np.testing.assert_array_equal(flat, original_flat)


@pytest.mark.parametrize("dark,flat", [
    (np.zeros(3), np.zeros(3)),
    (np.zeros((2, 2)), np.zeros((3, 2))),
])
def test_mask_rejects_invalid_flux_shapes(dark, flat):
    with pytest.raises(ValueError, match="matching 2-D"):
        persist.make_persistence_mask(dark, flat)


def test_mask_rejects_invalid_input_mask_shape():
    with pytest.raises(ValueError, match="input masks"):
        persist.make_persistence_mask(
            np.zeros((2, 2)), np.ones((2, 2)), dark_mask=np.zeros(3))


@pytest.mark.parametrize("threshold", [0, -1])
def test_mask_rejects_nonpositive_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        persist.make_persistence_mask(
            np.zeros((2, 2)), np.ones((2, 2)), threshold=threshold)


@pytest.mark.parametrize("size", [(0, 1), (1,), (1.5, 2)])
def test_mask_rejects_invalid_smoothing_size(size):
    with pytest.raises(ValueError, match="smooth_size"):
        persist.make_persistence_mask(
            np.zeros((2, 2)), np.ones((2, 2)), smooth_size=size)


def test_fake_load_frame_reads_all_chips(tmp_path):
    load = FakeLoad(tmp_path)
    for index, chip in enumerate("abc"):
        filename = Path(load.filename("2D", num=12, chip=chip))
        filename.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList([
            fits.PrimaryHDU(header=fits.Header({"FRAME": 12})),
            fits.ImageHDU(np.full((2, 3), index + 1.0)),
            fits.ImageHDU(np.ones((2, 3))),
            fits.ImageHDU(np.full((2, 3), 9, dtype=np.int16)),
        ]).writeto(filename)
    frames = load.frame(12)
    assert set(frames) == {"a", "b", "c"}
    np.testing.assert_array_equal(frames["b"]["flux"], 2)


def test_write_persist_data_model(tmp_path):
    filename = tmp_path / "persist.fits"
    persist._write_persist(
        filename, np.ones((2, 3)), np.full((2, 3), 0.2),
        fits.Header({"CHIP": "b"}), apred="daily", threshold=0.1)
    with fits.open(filename) as hdul:
        assert hdul[0].data.dtype.kind == "i"
        assert hdul[1].data.dtype.kind == "f"
        assert hdul[0].header["EXTNAME"] == "PERSIST"
        assert hdul[0].header["PTHRESH"] == 0.1
        assert hdul[0].header["APRED"] == "daily"
        assert hdul[0].header["CHIP"] == "b"
        assert hdul[1].header["EXTNAME"] == "PERSIST_RATE"


def patch_builder(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    monkeypatch.setattr(persist.apload, "ApLoad", lambda **kwargs: load)
    lock_calls = []
    monkeypatch.setattr(
        cal_utils.lock, "lock",
        lambda *args, **kwargs: lock_calls.append((args, kwargs)))
    process_calls = []
    monkeypatch.setattr(
        persist, "_process",
        lambda *args, **kwargs: process_calls.append((args, kwargs)))

    frame_calls = []
    def load_frame(number, chip=None):
        frame_calls.append((int(number), chip))
        value = 0.2 if int(number) == 10 else 1.0
        frames = {
            current_chip: {
                "header": fits.Header({"CHIP": current_chip}),
                "flux": np.full((3, 4), value),
                "err": np.ones((3, 4)),
                "mask": np.zeros((3, 4), dtype=np.int16),
            }
            for current_chip in "abc"
        }
        return frames if chip is None else frames[chip]
    load.frame = load_frame
    load.frame_calls = frame_calls
    return load, lock_calls, process_calls


def test_build_persist_processes_and_writes_all_chips(monkeypatch, tmp_path):
    load, lock_calls, process_calls = patch_builder(monkeypatch, tmp_path)
    result = persist.build_persist(
        99, 10, 20, cmjd="60000", darkid=30, flatid=40,
        threshold=0.1, verbose=True)
    assert result is None
    args, kwargs = process_calls[0]
    assert args[1] == [10, 20]
    assert kwargs == {
        "cmjd": "60000", "darkid": 30, "flatid": 40,
        "clobber": False, "unlock": False, "verbose": True,
    }
    for chip, filename in zip("abc", load.product_files("persist", 99)):
        with fits.open(filename) as hdul:
            np.testing.assert_array_equal(hdul[0].data, 1)
            np.testing.assert_allclose(hdul[1].data, 0.2)
            assert hdul[0].header["CHIP"] == chip
    assert load.delete_calls == [("persist", "99", {"verbose": True})]
    assert load.frame_calls == [(10, None), (20, None)]
    assert lock_calls[-1][1] == {"clear": True}


def test_thresh_legacy_alias_overrides_threshold(monkeypatch, tmp_path):
    load, _, _ = patch_builder(monkeypatch, tmp_path)
    persist.build_persist(99, 10, 20, threshold=9, thresh=0.3)
    with fits.open(load.product_files("persist", 99)[0]) as hdul:
        assert hdul[0].header["PTHRESH"] == 0.3
        np.testing.assert_array_equal(hdul[0].data, 2)


def test_existing_product_short_circuits_without_lock(monkeypatch, tmp_path,
                                                       capsys):
    load, lock_calls, process_calls = patch_builder(monkeypatch, tmp_path)
    for filename in load.product_files("persist", 99):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"existing")
    assert persist.build_persist(99, 10, 20, verbose=True) is None
    assert not process_calls
    assert not lock_calls
    assert not load.delete_calls
    assert "persist product 99 already exists" in capsys.readouterr().out


def test_partial_product_is_deleted_and_rebuilt(monkeypatch, tmp_path):
    load, _, process_calls = patch_builder(monkeypatch, tmp_path)
    first = Path(load.product_files("persist", 99)[0])
    first.parent.mkdir(parents=True)
    first.write_bytes(b"partial")
    persist.build_persist(99, 10, 20)
    assert process_calls
    assert load.delete_calls
    assert all(Path(filename).stat().st_size > 0
               for filename in load.product_files("persist", 99))


def test_clobber_rebuilds_complete_product(monkeypatch, tmp_path):
    load, _, process_calls = patch_builder(monkeypatch, tmp_path)
    outputs = load.product_files("persist", 99)
    for filename in outputs:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"old")
    persist.build_persist(99, 10, 20, clobber=True)
    assert process_calls[0][1]["clobber"] is True
    assert Path(outputs[0]).read_bytes() != b"old"


def test_lock_is_cleared_when_processing_fails(monkeypatch, tmp_path):
    _, lock_calls, _ = patch_builder(monkeypatch, tmp_path)
    monkeypatch.setattr(
        persist, "_process",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        persist.build_persist(99, 10, 20)
    assert lock_calls[-1][1] == {"clear": True}


def test_unexpected_registry_file_count_fails(monkeypatch, tmp_path):
    load, _, _ = patch_builder(monkeypatch, tmp_path)
    monkeypatch.setattr(
        load, "product_files",
        lambda product, name: [load.filename("Persist", num=name, chip="a")])
    with pytest.raises(RuntimeError, match="expected 3"):
        persist.build_persist(99, 10, 20)


def test_makecal_persist_dispatches_current_builder(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        makecal, "build_persist",
        lambda *args, **kwargs: calls.append((args, kwargs)))
    context = makecal.CalibrationContext(
        load=FakeLoad(tmp_path), calfile="cal.par", allcaldict={},
        clobber=True, unlock=True, verbose=True)
    monkeypatch.setattr(
        context, "row",
        lambda *args, **kwargs: {
            "darkid": 10, "flatid": 20, "thresh": 0.2})
    monkeypatch.setattr(
        context, "calibrations",
        lambda mjd: {
            "darkid": 30, "flatid": 40,
            "sparseid": 50, "fiberid": 60})
    makecal.persist("99", context)
    assert calls[0][0] == ("99", 10, 20)
    assert calls[0][1]["thresh"] == 0.2
    assert calls[0][1]["cmjd"] == "60000"
    assert calls[0][1]["clobber"] is True
    assert calls[0][1]["unlock"] is True
    assert calls[0][1]["verbose"] is True

