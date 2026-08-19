"""Tests for the numbered persistence-mask implementation."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from apogee_drp.apred.cal import makecal
from apogee_drp.apred.cal import persistence as persist


class FakeLoad:
    apred = "daily"
    telescope = "apo25m"

    def __init__(self, root):
        self.root = Path(root)

    def cmjd(self, number):
        return "60000"

    def filename(self, kind, num=None, chips=False, **kwargs):
        directory = self.root / ("exp" if kind == "2D" else "cal")
        return str(directory / f"ap{kind}-{int(num):08d}.fits")


def test_product_files_are_three_chip_products(tmp_path):
    files = persist.product_files(FakeLoad(tmp_path), 123)
    assert [Path(name).name for name in files] == [
        "apPersist-a-00000123.fits",
        "apPersist-b-00000123.fits",
        "apPersist-c-00000123.fits",
    ]


def test_mask_severity_matches_idl_threshold_order():
    dark = np.array([[0.0, 0.026, 0.051, 0.101]])
    mask, rate = persist.make_persistence_mask(
        dark, np.ones_like(dark), threshold=0.1, smooth_size=(1, 1))
    np.testing.assert_array_equal(mask, [[0, 4, 2, 1]])
    np.testing.assert_allclose(rate, dark)


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


def test_unselected_mask_bits_are_retained():
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
    original = dark.copy(), flat.copy()
    persist.make_persistence_mask(dark, flat, smooth_size=(1, 1))
    np.testing.assert_array_equal(dark, original[0])
    np.testing.assert_array_equal(flat, original[1])


@pytest.mark.parametrize("dark, flat", [
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


def test_load_2d_reads_flux_mask_and_header(tmp_path):
    load = FakeLoad(tmp_path)
    filename = Path(persist._chip_filename(load, "2D", 12, "b"))
    filename.parent.mkdir(parents=True)
    fits.HDUList([
        fits.PrimaryHDU(header=fits.Header({"FRAME": 12})),
        fits.ImageHDU(np.full((2, 3), 7.0)),
        fits.ImageHDU(),
        fits.ImageHDU(np.full((2, 3), 9, dtype=np.int16)),
    ]).writeto(filename)
    frame = persist._load_2d(load, 12, "b")
    assert frame["header"]["FRAME"] == 12
    np.testing.assert_array_equal(frame["flux"], 7)
    np.testing.assert_array_equal(frame["mask"], 9)


def patch_builder(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    monkeypatch.setattr(persist, "_make_load", lambda **kwargs: load)
    lock_calls = []
    monkeypatch.setattr(
        persist.lock, "lock",
        lambda *args, **kwargs: lock_calls.append((args, kwargs)))
    process_calls = []
    monkeypatch.setattr(
        persist, "_process",
        lambda *args, **kwargs: process_calls.append((args, kwargs)))

    def load_2d(actual_load, number, chip):
        value = 0.2 if int(number) == 10 else 1.0
        return {
            "header": fits.Header({"CHIP": chip}),
            "flux": np.full((3, 4), value),
            "mask": np.zeros((3, 4), dtype=np.int16),
        }

    monkeypatch.setattr(persist, "_load_2d", load_2d)
    return load, lock_calls, process_calls


def test_build_persist_processes_and_writes_idl_layout(monkeypatch, tmp_path):
    load, lock_calls, process_calls = patch_builder(monkeypatch, tmp_path)
    outputs = persist.build_persist(
        99, 10, 20, cmjd="60000", darkid=30, flatid=40,
        threshold=0.1, verbose=True)
    assert outputs == persist.product_files(load, 99)
    args, kwargs = process_calls[0]
    assert args[1] == [10, 20]
    assert kwargs == {
        "cmjd": "60000", "darkid": 30, "flatid": 40,
        "clobber": False, "unlock": False, "verbose": True,
    }
    for chip, filename in zip("abc", outputs):
        with fits.open(filename) as hdul:
            np.testing.assert_array_equal(hdul[0].data, 1)
            np.testing.assert_allclose(hdul[1].data, 0.2)
            assert hdul[0].header["EXTNAME"] == "PERSIST"
            assert hdul[0].header["PTHRESH"] == 0.1
            assert hdul[0].header["CHIP"] == chip
            assert hdul[1].header["EXTNAME"] == "PERSIST_RATE"
    assert lock_calls[-1][1] == {"clear": True}


def test_thresh_legacy_alias_overrides_threshold(monkeypatch, tmp_path):
    load, _, _ = patch_builder(monkeypatch, tmp_path)
    outputs = persist.build_persist(99, 10, 20, threshold=9, thresh=0.3)
    with fits.open(outputs[0]) as hdul:
        assert hdul[0].header["PTHRESH"] == 0.3
        np.testing.assert_array_equal(hdul[0].data, 2)


def test_existing_products_short_circuit(monkeypatch, tmp_path, capsys):
    load, lock_calls, process_calls = patch_builder(monkeypatch, tmp_path)
    outputs = persist.product_files(load, 99)
    for filename in outputs:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(b"existing")
    assert persist.build_persist(99, 10, 20, verbose=True) == outputs
    assert not process_calls
    assert "already made" in capsys.readouterr().out
    assert all(kwargs != {"clear": True} for _, kwargs in lock_calls)


def test_partial_products_are_removed_and_rebuilt(monkeypatch, tmp_path):
    load, _, process_calls = patch_builder(monkeypatch, tmp_path)
    outputs = persist.product_files(load, 99)
    Path(outputs[0]).parent.mkdir(parents=True)
    Path(outputs[0]).write_bytes(b"partial")
    persist.build_persist(99, 10, 20)
    assert process_calls
    assert all(Path(filename).stat().st_size > 0 for filename in outputs)


def test_clobber_rebuilds_complete_product(monkeypatch, tmp_path):
    load, _, process_calls = patch_builder(monkeypatch, tmp_path)
    outputs = persist.product_files(load, 99)
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


def test_makecal_persist_dispatches_numbered_builder(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        makecal_v7, "build_persist",
        lambda *args, **kwargs: calls.append((args, kwargs)))
    context = makecal_v7.CalibrationContext(
        load=FakeLoad(tmp_path), calfile="cal.par", allcaldict={},
        clobber=True, unlock=True, verbose=True)
    monkeypatch.setattr(
        context, "row",
        lambda *args, **kwargs: {"darkid": 10, "flatid": 20, "thresh": 0.2})
    monkeypatch.setattr(
        context, "calibrations",
        lambda mjd: {"darkid": 30, "flatid": 40,
                     "sparseid": 50, "fiberid": 60})
    makecal_v7.persist("99", context)
    assert calls[0][0] == ("99", 10, 20)
    assert calls[0][1]["thresh"] == 0.2
    assert calls[0][1]["cmjd"] == "60000"
    assert calls[0][1]["verbose"] is True


def test_legacy_mkpersist_wrapper(monkeypatch):
    from apogee_drp.apred.cal import mkpersist
    calls = []
    monkeypatch.setattr(
        mkpersist, "build_persist",
        lambda *args, **kwargs: calls.append((args, kwargs)) or ["done"])
    assert mkpersist.mkpersist(1, 2, 3, threshold=0.4) == ["done"]
    assert calls == [((1, 2, 3), {"clobber": False, "threshold": 0.4})]
