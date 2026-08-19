"""Tests for the shared Fiber/Sparse/PSF calibration implementation."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from apogee_drp.apred.cal import psfcal as pc


class FakeLoad:
    apred = "daily"
    telescope = "apo25m"
    instrument = "apogee-n"

    def __init__(self, root):
        self.root = Path(root)

    def filename(self, kind, num, chips=True):
        return str(self.root / f"ap{kind}-{int(num):08d}.fits")


def frame(ny=64, nx=80, centers=(15, 31, 47), amplitude=1000.0):
    y = np.arange(ny)[:, None]
    x = np.arange(nx)[None, :]
    flux = np.zeros((ny, nx), float)
    for center in centers:
        trace = center + 0.015 * (x - nx / 2) + 0.0001 * (x - nx / 2) ** 2
        flux += amplitude * np.exp(-0.5 * ((y - trace) / 1.1) ** 2)
    return {
        "flux": flux, "err": np.ones_like(flux),
        "mask": np.zeros_like(flux, dtype=np.uint16),
        "header": fits.Header({"EXPTYPE": "QUARTZFLAT"}),
    }


@pytest.mark.parametrize("kind,count", [("fiber", 3), ("sparse", 4), ("psf", 9)])
def test_product_files_contract(tmp_path, kind, count):
    files = pc.product_files(FakeLoad(tmp_path), kind, 123, )
    assert len(files) == count
    assert len(set(files)) == count
    assert all("00000123" in filename for filename in files)


def test_product_files_rejects_unknown(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        pc.product_files(FakeLoad(tmp_path), "bogus", 1)


def test_quadratic_peak_subpixel():
    profile = 10 - (np.arange(7) - 3.25) ** 2
    assert pc._quadratic_peak(profile, 3) == pytest.approx(3.25)
    assert pc._quadratic_peak(profile, 0) == 0


def test_match_peaks_is_unique_and_ordered():
    fibers, positions, distances = pc._match_peaks(
        [20.9, 10.2, 10.8, 50], np.array([10, 21, 30]), tolerance=1.5)
    np.testing.assert_array_equal(fibers, [0, 1])
    np.testing.assert_allclose(positions, [10.2, 20.9])
    np.testing.assert_allclose(distances, [0.2, 0.1])


def test_centroid_and_empty_centroid():
    assert pc._centroid(np.array([0, 1, 2, 1, 0]), 2) == pytest.approx(2)
    assert np.isnan(pc._centroid(np.zeros(5), 2))


def test_find_traces_recovers_curved_centers():
    result = pc.find_traces(frame(), np.array([15, 31, 47]), average=5)
    np.testing.assert_array_equal(result.fibers, [0, 1, 2])
    assert result.trace.shape == (3, 80)
    np.testing.assert_allclose(result.trace[:, 40], [15, 31, 47], atol=0.15)
    assert result.mean_distance < 0.2


def test_find_traces_ignores_masked_pixels():
    data = frame()
    data["mask"][0:8, :] = 1
    result = pc.find_traces(data, np.array([15, 31, 47]), average=5)
    assert len(result.fibers) == 3


@pytest.mark.parametrize("bad_average", [0, 40, 100])
def test_find_traces_validates_average(bad_average):
    with pytest.raises(ValueError, match="average"):
        pc.find_traces(frame(), [15], average=bad_average)


def test_find_traces_requires_matches():
    with pytest.raises(ValueError, match="No detected traces"):
        pc.find_traces(frame(), [2, 5], average=5)


def test_empirical_psf_is_normalized():
    data = frame()
    solution = pc.find_traces(data, [15, 31, 47], average=5)
    profiles = pc.build_empirical_psf(data, solution, half_width=5, smooth_columns=3)
    assert [profile["fiber"] for profile in profiles] == [0, 1, 2]
    for profile in profiles:
        totals = profile["img"].sum(axis=1)
        np.testing.assert_allclose(totals[totals > 0], 1, atol=1e-6)


def test_combine_profiles_uses_dense_core_and_sparse_wings():
    cent = np.full(4, 10.0)
    dense = [{"fiber": 2, "cent": cent, "lo": 4, "hi": 16,
              "img": np.ones((4, 13))}]
    sparse = [{"fiber": 2, "lo": 0, "hi": 20,
               "img": np.full((4, 21), 2.0)}]
    result = pc._combine_profiles(dense, sparse)[0]
    assert (result["lo"], result["hi"]) == (0, 20)
    np.testing.assert_allclose(result["img"].sum(axis=1), 1)
    assert result["img"][0, 10] < result["img"][0, 0]  # 1 in core, 2 in wing


def test_epsf_round_trip(tmp_path):
    solution = pc.find_traces(frame(), [15, 31, 47], average=5)
    profiles = pc.build_empirical_psf(frame(), solution, smooth_columns=3)
    filename = tmp_path / "epsf.fits"
    pc._write_epsf(filename, profiles, fits.Header({"TEST": 1}))
    loaded = pc._load_epsf(filename)
    assert fits.getheader(filename)["NTRACE"] == 3
    assert fits.getheader(filename)["TEST"] == 1
    np.testing.assert_allclose(loaded[1]["img"], profiles[1]["img"])


def test_average_frames_masks_and_combines(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    frames = [frame(8, 10, centers=(), amplitude=0) for _ in range(2)]
    frames[0]["flux"][:] = 2
    frames[1]["flux"][:] = 4
    frames[1]["mask"][0, 0] = 1
    monkeypatch.setattr(pc, "_load_frame", lambda load, exposure, chip: frames[exposure - 1])
    result = pc._average_frames(load, [1, 2], "a")
    assert result["flux"][1, 1] == 3
    assert result["flux"][0, 0] == 2


def test_reduce_forwards_chip_maxread(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    calls = []
    monkeypatch.setattr(pc.ap3d, "process_file", lambda *args, **kwargs: calls.append((args, kwargs)))
    pc._reduce(load, [11, 12], darkid=1, flatid=2, bpmid=3,
               maxread=np.array([4, 5, 6]))
    assert len(calls) == 6
    assert [calls[index * 2][1]["max_read"] for index in range(3)] == [4, 5, 6]
    assert calls[0][1]["dark"].endswith("Dark-a-00000001.fits")


def patch_workflow(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    monkeypatch.setattr(pc, "_make_load", lambda **kwargs: load)
    monkeypatch.setattr(pc.lock, "lock", lambda *args, **kwargs: None)
    monkeypatch.setattr(pc, "_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(pc, "_load_frame", lambda *args: frame())
    return load


def test_build_fiber_writes_three_trace_files(monkeypatch, tmp_path):
    patch_workflow(monkeypatch, tmp_path)
    outputs = pc.build_fiber(
        101, reference_positions={chip: np.array([15, 31, 47]) for chip in pc.CHIPS},
        average=5)
    assert len(outputs) == 3
    assert all(Path(filename).is_file() for filename in outputs)
    assert all(fits.getdata(filename).shape == (3, 80) for filename in outputs)


def test_build_fiber_existing_is_not_rebuilt(monkeypatch, tmp_path):
    load = patch_workflow(monkeypatch, tmp_path)
    outputs = pc.product_files(load, "fiber", 101)
    for filename in outputs:
        Path(filename).write_bytes(b"existing")
    monkeypatch.setattr(pc, "_reduce", lambda *args, **kwargs: pytest.fail("rebuilt"))
    assert pc.build_fiber(101) == outputs


def test_build_sparse_writes_four_products(monkeypatch, tmp_path):
    load = patch_workflow(monkeypatch, tmp_path)
    for chip in pc.CHIPS:
        fits.writeto(pc._chip_filename(load, "Fiber", 9, chip),
                     np.tile(np.array([15, 31, 47])[:, None], (1, 80)))
    outputs = pc.build_sparse([201, 202], fiberid=9, average=5, dmax=5)
    assert len(outputs) == 4
    assert fits.getdata(outputs[0]).shape == (3, 64, 80)
    assert all(fits.getheader(filename)["NTRACE"] == 3 for filename in outputs[1:])


def test_build_psf_requires_sparse(monkeypatch, tmp_path):
    patch_workflow(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="sparseid"):
        pc.build_psf(301)


def test_build_psf_writes_nine_products(monkeypatch, tmp_path):
    load = patch_workflow(monkeypatch, tmp_path)
    references = np.tile(np.array([15, 31, 47])[:, None], (1, 80))
    solution = pc.find_traces(frame(), [15, 31, 47], average=5)
    sparse_profiles = pc.build_empirical_psf(frame(), solution, half_width=10,
                                              smooth_columns=3)
    for chip in pc.CHIPS:
        fits.writeto(pc._chip_filename(load, "Fiber", 9, chip), references)
        pc._write_epsf(pc._chip_filename(load, "EPSF", 8, chip),
                       sparse_profiles, fits.Header())
    outputs = pc.build_psf(301, sparseid=8, fiberid=9, average=5)
    assert len(outputs) == 9
    assert all(Path(filename).is_file() for filename in outputs)
    for chip in pc.CHIPS:
        assert len(fits.open(pc._chip_filename(load, "PSF", 301, chip))) == 5


def test_prepare_outputs_treats_partial_product_as_incomplete(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    monkeypatch.setattr(pc.lock, "lock", lambda *args, **kwargs: None)
    files = pc.product_files(load, "fiber", 4)
    Path(files[0]).write_bytes(b"partial")
    outputs, should_build = pc._prepare_outputs(
        load, "fiber", 4, clobber=False, unlock=False, verbose=False)
    assert should_build
    assert outputs == files
    assert not Path(files[0]).exists()
