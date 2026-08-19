"""Focused tests for detector linearity and product construction."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from apogee_drp.apred.cal import detector


class FakeLoad:
    """Small ApLoad stand-in that writes every product below ``root``."""

    prefix = "ap"

    def __init__(self, root):
        self.root = Path(root)

    def filename(self, kind, num=None, chips=False):
        return str(self.root / f"ap{kind}-{int(num):08d}.fits")


@pytest.fixture
def detector_environment(monkeypatch, tmp_path):
    load = FakeLoad(tmp_path)
    lock_calls = []
    monkeypatch.setattr(detector, "_make_load", lambda **kwargs: load)
    monkeypatch.setattr(
        detector.lock, "lock",
        lambda filename, **kwargs: lock_calls.append((Path(filename), kwargs)),
    )
    return load, lock_calls


def synthetic_measurements(chip=0, *, coefficients=(1.0, -2e-6, 3e-11),
                           iy=5, reads=3):
    data = np.zeros(12, dtype=detector.LINEARITY_DTYPE)
    data["chip"] = chip
    data["ix"] = np.repeat([0, 5, 10], 4)
    data["iy"] = iy
    data["read"] = reads
    data["counts"] = np.linspace(1000, 40_000, len(data))
    data["rate"] = np.polynomial.polynomial.polyval(
        data["counts"], coefficients)
    data["instantaneous_rate"] = data["rate"]
    return data


def synthetic_ramp(nread=11, step=350.0):
    """Return a memory-cheap, spatially uniform full-size ramp view."""
    reads = np.arange(nread, dtype=float)[:, None, None] * step
    return np.broadcast_to(reads, (nread, 1984, 1984))


def test_linearity_dtype_matches_disk_column_contract():
    assert detector.LINEARITY_DTYPE.names == (
        "read", "chip", "ix", "iy", "counts", "rate",
        "instantaneous_rate",
    )
    assert np.issubdtype(detector.LINEARITY_DTYPE["read"], np.integer)
    assert np.issubdtype(detector.LINEARITY_DTYPE["rate"], np.floating)


def test_sample_linearity_uniform_ramp():
    result = detector.sample_linearity(synthetic_ramp(), 2, nskip=4)
    assert result.dtype == detector.LINEARITY_DTYPE
    assert len(result) == 8 * 8 * 3
    np.testing.assert_array_equal(np.unique(result["read"]), [2, 6, 10])
    np.testing.assert_array_equal(np.unique(result["chip"]), [2])
    np.testing.assert_array_equal(np.unique(result["ix"]), np.arange(0, 40, 5))
    np.testing.assert_array_equal(np.unique(result["iy"]), np.arange(0, 40, 5))
    np.testing.assert_allclose(result["rate"], 1.0, atol=1e-12)
    np.testing.assert_allclose(result["instantaneous_rate"], 1.0, atol=1e-12)


def test_sample_linearity_respects_nskip():
    result = detector.sample_linearity(synthetic_ramp(nread=12), 0, nskip=3)
    np.testing.assert_array_equal(np.unique(result["read"]), [2, 5, 8, 11])


@pytest.mark.parametrize(
    "shape, message",
    [
        ((4, 20), "shape"),
        ((4, 1983, 1984), "shape"),
        ((3, 1984, 1984), "four reads"),
    ],
)
def test_sample_linearity_rejects_invalid_cubes(shape, message):
    cube = np.broadcast_to(0.0, shape)
    with pytest.raises(ValueError, match=message):
        detector.sample_linearity(cube, 0)


def test_sample_linearity_rejects_nonpositive_nskip():
    with pytest.raises(ValueError, match="positive"):
        detector.sample_linearity(synthetic_ramp(4), 0, nskip=0)


def test_sample_linearity_returns_empty_when_reference_range_is_missing():
    result = detector.sample_linearity(synthetic_ramp(step=10_000), 0)
    assert result.shape == (0,)
    assert result.dtype == detector.LINEARITY_DTYPE


def test_fit_linearity_recovers_each_supported_order():
    for coefficients in ([1.2, -3e-6], [1.0, -2e-6, 3e-11]):
        data = synthetic_measurements(coefficients=coefficients)
        fitted = detector.fit_linearity(data, order=len(coefficients) - 1)
        np.testing.assert_allclose(fitted, coefficients, rtol=1e-8, atol=1e-12)


def test_fit_linearity_applies_read_count_and_finite_filters():
    good = synthetic_measurements()
    rejected = good[:4].copy()
    rejected["read"] = [1, 3, 3, 3]
    rejected["counts"] = [2000, 60_000, np.nan, 3000]
    rejected["rate"] = [100, 100, 100, np.nan]
    combined = np.concatenate([good, rejected])
    np.testing.assert_allclose(
        detector.fit_linearity(combined), detector.fit_linearity(good),
        rtol=1e-10,
    )


def test_fit_linearity_apo_uses_low_y_chip_c_only():
    chip_a = synthetic_measurements(0)
    chip_c_bad = synthetic_measurements(2, coefficients=(100, 0, 0), iy=20)
    np.testing.assert_allclose(
        detector.fit_linearity(np.concatenate([chip_a, chip_c_bad])),
        detector.fit_linearity(chip_a), rtol=1e-10,
    )


def test_fit_linearity_lco_includes_chip_b():
    chip_b = synthetic_measurements(1, coefficients=(2, 1e-6, 0))
    np.testing.assert_allclose(
        detector.fit_linearity(chip_b, telescope="lco25m"),
        [2, 1e-6, 0], rtol=1e-8, atol=1e-12,
    )
    with pytest.raises(ValueError, match="not enough"):
        detector.fit_linearity(chip_b, telescope="apo25m")


def test_fit_linearity_rejects_unstructured_or_incomplete_data():
    with pytest.raises(ValueError, match="fields"):
        detector.fit_linearity(np.ones((5, 7)))
    incomplete = np.zeros(5, dtype=[("counts", float), ("rate", float)])
    with pytest.raises(ValueError, match="fields"):
        detector.fit_linearity(incomplete)


def test_fit_linearity_requires_more_points_than_polynomial_order():
    with pytest.raises(ValueError, match="not enough"):
        detector.fit_linearity(synthetic_measurements()[:2], order=2)


def test_measure_linearity_reads_all_chips_and_forwards_options(
        detector_environment, monkeypatch):
    _, lock_calls = detector_environment
    reader_calls = []
    sampled_chips = []

    def reader(filename, **kwargs):
        reader_calls.append((filename, kwargs))
        return np.zeros((4, 2, 2))

    def sample(cube, chip, **kwargs):
        sampled_chips.append((chip, kwargs))
        return synthetic_measurements(chip)

    monkeypatch.setattr(detector, "sample_linearity", sample)
    result = detector.measure_linearity(
        123, nread=9, nskip=3, ramp_reader=reader, verbose=True)
    assert len(reader_calls) == 3
    assert [Path(call[0]).name for call in reader_calls] == [
        "apR-a-00000123.fits", "apR-b-00000123.fits", "apR-c-00000123.fits"]
    assert all(call[1] == {"apred": "daily", "nread": 9} for call in reader_calls)
    assert sampled_chips == [(0, {"nskip": 3}), (1, {"nskip": 3}),
                             (2, {"nskip": 3})]
    np.testing.assert_allclose(result, [1, -2e-6, 3e-11], rtol=1e-8)
    assert lock_calls[-1][1] == {"clear": True}


def test_measure_linearity_single_chip(detector_environment, monkeypatch):
    calls = []
    monkeypatch.setattr(
        detector, "sample_linearity",
        lambda cube, chip, **kwargs: calls.append(chip) or synthetic_measurements(chip),
    )
    detector.measure_linearity(
        124, chip=2, ramp_reader=lambda *args, **kwargs: np.zeros((4, 2, 2)))
    assert calls == [2]


def test_measure_linearity_reuses_cached_measurements(
        detector_environment, monkeypatch, capsys):
    _, lock_calls = detector_environment
    monkeypatch.setattr(
        detector, "sample_linearity",
        lambda *args, **kwargs: synthetic_measurements(),
    )
    detector.measure_linearity(
        125, chip=0,
        ramp_reader=lambda *args, **kwargs: np.zeros((4, 2, 2)),
    )
    lock_calls.clear()

    def should_not_read(*args, **kwargs):
        raise AssertionError("cached measurements should avoid ramp I/O")

    result = detector.measure_linearity(125, ramp_reader=should_not_read, verbose=True)
    # The on-disk diagnostic table intentionally rounds rates to seven digits.
    np.testing.assert_allclose(result, [1, -2e-6, 3e-11], rtol=3e-6)
    assert "already exist" in capsys.readouterr().out
    assert not any(options.get("lock") for _, options in lock_calls)
    assert not any(options.get("clear") for _, options in lock_calls)


def test_measure_linearity_clobber_recomputes_cache(
        detector_environment, monkeypatch):
    monkeypatch.setattr(
        detector, "sample_linearity",
        lambda *args, **kwargs: synthetic_measurements(coefficients=(5, 0, 0)),
    )
    detector.measure_linearity(
        126, chip=0,
        ramp_reader=lambda *args, **kwargs: np.zeros((4, 2, 2)),
    )
    monkeypatch.setattr(
        detector, "sample_linearity",
        lambda *args, **kwargs: synthetic_measurements(),
    )
    result = detector.measure_linearity(
        126, chip=0, clobber=True,
        ramp_reader=lambda *args, **kwargs: np.zeros((4, 2, 2)),
    )
    np.testing.assert_allclose(result, [1, -2e-6, 3e-11], rtol=1e-8)


def test_measure_linearity_rejects_invalid_chip_and_clears_lock(
        detector_environment):
    _, lock_calls = detector_environment
    with pytest.raises(ValueError, match="chip"):
        detector.measure_linearity(127, chip=4)
    assert lock_calls[-1][1] == {"clear": True}


def test_measure_linearity_clears_lock_when_reader_fails(detector_environment):
    _, lock_calls = detector_environment

    def fail(*args, **kwargs):
        raise RuntimeError("bad ramp")

    with pytest.raises(RuntimeError, match="bad ramp"):
        detector.measure_linearity(128, chip=0, ramp_reader=fail)
    assert lock_calls[-1][1] == {"clear": True}


def test_read_and_correct_ramp_limits_reads(monkeypatch, tmp_path):
    outdir = tmp_path / "daily"
    outdir.mkdir()
    unpacked = outdir / "apR-a-00000001.fits"
    fits.HDUList([fits.PrimaryHDU(header=fits.Header({"TEST": 42}))] + [
        fits.ImageHDU(np.full((3, 4), value)) for value in range(5)
    ]).writeto(unpacked)
    monkeypatch.setattr(detector.utils, "localdir", lambda: str(tmp_path))

    import apogee_drp.apred.ap3d as ap3d
    calls = []

    def reference_correct(cube, header, **kwargs):
        calls.append((cube.copy(), header, kwargs))
        return cube + 1, None, None, None

    monkeypatch.setattr(ap3d, "reference_correct", reference_correct)
    result = detector._read_and_correct_ramp(
        "/raw/apR-a-00000001.fits", apred="daily", nread=3)
    assert result.shape == (3, 3, 4)
    np.testing.assert_array_equal(result[:, 0, 0], [1, 2, 3])
    assert calls[0][1]["TEST"] == 42
    assert calls[0][2] == {"indiv": 0, "cds": True}


@pytest.mark.parametrize(
    "telescope, gain, read_noise",
    [
        ("apo25m", 1.9, [13 * 1.9, 11 * 1.9, 10 * 1.9]),
        ("lco25m", 3.0, [7 * 3.0, 8 * 3.0, 4 * 3.0]),
    ],
)
def test_build_detector_fits_contract(detector_environment, telescope, gain,
                                      read_noise):
    _, lock_calls = detector_environment
    coefficients = np.array([1.0, -2e-6, 3e-11])
    linearity_calls = []

    def linearity(frameid, **kwargs):
        linearity_calls.append((frameid, kwargs))
        return coefficients

    outputs = detector.build_detector(
        13390003, linid=13390001, telescope=telescope,
        linearity_function=linearity, verbose=True)
    assert [Path(path).name for path in outputs] == [
        "apDetector-a-13390003.fits", "apDetector-b-13390003.fits",
        "apDetector-c-13390003.fits"]
    assert linearity_calls[0][0] == 13390001
    for index, output in enumerate(outputs):
        with fits.open(output) as hdul:
            assert [hdu.name for hdu in hdul] == [
                "PRIMARY", "READNOISE", "GAIN", "LINEARITY CORRECTION"]
            np.testing.assert_allclose(hdul[1].data, read_noise[index])
            np.testing.assert_allclose(hdul[2].data, gain)
            assert hdul[3].data.shape == (4, 3)
            np.testing.assert_allclose(hdul[3].data,
                                       np.tile(coefficients, (4, 1)))
    assert lock_calls[-1][1] == {"clear": True}


@pytest.mark.parametrize("linid", [None, 0])
def test_build_detector_identity_linearity_without_exposure(
        detector_environment, linid):
    def should_not_run(*args, **kwargs):
        raise AssertionError("linearity measurement should not run")

    outputs = detector.build_detector(
        200, linid=linid, linearity_function=should_not_run)
    with fits.open(outputs[0]) as hdul:
        np.testing.assert_array_equal(
            hdul[3].data, np.tile([1.0, 0.0, 0.0], (4, 1)))


def test_build_detector_existing_products_are_reused(
        detector_environment, capsys):
    load, lock_calls = detector_environment
    outputs = [load.root / f"apDetector-{chip}-00000201.fits" for chip in "abc"]
    for output in outputs:
        output.touch()

    def should_not_run(*args, **kwargs):
        raise AssertionError("existing products should skip linearity")

    actual = detector.build_detector(
        201, linid=99, linearity_function=should_not_run, verbose=True)
    assert list(map(Path, actual)) == outputs
    assert "already exists" in capsys.readouterr().out
    assert not any(options.get("lock") for _, options in lock_calls)


def test_build_detector_partial_products_are_rebuilt(detector_environment):
    load, _ = detector_environment
    old = load.root / "apDetector-a-00000202.fits"
    old.write_bytes(b"old")
    outputs = detector.build_detector(202)
    assert all(Path(path).stat().st_size > 3 for path in outputs)
    with fits.open(old) as hdul:
        assert hdul[1].name == "READNOISE"


def test_build_detector_clobber_rewrites_complete_product_set(
        detector_environment):
    first = detector.build_detector(203)
    for output in first:
        Path(output).write_bytes(b"old")
    second = detector.build_detector(203, clobber=True)
    assert second == first
    for output in second:
        with fits.open(output) as hdul:
            assert len(hdul) == 4


def test_build_detector_forwards_linearity_options(detector_environment):
    calls = []

    def linearity(frameid, **kwargs):
        calls.append((frameid, kwargs))
        return [1, 0, 0]

    detector.build_detector(
        204, linid=55, apred="test", telescope="lco25m", unlock=True,
        clobber=True, verbose=True, linearity_function=linearity)
    assert calls == [(55, {
        "apred": "test", "telescope": "lco25m", "unlock": True,
        "clobber": True, "verbose": True,
    })]


def test_build_detector_clears_lock_when_linearity_fails(detector_environment):
    _, lock_calls = detector_environment

    def fail(*args, **kwargs):
        raise RuntimeError("fit failed")

    with pytest.raises(RuntimeError, match="fit failed"):
        detector.build_detector(205, linid=10, linearity_function=fail)
    assert lock_calls[-1][1] == {"clear": True}
