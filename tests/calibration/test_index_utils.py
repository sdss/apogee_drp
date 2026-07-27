"""Tests for calibration-index and shared numerical helpers."""

import numpy as np
import pytest

from apogee_drp.apred.cal.index import readcalstr
from apogee_drp.apred.cal.utils import flatsmooth, robust_slope


def _records(rows):
    return np.array(
        rows, dtype=[("mjd1", "i4"), ("mjd2", "i4"), ("name", "U20")]
    )


def test_readcalstr_empty_returns_zero():
    assert readcalstr(None, 60000) == 0
    assert readcalstr(_records([]), 60000) == 0


def test_readcalstr_inclusive_date_boundaries():
    records = _records([(59000, 60000, "123")])
    assert readcalstr(records, 59000) == 123
    assert readcalstr(records, 60000) == 123
    assert readcalstr(records, 60001) == 0


def test_readcalstr_last_overlap_wins(capsys):
    records = _records([(59000, 60000, "1"), (59500, 60500, "2")])
    assert readcalstr(records, 59700) == 2
    assert "will use last" in capsys.readouterr().out


def test_readcalstr_silent_overlap(capsys):
    records = _records([(59000, 60000, "1"), (59500, 60500, "2")])
    assert readcalstr(records, 59700, verbose=False) == 2
    assert capsys.readouterr().out == ""


def test_readcalstr_preserves_non_numeric_names():
    records = _records([(59000, 60000, "daily-60000")])
    assert readcalstr(records, 59500) == "daily-60000"


def test_readcalstr_rejects_missing_fields():
    records = np.array([(1,)], dtype=[("mjd1", "i4")])
    with pytest.raises(ValueError, match="mjd1, mjd2, and name"):
        readcalstr(records, 1)


def test_robust_slope_exact_line():
    x = np.arange(100.0)
    assert np.isclose(robust_slope(x, 4.2 * x - 19), 4.2)


def test_robust_slope_handles_unsorted_multidimensional_input():
    x = np.array([[4, 1, 3], [0, 5, 2]], dtype=float)
    y = -2 * x + 7
    assert np.isclose(robust_slope(x, y), -2)


def test_robust_slope_ignores_nonfinite_pairs():
    x = np.arange(10.0)
    y = 3 * x
    x[2], y[7] = np.nan, np.inf
    assert np.isclose(robust_slope(x, y), 3)


def test_robust_slope_resists_large_outlier():
    x = np.arange(100.0)
    y = 1.5 * x + 4
    y[[25, 50, 75]] += [1e6, -1e6, 1e6]
    # Preserve the exact IDL quartile estimator rather than silently replacing
    # it with a different robust regression algorithm.
    assert np.isclose(robust_slope(x, y), 1.5, rtol=0.01)


def test_robust_slope_rejects_different_lengths():
    with pytest.raises(ValueError, match="same number"):
        robust_slope(np.arange(5), np.arange(6))


def test_robust_slope_rejects_too_few_points():
    with pytest.raises(ValueError, match="four"):
        robust_slope([1, 2, 3], [2, 4, 6])


def test_flatsmooth_recovers_plane_to_numerical_precision():
    yy, xx = np.indices((128, 160))
    plane = 0.7 + 1e-3 * xx - 7e-4 * yy
    result = flatsmooth(plane, xstep=16, ystep=16, xbin=31, ybin=31)
    np.testing.assert_allclose(result, plane, atol=2e-6)


def test_flatsmooth_rejects_outliers():
    yy, xx = np.indices((128, 128))
    plane = 0.9 + 2e-4 * xx + 3e-4 * yy
    corrupted = plane.copy()
    corrupted[30:35, 50:55] = 100
    result = flatsmooth(corrupted, xstep=16, ystep=16, xbin=31, ybin=31)
    assert np.nanmedian(np.abs(result - plane)) < 1e-4


def test_flatsmooth_ignores_values_outside_valid_range():
    yy, xx = np.indices((96, 96))
    plane = 0.8 + 1e-3 * xx
    image = plane.copy()
    image[:10] = 0
    image[-10:] = 2
    result = flatsmooth(
        image, xstep=16, ystep=16, xbin=31, ybin=31, lobad=0.1, hibad=1.5
    )
    np.testing.assert_allclose(result[20:-20], plane[20:-20], atol=2e-4)


def test_flatsmooth_rejects_non_image():
    with pytest.raises(ValueError, match="two-dimensional"):
        flatsmooth(np.ones(10))


@pytest.mark.parametrize(
    "kwargs",
    [{"xstep": 0}, {"ystep": 0}, {"xbin": 0}, {"ybin": -1}],
)
def test_flatsmooth_rejects_invalid_window_parameters(kwargs):
    with pytest.raises(ValueError, match="positive"):
        flatsmooth(np.ones((32, 32)), **kwargs)


def test_flatsmooth_rejects_insufficient_valid_regions():
    with pytest.raises(ValueError, match="not enough"):
        flatsmooth(np.zeros((32, 32)), xstep=16, ystep=16)
