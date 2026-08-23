import numpy as np

from apogee_drp.utils.numerics import median_absolute_deviation, robust_polyfit


def test_median_absolute_deviation_ignores_nonfinite_values():
    assert median_absolute_deviation([1, 2, 3, np.nan, np.inf]) == 1
    assert np.isnan(median_absolute_deviation([np.nan, np.inf]))


def test_robust_polyfit_rejects_an_outlier():
    x = np.arange(10.0)
    y = 2 + 3 * x
    y[5] = 1000
    np.testing.assert_allclose(robust_polyfit(x, y, 1), [2, 3], atol=1e-10)
