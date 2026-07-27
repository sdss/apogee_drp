"""Focused regression tests for the IDL calibration translations."""

import numpy as np

from apogee_drp.apred.cal.aplincorr import aplincorr
from apogee_drp.apred.cal.flatsmooth import flatsmooth
from apogee_drp.apred.cal.getrn import fowler_sample
from apogee_drp.apred.cal.readcalstr import readcalstr
from apogee_drp.apred.cal.robust_slope import robust_slope


def test_aplincorr_constant_polynomial():
    data = np.arange(2048 * 4, dtype=float).reshape(2048, 4)
    coefficients = np.zeros((4, 2))
    coefficients[:, 0] = 2
    corrected = aplincorr(data, coefficients)
    np.testing.assert_allclose(corrected, data / 2)


def test_aplincorr_legacy_corrects_first_read_only():
    data = np.ones((2048, 4)) * 10
    coefficients = np.zeros((4, 2))
    coefficients[:, 0] = 2
    corrected = aplincorr(data, coefficients, legacy=True)
    np.testing.assert_allclose(corrected[:, 0], 5)
    np.testing.assert_allclose(corrected[:, 1:], 10)


def test_robust_slope_recovers_line_with_outlier():
    x = np.arange(100, dtype=float)
    y = 3.5 * x - 7
    y[50] += 1e5
    assert np.isclose(robust_slope(x, y), 3.5)


def test_readcalstr_uses_last_overlapping_record():
    records = np.array(
        [(59000, 60000, "1"), (59500, 60500, "2")],
        dtype=[("mjd1", "i4"), ("mjd2", "i4"), ("name", "U8")],
    )
    assert readcalstr(records, 59700, verbose=False) == 2
    assert readcalstr(records, 70000, verbose=False) == 0


def test_fowler_sample():
    cube = np.arange(6.0)[None, None, :]
    np.testing.assert_allclose(fowler_sample(cube, 2), [[4.0]])
    np.testing.assert_allclose(fowler_sample(cube, 0), [[5.0]])


def test_flatsmooth_plane():
    yy, xx = np.indices((128, 128))
    plane = 0.8 + xx * 1e-3 + yy * 2e-3
    smooth = flatsmooth(plane, xstep=16, ystep=16, xbin=31, ybin=31)
    np.testing.assert_allclose(smooth, plane, atol=2e-6)
