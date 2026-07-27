"""Numerical tests for detector calibration utilities."""

import numpy as np
import pytest

from apogee_drp.apred.cal.detector import (
    aplincorr,
    fowler_sample,
    getrn,
    noise,
)
from apogee_drp.apred.cal.getrn import _robust_sigma


def test_aplincorr_rejects_wrong_detector_width():
    with pytest.raises(ValueError, match="2048"):
        aplincorr(np.ones((2047, 4)), np.ones((4, 2)))


def test_aplincorr_rejects_too_few_reads():
    with pytest.raises(ValueError, match="three reads"):
        aplincorr(np.ones((2048, 2)), np.ones((4, 2)))


@pytest.mark.parametrize("shape", [(3, 2), (4,), (4, 2, 1)])
def test_aplincorr_rejects_bad_coefficient_shape(shape):
    with pytest.raises(ValueError, match="lindata"):
        aplincorr(np.ones((2048, 4)), np.ones(shape))


def test_aplincorr_constant_polynomial_all_reads():
    data = np.arange(2048 * 5, dtype=float).reshape(2048, 5)
    coefficients = np.zeros((4, 3))
    coefficients[:, 0] = 2.0
    np.testing.assert_array_equal(aplincorr(data, coefficients), data / 2.0)


def test_aplincorr_quadrants_use_independent_coefficients():
    data = np.ones((2048, 4), dtype=float) * 12
    coefficients = np.zeros((4, 1))
    coefficients[:, 0] = [1, 2, 3, 4]
    corrected = aplincorr(data, coefficients)
    for quadrant, divisor in enumerate([1, 2, 3, 4]):
        np.testing.assert_allclose(
            corrected[quadrant * 512 : (quadrant + 1) * 512],
            data[quadrant * 512 : (quadrant + 1) * 512] / divisor,
        )


def test_aplincorr_first_two_corrections_equal_third():
    data = np.tile(np.array([100.0, 110.0, 120.0, 140.0]), (2048, 1))
    coefficients = np.tile(np.array([1.0, 1e-3]), (4, 1))
    corrected = aplincorr(data, coefficients)
    correction = data / corrected
    np.testing.assert_allclose(correction[:, 0], correction[:, 2])
    np.testing.assert_allclose(correction[:, 1], correction[:, 2])


def test_aplincorr_nonfinite_level_uses_zero():
    data = np.ones((2048, 4), dtype=float)
    data[0, 3] = np.nan
    coefficients = np.tile(np.array([2.0, 0.5]), (4, 1))
    corrected = aplincorr(data, coefficients)
    assert np.isnan(corrected[0, 3])
    assert np.isclose(data[1, 3] / corrected[1, 3], 2.0)


def test_aplincorr_legacy_changes_only_first_read():
    data = np.ones((2048, 5), dtype=float) * 10
    coefficients = np.zeros((4, 1))
    coefficients[:, 0] = 2
    corrected = aplincorr(data, coefficients, legacy=True)
    np.testing.assert_allclose(corrected[:, 0], 5)
    np.testing.assert_array_equal(corrected[:, 1:], data[:, 1:])


def test_fowler_sample_linear_ramp_utr():
    ramp = np.arange(8.0)[None, None, :] * 3.25 + 11
    np.testing.assert_allclose(fowler_sample(ramp, 0), [[3.25 * 7]])


@pytest.mark.parametrize(
    ("nfowler", "expected"),
    [(1, 5.0), (2, 4.0), (3, 3.0)],
)
def test_fowler_sample_linear_ramp(nfowler, expected):
    ramp = np.arange(6.0)[None, None, :]
    np.testing.assert_allclose(fowler_sample(ramp, nfowler), [[expected]])


@pytest.mark.parametrize("nfowler", [-1, 4, 99])
def test_fowler_sample_rejects_invalid_count(nfowler):
    with pytest.raises(ValueError):
        fowler_sample(np.ones((2, 3, 6)), nfowler)


def test_fowler_sample_rejects_non_cube():
    with pytest.raises(ValueError, match="cube"):
        fowler_sample(np.ones((2, 3)), 1)


def test_robust_sigma_ignores_nan_and_outlier():
    values = np.r_[np.arange(-10, 11), np.nan, 1e9]
    assert np.isfinite(_robust_sigma(values))
    assert _robust_sigma(values) < 20


def test_robust_sigma_empty_is_nan():
    assert np.isnan(_robust_sigma([np.nan, np.inf]))


def _three_chip_cubes(seed=12, shape=(32, 2048, 12)):
    rng = np.random.default_rng(seed)
    cubes1, cubes2 = [], []
    ramp = np.arange(shape[-1])[None, None, :] * 100.0
    for _ in range(3):
        cubes1.append(ramp + rng.normal(0, 3, shape))
        cubes2.append(ramp + rng.normal(0, 3, shape))
    return cubes1, cubes2


def test_getrn_output_schema_and_sampling_metadata():
    first, second = _three_chip_cubes()
    result = getrn(first, second)
    assert result.shape == (3,)
    assert result.dtype.names == (
        "n", "m", "rn1", "rn1corr", "rn2", "rn2corr", "rn3", "rn4"
    )
    np.testing.assert_array_equal(result["m"][0], [1, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(result["n"][0, 1:], 2)


def test_getrn_finite_values_for_all_chips_and_quadrants():
    first, second = _three_chip_cubes()
    result = getrn(first, second)
    assert np.all(np.isfinite(result["rn2"]))
    assert np.all(result["rn2"] > 0)


def test_getrn_rejects_wrong_chip_count():
    cube = np.zeros((32, 2048, 12))
    with pytest.raises(ValueError, match="three chips"):
        getrn([cube], [cube])


def test_getrn_rejects_mismatched_cube_shapes():
    first, second = _three_chip_cubes()
    second[1] = second[1][..., :-1]
    with pytest.raises(ValueError, match="identical"):
        getrn(first, second)


def test_noise_returns_requested_bins_and_counts():
    rng = np.random.default_rng(3)
    signal = np.linspace(100, 10000, 20 * 30).reshape(20, 30)
    stack = signal + rng.normal(0, 5, (6, 20, 30))
    result = noise(stack, bins=8)
    assert len(result["edges"]) == 9
    assert result["npix"].sum() <= signal.size
    assert np.count_nonzero(result["npix"]) >= 6


def test_noise_explicit_edges_and_model_variance():
    rng = np.random.default_rng(4)
    stack = 100 + rng.normal(0, 2, (4, 10, 12))
    errors = np.full_like(stack, 2.5)
    result = noise(stack, errors, bins=[1, 50, 150, 1000])
    assert "model_variance" in result
    assert np.isclose(result["model_variance"][1], 6.25)


def test_noise_bad_mask_removes_pixels():
    stack = np.ones((3, 5, 6)) * 100
    stack[:, 2, 3] += [0, 100, -100]
    bad = np.zeros((5, 6), bool)
    bad[2, 3] = True
    result = noise(stack, bad=bad, bins=[1, 200])
    assert result["npix"][0] == 29


@pytest.mark.parametrize("shape", [(4, 5), (1, 4, 5)])
def test_noise_rejects_invalid_stack(shape):
    with pytest.raises(ValueError, match="nimage"):
        noise(np.ones(shape))


def test_noise_rejects_mismatched_error_shape():
    with pytest.raises(ValueError, match="same shape"):
        noise(np.ones((3, 4, 5)), np.ones((2, 4, 5)), bins=[1, 2])

