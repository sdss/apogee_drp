"""Unit tests for the numerical mkdark implementation."""

import numpy as np
import pytest

from apogee_drp.apred.cal import mkdark


def make_ramps(nframe=3, nread=8, ny=16, nx=20, rate=2.0):
    reads = np.arange(nread, dtype=float)
    ramps = rate * reads[None, :, None, None]
    return np.broadcast_to(ramps, (nframe, nread, ny, nx)).copy()


def test_apvariance_matches_idl_formula():
    data = np.array([-5.0, 0.0, 19.0])
    expected = np.maximum(data / 1.9, 0) + (18.0 / 1.9) ** 2
    np.testing.assert_allclose(mkdark.apvariance(data), expected)


def test_make_dark_recovers_exact_ramp():
    ramps = make_ramps()
    dark, chi2, mask, rate, stats = mkdark.make_dark(ramps, row_block=5)
    np.testing.assert_allclose(dark, ramps[0])
    np.testing.assert_allclose(chi2, 0)
    np.testing.assert_allclose(rate, 2.0)
    assert not np.any(mask)
    assert stats["nframes"] == 3
    assert stats["nreads"] == 8


def test_make_dark_uses_frame_median():
    ramps = make_ramps()
    ramps[0] += 1000
    dark, _, _, _, _ = mkdark.make_dark(ramps)
    np.testing.assert_allclose(dark, ramps[1])


def test_make_dark_flags_hot_pixel_and_neighbor():
    ramps = make_ramps(rate=1.0)
    reads = np.arange(ramps.shape[1], dtype=float)
    ramps[:, :, 8, 8] = 20.0 * reads[None, :]
    ramps[:, :, 8, 9] = 3.0 * reads[None, :]
    _, _, mask, rate, stats = mkdark.make_dark(ramps)
    assert rate[8, 8] > 10
    assert mask[8, 8] & 2
    assert mask[8, 9] & 4
    assert stats["nhot"] == 1
    assert stats["nhotneigh"] == 1


def test_make_dark_flags_nonfinite_rate():
    ramps = make_ramps()
    ramps[:, -1, 4, 5] = np.nan
    _, _, mask, _, stats = mkdark.make_dark(ramps)
    assert mask[4, 5] & 1
    assert stats["nsat"] == 1


def test_make_dark_does_not_modify_input():
    ramps = make_ramps()
    original = ramps.copy()
    mkdark.make_dark(ramps)
    np.testing.assert_array_equal(ramps, original)


@pytest.mark.parametrize("shape", [(3, 8, 10), (0, 8, 10, 10),
                                    (2, 2, 10, 10)])
def test_make_dark_rejects_invalid_ramps(shape):
    with pytest.raises(ValueError):
        mkdark.make_dark(np.zeros(shape))


def test_mkdark_validates_empty_input_before_io():
    with pytest.raises(ValueError, match="at least one"):
        mkdark.mkdark([])


def test_mkdark_rejects_unported_psf_branch_before_io():
    with pytest.raises(NotImplementedError, match="psfid"):
        mkdark.mkdark([12345678], psfid=12345679)
