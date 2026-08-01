"""Focused numerical tests for mkflat."""

import numpy as np
import pytest

from apogee_drp.apred.cal import mkflat


def test_make_flat_chip_does_not_modify_input():
    flat = np.ones((64, 64), dtype=float)
    flatmask = np.zeros_like(flat, dtype=np.uint32)
    original = flat.copy()
    mkflat.make_flat_chip(flat, flatmask, bad_pixel_bits=0)
    np.testing.assert_array_equal(flat, original)


def test_make_flat_chip_marks_reduction_and_zero_pixels():
    flat = np.ones((64, 64), dtype=float)
    flatmask = np.zeros_like(flat, dtype=np.uint32)
    flatmask[10, 10] = 4
    flat[20, 20] = 0

    output, spectral, mask = mkflat.make_flat_chip(
        flat,
        flatmask,
        bad_pixel_bits=4,
    )

    assert output[10, 10] == 0
    assert mask[10, 10] & 1
    assert output[20, 20] == 0
    assert mask[20, 20] & 8
    assert output.shape == spectral.shape == mask.shape


def test_make_flat_chip_flags_outlier_and_neighbor():
    flat = np.ones((64, 64), dtype=float)
    flat[30, 30] = 2.0
    flat[30, 31] = 1.10
    flatmask = np.zeros_like(flat, dtype=np.uint32)

    _, _, mask = mkflat.make_flat_chip(
        flat,
        flatmask,
        bad_pixel_bits=0,
    )

    assert mask[30, 30] & 2
    assert mask[30, 31] & 4


def test_make_flat_chip_validates_shapes():
    with pytest.raises(ValueError, match="identical"):
        mkflat.make_flat_chip(
            np.ones((10, 10)),
            np.zeros((9, 10)),
            bad_pixel_bits=0,
        )


def test_non_dithered_spectral_flat_is_quadratic_model():
    x = np.arange(64, dtype=float)
    profile = 1.0 + 1e-3 * x + 2e-5 * x**2
    flat = np.broadcast_to(profile[:, None], (64, 64)).copy()

    output, spectral, mask = mkflat.make_flat_chip(
        flat,
        np.zeros_like(flat, dtype=np.uint32),
        bad_pixel_bits=0,
    )

    np.testing.assert_allclose(output, flat)
    np.testing.assert_allclose(spectral[:, 0], profile, rtol=1e-10)
    assert not np.any(mask)


def test_nrep_validation_happens_before_pipeline_io():
    with pytest.raises(ValueError, match="nrep"):
        mkflat.mkflat([12345678], nrep=0)

