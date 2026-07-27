import numpy as np

from apogee_drp.apred.visit.combine import (
    DitherPair,
    combine_spectra,
    convert_lsf_to_half_pixel,
    convert_wcoef_to_half_pixel,
    dither_combine,
    interlace_frame_pair,
    interlace_pair,
    sinc_interlaced,
)


def test_sinc_shortcuts_return_original_samples_and_variance():
    a = np.arange(6, dtype=np.float32)
    b = a + 10
    e1 = np.full(6, 2, dtype=np.float32)
    e2 = np.full(6, 3, dtype=np.float32)
    out, var = sinc_interlaced(a, b, 0.5, 0.5, err1=e1, err2=e2)
    np.testing.assert_array_equal(out, a)
    np.testing.assert_array_equal(var, e1**2)
    out, var = sinc_interlaced(a, b, 0.5, 0.0, err1=e1, err2=e2)
    np.testing.assert_array_equal(out, b)
    np.testing.assert_array_equal(var, e2**2)


def test_half_pixel_pair_interlaces_without_interpolation():
    left = np.array([1, 2, 3, 4], dtype=np.float32)
    right = np.array([11, 12, 13, 14], dtype=np.float32)
    err = np.ones(4, dtype=np.float32)
    flux, error = interlace_pair(
        left, right, err, err, shift=0.5, absolute_shift=0
    )
    np.testing.assert_array_equal(flux[0::2], left)
    np.testing.assert_array_equal(flux[1::2], right)
    np.testing.assert_array_equal(error, 1)


def test_combine_spectra_sums_equal_exposures():
    flux = np.array([[10, 20], [14, 24]], dtype=np.float32)
    error = np.full_like(flux, 2)
    mask = np.zeros_like(flux, dtype=np.int16)
    result = combine_spectra(flux, error, mask)
    # Weighted mean 12/22, multiplied by sum(scales)=2.
    np.testing.assert_allclose(result.flux, [24, 44])
    np.testing.assert_allclose(result.error, 2 * np.sqrt(2))


def test_combine_masks_bad_samples_when_requested():
    flux = np.array([[10.0], [1000.0]])
    error = np.ones_like(flux)
    mask = np.array([[0], [1]], dtype=np.int16)
    result = combine_spectra(flux, error, mask, bad_mask=1)
    # One accepted normalized exposure, rescaled by both continua.
    np.testing.assert_allclose(result.flux, [20])
    assert result.mask[0] == 1


def test_wcoef_half_pixel_conversion():
    coef = np.ones((1, 10), dtype=np.float64)
    out = convert_wcoef_to_half_pixel(coef)
    np.testing.assert_array_equal(out[0, [0, 2, 3, 5]], 2)
    np.testing.assert_allclose(out[0, 6:], [1, 0.5, 0.25, 0.125])


def test_lsf_core_half_pixel_conversion():
    # binsize, xoffset, horder=0, porder(sigma)=1, then two sigma coefs.
    coef = np.array([[1, 3, 0, 1, 4, 2]], dtype=np.float64)
    out = convert_lsf_to_half_pixel(coef)
    np.testing.assert_allclose(out, [[2, 6, 0, 1, 8, 2]])


def _synthetic_frame(offset, value):
    frame = {"shift": {"chipfit": np.array([0, offset, offset, offset])}}
    for chip in "abc":
        shape = (1, 4)
        wcoef = np.zeros((1, 10), dtype=np.float64)
        wcoef[:, 3] = 1
        wcoef[:, 6] = 16000
        wcoef[:, 7] = 100
        frame["chip" + chip] = {
            "header": {"EXPTIME": 500.0},
            "filename": f"frame-{value}.fits",
            "flux": np.full(shape, value, dtype=np.float32),
            "err": np.ones(shape, dtype=np.float32),
            "mask": np.zeros(shape, dtype=np.int16),
            "wavelength": np.tile(np.arange(4), (1, 1)).astype(float),
            "sky": np.full(shape, 2, dtype=np.float32),
            "skyerr": np.full(shape, 0.5, dtype=np.float32),
            "telluric": np.full(shape, 0.9, dtype=np.float32),
            "telluricerr": np.full(shape, 0.01, dtype=np.float32),
            "wcoef": wcoef,
            "lsfcoef": np.array([[2, 0, 0, 0, 1]], dtype=float),
        }
    return frame


def test_interlace_frame_pair_complete_fields():
    frames = [_synthetic_frame(0.0, 10), _synthetic_frame(-0.5, 20)]
    pair = DitherPair(
        framename=np.array(["1", "2"]),
        framenum=np.array([1, 2]),
        oldshift=np.array([0, 0.5], dtype=np.float32),
        shift=np.array([0.5, 0], dtype=np.float32),
        sn=np.array([10, 10], dtype=np.float32),
        refshift=np.float32(0),
        relshift=np.float32(0.5),
        nused=np.array([1, 1]),
        index=np.array([0, 1]),
    )
    out = interlace_frame_pair(
        frames,
        pair,
        reference_index=0,
        no_scale=True,
        npad=0,
    )
    assert out["chipa"]["flux"].shape == (1, 8)
    np.testing.assert_array_equal(out["chipa"]["flux"][0, 0::2], 10)
    np.testing.assert_array_equal(out["chipa"]["flux"][0, 1::2], 20)
    np.testing.assert_array_equal(out["chipa"]["sky"][0, 0::2], 2)
    assert out["chipa"]["wcoef"][0, 0] == 0


def test_dither_combine_no_dither_sums_frames():
    frames = [_synthetic_frame(0.0, 10), _synthetic_frame(0.0, 14)]
    dtype = [("index", "i4"), ("framenum", "U8"), ("shift", "f4"), ("sn", "f4")]
    shifts = np.zeros(2, dtype=dtype)
    shifts["index"] = [0, 1]
    shifts["framenum"] = ["1", "2"]
    shifts["sn"] = 10
    out, pairs = dither_combine(
        frames, shifts, no_dither=True, no_scale=True
    )
    assert pairs is None
    np.testing.assert_allclose(out["chipa"]["flux"], 24)
    np.testing.assert_allclose(out["chipa"]["err"], np.sqrt(2))
    np.testing.assert_allclose(out["chipa"]["sky"], 4)
    np.testing.assert_allclose(out["chipa"]["telluric"], 0.9)
