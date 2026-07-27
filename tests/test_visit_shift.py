import numpy as np

from apogee_drp.apred.visit.shift import LinePeak, dither_shift


def _chip(flux, peaks=None):
    return {
        "header": {},
        "flux": np.asarray(flux, dtype=np.float32),
        "err": np.ones_like(flux, dtype=np.float32),
        "mask": np.zeros_like(flux, dtype=np.int32),
        "peaks": peaks,
    }


def _frame(fluxes, peaks=None):
    if peaks is None:
        peaks = [None, None, None]
    return {
        name: _chip(flux, chip_peaks)
        for name, flux, chip_peaks in zip(
            ("chipa", "chipb", "chipc"), fluxes, peaks
        )
    }


def _shifted_spectra(nfiber=16, npix=320, offsets=(0.42, 0.42, 0.42), slope=0):
    rng = np.random.default_rng(2718)
    x = np.arange(npix, dtype=float)
    first_chips, second_chips = [], []
    for chip_index in range(3):
        first = np.empty((nfiber, npix))
        second = np.empty_like(first)
        for fiber in range(nfiber):
            spectrum = np.zeros(npix)
            for center, height, width in zip(
                rng.uniform(35, npix - 35, 18),
                rng.uniform(0.5, 2.0, 18),
                rng.uniform(0.7, 2.3, 18),
            ):
                spectrum += height * np.exp(-0.5 * ((x - center) / width) ** 2)
            spectrum += rng.normal(0, 0.003, npix)
            shift = offsets[chip_index] + slope * fiber
            first[fiber] = spectrum
            second[fiber] = np.interp(x - shift, x, spectrum, left=0, right=0)
        first_chips.append(first)
        second_chips.append(second)
    return _frame(first_chips), _frame(second_chips)


def test_xcorr_recovers_positive_subpixel_shift():
    frame1, frame2 = _shifted_spectra()
    result = dither_shift(frame1, frame2, xcorr=True, return_shiftarr=True)
    assert result.type == "xcorr"
    assert abs(result.shiftfit[0] - 0.42) < 0.025
    assert abs(result.shiftfit[1]) < 0.002
    assert result.shiftarr.shape == (16, 3)
    np.testing.assert_allclose(result.chipfit[1:], 0.42, atol=0.03)


def test_xcorr_fits_common_slope_and_chip_offsets():
    offsets = (-0.25, 0.10, 0.48)
    frame1, frame2 = _shifted_spectra(offsets=offsets, slope=0.003)
    result = dither_shift(frame1, frame2, xcorr=True)
    np.testing.assert_allclose(result.chipfit[0], 0.003, atol=0.001)
    np.testing.assert_allclose(result.chipfit[1:], offsets, atol=0.035)
    for chip_index, offset in enumerate(offsets):
        np.testing.assert_allclose(
            result.chipshift[chip_index], [offset, 0.003], atol=0.035
        )


def test_xcorr_nofit_returns_per_chip_means():
    offsets = (-0.20, 0.05, 0.35)
    frame1, frame2 = _shifted_spectra(offsets=offsets)
    result = dither_shift(frame1, frame2, xcorr=True, nofit=True)
    assert result.shiftfit[1] == 0
    assert result.chipfit[0] == 0
    np.testing.assert_allclose(result.chipfit[1:], offsets, atol=0.03)


def test_line_mode_matches_peaks_and_rejects_large_centroid_errors():
    first = [
        [LinePeak(0, 100.1, 0.1, 0.2), LinePeak(1, 120.2, 3.0, 0.2)],
        [LinePeak(0, 90.3, 0.2, 0.2)],
        [LinePeak(201, 80.4, 0.2, 0.2)],
    ]
    second = [
        [LinePeak(0, 100.47, 0.1, 0.2), LinePeak(1, 120.57, 0.1, 0.2)],
        [LinePeak(0, 90.67, 0.2, 0.2)],
        [LinePeak(201, 80.77, 0.2, 0.2)],
    ]
    zeros = [np.zeros((2, 64)) for _ in range(3)]
    frame1 = _frame(zeros, first)
    frame2 = _frame(zeros, second)
    result = dither_shift(
        frame1, frame2, peak_finder=lambda chip: chip["peaks"]
    )
    np.testing.assert_allclose(result.shiftfit, [0.37, 0], atol=1e-6)
    assert result.shifterr < 1e-6


def test_invalid_frame_is_rejected():
    frame1, frame2 = _shifted_spectra()
    del frame2["chipc"]["mask"]
    try:
        dither_shift(frame1, frame2, xcorr=True)
    except ValueError as error:
        assert "mask" in str(error)
    else:
        raise AssertionError("missing required field was accepted")
