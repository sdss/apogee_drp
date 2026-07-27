import numpy as np

from apogee_drp.apred.visit.flux import (
    BADERR,
    H_ZEROPOINT_FLAMBDA,
    flux_calibrate,
)


def _frame(npix=240):
    wavelength_ranges = ((15100, 15800), (15850, 16400), (16450, 16950))
    frame = {}
    for letter, limits in zip("abc", wavelength_ranges):
        wavelength = np.broadcast_to(
            np.linspace(*limits, npix), (300, npix)
        ).copy()
        flux = np.full((300, npix), 1000.0, dtype=np.float32)
        frame[f"chip{letter}"] = {
            "header": {},
            "flux": flux,
            "err": np.full_like(flux, 10.0),
            "mask": np.zeros_like(flux, dtype=np.int16),
            "wavelength": wavelength,
            "sky": np.full_like(flux, 100.0),
            "skyerr": np.full_like(flux, 2.0),
        }
    return frame


def _plugmap(fiberid, objtype, hmag):
    n = len(fiberid)
    mag = np.zeros((n, 3), dtype=float)
    mag[:, 1] = hmag
    return {
        "fiberdata": {
            "spectrographid": np.full(n, 2),
            "holetype": np.full(n, "OBJECT"),
            "objtype": np.asarray(objtype),
            "fiberid": np.asarray(fiberid),
            "mag": mag,
        }
    }


def test_absolute_normalization_and_sky_use_original_chip_b_counts():
    frame = _frame()
    # Physical fibers 1, 2, and 3 map to rows 299, 298, and 297.
    for chip in frame.values():
        chip["flux"][299] = 1100
        chip["flux"][298] = 2100
        chip["flux"][297] = 100
        chip["err"][299, 10] = BADERR
    plugmap = _plugmap([1, 2, 3], ["STAR", "STAR", "SKY"], [10, 0, 0])

    output = flux_calibrate(frame, plugmap)
    expected = 10 ** (-0.4 * 10) * H_ZEROPOINT_FLAMBDA / 1000
    assert np.isclose(output["fluxcorr"][299], expected)
    # Missing H magnitude and sky fibers receive the median valid norm.
    assert np.isclose(output["fluxcorr"][298], expected)
    assert np.isclose(output["fluxcorr"][297], expected)
    assert np.isclose(output["chipb"]["flux"][299, 0], 1100 * expected)
    assert output["chipa"]["err"][299, 10] == BADERR
    assert np.isclose(output["chipc"]["sky"][297, 0], 100 * expected)
    assert frame["chipb"]["flux"][299, 0] == 1100


def test_relative_response_recovers_fifth_order_model():
    frame = _frame(npix=260)
    coefficients = np.array([2e-17, -1e-14, 3e-11, -2e-8, 4e-5])
    offsets = (4.0, 4.25)
    fibers = (1, 2)
    for chip in frame.values():
        wavelength = chip["wavelength"]
        x = wavelength - 16000.0
        log_response = sum(
            coefficients[index] * x ** (5 - index) for index in range(5)
        )
        for fiber, offset in zip(fibers, offsets):
            chip["flux"][300 - fiber] = 10 ** (log_response[300 - fiber] + offset)
    plugmap = _plugmap(fibers, ["HOT_STD", "HOT_STD"], [0, 0])

    output = flux_calibrate(frame, plugmap)
    fitted = np.array(
        [output["chipa"]["header"][f"FLXPAR{i}"] for i in range(1, 6)]
    )
    np.testing.assert_allclose(fitted, coefficients, rtol=2e-3, atol=2e-20)
    assert output["chipa"]["header"]["FLUXNORM"] == 1.0
    assert len(output["chipa"]["header"]["HISTORY"]) == 5


def test_hot_standard_with_500_bad_pixels_is_excluded():
    frame = _frame(npix=600)
    frame["chipb"]["mask"][299, :500] = 1
    plugmap = _plugmap([1], ["HOT_STD"], [0])
    output = flux_calibrate(frame, plugmap)
    assert "FLXPAR1" not in output["chipa"]["header"]
    assert output["fluxcorr"][299] == 1


def test_missing_required_field_is_rejected():
    frame = _frame()
    del frame["chipc"]["skyerr"]
    try:
        flux_calibrate(frame, _plugmap([1], ["STAR"], [10]))
    except ValueError as error:
        assert "skyerr" in str(error)
    else:
        raise AssertionError("missing skyerr was accepted")
