import numpy as np

from apogee_drp.apred.sky.subtract import BITS, sky_subtract
from apogee_drp.apred.visit.io import BADERR


def _synthetic_frame(npix=320):
    x = np.arange(npix, dtype=float)
    line = (
        1200 * np.exp(-0.5 * ((x - 100) / 3.0) ** 2)
        + 800 * np.exp(-0.5 * ((x - 220) / 4.0) ** 2)
    )
    frame = {"shift": {"shiftfit": np.array([0.5, 0])}}
    for chip_index, letter in enumerate("abc"):
        wavelength = np.broadcast_to(
            np.linspace(15100 + 600 * chip_index, 15600 + 600 * chip_index, npix),
            (300, npix),
        ).copy()
        flux = np.zeros((300, npix), dtype=np.float32)
        error = np.full((300, npix), 5.0, dtype=np.float32)
        mask = np.zeros((300, npix), dtype=np.int16)
        # Fiber 150 is the science target. Fibers 140, 160, and 170 are sky.
        flux[150] = 250 + 1.2 * line
        for fiber_id in (140, 160, 170):
            flux[300 - fiber_id] = 50 + line
            error[300 - fiber_id] = 10
        frame[f"chip{letter}"] = {
            "header": {"EXPTIME": 373},
            "flux": flux,
            "err": error,
            "mask": mask,
            "wavelength": wavelength,
            "lsfcoef": np.zeros((300, 8)),
            "wcoef": np.zeros((300, 14)),
        }
    return frame, line


def _plugmap(include_sky=True):
    fiberid = np.array([150, 140, 160, 170])
    objtype = np.array(["STAR", "SKY", "SKY", "SKY"])
    if not include_sky:
        objtype[1:] = "STAR"
    return {
        "mjd": 60000,
        "fiberdata": {
            "fiberid": fiberid,
            "spectrographid": np.full(4, 2),
            "holetype": np.full(4, "OBJECT"),
            "objtype": objtype,
            "ra": np.array([10.0, 10.01, 9.99, 10.02]),
            "dec": np.array([0.0, 0.01, -0.01, 0.02]),
            "mag": np.full((4, 3), 12.0),
        },
    }


def test_nearest_sky_recovers_continuum_and_error():
    frame, line = _synthetic_frame()
    output, metrics = sky_subtract(
        frame, _plugmap(), return_metrics=True
    )
    target = output["chipa"]
    # Input was stellar continuum 200 + sky continuum 50 + 1.2*sky lines.
    np.testing.assert_allclose(np.median(target["flux"][150]), 200, atol=1)
    np.testing.assert_allclose(target["sky"][150, 100], 50 + 1.2 * line[100], rtol=0.02)
    expected_error = np.sqrt(5**2 + (10 / np.sqrt(3)) ** 2)
    np.testing.assert_allclose(
        np.median(target["err"][150]), expected_error, rtol=0.02
    )
    assert metrics.nskies == 3
    assert target["header"]["NSKIES"] == 3
    # Default operation is non-mutating.
    assert "sky" not in frame["chipa"]


def test_bad_sky_coverage_sets_nosky_and_baderr():
    frame, _ = _synthetic_frame()
    for fiber_id in (140, 160, 170):
        for chip in (frame["chipa"], frame["chipb"], frame["chipc"]):
            chip["mask"][300 - fiber_id, :10] = 1
    output = sky_subtract(frame, _plugmap())
    bit = int(BITS.getval("NOSKY"))
    assert np.all(output["chipa"]["mask"][150, :10] & bit)
    assert np.all(output["chipa"]["err"][150, :10] == BADERR)
    assert np.all(output["chipa"]["skyerr"][150, :10] == BADERR)


def test_significant_skyline_bit_is_set_for_science_target():
    frame, _ = _synthetic_frame()
    output = sky_subtract(frame, _plugmap())
    bit = int(BITS.getval("SIG_SKYLINE"))
    assert np.count_nonzero(output["chipb"]["mask"][150] & bit) > 0
    # The dedicated sky spectra are not tagged SIG_SKYLINE.
    assert np.count_nonzero(output["chipb"]["mask"][160] & bit) == 0


def test_no_sky_and_missing_model_linelist_fail_explicitly():
    frame, _ = _synthetic_frame()
    try:
        sky_subtract(frame, _plugmap(include_sky=False))
    except ValueError as error:
        assert "no sky fibers" in str(error)
    else:
        raise AssertionError("frame without sky fibers was accepted")
    try:
        sky_subtract(frame, _plugmap(), suboption=2)
    except ValueError as error:
        assert "airglow line list" in str(error)
    else:
        raise AssertionError("model mode without a line list was accepted")
