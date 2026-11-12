import numpy as np
from scipy.special import erf

from apogee_drp.apred.sky.model import (
    AirglowLine,
    MeasuredSkyLine,
    build_airglow_model,
    evaluate_poly2d,
    lsf_gh,
    poly2d_basis,
)
from apogee_drp.apred.sky.subtract import sky_subtract


def _model_frame(npix=320):
    frame = {"shift": {"shiftfit": np.array([0.5, 0])}}
    lines = [
        AirglowLine(1, 15250.0, "OH"),
        AirglowLine(2, 15400.0, "O2"),
    ]
    fiberids = np.array([150, 140, 160, 170])
    rows = 300 - fiberids
    for chip_index, letter in enumerate("abc"):
        lower = 15100 + chip_index * 600
        wavelength = np.broadcast_to(
            np.linspace(lower, lower + 500, npix), (300, npix)
        ).copy()
        flux = np.zeros((300, npix), dtype=np.float32)
        error = np.full((300, npix), 5.0, dtype=np.float32)
        lsfcoef = np.broadcast_to(
            # binsize, Xoffset, Horder, sigma Porder, sigma coefficient
            np.array([1.0, 0.0, 0.0, 0.0, 1.2]), (300, 5)
        ).copy()
        for row_number, fiber_id in zip(rows, fiberids):
            continuum = 200.0 if fiber_id == 150 else 50.0
            flux[row_number] = continuum
            if chip_index == 0:
                pixel = np.arange(npix)
                for line in lines:
                    center = np.interp(line.wave, wavelength[row_number], pixel)
                    flux[row_number] += 1000 * lsf_gh(
                        pixel, center, lsfcoef[row_number]
                    )
        frame[f"chip{letter}"] = {
            "header": {"EXPTIME": 373},
            "flux": flux,
            "err": error,
            "mask": np.zeros((300, npix), dtype=np.int16),
            "wavelength": wavelength,
            "lsfcoef": lsfcoef,
            "wcoef": np.zeros((300, 14)),
        }
    plugmap = {
        "mjd": 60000,
        "fiberdata": {
            "fiberid": fiberids,
            "spectrographid": np.full(4, 2),
            "holetype": np.full(4, "OBJECT"),
            "objtype": np.array(["STAR", "SKY", "SKY", "SKY"]),
            "ra": np.array([10.0, 10.01, 9.99, 10.02]),
            "dec": np.array([0.0, 0.01, -0.01, 0.02]),
            "zeta": np.array([0.0, -0.5, 0.4, 0.8]),
            "eta": np.array([0.0, 0.2, -0.3, 0.5]),
            "mag": np.full((4, 3), 12.0),
        },
    }
    return frame, plugmap, lines


def test_lsf_gh_matches_binned_gaussian():
    x = np.arange(-12, 13, dtype=float)
    sigma = 1.4
    parameters = np.array([1.0, 0.0, 0.0, 0.0, sigma])
    actual = lsf_gh(x, 0.2, parameters)
    w1 = (x - 0.5 - 0.2) / sigma
    w2 = (x + 0.5 - 0.2) / sigma
    expected = 0.5 * (
        erf(w2 / np.sqrt(2)) - erf(w1 / np.sqrt(2))
    )
    np.testing.assert_allclose(actual, expected, atol=2e-15)
    np.testing.assert_allclose(actual.sum(), 1, atol=1e-12)


def test_poly2d_uses_idl_eleven_term_order():
    coefficients = np.arange(1, 12, dtype=float)
    x, y = 0.3, -0.4
    assert evaluate_poly2d(x, y, coefficients) == (
        poly2d_basis(x, y) @ coefficients
    )
    assert poly2d_basis(np.arange(3), 2).shape == (3, 11)


def test_model_mode_fits_and_subtracts_airglow_lines():
    frame, plugmap, lines = _model_frame()
    output, metrics, model = sky_subtract(
        frame,
        plugmap,
        suboption=2,
        airglow=lines,
        return_metrics=True,
    )
    assert {line.id for line in model.lines} == {1, 2}
    np.testing.assert_allclose(model.median_flux[1], 1000, rtol=0.03)
    np.testing.assert_allclose(model.median_flux[2], 1000, rtol=0.03)
    target = output["chipa"]
    np.testing.assert_allclose(np.median(target["flux"][150]), 200, atol=0.5)
    assert np.max(np.abs(target["flux"][150] - 200)) < 5
    assert np.max(target["sky"][150]) > 100
    assert target["header"]["NSKIES"] == 3
    assert "SKPR1_1" in target["header"]
    assert metrics.nskies == 3


def test_model_mode_requires_detected_lines():
    frame, plugmap, _ = _model_frame()
    outside = [AirglowLine(99, 18000, "OH")]
    try:
        sky_subtract(frame, plugmap, suboption=2, airglow=outside)
    except ValueError as error:
        assert "no airglow lines" in str(error)
    else:
        raise AssertionError("undetected line list was accepted")


def test_species_normalization_spatial_model():
    rng = np.random.default_rng(51)
    nsky = 20
    zeta = rng.uniform(-1, 1, nsky)
    eta = rng.uniform(-1, 1, nsky)
    spatial = 1 + 0.08 * zeta - 0.04 * eta + 0.02 * zeta * eta
    lines = [
        AirglowLine(1, 15200, "OH"),
        AirglowLine(2, 15300, "OH"),
        AirglowLine(3, 15900, "O2"),
        AirglowLine(4, 16000, "O2"),
    ]
    measurements = []
    for skyfiber in range(nsky):
        for line in lines:
            base = 1000 + 100 * line.id
            measurements.append(
                MeasuredSkyLine(
                    skyfiber,
                    skyfiber,
                    0,
                    line.id,
                    line.wave,
                    line.species,
                    base * spatial[skyfiber],
                    2.0,
                    line.wave,
                    1,
                )
            )
    model = build_airglow_model(
        measurements, lines, np.arange(nsky), zeta, eta
    )
    for species in ("OH", "O2"):
        prediction = evaluate_poly2d(
            zeta, eta, model.species_coefficients[species]
        )
        # The line/fiber normalization has one arbitrary global scale, so
        # compare each field after matching its median.
        prediction *= np.median(spatial) / np.median(prediction)
        np.testing.assert_allclose(prediction, spatial, atol=2e-4)
