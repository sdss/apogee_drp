import numpy as np
from astropy.io import fits

from apogee_drp.apred.sky.telluric import (
    apply_telluric_correction,
    fit_spatial_scales,
    fit_telluric_star,
    load_preconvolved_telluric,
    normalize_preconvolved_models,
    select_telluric_standards,
    telluric_correct_frame,
    telluric_transmission,
)


def _models(npix=600):
    x = np.arange(npix)
    centers = (120, 300, 470)
    widths = (10, 18, 8)
    depth = (0.25, 0.18, 0.35)
    return np.stack(
        [
            1 - d * np.exp(-0.5 * ((x - c) / w) ** 2)
            for c, w, d in zip(centers, widths, depth)
        ],
        axis=1,
    )


def test_transmission_is_multiplicative():
    models = np.array([[0.8, 0.9, 0.7]])
    got = telluric_transmission(models, [2.0, 0.5, 1.0])
    expected = (1 + 2 * -0.2) * (1 + 0.5 * -0.1) * (1 - 0.3)
    np.testing.assert_allclose(got, expected)


def test_star_fit_recovers_species_scales():
    models = _models()
    expected = np.array([0.8, 1.3, 0.55])
    continuum = 800 + 0.2 * np.arange(models.shape[0])
    flux = continuum * telluric_transmission(models, expected)
    result = fit_telluric_star(flux, np.full(flux.size, 2.0), models)
    assert result.status == 1
    np.testing.assert_allclose(result.scale, expected, atol=0.05)


def test_spatial_fit_and_correction():
    rng = np.random.default_rng(12)
    zeta = rng.uniform(-1.3, 1.3, 30)
    eta = rng.uniform(-1.3, 1.3, 30)
    scales = np.column_stack(
        (
            1.1 + 0.1 * zeta - 0.05 * eta,
            0.9 - 0.04 * zeta + 0.08 * eta,
            1.0 + 0.1 * zeta + 0.02 * zeta**2 - 0.03 * zeta * eta
            + 0.04 * eta,
        )
    )
    fits = fit_spatial_scales(zeta, eta, scales)
    assert [len(f.coefficients) for f in fits] == [3, 3, 6]
    models = _models(100)
    true_scale = np.array([1.1, 0.9, 1.0])
    intrinsic = np.linspace(500, 600, 100)
    observed = intrinsic * telluric_transmission(models, true_scale)
    result = apply_telluric_correction(
        observed, np.ones(100), models, 0.0, 0.0, fits
    )
    np.testing.assert_allclose(result.scale, true_scale, atol=1e-10)
    np.testing.assert_allclose(result.flux, intrinsic, rtol=1e-12)
    assert np.all(result.error >= 0)


def test_preconvolved_layout_and_standard_selection():
    raw = np.ones((60, 7, 3, 2))
    normalized = normalize_preconvolved_models(raw, nfibers=7, npix=20)
    assert normalized.shape == (7, 60, 3, 2)
    plugmap = {
        "fiberdata": {
            "fiberid": np.array([300, 299, 298]),
            "spectrographid": np.array([2, 2, 2]),
            "holetype": np.array(["OBJECT", "OBJECT", "OBJECT"]),
            "objtype": np.array(["HOT_STD", "STAR", "SKY"]),
        }
    }
    rows, detectors = select_telluric_standards(plugmap)
    np.testing.assert_array_equal(rows, [0])
    np.testing.assert_array_equal(detectors, [0])
    rows, detectors = select_telluric_standards(plugmap, starfitopt=2)
    np.testing.assert_array_equal(rows, [0, 1])
    np.testing.assert_array_equal(detectors, [0, 1])


def test_full_frame_preconvolved_path():
    nfiber, npix = 14, 600
    one_chip_models = _models(npix)
    models = np.tile(one_chip_models, (nfiber, 3, 1, 1))
    zeta = np.linspace(-1, 1, nfiber)
    eta = np.linspace(-0.8, 0.8, nfiber) ** 2
    scales = np.column_stack(
        (
            0.9 + 0.08 * zeta,
            1.1 - 0.05 * eta,
            0.8 + 0.04 * zeta + 0.03 * eta,
        )
    )
    frame = {"shift": {}}
    for chip_index, letter in enumerate("abc"):
        transmission = np.stack(
            [
                telluric_transmission(models[fiber, chip_index], scales[fiber])
                for fiber in range(nfiber)
            ]
        )
        intrinsic = np.tile(
            600 + np.linspace(0, 30, npix), (nfiber, 1)
        )
        frame[f"chip{letter}"] = {
            "header": {},
            "flux": intrinsic * transmission,
            "err": np.full((nfiber, npix), 2.0),
            "mask": np.zeros((nfiber, npix), np.int32),
            "sky": np.zeros((nfiber, npix)),
        }
    plugmap = {
        "mjd": 59000,
        "fiberdata": {
            "fiberid": 300 - np.arange(nfiber),
            "spectrographid": np.full(nfiber, 2),
            "holetype": np.full(nfiber, "OBJECT"),
            "objtype": np.array(["HOT_STD"] * 12 + ["STAR"] * 2),
            "zeta": zeta,
            "eta": eta,
            "mag": np.tile([12, 11, 10, 9, 8], (nfiber, 1)),
        },
    }
    result = telluric_correct_frame(frame, plugmap, models)
    assert len(result.star_fits) == 12
    assert result.tellstar["nfit"].min() >= 4
    assert result.tellstar["bestmod"].tolist() == [0, 0, 0]
    for letter in "abc":
        chip = result.frame[f"chip{letter}"]
        assert np.all(chip["telluric"] > 0)
        np.testing.assert_allclose(
            np.median(chip["flux"], axis=1), 615, atol=4
        )


def test_preconvolved_fits_airmass_and_wavelength_interpolation(tmp_path):
    nfiber, npix, ngrid = 2, 5, 9
    frame = {}
    files = []
    target = np.tile(np.linspace(1.1, 7.1, npix), (nfiber, 1))
    source_wave = np.tile(np.arange(ngrid), (nfiber, 1))
    for chip_index, letter in enumerate("abc"):
        frame[f"chip{letter}"] = {
            "header": {"ALT": 30.0},
            "flux": np.ones((nfiber, npix)),
            "wavelength": target + chip_index * 10,
        }
        grid = source_wave + chip_index * 10
        # FITS layout: [model,species,fiber,wave].
        low = np.empty((1, 3, nfiber, ngrid))
        high = np.empty_like(low)
        for species in range(3):
            low[0, species] = 0.90 + 0.01 * species + 0.001 * grid
            high[0, species] = low[0, species] - 0.10
        primary = fits.PrimaryHDU(grid)
        primary.header["AIR0"] = 1.0
        primary.header["DAIR"] = 1.0
        filename = tmp_path / f"telluric-{letter}.fits"
        fits.HDUList(
            [primary, fits.ImageHDU(low), fits.ImageHDU(high)]
        ).writeto(filename)
        files.append(filename)
    loaded = load_preconvolved_telluric(files, frame)
    assert loaded.shape == (nfiber, 3 * npix, 3, 1)
    # ALT=30 gives airmass=2, hence the second extension exactly.
    expected = 0.80 + 0.001 * target[0]
    np.testing.assert_allclose(loaded[0, :npix, 0, 0], expected)
