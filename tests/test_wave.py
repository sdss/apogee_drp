"""Unit and regression tests for :mod:`apogee_drp.apred.wave`.

The tests in this file deliberately use synthetic data.  They test the
numerical building blocks of the wavelength calibration without requiring
APOGEE raw data, a reduction tree, or database access.
"""

import numpy as np
import pytest

from apogee_drp.apred import wave


def numerical_jacobian(function, pars, relative_step=1e-6):
    """Return a central finite-difference Jacobian."""
    pars = np.asarray(pars, dtype=float)
    value = np.asarray(function(pars), dtype=float)
    jacobian = np.empty((value.size, pars.size), dtype=float)

    for column in range(pars.size):
        step = relative_step * max(1.0, abs(pars[column]))
        upper = pars.copy()
        lower = pars.copy()
        upper[column] += step
        lower[column] -= step
        jacobian[:, column] = (
            np.ravel(function(upper)) - np.ravel(function(lower))
        ) / (2.0 * step)

    return jacobian


def make_synthetic_peak(
        size=101,
        amplitude=500.0,
        center=50.37,
        width=1.15,
        background=20.0,
        read_noise=5.0,
        noise=False,
        seed=12345,
):
    """Construct one synthetic pixel-integrated Gaussian line."""
    x = np.arange(size, dtype=float)
    truth = np.array([amplitude, center, width, background], dtype=float)
    model = wave.gaussbin(x, *truth)
    error = np.sqrt(np.maximum(model, 0.0) + read_noise**2)

    if noise:
        rng = np.random.default_rng(seed)
        spectrum = model + rng.normal(0.0, error)
    else:
        spectrum = model.copy()

    return x, spectrum, error, truth


def assert_peak_close(fitted, truth, center_atol=1e-3, width_atol=1e-3):
    """Check the scientifically important peak parameters."""
    assert np.isfinite(fitted).all()
    assert fitted[0] > 0.0
    assert fitted[1] == pytest.approx(truth[1], abs=center_atol)
    assert fitted[2] == pytest.approx(truth[2], abs=width_atol)


class TestGaussianModel:
    def test_profile_matches_unit_amplitude_gaussbin(self):
        x = np.linspace(10.0, 30.0, 51)
        center = 19.73
        width = 1.24

        profile = wave._gaussbin_profile(x, center, width)
        model = wave.gaussbin(x, 1.0, center, width, 0.0)

        np.testing.assert_allclose(profile, model, rtol=1e-13, atol=1e-13)

    def test_background_is_constant(self):
        x = np.arange(21, dtype=float)
        background = 17.3
        model = wave.gaussbin(x, 0.0, 10.0, 1.1, background)
        np.testing.assert_allclose(model, background)

    def test_multiple_components_add(self):
        x = np.arange(51, dtype=float)
        first = (100.0, 20.2, 0.9, 4.0)
        second = (60.0, 31.7, 1.4, 0.0)

        combined = wave.gaussbin(x, *(first + second))
        separate = wave.gaussbin(x, *first) + wave.gaussbin(x, *second)

        np.testing.assert_allclose(combined, separate, rtol=1e-13, atol=1e-13)

    @pytest.mark.parametrize("width", [0.55, 0.8, 1.2, 2.5, 4.5])
    def test_integrated_flux_on_large_grid(self, width):
        x = np.arange(-100, 101, dtype=float)
        amplitude = 23.0
        profile_sum = np.sum(wave.gaussbin(x, amplitude, 0.37, width, 0.0))
        expected = amplitude * np.sqrt(2.0 * np.pi) * width
        assert profile_sum == pytest.approx(expected, rel=1e-12)

    def test_gaussbin_jacobian_when_available(self):
        if not hasattr(wave, "gaussbin_jac"):
            pytest.skip("This wave.py does not expose gaussbin_jac")

        x = np.arange(15, dtype=float)
        pars = np.array([250.0, 7.31, 1.17, 13.0])
        analytic = wave.gaussbin_jac(x, *pars)
        numeric = numerical_jacobian(lambda p: wave.gaussbin(x, *p), pars)

        np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-6)


class TestVariableProjection:
    def test_linear_solution_matches_weighted_lstsq(self):
        x, y, yerr, truth = make_synthetic_peak(noise=True)
        data = wave._peakfit_linear_data(y, yerr)
        amplitude, background, profile = wave._peakfit_linear_pars(
            x, truth[1], truth[2], data
        )

        design = np.column_stack((profile, np.ones(x.size))) / yerr[:, None]
        target = y / yerr
        expected, _, _, _ = np.linalg.lstsq(design, target, rcond=None)

        np.testing.assert_allclose(
            [amplitude, background], expected, rtol=1e-11, atol=1e-11
        )

    def test_linear_solution_residual_is_orthogonal_to_design(self):
        x, y, yerr, truth = make_synthetic_peak(noise=True)
        data = wave._peakfit_linear_data(y, yerr)
        amplitude, background, profile = wave._peakfit_linear_pars(
            x, truth[1] + 0.1, truth[2] * 1.05, data
        )
        residual = (amplitude * profile + background - y) / yerr

        assert np.dot(residual, profile / yerr) == pytest.approx(0.0, abs=1e-9)
        assert np.dot(residual, 1.0 / yerr) == pytest.approx(0.0, abs=1e-9)

    def test_residual_is_zero_for_exact_model(self):
        x, y, yerr, truth = make_synthetic_peak(noise=False)
        data = wave._peakfit_linear_data(y, yerr)
        nonlinear_pars = [truth[1], np.log(truth[2])]
        residual = wave._peakfit_residual(nonlinear_pars, x, data)
        np.testing.assert_allclose(residual, 0.0, atol=1e-11)

    def test_singular_linear_problem_returns_nan(self, monkeypatch):
        x = np.arange(10, dtype=float)
        y = np.ones(10)
        yerr = np.ones(10)
        data = wave._peakfit_linear_data(y, yerr)

        # A constant profile is exactly degenerate with the background.
        monkeypatch.setattr(
            wave,
            "_gaussbin_profile",
            lambda x, center, width: np.ones_like(x, dtype=float),
        )

        amplitude, background, _ = wave._peakfit_linear_pars(
            x, 4.5, 1.0, data
        )

        assert np.isnan(amplitude)
        assert np.isnan(background)

class TestFastPeakFit:
    @pytest.mark.parametrize(
        "amplitude,center,width,background",
        [
            (30.0, 50.13, 0.65, -5.0),
            (100.0, 49.50, 0.90, 0.0),
            (500.0, 50.37, 1.15, 20.0),
            (5000.0, 50.82, 2.50, 100.0),
        ],
    )
    def test_exact_synthetic_lines(self, amplitude, center, width, background):
        x, y, yerr, truth = make_synthetic_peak(
            amplitude=amplitude,
            center=center,
            width=width,
            background=background,
        )
        initial = np.array([amplitude * 0.9, center - 0.25, width * 1.1, background])
        fitted, error = wave._peakfit_fast(x, y, yerr, initial, xwid=5)

        assert_peak_close(fitted, truth, center_atol=2e-4, width_atol=2e-4)
        assert np.isfinite(error[1])
        assert np.isfinite(error[2])
        assert error[1] >= 0.0
        assert error[2] >= 0.0

    def test_noisy_line_agrees_with_fully_converged_fit(self):
        from scipy.optimize import least_squares

        x, y, yerr, truth = make_synthetic_peak(noise=True)
        initial = np.array([450.0, 50.0, 1.0, 20.0])
        fast, _ = wave._peakfit_fast(x, y, yerr, initial, xwid=5)

        data = wave._peakfit_linear_data(y, yerr)
        reference_result = least_squares(
            wave._peakfit_residual,
            [initial[1], np.log(initial[2])],
            args=(x, data),
            method="lm",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=200,
        )
        center = reference_result.x[0]
        width = np.exp(reference_result.x[1])
        amplitude, background, _ = wave._peakfit_linear_pars(
            x, center, width, data
        )
        reference = np.array([amplitude, center, width, background])

        assert fast[1] == pytest.approx(reference[1], abs=1e-3)
        assert fast[2] == pytest.approx(reference[2], abs=1e-3)

    def test_rejects_nonphysical_constant_spectrum(self):
        x = np.arange(21, dtype=float)
        y = np.full(21, 10.0)
        yerr = np.ones(21)
        initial = np.array([1.0, 10.0, 1.0, 10.0])

        with pytest.raises(RuntimeError, match="Invalid fast peak fit"):
            wave._peakfit_fast(x, y, yerr, initial, xwid=5)

    def test_rejects_center_shift_outside_allowed_window(self):
        x, y, yerr, truth = make_synthetic_peak(center=52.0)
        initial = np.array([truth[0], 49.0, truth[2], truth[3]])

        with pytest.raises(RuntimeError, match="Invalid fast peak fit"):
            wave._peakfit_fast(x, y, yerr, initial, xwid=1)


class TestPeakFitInterface:
    @pytest.mark.parametrize("center", [2.2, 40.37, 77.6])
    def test_peakfit_handles_detector_edges(self, center):
        x, spectrum, error, truth = make_synthetic_peak(
            size=80,
            center=center,
            amplitude=1000.0,
            width=1.0,
            background=0.0,
        )
        fitted, fit_error = wave.peakfit(
            spectrum,
            center + 0.3,
            estsig=1.0,
            sigma=error,
        )

        assert_peak_close(fitted, truth, center_atol=2e-3, width_atol=2e-3)
        assert np.isfinite(fit_error[1])

    def test_fast_failure_uses_curve_fit_fallback(self, monkeypatch):
        _, spectrum, error, truth = make_synthetic_peak()
        original_curve_fit = wave.curve_fit
        calls = {"count": 0}

        def forced_failure(*args, **kwargs):
            raise RuntimeError("deliberate test failure")

        def counting_curve_fit(*args, **kwargs):
            calls["count"] += 1
            return original_curve_fit(*args, **kwargs)

        monkeypatch.setattr(wave, "_peakfit_fast", forced_failure)
        monkeypatch.setattr(wave, "curve_fit", counting_curve_fit)

        fitted, fit_error = wave.peakfit(
            spectrum,
            truth[1],
            estsig=truth[2],
            sigma=error,
        )

        assert calls["count"] >= 1
        assert_peak_close(fitted, truth, center_atol=1e-4, width_atol=1e-4)
        assert np.isfinite(fit_error).all()

    def test_custom_model_uses_curve_fit(self, monkeypatch):
        _, spectrum, error, truth = make_synthetic_peak()
        original_curve_fit = wave.curve_fit
        calls = {"count": 0}

        def custom_model(x, amplitude, center, width, background):
            return wave.gaussbin(x, amplitude, center, width, background)

        def counting_curve_fit(*args, **kwargs):
            calls["count"] += 1
            return original_curve_fit(*args, **kwargs)

        monkeypatch.setattr(wave, "curve_fit", counting_curve_fit)

        fitted, _ = wave.peakfit(
            spectrum,
            truth[1],
            estsig=truth[2],
            sigma=error,
            func=custom_model,
        )

        assert calls["count"] >= 1
        assert_peak_close(fitted, truth, center_atol=1e-4, width_atol=1e-4)

    def test_blended_peak_fit_recovers_centers(self):
        x = np.arange(2048, dtype=float)
        main = np.array([500.0, 1000.25, 1.10, 20.0])
        neighbor = np.array([180.0, 1004.10, 1.25, 0.0])
        spectrum = wave.gaussbin(x, *(tuple(main) + tuple(neighbor)))
        error = np.sqrt(np.maximum(spectrum, 0.0) + 25.0)

        fitted, fit_error = wave.peakfit_multi(
            spectrum,
            main.copy(),
            neighbor.copy(),
            sigma=error,
        )

        assert fitted[1] == pytest.approx(main[1], abs=2e-3)
        assert fitted[5] == pytest.approx(neighbor[1], abs=2e-3)
        assert np.isfinite(fit_error).all()


class TestWavelengthCoordinates:
    def test_func_multi_poly_without_offsets(self):
        pixel = np.array([0.0, 1023.5, 2047.0] * 3)
        chip = np.repeat([1.0, 2.0, 3.0], 3)
        group = np.zeros(pixel.size)
        x = np.vstack((pixel, chip, group))
        coefficients = np.array([2e-7, 0.21, 16000.0])
        offsets = np.zeros(3)

        actual = wave.func_multi_poly(
            x, *(tuple(coefficients) + tuple(offsets)), npoly=3
        )
        xglobal = pixel - 1023.5 + (chip - 2.0) * 2048.0
        expected = np.polyval(coefficients, xglobal)

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    def test_func_multi_poly_applies_group_and_chip_offsets(self):
        pixel = np.array([100.0, 100.0, 100.0, 100.0])
        chip = np.array([1.0, 2.0, 3.0, 1.0])
        group = np.array([0.0, 0.0, 0.0, 1.0])
        x = np.vstack((pixel, chip, group))
        coefficients = np.array([0.2, 16000.0])
        offsets = np.array([1.0, 2.0, 3.0, -2.0, -3.0, -4.0])

        actual = wave.func_multi_poly(
            x, *(tuple(coefficients) + tuple(offsets)), npoly=2
        )
        selected_offsets = np.array([1.0, 2.0, 3.0, -2.0])
        xglobal = pixel - 1023.5 + (chip - 2.0) * 2048.0 + selected_offsets
        expected = np.polyval(coefficients, xglobal)

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    def test_getgroup_makes_consecutive_ids(self):
        groups = np.array([10, 3, 10, 27, 3, 27, 10])
        mapped, unique = wave.getgroup(groups)
        np.testing.assert_array_equal(unique, [3, 10, 27])
        np.testing.assert_array_equal(mapped, [1, 0, 1, 2, 0, 2, 1])

    def test_pixel_wavelength_roundtrip_for_nonlinear_solution(self):
        pixel_grid = np.arange(2048, dtype=float)
        centered = pixel_grid - 1023.5
        wavelength_grid = 16000.0 + 0.22 * centered + 2e-8 * centered**2
        test_pixel = np.array([0.0, 1.5, 100.25, 1023.5, 1900.75, 2047.0])

        test_wave = wave.pix2wave(test_pixel, wavelength_grid)
        recovered_pixel = wave.wave2pix(test_wave, wavelength_grid)

        np.testing.assert_allclose(recovered_pixel, test_pixel, atol=2e-3)

    def test_scalar_coordinate_conversion_returns_scalar(self):
        wavelength_grid = np.linspace(15100.0, 15800.0, 2048)
        wavelength = wave.pix2wave(100.5, wavelength_grid)
        pixel = wave.wave2pix(wavelength, wavelength_grid)

        assert np.isscalar(wavelength)
        assert np.isscalar(pixel)
        assert pixel == pytest.approx(100.5, abs=2e-3)

    def test_coordinate_conversion_returns_nan_outside_detector(self):
        wavelength_grid = np.linspace(15100.0, 15800.0, 2048)
        output_wave = wave.pix2wave(np.array([-1.0, 2048.0]), wavelength_grid)
        output_pixel = wave.wave2pix(
            np.array([15099.0, 15801.0]), wavelength_grid
        )

        assert np.isnan(output_wave).all()
        assert np.isnan(output_pixel).all()

    def test_wave2pix_accepts_descending_wavelength_grid(self):
        ascending = np.linspace(15100.0, 15800.0, 2048)
        descending = ascending[::-1]
        requested = np.array([15200.0, 15500.0, 15700.0])

        expected = wave.wave2pix(requested, ascending)
        actual = wave.wave2pix(requested, descending)

        np.testing.assert_allclose(actual, 2047.0 - expected, atol=2e-3)


class TestArcPairs:
    def test_removes_single_exposure_groups_and_renumbers(self):
        frame_dtype = [
            ("num", "i8"),
            ("group", "i8"),
            ("lamptype", "U10"),
            ("dithpix", "f8"),
        ]
        line_dtype = [
            ("frameid", "i8"),
            ("group", "i8"),
            ("pixel", "f8"),
        ]

        frameinfo = np.array(
            [
                (100, 2, "UNE", 12.0),
                (101, 2, "THARNE", 12.0),
                (200, 8, "UNE", 12.0),
                (300, 11, "UNE", 12.0),
                (301, 11, "THARNE", 12.0),
            ],
            dtype=frame_dtype,
        )
        lines = np.array(
            [
                (100, 2, 10.0),
                (101, 2, 11.0),
                (200, 8, 12.0),
                (300, 11, 13.0),
                (301, 11, 14.0),
            ],
            dtype=line_dtype,
        )

        output_frames, output_lines = wave.getarcpairs(
            frameinfo.copy(), lines.copy()
        )

        np.testing.assert_array_equal(output_frames["num"], [100, 101, 300, 301])
        np.testing.assert_array_equal(output_lines["frameid"], [100, 101, 300, 301])
        np.testing.assert_array_equal(output_frames["group"], [0, 0, 1, 1])
        np.testing.assert_array_equal(output_lines["group"], [0, 0, 1, 1])

