"""Numerical checks for native-pixel integration and variable-projection fitting."""

import os, unittest
from pathlib import Path
import numpy as np
from apogee_drp.utils.calibration_lsf import (
    LineWindows,
    NativePolynomialLSF,
    edge_profile,
    project_lines,
    fit_native_polynomial,
    evaluate_held_out,
    load_calibration_windows,
)


def synthetic_lines(noise=0.02):
    n = 12
    npix = 29
    q = np.linspace(600, 7800, n)
    width = np.linspace(1.7, 0.9, n)
    x = np.tile(np.arange(npix) - npix // 2, (n, 1)).astype(float)
    lo = q[:, None] + (x - 0.5) * width[:, None]
    hi = q[:, None] + (x + 0.5) * width[:, None]
    coefs = np.zeros((6, 2))
    coefs[:, 0] = [1.13, 0.025, 0.04, -0.015, 0.04, 4.0]
    coefs[0, 1] = 0.08
    coefs[1, 1] = -0.01
    truth = NativePolynomialLSF(coefs, 4200.0, 3600.0, 5, {})
    lines = LineWindows(
        lo,
        hi,
        np.zeros((n, npix)),
        np.full((n, npix), 2.0),
        np.ones((n, npix), bool),
        q,
        np.where(np.arange(n) % 3 == 0, 0.6, 0.0),
        np.arange(n) // 4,
        np.arange(n),
        np.arange(n) % 4 != 2,
        x,
        {},
    )
    amplitude = np.linspace(5000, 13000, n)
    profile = truth.profile(lines, np.zeros(n))
    rng = np.random.default_rng(813)
    lines.flux = (
        amplitude[:, None] * profile
        + np.linspace(20, 40, n)[:, None]
        + 0.12 * x
        + rng.normal(0, noise, profile.shape)
    )
    return lines, truth


class CalibrationLSFTests(unittest.TestCase):
    def test_native_profile_analytic_derivatives(self):
        lines, model = synthetic_lines()
        shift = np.linspace(-0.04, 0.04, len(lines.centers))
        value, jac, dc = model.profile(lines, shift, True)
        step = 1e-6
        for i in range(model.coefficients.size):
            a, b = np.unravel_index(i, model.coefficients.shape)
            model.coefficients[a, b] += step
            upper = model.profile(lines, shift)
            model.coefficients[a, b] -= 2 * step
            lower = model.profile(lines, shift)
            model.coefficients[a, b] += step
            np.testing.assert_allclose(
                jac[:, :, i], (upper - lower) / (2 * step), atol=2e-9, rtol=1e-5
            )
        np.testing.assert_allclose(
            dc,
            (model.profile(lines, shift + step) - model.profile(lines, shift - step))
            / (2 * step),
            atol=2e-8,
            rtol=2e-5,
        )

    def test_variable_projection_derivatives(self):
        lines, model = synthetic_lines()
        shift = np.zeros(len(lines.centers))
        profile, derivative, _ = model.profile(lines, shift, True)
        residual, jac, linear = project_lines(profile, lines, derivative)
        step = 1e-6
        for i in range(model.coefficients.size):
            a, b = np.unravel_index(i, model.coefficients.shape)
            model.coefficients[a, b] += step
            upper = project_lines(model.profile(lines, shift), lines)[0]
            model.coefficients[a, b] -= 2 * step
            lower = project_lines(model.profile(lines, shift), lines)[0]
            model.coefficients[a, b] += step
            np.testing.assert_allclose(
                jac[:, :, i], (upper - lower) / (2 * step), atol=3e-6, rtol=1e-5
            )

    def test_shared_polynomial_recovers_unseen_synthetic_lines(self):
        lines, truth = synthetic_lines(noise=0.001)
        fitted = fit_native_polynomial(lines, degree=1, horder=5)
        metrics, pred, data = evaluate_held_out(fitted, lines)
        self.assertLess(max(r["rms_peak_relative"] for r in metrics), 1e-5)

    def test_native_polynomial_file_roundtrip(self):
        import tempfile

        lines, model = synthetic_lines()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.npz"
            model.save(path)
            loaded = NativePolynomialLSF.read(path)
            np.testing.assert_array_equal(
                loaded.fields(lines.centers), model.fields(lines.centers)
            )
            with self.assertRaises(FileExistsError):
                model.save(path)

    def test_native_polynomial_propagation_matches_affine_native_grid(self):
        from apogee_drp.utils.apstar_lsf import VisitLSF
        from apogee_drp.utils.lsf_parameters import pack_parameters
        from apogee_drp.utils.calibration_lsf import LOGW0, DLOGW

        star_wave = 10 ** (LOGW0 + np.arange(8575) * DLOGW)
        q = 5000.0 - 0.5 * np.arange(4096)
        native_wave = 10 ** (LOGW0 + q * DLOGW)
        gh = np.zeros((6, 2))
        gh[0] = [2.3, -0.00005]
        gh[3, 0] = -0.02
        gh[4, 0] = 0.03
        gh[5, 0] = 0.01
        packed = pack_parameters(
            dict(
                binsize=2.0,
                Xoffset=0.0,
                Horder=5,
                Porder=np.array([1, 0, 0, 0, 0, 0]),
                GHcoefs=gh,
                Wproftype=1,
                nWpar=2,
                WPorder=np.zeros(2, dtype=int),
                Wcoefs=np.array([[0.04, 0], [6.0, 0]]),
            )
        )
        visit = VisitLSF(
            "synthetic",
            np.tile(native_wave, (3, 1)),
            np.tile(packed, (3, 1)),
            -145.0,
            np.array([5.0, 4.25, 3.5]),
            np.ones(3),
        )
        coefs = np.zeros((6, 2))
        coefs[:, 0] = [1.1, 0.02, 0.03, -0.01, 0.04, 3.0]
        coefs[0, 1] = 0.05
        model = NativePolynomialLSF(coefs, 4000.0, 1000.0, 5, {})
        center = 4000.37
        output = int(center) + np.arange(-48, 49)
        expected = visit.response(star_wave, center, output, 0)
        actual = model.visit_response(visit, star_wave, center, output, 0)
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=1e-9)

    @unittest.skipUnless(
        os.getenv("APOGEE_CALIBRATION_DIR"),
        "Set APOGEE_CALIBRATION_DIR for measured-line tests",
    )
    def test_measured_training_validation_pixels_do_not_overlap(self):
        root = Path(os.environ["APOGEE_CALIBRATION_DIR"])
        lines = load_calibration_windows(
            [root / f"ap1D-{c}-40030031.fits" for c in "abc"],
            root / "apLSF-40030031.sav",
        )
        for chip in range(3):
            a = lines.select((lines.chip == chip) & lines.train)
            b = lines.select((lines.chip == chip) & ~lines.train)
            training = np.round(a.lo[a.valid], 9)
            validation = np.round(b.lo[b.valid], 9)
            self.assertEqual(len(np.intersect1d(training, validation)), 0)
        self.assertGreater(len(lines.centers), 12)


if __name__ == "__main__":
    unittest.main()
