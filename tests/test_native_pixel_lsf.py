"""Test direct pixel GH fitting, algebraic export and explicit coordinate I/O."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from astropy.table import Table
import numpy as np
from astropy.io import fits
from apogee_drp.utils.pixel_lsf import NativePixelLSF
from apogee_drp.utils.calibration_lsf import LineWindows, NativePolynomialLSF
from apogee_drp.utils.native_lsf import NativeLSFCalibration
from apogee_drp.utils.apstar_lsf import evaluate_apogee_gh, VisitLSF


def model():
    co = np.zeros((6, 4))
    co[:, 0] = [1.2, 0.02, -0.01, 0.005, 0.04, 4.0]
    co[0, 1] = 0.08
    co[1, 2] = 0.003
    co[-1, 1] = 0.1
    return NativePixelLSF(
        co, 0.0, 4000.0, 5, {"fiber": 22}, np.array([-3200.0, -1000.0, 1200.0]), 2048
    )


def windows(m):
    chip = np.array([0, 1, 2])
    native = np.array([500.2, 1000.4, 1500.1])
    c = native + m.chip_offsets
    x = np.broadcast_to(np.arange(-16, 17), (3, 33)).copy()
    lo = c[:, None] + x - 0.5
    return LineWindows(
        lo,
        lo + 1,
        np.ones_like(lo),
        np.ones_like(lo),
        np.ones_like(lo, dtype=bool),
        c,
        np.array([0.2, 0.4, 0.0]),
        chip,
        np.arange(3),
        np.ones(3, dtype=bool),
        x,
        {},
    )


class PixelNativeTests(unittest.TestCase):
    def test_profile_is_standard_pixel_gh_including_doublets(self):
        m = model()
        lines = windows(m)
        shift = np.array([0.1, -0.03, 0.02])
        p = m.profile(lines, shift)
        for i, chip in enumerate(lines.chip):
            local = lines.centers[i] - m.chip_offsets[chip] + shift[i]
            outputs = lines.centers[i] - m.chip_offsets[chip] + lines.x[i]
            expected = 0
            for sign in (-0.5, 0.5):
                c = local + sign * lines.separation[i]
                expected = expected + 0.5 * evaluate_apogee_gh(
                    outputs - c, [c], m.parameters(int(chip))
                )
            np.testing.assert_allclose(p[i], expected, atol=2e-14)

    def test_profile_analytic_derivatives(self):
        m = model()
        lines = windows(m)
        shift = np.array([0.01, -0.02, 0.03])
        value, jac, dc = m.profile(lines, shift, True)
        step = 1e-6
        for k in (0, 1, 5, 11, 20, 23):
            old = m.coefficients.ravel()[k]
            m.coefficients.ravel()[k] = old + step
            plus = m.profile(lines, shift)
            m.coefficients.ravel()[k] = old - step
            minus = m.profile(lines, shift)
            m.coefficients.ravel()[k] = old
            np.testing.assert_allclose(
                jac[:, :, k], (plus - minus) / (2 * step), atol=2e-9, rtol=2e-5
            )
        for i in range(3):
            plus = shift.copy()
            minus = shift.copy()
            plus[i] += step
            minus[i] -= step
            numerical = (m.profile(lines, plus) - m.profile(lines, minus)) / (2 * step)
            np.testing.assert_allclose(dc[i], numerical[i], atol=1e-8, rtol=2e-5)

    def test_exact_half_pixel_coefficient_scaling(self):
        m = model()
        centers = np.array([300.2, 900.3, 1500.4])
        offsets = np.arange(-20, 21)
        for chip in range(3):
            original = evaluate_apogee_gh(offsets, centers, m.parameters(chip))
            fine = evaluate_apogee_gh(
                offsets * 2, centers * 2, m.parameters(chip, pixel_scale=2, binsize=2)
            )
            np.testing.assert_allclose(original, fine, atol=3e-15)

    def test_explicit_fiber_degree_overrides(self):
        lines = SimpleNamespace(provenance={})

        def fit(lines, files, fiber, degree, horder):
            m = model()
            m.coefficients = np.pad(m.coefficients, ((0, 0), (0, degree - 3)))
            m.diagnostics = {"fiber": fiber}
            return m, lines

        with patch(
            "apogee_drp.utils.native_lsf.load_calibration_windows", return_value=lines
        ), patch(
            "apogee_drp.utils.native_lsf.fit_native_pixel", side_effect=fit
        ) as fitting, patch(
            "apogee_drp.utils.native_lsf.evaluate_held_out", return_value=([], [], None)
        ):
            calibration = NativeLSFCalibration.fit(
                ["a", "b", "c"],
                line_catalog=Table(),
                fibers=[22, 23],
                degree=3,
                degree_overrides={23: 4},
            )
        self.assertEqual(fitting.call_args_list[0].kwargs["degree"], 3)
        self.assertEqual(fitting.call_args_list[1].kwargs["degree"], 4)
        self.assertEqual(calibration.provenance["degree_overrides"], {23: 4})

    def test_global_file_preserves_different_degrees_between_fibers(self):
        m = model()
        extended = NativePixelLSF(
            np.pad(m.coefficients, ((0, 0), (0, 1))),
            m.origin,
            m.scale,
            m.horder,
            {"fiber": 23},
            m.chip_offsets.copy(),
            m.npix,
        )
        extended.coefficients[0, 4] = 0.001
        calibration = NativeLSFCalibration({22: m, 23: extended})
        h = fits.HDUList([fits.PrimaryHDU(), calibration.global_hdu()])
        restored = NativeLSFCalibration.from_hdul(h)
        self.assertEqual(restored.model(22).coefficients.shape, (6, 4))
        self.assertEqual(restored.model(23).coefficients.shape, (6, 5))
        np.testing.assert_array_equal(restored.model(22).parameters(0), m.parameters(0))
        np.testing.assert_array_equal(
            restored.model(23).parameters(0), extended.parameters(0)
        )

    def test_pixel_npz_and_global_fits_roundtrip_preserve_coordinate_type(self):
        m = model()
        with TemporaryDirectory() as temp:
            p = Path(temp) / "pixel.npz"
            m.save(p)
            loaded = NativePixelLSF.read(p)
            np.testing.assert_array_equal(loaded.chip_offsets, m.chip_offsets)
            with self.assertRaises(ValueError):
                NativePolynomialLSF.read(p)
            h = NativeLSFCalibration({22: m}).global_hdu()
            self.assertEqual(h.header["LSFVER"], 2)
            self.assertNotIn("LOGW0", h.header)
            fits.HDUList([fits.PrimaryHDU(), h]).writeto(Path(temp) / "cal.fits")
            restored = NativeLSFCalibration.read(Path(temp) / "cal.fits").model(22)
            self.assertIsInstance(restored, NativePixelLSF)
            for chip in range(3):
                np.testing.assert_array_equal(
                    restored.parameters(chip), m.parameters(chip)
                )


if __name__ == "__main__":
    unittest.main()
