"""Sigma-only refinement must recover pixel variation and preserve shape fields."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import numpy as np
from astropy.io import fits
from astropy.table import Table
from apogee_drp.utils.pixel_lsf import NativePixelLSF
from apogee_drp.utils.calibration_lsf import LineWindows, LOGW0, DLOGW
from apogee_drp.utils.native_lsf import NativeLSFCalibration
from apogee_drp.utils.sigma_refinement import refine_sigma


def models():
    co = np.zeros((6, 4))
    co[:, 0] = [1.2, 0.015, -0.01, 0.004, 0.03, 4.2]
    co[0, 1:] = [0.08, 0.025, -0.012]
    true = NativePixelLSF(
        co, 0.0, 4096.0, 5, {"fiber": 22}, np.array([-3200.0, -1000.0, 1200.0]), 2048
    )
    guess = NativePixelLSF(
        co.copy(),
        0.0,
        4096.0,
        5,
        {"fiber": 22, "filled": True},
        true.chip_offsets.copy(),
        2048,
    )
    guess.coefficients[0] *= 1.08
    return true, guess


def lines():
    true, guess = models()
    rng = np.random.default_rng(18)
    chip = np.repeat(np.arange(3), 10)
    p = np.tile(np.linspace(100, 1900, 10) + 0.37, 3)
    c = p + true.chip_offsets[chip]
    x = np.floor(p)[:, None] + np.arange(-14, 15) - p[:, None]
    lo = c[:, None] + x - 0.5
    data = LineWindows(
        lo,
        lo + 1,
        np.zeros_like(lo),
        np.full_like(lo, 20.0),
        np.ones_like(lo, dtype=bool),
        c,
        np.zeros(len(c)),
        chip,
        np.arange(len(c)),
        np.arange(len(c)) % 4 != 1,
        x,
        {},
    )
    profile = true.profile(data, np.zeros(len(c)))
    data.flux = 80.0 + 0.05 * x + 30000.0 * profile + rng.normal(0, 20.0, profile.shape)
    return true, guess, data


class SigmaRefinementTests(unittest.TestCase):
    def test_recovers_sigma_curve_without_changing_other_fields(self):
        true, guess, data = lines()
        original = guess.coefficients.copy()
        result, info = refine_sigma(guess, data)
        self.assertTrue(info["accepted"], info)
        x = np.linspace(-3100, 3100, 100)
        self.assertLess(np.max(abs(result.fields(x)[0] / true.fields(x)[0] - 1)), 0.007)
        np.testing.assert_array_equal(result.coefficients[1:], original[1:])
        np.testing.assert_array_equal(guess.coefficients, original)
        self.assertLess(info["after"]["chi_square"], info["before"]["chi_square"])
        self.assertEqual(len(info["sigma_curves"]), 3)

    def test_diagnostic_pdf_preserves_models_and_training_split(self):
        from apogee_drp.utils.lsf_diagnostics import plot_sigma_check

        _, guess, data = lines()
        result, _ = refine_sigma(guess, data)
        coefficients = result.coefficients.copy()
        split = data.train.copy()
        with TemporaryDirectory() as directory:
            for name, windows in [("peaks", data), ("empty", None)]:
                output = plot_sigma_check(
                    result, windows, Path(directory) / f"{name}.pdf"
                )
                self.assertTrue(output.read_bytes().startswith(b"%PDF"))
                self.assertGreater(output.stat().st_size, 5000)
        np.testing.assert_array_equal(result.coefficients, coefficients)
        np.testing.assert_array_equal(data.train, split)

    def test_run_overview_includes_missing_fiber_without_changing_models(self):
        _, guess, data = lines()
        result, _ = refine_sigma(guess, data)
        cal = NativeLSFCalibration({22: result}, {22: "FILLED", 23: "FAILED"})
        before = result.coefficients.copy()
        with TemporaryDirectory() as directory:
            path = cal.plot_diagnostics(Path(directory) / "overview.pdf")
            self.assertTrue(path.read_bytes().startswith(b"%PDF"))
            self.assertGreater(path.stat().st_size, 10000)
        np.testing.assert_array_equal(before, result.coefficients)

    def test_bad_held_out_result_retains_the_original_sigma(self):
        _, guess, data = lines()
        before = dict(
            lines=8, pooled_reduced_chisq=1.0, chi_square=100.0, degrees_of_freedom=100
        )
        after = dict(
            lines=8,
            pooled_reduced_chisq=100.0,
            chi_square=10000.0,
            degrees_of_freedom=100,
        )
        with patch(
            "apogee_drp.utils.sigma_refinement.validation_statistics",
            side_effect=[before, after],
        ):
            result, info = refine_sigma(guess, data)
        self.assertEqual(info["status"], "REJECTED_VALIDATION")
        np.testing.assert_array_equal(result.coefficients, guess.coefficients)

    def test_sparse_data_do_not_free_unconstrained_sigma_parameters(self):
        _, guess, data = lines()
        data.train[:] = False
        data.train[:3] = True
        result, info = refine_sigma(guess, data)
        self.assertEqual(info["status"], "INSUFFICIENT_PEAKS")
        np.testing.assert_array_equal(result.coefficients, guess.coefficients)

    def test_full_checker_refines_data_leaves_empty_fiber_and_is_idempotent(self):
        true, guess = models()
        rng = np.random.default_rng(5)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            files = []
            catalog = []
            p = np.linspace(100, 1900, 10) + 0.37
            for chip, offset in enumerate(true.chip_offsets):
                wave = 10 ** (LOGW0 + DLOGW * (5000 - (np.arange(2048) + offset)))
                f = np.zeros((300, 2048))
                e = np.full_like(f, 1e10)
                mask = np.zeros_like(f, dtype=np.int16)
                e[278] = 20.0
                f[278] = 80.0 + rng.normal(0, 20.0, 2048)
                for j, center in enumerate(p):
                    pix = int(np.floor(center)) + np.arange(-18, 19)
                    f[278, pix] += 30000.0 * true.native_profiles(
                        np.array([center]), pix - center, chip
                    )
                    catalog.append(
                        (
                            chip * 10 + j,
                            10 ** (LOGW0 + DLOGW * (5000 - (center + offset))),
                            1,
                            0,
                            0.0,
                        )
                    )
                coeff = np.zeros((14, 300))
                coeff[0] = offset
                path = root / f"ap1D-{chip}.fits"
                files.append(path)
                fits.HDUList(
                    [
                        fits.PrimaryHDU(),
                        fits.ImageHDU(f),
                        fits.ImageHDU(e),
                        fits.ImageHDU(mask),
                        fits.ImageHDU(np.broadcast_to(wave, (300, 2048))),
                        fits.ImageHDU(coeff),
                    ]
                ).writeto(path)
            tab = Table(
                rows=catalog, names=["ID", "WAVE", "USELSF", "DOUBLET", "DBL_WSEP"]
            )
            empty = NativePixelLSF(
                guess.coefficients.copy(),
                0.0,
                4096.0,
                5,
                {"fiber": 25, "filled": True},
                guess.chip_offsets.copy(),
                2048,
            )
            cal = NativeLSFCalibration({22: guess, 25: empty})
            cal.check_filled_peaks(files, line_catalog=tab)
            self.assertTrue(cal.model(22).diagnostics["sigma_data_constrained"])
            self.assertEqual(
                cal.model(25).diagnostics["sigma_refinement"]["status"],
                "NO_USABLE_DATA",
            )
            self.assertTrue(cal.model(22).diagnostics["filled"])
            self.assertTrue(cal.global_hdu([22]).header["SIGREFIN"])
            saved = cal.model(22).coefficients.copy()
            cal.check_filled_peaks(
                files, line_catalog=tab, diagnostic_directory=root / "diagnostics"
            )
            self.assertTrue((root / "diagnostics/lsf-fiber-022.pdf").is_file())
            self.assertTrue((root / "diagnostics/lsf-fiber-025.pdf").is_file())
            np.testing.assert_array_equal(cal.model(22).coefficients, saved)
            path = root / "checked.fits"
            fits.HDUList([fits.PrimaryHDU(), cal.global_hdu()]).writeto(path)
            reread = NativeLSFCalibration.read(path)
            self.assertTrue(reread.model(22).diagnostics["sigma_data_constrained"])
            np.testing.assert_array_equal(reread.model(22).coefficients, saved)


if __name__ == "__main__":
    unittest.main()
