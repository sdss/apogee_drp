"""Independent resampler, packed-coefficient and real-file regression checks."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from astropy.io import fits
from apogee_drp.apred.sincint import sincint
from apogee_drp.utils.apstar_lsf import (
    ApStarLSF,
    ChipLSF,
    VisitLSF,
    fit_chip,
    fit_polynomial_profiles,
    integrated_basis,
    sinc_weights,
    wave_to_pixel,
    derive_apstar_lsf,
    evaluate_apogee_gh,
    CSPEED,
)


class LSFTests(unittest.TestCase):
    def test_matches_actual_pipeline_sinc(self):
        rng = np.random.default_rng(81)
        signal = rng.normal(size=4096)
        pixel = np.r_[rng.uniform(60, 4030, 30), 100.0, 2030.0]
        for nres in [3.5, 4.25, 5.0]:
            index, weight = sinc_weights(pixel, nres, len(signal))
            expected = sincint(pixel, nres, [[signal, None]])[0][0]
            np.testing.assert_allclose(
                np.sum(weight * signal[index], axis=1), expected, rtol=2e-14, atol=2e-14
            )

    def test_packed_doppler_roundtrip_and_axis_orientation(self):
        coefs = np.zeros((7, 3))
        coefs[:, 0] = [1.2, 0.03, -0.07, 0.08, -0.02, 0.04, 4.5]
        coefs[0, 1] = 0.15
        coefs[2, 1] = -0.02
        coefs[-1, 2] = 0.2
        model = ChipLSF(0, 4500.0, 900.0, coefs, 5, (3600, 5400), {})
        centers = np.array([3750.13, 4500.43, 5150.83])
        offset = np.arange(-40, 41)[None, :] - np.array([0.13, 0.43, 0.83])[:, None]
        packed = model.doppler_parameters()
        expected = evaluate_apogee_gh(offset, centers, packed)
        expected /= expected.sum(axis=1)[:, None]
        np.testing.assert_allclose(
            model.evaluate(centers, offset), expected, rtol=1e-10, atol=2e-14
        )
        reflected = model.evaluate(centers, -offset)
        self.assertGreater(np.max(np.abs(reflected - expected)), 0.001)

    def test_native_basis_against_numerical_quadrature(self):
        from scipy.integrate import quad
        from scipy.special import eval_hermitenorm
        import math

        for sigma in [0.8, 2.3]:
            for binsize in [1.0, 2.0]:
                x = np.array([-2.13, 0.37, 3.21])
                basis = integrated_basis(x, sigma, 8, binsize)
                for m in [0, 1, 3, 5, 8]:
                    expected = [
                        quad(
                            lambda y: np.exp(-0.5 * (y / sigma) ** 2)
                            / (sigma * np.sqrt(2 * np.pi))
                            * eval_hermitenorm(m, y / sigma)
                            / np.sqrt(math.factorial(m)),
                            q - binsize / 2,
                            q + binsize / 2,
                            epsabs=1e-13,
                        )[0]
                        for q in x
                    ]
                    np.testing.assert_allclose(
                        basis[:, m], expected, rtol=1e-11, atol=2e-14
                    )

    def test_conditioned_fit_recovers_known_response(self):
        coefs = np.zeros((7, 2))
        coefs[:, 0] = [1.25, 0.02, -0.05, 0.07, 0.01, 0.03, 4.0]
        coefs[0, 1] = 0.1
        truth = ChipLSF(1, 2500, 1000, coefs, 5, (1500, 3500), {})
        centers = np.linspace(1500.17, 3500.81, 16)
        offset = np.arange(-30, 31)[None, :] - (centers % 1)[:, None]
        data = truth.evaluate(centers, offset)
        fitted = fit_chip(
            1,
            centers,
            offset,
            data,
            degree=1,
            initial_sigma=1.2,
            initial_wing_sigma=4.0,
        )
        np.testing.assert_allclose(fitted.evaluate(centers, offset), data, atol=1e-7)

    def test_joint_polynomial_recovers_shared_profiles(self):
        coefs = np.zeros((7, 2))
        coefs[:, 0] = [1.2, 0.02, -0.04, 0.05, 0.01, 0.03, 4.5]
        coefs[0, 1] = 0.1
        coefs[2, 1] = 0.01
        truth = ChipLSF(-1, 350.0, 250.0, coefs, 5, (100.0, 600.0), {})
        profiles = {}
        for chip, domain in enumerate([(100.0, 200.0), (230.0, 350.0), (400.0, 600.0)]):
            q = np.linspace(*domain, 12) + 0.17
            offset = np.arange(-30, 31)[None, :] - (q % 1)[:, None]
            profiles[chip] = dict(
                centers=q,
                offsets=offset,
                reference=truth.evaluate(q, offset),
                training=np.arange(len(q)) % 3 != 0,
            )
        fitted, diagnostics = fit_polynomial_profiles(
            profiles, degree=1, horder=5, initial_sigma=1.2, initial_wing_sigma=4.5
        )
        for p in profiles.values():
            np.testing.assert_allclose(p["model"], p["reference"], atol=1e-7)
        self.assertEqual(fitted.doppler_parameters().ndim, 1)

    def test_analytic_coefficient_jacobian(self):
        coefs = np.zeros((7, 3))
        coefs[:, 0] = [1.2, 0.02, -0.04, 0.05, 0.01, 0.03, 4.5]
        coefs[0, 1] = 0.1
        coefs[2, 2] = 0.02
        model = ChipLSF(0, 500.0, 300.0, coefs, 5, (200, 800), {})
        centers = np.linspace(250.13, 750.67, 9)
        offsets = np.arange(-25, 26)[None, :] - (centers % 1)[:, None]
        value, jac = model.evaluate_with_jacobian(centers, offsets)
        for index in range(coefs.size):
            row, col = np.unravel_index(index, coefs.shape)
            step = 1e-6
            model.coefficients[row, col] += step
            upper = model.evaluate(centers, offsets)
            model.coefficients[row, col] -= 2 * step
            lower = model.evaluate(centers, offsets)
            model.coefficients[row, col] += step
            np.testing.assert_allclose(
                jac[:, index],
                ((upper - lower) / (2 * step)).ravel(),
                atol=2e-9,
                rtol=3e-5,
            )

    def test_public_api_does_not_load_doppler_models(self):
        import subprocess, sys

        r = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from apogee_drp.utils.lsf import derive_apstar_lsf; assert 'doppler' not in sys.modules",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_saved_result_roundtrip_and_gap_handling(self):
        coefs = np.zeros((7, 2))
        coefs[:, 0] = [1.2, 0.02, -0.04, 0.05, 0.01, 0.03, 4.0]
        chip = ChipLSF(0, 50.0, 30.0, coefs, 5, (20.0, 80.0), {"rms": 0.0})
        centers = np.array([40.2, 60.3])
        offsets = np.arange(-8, 9)[None, :] - (centers % 1)[:, None]
        model = chip.evaluate(centers, offsets)
        profiles = {
            0: dict(
                centers=centers,
                offsets=offsets,
                reference=model,
                model=model,
                training=np.array([True, False]),
            )
        }
        result = ApStarLSF(
            np.linspace(15000, 15100, 100),
            [chip],
            profiles,
            dict(halfwidth=8, diagnostics={"0": chip.diagnostics}),
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lsf.fits"
            result.save(path)
            loaded = ApStarLSF.read(path)
            np.testing.assert_allclose(
                loaded(centers), result(centers), atol=2e-14, rtol=1e-12
            )
            self.assertTrue(np.isnan(loaded([5.0])).all())
            with self.assertRaises(OSError):
                result.save(path)

    def test_invalid_weights_and_unrecognized_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            visit = Path(temp) / "visit.fits"
            hdus = [fits.PrimaryHDU()] + [fits.ImageHDU() for _ in range(10)]
            hdus[4] = fits.ImageHDU(np.tile(np.linspace(16000, 15000, 100), (3, 1)))
            hdus[10] = fits.ImageHDU(np.ones((3, 26)))
            fits.HDUList(hdus).writeto(visit)
            with self.assertRaisesRegex(ValueError, "Unrecognized"):
                VisitLSF.read(visit, 0.0)
            star = Path(temp) / "star.fits"
            h = fits.PrimaryHDU()
            h.header["NVISITS"] = 2
            h.header["SFILE1"] = "visit.fits"
            h.header["SFILE2"] = "visit2.fits"
            h.header["VREL1"] = 0.0
            h.header["VREL2"] = 0.0
            flux = fits.ImageHDU(np.ones((3, 8575)))
            flux.header["CRVAL1"] = 4.179
            flux.header["CDELT1"] = 6e-6
            fits.HDUList([h, flux]).writeto(star)
            with self.assertRaisesRegex(ValueError, "every visit"):
                derive_apstar_lsf(star, [visit])

    @unittest.skipUnless(
        os.getenv("APOGEE_LSF_APSTAR") and os.getenv("APOGEE_LSF_APVISIT"),
        "Set APOGEE_LSF_APSTAR and APOGEE_LSF_APVISIT for actual-file checks",
    )
    def test_real_file_response_against_full_native_resampling(self):
        star = os.environ["APOGEE_LSF_APSTAR"]
        visit = os.environ["APOGEE_LSF_APVISIT"]
        with fits.open(star) as h:
            wave = 10 ** (
                h[1].header["CRVAL1"] + np.arange(8575) * h[1].header["CDELT1"]
            )
            vrel = h[0].header["VREL1"]
        native = VisitLSF.read(visit, vrel)
        self.assertEqual(native.pars[:, 0].tolist(), [2.0, 2.0, 2.0])
        for chip in range(3):
            wc = np.median(native.wave[chip]) / (1 + vrel / CSPEED)
            center = (
                np.interp(np.log10(wc), np.log10(wave), np.arange(len(wave))) + 0.173
            )
            output = np.floor(center).astype(int) + np.arange(-48, 49)
            # Independent reference: evaluate all 4096 native samples, then call
            # the pipeline sincint itself rather than the fast sparse operator.
            line_wave = 10 ** np.interp(center, np.arange(len(wave)), np.log10(wave))
            xc = float(
                wave_to_pixel(line_wave * (1 + vrel / CSPEED), native.wave[chip])
            )
            raw = evaluate_apogee_gh(
                np.arange(4096) - xc, np.array([xc]), native.pars[chip]
            )
            xout = wave_to_pixel(wave[output] * (1 + vrel / CSPEED), native.wave[chip])
            reference = sincint(xout, native.nres[chip], [[raw, None]])[0][0]
            reference /= reference.sum()
            np.testing.assert_allclose(
                native.response(wave, center, output, chip),
                reference,
                rtol=1e-12,
                atol=2e-14,
            )


if __name__ == "__main__":
    unittest.main()
