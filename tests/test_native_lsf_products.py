"""Regression tests for authoritative native models and row-matched apStar LSFs."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from astropy.table import Table

from apogee_drp.utils.calibration_lsf import NativePolynomialLSF, LOGW0, DLOGW
from apogee_drp.utils.native_lsf import (
    NativeLSFCalibration,
    project_native_parameters,
    visit_global_lsf_hdu,
)
from apogee_drp.utils.apstar_lsf import (
    ApStarLSF,
    ChipLSF,
    fit_polynomial_profiles,
    evaluate_apogee_gh,
)
from apogee_drp.utils.resampled_lsf import (
    build_apstar_lsf_table,
    inverse_variance_weights,
    evaluate_lsf_table,
)


def native(fiber=300):
    return NativePolynomialLSF(
        np.array([[1.2], [0.01], [0.005], [-0.002], [0.03], [4.0]]),
        4000.0,
        4000.0,
        5,
        {"fiber": fiber},
    )


class NativeProductTests(unittest.TestCase):
    def test_exact_global_roundtrip_and_convention_guard(self):
        c = NativeLSFCalibration({300: native()}, {300: "OK", 299: "bad lines"})
        h = fits.HDUList([fits.PrimaryHDU(), c.global_hdu()])
        d = NativeLSFCalibration.from_hdul(h)
        np.testing.assert_array_equal(d.model(300).coefficients, native().coefficients)
        with self.assertRaises(ValueError):
            d.model(299)
        h["GLOBAL_LSF"].header["DLOGW"] = 1.0
        with self.assertRaises(ValueError):
            NativeLSFCalibration.from_hdul(h)

    def test_unavailable_fiber_is_preserved_without_substituting_a_model(self):
        with TemporaryDirectory() as temp:
            paths = [Path(temp) / f"{c}.fits" for c in "abc"]
            for path in paths:
                fits.HDUList(
                    [
                        fits.PrimaryHDU(),
                        NativeLSFCalibration({300: native()}).global_hdu(),
                    ]
                ).writeto(path)
            hdu = visit_global_lsf_hdu(paths, 299)
            self.assertEqual(hdu.header["LSFSTAT"], "NO_LSF")
            self.assertEqual(len(hdu.data), 0)
            calibration = NativeLSFCalibration.from_hdul(
                fits.HDUList([fits.PrimaryHDU(), hdu])
            )
            with self.assertRaises(ValueError):
                calibration.model(299)

    def test_native_pixel_projection_increasing_and_decreasing_wave(self):
        for sign in (1, -1):
            wave = 10 ** (LOGW0 + DLOGW * (4000 + sign * np.arange(256) * 0.7))
            pars, error = project_native_parameters(native(), wave)
            self.assertLess(error, 1e-5)
            self.assertTrue(
                np.isfinite(
                    evaluate_apogee_gh(np.arange(-16, 17), [60.0, 150.0], pars)
                ).all()
            )

    def test_standard_files_full_cube_failures_and_no_clobber(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = []
            outputs = []
            wave = 10 ** (LOGW0 + DLOGW * (4000 + np.arange(128) * 0.7))
            for chip in "abc":
                path = root / f"ap1D-{chip}.fits"
                inputs.append(path)
                outputs.append(root / f"apLSF-{chip}.fits")
                fits.HDUList(
                    [
                        fits.PrimaryHDU(),
                        fits.ImageHDU(np.ones((300, 128))),
                        fits.ImageHDU(np.ones((300, 128))),
                        fits.ImageHDU(np.zeros((300, 128), int)),
                        fits.ImageHDU(np.broadcast_to(wave, (300, 128))),
                    ]
                ).writeto(path)
            c = NativeLSFCalibration(
                {300: native()}, {300: "OK", 299: "insufficient lines"}
            )
            c.write(inputs, outputs, global_output=root / "global.fits")
            with fits.open(outputs[0], checksum=True) as h:
                self.assertEqual(h[0].data.shape[1], 300)
                self.assertEqual(h[1].data.shape, (33, 300, 128))
                self.assertTrue(np.isnan(h[0].data[:, 1]).all())
                self.assertTrue(np.isfinite(h[1].data[:, 0]).all())
                self.assertEqual(h[0].header["LSFMETHD"], 3)
                self.assertEqual(h[0].header["BINSIZE"], 1.0)
            loaded = NativeLSFCalibration.read(root / "global.fits")
            self.assertEqual(loaded.status[299], "insufficient lines")
            before = outputs[0].read_bytes()
            with self.assertRaises(FileExistsError):
                c.write(inputs, outputs)
            self.assertEqual(outputs[0].read_bytes(), before)

    def test_visit_carries_exact_model_and_rejects_mixed_calibrations(self):
        with TemporaryDirectory() as temp:
            paths = [Path(temp) / f"{c}.fits" for c in "abc"]
            for p in paths:
                fits.HDUList(
                    [
                        fits.PrimaryHDU(),
                        NativeLSFCalibration({300: native()}).global_hdu(),
                    ]
                ).writeto(p)
            h = visit_global_lsf_hdu(paths, 300)
            self.assertEqual(len(h.data), 1)
            self.assertEqual(int(h.data["FIBER"][0]), 300)
            fits.HDUList([fits.PrimaryHDU()]).writeto(paths[2], overwrite=True)
            with self.assertRaises(ValueError):
                visit_global_lsf_hdu(paths, 300)

    def test_visit_writer_preserves_hdu_indices_and_global_model(self):
        # Reuse the existing visit-product fixture, exercising the actual writer.
        path = Path(__file__).parent / "visit/test_visit_output.py"
        spec = importlib.util.spec_from_file_location("lsf_visit_fixtures", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from apogee_drp.apred.visit.output import build_visit_hdul

        frame = module._frame()
        plug = module._plugmap()
        with TemporaryDirectory() as temp:
            for letter in "abc":
                chip = frame.chip(letter)
                p = Path(temp) / f"{letter}.fits"
                fits.HDUList(
                    [
                        fits.PrimaryHDU(),
                        NativeLSFCalibration({300: native()}).global_hdu(),
                    ]
                ).writeto(p)
                chip.lsffile = str(p)
            h = build_visit_hdul(frame, plug, 0)
            self.assertEqual(h[4].name, "WAVELENGTH")
            self.assertEqual(h[10].name, "LSF COEFFICIENTS")
            self.assertIn("GLOBAL_LSF", h)
            self.assertEqual(NativeLSFCalibration.from_hdul(h).model(300).horder, 5)


class ResampledProductTests(unittest.TestCase):
    def test_invalid_errors_contribute_no_weight(self):
        w = inverse_variance_weights([2.0, 0.0, np.nan, np.inf, -1.0])
        np.testing.assert_array_equal(w, [0.25, 0, 0, 0, 0])

    def test_mixed_fit_and_compact_fits_roundtrip(self):
        co = np.zeros((7, 4))
        co[:, 0] = [1.2, 0.001, 0.02, 0.01, -0.003, 0.03, 4.0]
        co[0, 1] = 0.04
        co[2, 3] = 0.001
        co[3, 2] = 0.001
        model = ChipLSF(-1, 100.0, 100.0, co, 5, (0.0, 200.0), {})
        centers = np.linspace(0, 200, 45)
        offsets = np.broadcast_to(np.arange(-40, 41), (45, 81))
        profile = dict(
            centers=centers,
            offsets=offsets,
            reference=model.evaluate(centers, offsets),
            training=np.arange(45) % 4 != 1,
        )
        degrees = [3, 0, 3, 2, 3, 0, 0]
        fitted, qa = fit_polynomial_profiles(
            {0: profile}, horder=5, degree=3, polynomial_degrees=degrees
        )
        self.assertEqual(fitted.diagnostics["fitted_coefficients"], 18)
        mask = np.arange(4)[None, :] > np.array(degrees)[:, None]
        np.testing.assert_array_equal(fitted.coefficients[mask], 0.0)
        self.assertLess(qa["0"]["validation_max_peak_relative"], 1e-4)
        a = ApStarLSF(
            np.arange(250), [], {0: profile}, {}, fitted, np.array([[0.0, 200.0]])
        )
        with TemporaryDirectory() as temp:
            path = Path(temp) / "lsf.fits"
            a.save(path)
            b = ApStarLSF.read(path)
            np.testing.assert_allclose(
                b.global_model.evaluate(centers, offsets),
                fitted.evaluate(centers, offsets),
                atol=1e-13,
            )

    def test_combination_rows_use_corresponding_weights_and_shifts(self):
        target = SimpleNamespace(
            header=fits.Header({"NVISITS": 2, "VREL1": 10.0, "VREL2": 20.0}),
            wave=np.arange(8.0),
            flux=np.zeros((4, 8)),
        )
        pixel = np.array([[1.0] * 8, [2.0] * 8])
        glob = np.array([[3.0] * 8, [4.0] * 8])
        co = np.zeros((7, 4))
        co[:, 0] = [1.2, 0, 0, 0, 0, 0.03, 4.0]
        model = ChipLSF(-1, 4.0, 4.0, co, 5, (0, 7), {})
        result = SimpleNamespace(
            parameters=model.doppler_parameters(),
            domains=np.array([[0, 1], [3, 4], [6, 7]]),
            provenance={"diagnostics": {"0": {"validation_max_peak_relative": 0.012}}},
        )
        with patch(
            "apogee_drp.utils.resampled_lsf.derive_apstar_lsf", return_value=result
        ) as derive:
            tab = build_apstar_lsf_table(
                target,
                ["v1.fits", "v2.fits"],
                pixel,
                global_weights=glob,
                vrel=[-30.0, 40.0],
            )
        self.assertEqual(tab["ROW"].tolist(), [0, 1, 2, 3])
        self.assertEqual(tab["KIND"].tolist(), ["PIXEL", "GLOBAL", "VISIT", "VISIT"])
        self.assertEqual(tab["STATUS"].tolist(), ["WARN"] * 4)
        np.testing.assert_array_equal(derive.call_args_list[0].kwargs["weights"], pixel)
        np.testing.assert_array_equal(derive.call_args_list[1].kwargs["weights"], glob)
        self.assertEqual(derive.call_args_list[0].args[0].header["VREL1"], -30.0)
        with patch(
            "apogee_drp.utils.resampled_lsf.derive_apstar_lsf", return_value=result
        ):
            with self.assertRaises(ValueError):
                build_apstar_lsf_table(
                    target,
                    ["v1.fits", "v2.fits"],
                    pixel,
                    global_weights=glob,
                    options={"max_error": 0.01},
                )
        values = evaluate_lsf_table(tab, 0, [0.5, 2.0, 3.5], np.arange(-8, 9))
        self.assertTrue(np.isfinite(values[[0, 2]]).all())
        self.assertTrue(np.isnan(values[1]).all())

    def test_unavailable_native_calibration_marks_rows_without_losing_spectra(self):
        from apogee_drp.utils.native_lsf import NativeLSFUnavailable

        target = SimpleNamespace(
            header=fits.Header({"VREL1": 0.0}),
            wave=np.arange(8.0),
            flux=np.zeros((1, 8)),
        )
        with patch(
            "apogee_drp.utils.resampled_lsf.derive_apstar_lsf",
            side_effect=NativeLSFUnavailable("failed fiber"),
        ):
            table = build_apstar_lsf_table(target, ["v1.fits"], np.ones((1, 8)))
        self.assertEqual(table["STATUS"][0], "NO_LSF")
        self.assertEqual(table["NPARS"][0], 0)
        with TemporaryDirectory() as temp:
            fits.BinTableHDU(table).writeto(Path(temp) / "unavailable.fits")

    def test_zero_weight_visits_are_excluded_from_combined_lsf(self):
        target = SimpleNamespace(
            header=fits.Header({"VREL1": 0.0, "VREL2": 1.0}),
            wave=np.arange(8.0),
            flux=np.zeros((4, 8)),
        )
        result = SimpleNamespace(
            parameters=np.array([1.0, 2.0]),
            domains=np.array([[0, 1], [3, 4], [6, 7]]),
            provenance={"diagnostics": {"0": {"validation_max_peak_relative": 0.001}}},
        )
        weights = np.array([[1.0] * 8, [0.0] * 8])
        with patch(
            "apogee_drp.utils.resampled_lsf.derive_apstar_lsf", return_value=result
        ) as derive:
            table = build_apstar_lsf_table(
                target, ["v1.fits", "v2.fits"], weights, global_weights=weights
            )
        self.assertEqual(derive.call_args_list[0].args[1], ["v1.fits"])
        self.assertEqual(derive.call_args_list[1].args[1], ["v1.fits"])
        with patch(
            "apogee_drp.utils.resampled_lsf.derive_apstar_lsf", return_value=result
        ):
            table = build_apstar_lsf_table(
                target, ["v1.fits", "v2.fits"], weights * 0, global_weights=weights
            )
        self.assertEqual(table["STATUS"][0], "NO_DATA")

    def test_missing_rv_visit_has_explicit_unavailable_row(self):
        target = SimpleNamespace(
            header=fits.Header({"VREL1": 0.0, "VREL2": "NaN"}),
            wave=np.arange(8.0),
            flux=np.zeros((4, 8)),
        )
        result = SimpleNamespace(
            parameters=np.array([1.0, 2.0]),
            domains=np.array([[0, 1], [3, 4], [6, 7]]),
            provenance={"diagnostics": {"0": {"validation_max_peak_relative": 0.001}}},
        )
        with patch(
            "apogee_drp.utils.resampled_lsf.derive_apstar_lsf", return_value=result
        ):
            tab = build_apstar_lsf_table(
                target,
                ["v1.fits", "v2.fits"],
                np.ones((2, 8)),
                global_weights=np.ones((2, 8)),
                vrel=[0.0, np.nan],
            )
        self.assertEqual(tab["STATUS"][-1], "NO_RV")
        self.assertEqual(tab["NPARS"][-1], 0)
        with self.assertRaises(ValueError):
            evaluate_lsf_table(tab, 3, [0.5], np.arange(-3, 4))


if __name__ == "__main__":
    unittest.main()
