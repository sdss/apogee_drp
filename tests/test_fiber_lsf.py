"""Tests for coefficient alignment, missing-fiber estimation and provenance."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import os
from unittest.mock import patch
import numpy as np
from numpy.polynomial import Chebyshev
from astropy.io import fits
from astropy.table import Table
from apogee_drp.utils.pixel_lsf import NativePixelLSF
from apogee_drp.utils.native_lsf import NativeLSFCalibration
from apogee_drp.utils.fiber_lsf import interpolate_fiber_model, fiber_weights
from apogee_drp.utils.resampled_lsf import build_apstar_lsf_table


def model(fiber, origin=0.0, scale=4096.0):
    physical = np.array(
        [
            [1.2 + 0.01 * fiber, 0.03, 0.001],
            [0.02 + 0.0002 * fiber, 0.002, 0],
            [-0.01, 0.001, 0],
            [0.005, 0, 0],
            [0.04, 0, 0],
            [4.0 + 0.02 * fiber, 0.1, 0.002],
        ]
    )
    coefficients = np.zeros_like(physical)
    for i, row in enumerate(physical):
        co = (
            Chebyshev(row, domain=[-4096, 4096])
            .convert(domain=[origin - scale, origin + scale])
            .coef
        )
        coefficients[i, : len(co)] = co
    offsets = np.array([-3200.0, -1000.0, 1200.0]) + fiber * 0.01
    return NativePixelLSF(
        coefficients, origin, scale, 5, {"fiber": fiber}, offsets, 2048
    )


class FiberFillTests(unittest.TestCase):
    def test_linear_interpolation_aligns_different_polynomial_bases(self):
        donors = {10: model(10, 100.0, 3100.0), 12: model(12, -150.0, 3300.0)}
        before = [m.coefficients.copy() for m in donors.values()]
        prediction = interpolate_fiber_model(donors, 11, "linear")
        true = model(11)
        x = np.linspace(-3200, 3248, 100)
        np.testing.assert_allclose(prediction.fields(x), true.fields(x), atol=3e-14)
        np.testing.assert_allclose(
            prediction.chip_offsets, true.chip_offsets, atol=1e-12
        )
        for m, co in zip(donors.values(), before):
            np.testing.assert_array_equal(m.coefficients, co)
        self.assertEqual(prediction.diagnostics["source_fibers"], [10, 12])

    def test_local_linear_recovers_linear_fiber_trend(self):
        donors = {f: model(f, 100 * f, 3000 + f) for f in [8, 9, 10, 12, 13, 14]}
        prediction = interpolate_fiber_model(donors, 11, "local_linear", 5)
        np.testing.assert_allclose(
            prediction.fields(np.arange(-3000, 3000, 150)),
            model(11).fields(np.arange(-3000, 3000, 150)),
            atol=2e-14,
        )
        self.assertEqual(len(prediction.diagnostics["source_fibers"]), 5)

    def test_consecutive_missing_fibers_and_repeated_calls_do_not_cascade(self):
        cal = NativeLSFCalibration(
            {10: model(10), 13: model(13)},
            {10: "OK", 11: "bad lines", 12: "bad lines", 13: "OK"},
        )
        measured = [cal.models[10], cal.models[13]]
        cal.fill_missing([11, 12], method="linear")
        self.assertEqual(cal.filled_fibers, [11, 12])
        self.assertEqual(cal.models[12].diagnostics["source_fibers"], [10, 13])
        cal.fill_missing([14], method="local_linear")
        self.assertEqual(cal.models[14].diagnostics["source_fibers"], [13])
        self.assertEqual(cal.models[14].diagnostics["fill_method"], "nearest")
        self.assertIs(cal.models[10], measured[0])
        self.assertIs(cal.models[13], measured[1])
        self.assertEqual(cal.models[11].diagnostics["original_failure"], "bad lines")

    def test_fitted_models_are_not_overwritten_and_bad_method_is_rejected(self):
        m = model(10)
        cal = NativeLSFCalibration({10: m})
        cal.fill_missing([10])
        self.assertIs(cal.models[10], m)
        with self.assertRaises(ValueError):
            cal.fill_missing([11], method="invalid")
        with self.assertRaises(ValueError):
            NativeLSFCalibration({}).fill_missing([11])

    def test_fill_flags_and_original_failure_roundtrip(self):
        cal = NativeLSFCalibration(
            {10: model(10), 12: model(12)}, {10: "OK", 11: "no data", 12: "OK"}
        )
        cal.fill_missing([11])
        self.assertFalse(cal.global_hdu([10]).header["LSFFILL"])
        self.assertTrue(cal.global_hdu([11]).header["LSFFILL"])
        with TemporaryDirectory() as temp:
            path = Path(temp) / "cal.fits"
            fits.HDUList([fits.PrimaryHDU(), cal.global_hdu()]).writeto(path)
            restored = NativeLSFCalibration.read(path)
        self.assertEqual(restored.filled_fibers, [11])
        self.assertEqual(restored.model(11).diagnostics["original_failure"], "no data")
        self.assertEqual(restored.model(11).diagnostics["source_fibers"], [10, 12])

    def test_fit_fills_only_requested_failed_fibers_and_can_disable_filling(self):
        def load(files, saved_fit, *, fiber, **kwargs):
            if fiber == 11:
                raise ValueError("no clean lines")
            return SimpleNamespace(provenance={"fiber": fiber})

        def fit(lines, files, fiber, **kwargs):
            return model(fiber), lines

        with patch(
            "apogee_drp.utils.native_lsf.load_calibration_windows", side_effect=load
        ), patch(
            "apogee_drp.utils.native_lsf.fit_native_pixel", side_effect=fit
        ), patch(
            "apogee_drp.utils.native_lsf.evaluate_held_out", return_value=([], [], None)
        ):
            c = NativeLSFCalibration.fit(
                ["a", "b", "c"], line_catalog=Table(), fibers=[10, 11, 12]
            )
            self.assertEqual(sorted(c.models), [10, 11, 12])
            self.assertEqual(c.filled_fibers, [11])
            unfilled = NativeLSFCalibration.fit(
                ["a", "b", "c"],
                line_catalog=Table(),
                fibers=[10, 11, 12],
                fill_missing=False,
            )
            self.assertNotIn(11, unfilled.models)

    @unittest.skipUnless(os.environ.get("APOGEE_CALIBRATION_DIR"), "Local calibration data not configured")
    def test_fixed_model_validation_can_use_fewer_lines_without_relaxing_default_fit(self):
        from apogee_drp.utils.calibration_lsf import load_calibration_windows
        from apogee_drp.utils.native_lsf import default_line_catalog
        root=Path(os.environ["APOGEE_CALIBRATION_DIR"])
        files=[root/f"ap1D-{chip}-40030031.fits" for chip in "abc"]
        with self.assertRaisesRegex(ValueError,"Too few"):
            load_calibration_windows(files,fiber=37,line_catalog=default_line_catalog())
        windows=load_calibration_windows(files,fiber=37,line_catalog=default_line_catalog(),min_lines_per_chip=0)
        self.assertEqual(len(windows.centers),28)
        self.assertEqual(np.count_nonzero(windows.chip==1),5)

    def test_resampled_table_preserves_estimated_native_flag(self):
        target = SimpleNamespace(
            header=fits.Header({"VREL1": 0.0}),
            wave=np.arange(8.0),
            flux=np.zeros((1, 8)),
        )
        result = SimpleNamespace(
            parameters=np.array([1.0, 2.0]),
            domains=np.array([[0, 1], [3, 4], [6, 7]]),
            provenance={
                "diagnostics": {"0": {"validation_max_peak_relative": 0.001}},
                "native_calibration_diagnostics": [{"filled": True}],
            },
        )
        with patch(
            "apogee_drp.utils.resampled_lsf.derive_apstar_lsf", return_value=result
        ):
            table = build_apstar_lsf_table(target, ["visit.fits"], np.ones((1, 8)))
        self.assertTrue(table["NATIVE_FILLED"][0])
        self.assertEqual(table["STATUS"][0], "OK")


if __name__ == "__main__":
    unittest.main()
