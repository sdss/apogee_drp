"""Exercise the production builder without site database/raw-ramp dependencies."""

import ast
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import numpy as np
from astropy.io import fits
from apogee_drp.utils.native_lsf import NativeLSFCalibration


class GlobalBuilderTests(unittest.TestCase):
    def test_default_builder_prepares_joint_native_inputs_and_full_products(self):
        # Execute the real builder source, substituting only the site loader,
        # raw-exposure preparation and product lock. Numerical modules are real.
        path = Path(__file__).parents[1] / "python/apogee_drp/apred/cal/lsf.py"
        tree = ast.parse(path.read_text())
        tree.body = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.ImportFrom)
                and node.module in ("utils", "apload")
                or isinstance(node, ast.ImportFrom)
                and any(a.name == "apload" for a in node.names)
            )
        ]
        namespace = {
            "__name__": "apogee_drp.apred.cal.lsf",
            "__package__": "apogee_drp.apred.cal",
        }
        exec(compile(tree, str(path), "exec"), namespace)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            outputs = [root / f"LSF-{c}.fits" for c in "abcd"]
            wavefiles = []
            for i in range(3):
                p = root / f"wave-{i}.fits"
                wavefiles.append(str(p))
                fits.HDUList(
                    [
                        fits.PrimaryHDU(),
                        fits.ImageHDU(np.zeros((14, 300))),
                        fits.ImageHDU(np.ones((300, 64)) * (15500 + i * 500)),
                    ]
                ).writeto(p)

            def spectrum(frame):
                return {
                    c: dict(
                        header=fits.Header({"DITHPIX": 0.5}),
                        flux=np.ones((64, 300)) * frame,
                        err=np.ones((64, 300)) * 2,
                        mask=np.zeros((64, 300), int),
                    )
                    for c in "abc"
                }

            load = SimpleNamespace(
                spectrum=spectrum,
                filename=lambda *a, **k: wavefiles,
                product_exists=lambda *a: all(p.exists() for p in outputs),
            )

            @contextmanager
            def product_lock(*args, **kwargs):
                yield True, list(map(str, outputs))

            namespace["apload"] = SimpleNamespace(ApLoad=lambda **kwargs: load)
            namespace["product_build_lock"] = product_lock
            seen = {}

            class Calibration:
                provenance = {}

                def write(self, inputs, chip_outputs, **kwargs):
                    seen["outputs"] = chip_outputs
                    seen["write_options"] = kwargs
                    for path in inputs:
                        with fits.open(path) as h:
                            np.testing.assert_array_equal(h[1].data, 3.0)
                            np.testing.assert_allclose(h[2].data, np.sqrt(8.0))
                            self.assert_equal_shape(h[4].data.shape)
                    for p in outputs:
                        fits.PrimaryHDU().writeto(p)

                def plot_diagnostics(self, output):
                    seen["diagnostic_output"] = output

                @staticmethod
                def assert_equal_shape(shape):
                    assert shape == (300, 64)

            cal = Calibration()

            def fit(inputs, **kwargs):
                seen["fit_options"] = kwargs
                return cal

            process = SimpleNamespace(
                process=lambda *a, **kwargs: seen.update(process_options=kwargs)
            )
            with patch.dict(
                sys.modules, {"apogee_drp.apred.process": process}
            ), patch.object(NativeLSFCalibration, "fit", side_effect=fit):
                result = namespace["build_lsf"]([1, 2], 123, psfid=456, fibers=[22])
            self.assertIs(result, cal)
            self.assertEqual(seen["diagnostic_output"], outputs[3].with_name(outputs[3].stem + "-diagnostics.pdf"))
            self.assertEqual(seen["fit_options"]["horder"], 5)
            self.assertEqual(seen["fit_options"]["degree"], 3)
            self.assertEqual(seen["fit_options"]["min_snr"], 15.0)
            self.assertEqual(seen["fit_options"]["fibers"], [22])
            self.assertEqual(seen["write_options"]["global_output"], str(outputs[3]))
            self.assertEqual(seen["process_options"]["waveid"], 123)
            self.assertEqual(cal.provenance["exposures"], [1, 2])


if __name__ == "__main__":
    unittest.main()
