"""Derive and save an apStar LSF without modifying either input spectrum."""

import argparse
import json
from pathlib import Path
import time
from apogee_drp.utils.apstar_lsf import derive_apstar_lsf

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("apstar")
p.add_argument("visits", nargs="+")
p.add_argument("--output", required=True)
p.add_argument("--degree", type=int, default=4)
p.add_argument("--horder", type=int, default=5)
p.add_argument("--centers", type=int, default=24)
p.add_argument("--fit-h2", action="store_true")
p.add_argument(
    "--polynomial-degrees",
    help="Per-field degrees, comma separated; default H5 mixed orders",
)
p.add_argument(
    "--native-calibration", help="Native polynomial fitted to measured lines (.npz)"
)
a = p.parse_args()
native = None
if a.native_calibration:
    from apogee_drp.utils.calibration_lsf import NativePolynomialLSF

    if Path(a.native_calibration).suffix == ".npz":
        import numpy as np

        with np.load(a.native_calibration, allow_pickle=False) as data:
            pixel_native = int(data["format_version"]) == 2
        if pixel_native:
            from apogee_drp.utils.pixel_lsf import NativePixelLSF

            native = NativePixelLSF.read(a.native_calibration)
        else:
            native = NativePolynomialLSF.read(a.native_calibration)
    else:
        from astropy.io import fits
        from apogee_drp.utils.native_lsf import NativeLSFCalibration

        if len(a.visits) != 1:
            raise ValueError(
                "For multiple visits, embed each native model in its apVisit file"
            )
        fiber = int(fits.getheader(a.visits[0])["FIBERID"])
        native = NativeLSFCalibration.read(a.native_calibration).model(fiber)
degrees = None
if a.polynomial_degrees:
    degrees = [int(s) for s in a.polynomial_degrees.split(",")]
elif a.degree == 4 and a.horder == 5 and not a.fit_h2:
    from apogee_drp.utils.resampled_lsf import H5_NATIVE_PIXEL_DEGREES

    degrees = H5_NATIVE_PIXEL_DEGREES
elif a.degree == 3 and a.horder == 5 and not a.fit_h2:
    from apogee_drp.utils.resampled_lsf import H5_MIXED_DEGREES

    degrees = H5_MIXED_DEGREES
start = time.perf_counter()
result = derive_apstar_lsf(
    a.apstar,
    a.visits,
    degree=a.degree,
    horder=a.horder,
    centers_per_chip=a.centers,
    fit_h2=a.fit_h2,
    native_calibration=native,
    polynomial_degrees=degrees,
)
result.provenance["elapsed_seconds"] = time.perf_counter() - start
result.save(a.output)
Path(a.output).with_suffix(".json").write_text(
    json.dumps(result.provenance, indent=2) + "\n"
)
print(json.dumps(result.provenance, indent=2), flush=True)
