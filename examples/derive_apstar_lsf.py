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
p.add_argument("--degree", type=int, default=3)
p.add_argument("--horder", type=int, default=8)
p.add_argument("--centers", type=int, default=24)
p.add_argument("--fit-h2", action="store_true")
p.add_argument(
    "--native-calibration", help="Native polynomial fitted to measured lines (.npz)"
)
a = p.parse_args()
native = None
if a.native_calibration:
    from apogee_drp.utils.calibration_lsf import NativePolynomialLSF

    native = NativePolynomialLSF.read(a.native_calibration)
start = time.perf_counter()
result = derive_apstar_lsf(
    a.apstar,
    a.visits,
    degree=a.degree,
    horder=a.horder,
    centers_per_chip=a.centers,
    fit_h2=a.fit_h2,
    native_calibration=native,
)
result.provenance["elapsed_seconds"] = time.perf_counter() - start
result.save(a.output)
Path(a.output).with_suffix(".json").write_text(
    json.dumps(result.provenance, indent=2) + "\n"
)
print(json.dumps(result.provenance, indent=2), flush=True)
