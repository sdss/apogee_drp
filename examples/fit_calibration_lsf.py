"""Fit one polynomial LSF to observed sky lines across all three detectors."""

import argparse, json
from pathlib import Path
from apogee_drp.utils.calibration_lsf import (
    load_calibration_windows,
    fit_native_polynomial,
    evaluate_held_out,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("directory", type=Path)
p.add_argument("--exposure", default="40030031")
p.add_argument("--fiber", type=int, default=22)
p.add_argument("--degree", type=int, default=3)
p.add_argument("--horder", type=int, default=5)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()


def locate(name):
    path = a.directory / name
    if path.exists():
        return path
    if Path(str(path) + ".gz").exists():
        return Path(str(path) + ".gz")
    raise FileNotFoundError(path)


files = [locate(f"ap1D-{chip}-{a.exposure}.fits") for chip in "abc"]
lines = load_calibration_windows(
    files, locate(f"apLSF-{a.exposure}.sav"), fiber=a.fiber
)
result = fit_native_polynomial(lines, degree=a.degree, horder=a.horder)
metrics, pred, data = evaluate_held_out(result, lines)
result.diagnostics.update(
    input_provenance=lines.provenance, validation=metrics, status="EXPERIMENTAL"
)
result.save(a.output)
a.output.with_suffix(".json").write_text(
    json.dumps(result.diagnostics, indent=2) + "\n"
)
print(json.dumps(result.diagnostics, indent=2))
