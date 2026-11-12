"""Fit global native LSFs and write standard apLSF calibration products."""

import argparse
from pathlib import Path
from apogee_drp.utils.native_lsf import NativeLSFCalibration

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("files", nargs=3, help="Wavelength-calibrated ap1D a, b, c files")
p.add_argument("--output-directory", required=True)
p.add_argument("--id", required=True, help="Calibration identifier, e.g. 40030031")
p.add_argument("--prefix", default="ap")
p.add_argument("--fibers", help="Physical fiber IDs, comma separated; default all 300")
p.add_argument(
    "--line-catalog", help="DRP airglow.txt catalog; located automatically by default"
)
p.add_argument("--saved-fit", help="Optional old .sav line positions for comparisons")
p.add_argument("--degree", type=int, default=3)
p.add_argument("--coordinate", choices=("pixel", "log_wavelength"), default="pixel")
p.add_argument("--horder", type=int, default=5)
p.add_argument("--min-snr", type=float, default=15.0)
p.add_argument("--strict", action="store_true", help="Abort on any failed fiber")
p.add_argument("--overwrite", action="store_true")
p.add_argument(
    "--no-fill", action="store_true", help="Leave failed fibers explicitly unavailable"
)
p.add_argument(
    "--no-peak-check",
    action="store_true",
    help="Keep filled LSFs entirely neighbor-based",
)
p.add_argument(
    "--diagnostic-directory", help="Save filled-fiber sigma and peak-fit PDFs"
)
a = p.parse_args()
fibers = None if a.fibers is None else [int(s) for s in a.fibers.split(",")]
calibration = NativeLSFCalibration.fit(
    a.files,
    fibers=fibers,
    line_catalog=a.line_catalog,
    saved_fit=a.saved_fit,
    degree=a.degree,
    horder=a.horder,
    min_snr=a.min_snr,
    on_error="raise" if a.strict else "record",
    coordinate=a.coordinate,
    fill_missing=False if a.no_fill else None,
    check_filled_peaks=not a.no_peak_check,
    sigma_options={"diagnostic_directory": a.diagnostic_directory},
)
out = Path(a.output_directory)
files = [out / f"{a.prefix}LSF-{chip}-{a.id}.fits" for chip in "abc"]
calibration.write(
    a.files,
    files,
    global_output=out / f"{a.prefix}LSF-{a.id}.fits",
    overwrite=a.overwrite,
)
print(
    f"{len(calibration.models)-len(calibration.filled_fibers)} fitted; "
    f"{len(calibration.filled_fibers)} filled; "
    f"{sum(f not in calibration.models for f in calibration.status)} unavailable"
)
for file in files:
    print(file)

if a.coordinate == "pixel":
    print(calibration.plot_diagnostics(out / f"{a.prefix}LSF-{a.id}-diagnostics.pdf"))
