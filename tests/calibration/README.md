# Calibration test suite

Run the self-contained calibration tests with:

```bash
pytest tests/calibration tests/test_cal_translation.py
```

The tests are divided into:

- `test_detector.py`: linearity, Fowler/UTR sampling, read noise, and
  photon-transfer measurements
- `test_index_utils.py`: calibration selection, robust slopes, and flat
  smoothing
- `test_workflows.py`: PSF selection, subprocess commands, and multi-night
  wavelength grouping
- `test_diagnostics.py`: plot and HTML products
- `test_builder_contracts.py`: source-level contracts for every `mk*` builder
- `test_fits_regression.py`: complete FITS product comparisons

## Real IDL-versus-Python products

Place corresponding IDL and Python FITS files in two directories. Filenames
must match. Then run:

```bash
export APOGEE_CAL_IDL_DIR=/path/to/idl/calibrations
export APOGEE_CAL_PYTHON_DIR=/path/to/python/calibrations
pytest -m regression tests/calibration/test_fits_regression.py
```

The comparison checks:

- the number of HDUs
- data presence and array shape in every extension
- matching NaN locations
- every finite pixel, with exact equality by default
- stable FITS headers, when `compare_stable_headers` is requested

The regression helper accepts explicit `rtol` and `atol` values for
diagnostics, but production acceptance should normally retain the zero
tolerances.

