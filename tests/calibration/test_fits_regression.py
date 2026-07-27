"""Tests and opt-in regression checks for complete calibration products."""

import os
from pathlib import Path

import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")

from .fits_compare import compare_fits_products, compare_stable_headers


def _write(path, arrays, *, object_name="calibration"):
    hdus = [fits.PrimaryHDU(header=fits.Header({"OBJECT": object_name}))]
    hdus.extend(fits.ImageHDU(array) for array in arrays)
    fits.HDUList(hdus).writeto(path)


def test_compare_fits_products_exact_match(tmp_path):
    arrays = [
        np.arange(30, dtype=np.float32).reshape(5, 6),
        np.arange(12, dtype=np.int16).reshape(3, 4),
    ]
    first, second = tmp_path / "idl.fits", tmp_path / "python.fits"
    _write(first, arrays)
    _write(second, arrays)
    results = compare_fits_products(first, second)
    assert len(results) == 2
    assert all(result.max_abs_difference == 0 for result in results)


def test_compare_fits_products_reports_pixel_difference(tmp_path):
    first, second = tmp_path / "idl.fits", tmp_path / "python.fits"
    _write(first, [np.zeros((4, 4), np.float32)])
    changed = np.zeros((4, 4), np.float32)
    changed[2, 3] = 1e-3
    _write(second, [changed])
    with pytest.raises(AssertionError, match="1 pixels differ"):
        compare_fits_products(first, second)


def test_compare_fits_products_optional_tolerance(tmp_path):
    first, second = tmp_path / "idl.fits", tmp_path / "python.fits"
    _write(first, [np.ones((4, 4), np.float64)])
    _write(second, [np.ones((4, 4), np.float64) + 1e-10])
    compare_fits_products(first, second, atol=2e-10)


def test_compare_fits_products_requires_same_shape(tmp_path):
    first, second = tmp_path / "idl.fits", tmp_path / "python.fits"
    _write(first, [np.zeros((4, 4))])
    _write(second, [np.zeros((4, 5))])
    with pytest.raises(AssertionError, match="shape differs"):
        compare_fits_products(first, second)


def test_compare_fits_products_requires_matching_nan_locations(tmp_path):
    first, second = tmp_path / "idl.fits", tmp_path / "python.fits"
    a, b = np.zeros((4, 4)), np.zeros((4, 4))
    a[0, 0], b[1, 1] = np.nan, np.nan
    _write(first, [a])
    _write(second, [b])
    with pytest.raises(AssertionError, match="NaN locations differ"):
        compare_fits_products(first, second)


def test_compare_stable_headers_ignores_provenance(tmp_path):
    first, second = tmp_path / "idl.fits", tmp_path / "python.fits"
    _write(first, [np.zeros(2)])
    _write(second, [np.zeros(2)])
    with fits.open(first, mode="update") as hdu:
        hdu[0].header["DATE"] = "2020-01-01"
        hdu[0].header["HISTORY"] = "IDL"
    with fits.open(second, mode="update") as hdu:
        hdu[0].header["DATE"] = "2030-01-01"
        hdu[0].header["HISTORY"] = "Python"
    compare_stable_headers(first, second)


@pytest.mark.regression
def test_real_idl_python_calibration_tree():
    """Compare externally generated products when both directories are set."""
    idl_dir = os.environ.get("APOGEE_CAL_IDL_DIR")
    python_dir = os.environ.get("APOGEE_CAL_PYTHON_DIR")
    if not idl_dir or not python_dir:
        pytest.skip(
            "set APOGEE_CAL_IDL_DIR and APOGEE_CAL_PYTHON_DIR "
            "to run real product comparisons"
        )
    idl_dir, python_dir = Path(idl_dir), Path(python_dir)
    idl_files = sorted(idl_dir.glob("*.fits"))
    assert idl_files, f"no FITS files found in {idl_dir}"
    for idl_file in idl_files:
        python_file = python_dir / idl_file.name
        assert python_file.exists(), f"missing Python product {python_file.name}"
        compare_fits_products(idl_file, python_file)

