"""Reusable IDL-versus-Python FITS comparison helpers."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ExtensionComparison:
    extension: int
    shape: tuple
    dtype_idl: str
    dtype_python: str
    finite_pixels: int
    exact_pixels: int
    max_abs_difference: float
    rms_difference: float


def compare_fits_products(idl_file, python_file, *, rtol=0.0, atol=0.0):
    """Compare matching image extensions in two FITS products.

    The defaults require machine-exact equality.  Header comparison is kept
    separate because provenance timestamps and software-version cards are
    expected to differ.
    """
    from astropy.io import fits

    idl_file, python_file = Path(idl_file), Path(python_file)
    comparisons = []
    with fits.open(idl_file, memmap=False) as idl_hdu, fits.open(
        python_file, memmap=False
    ) as python_hdu:
        if len(idl_hdu) != len(python_hdu):
            raise AssertionError(
                f"HDU count differs: IDL={len(idl_hdu)}, "
                f"Python={len(python_hdu)}"
            )
        for extension, (idl_ext, python_ext) in enumerate(
            zip(idl_hdu, python_hdu)
        ):
            if idl_ext.data is None and python_ext.data is None:
                continue
            if (idl_ext.data is None) != (python_ext.data is None):
                raise AssertionError(f"extension {extension}: data presence differs")
            idl_data = np.asarray(idl_ext.data)
            python_data = np.asarray(python_ext.data)
            if idl_data.shape != python_data.shape:
                raise AssertionError(
                    f"extension {extension}: shape differs "
                    f"{idl_data.shape} != {python_data.shape}"
                )
            finite = np.isfinite(idl_data) & np.isfinite(python_data)
            nan_match = np.isnan(idl_data) == np.isnan(python_data)
            if not np.all(nan_match):
                raise AssertionError(
                    f"extension {extension}: NaN locations differ"
                )
            difference = python_data[finite].astype(float) - idl_data[
                finite
            ].astype(float)
            close = np.isclose(
                python_data[finite], idl_data[finite],
                rtol=rtol, atol=atol, equal_nan=True,
            )
            comparisons.append(
                ExtensionComparison(
                    extension=extension,
                    shape=idl_data.shape,
                    dtype_idl=str(idl_data.dtype),
                    dtype_python=str(python_data.dtype),
                    finite_pixels=int(finite.sum()),
                    exact_pixels=int(close.sum()),
                    max_abs_difference=(
                        float(np.max(np.abs(difference)))
                        if difference.size else 0.0
                    ),
                    rms_difference=(
                        float(np.sqrt(np.mean(difference**2)))
                        if difference.size else 0.0
                    ),
                )
            )
            if not np.all(close):
                count = int((~close).sum())
                raise AssertionError(
                    f"extension {extension}: {count} pixels differ; "
                    f"max_abs={comparisons[-1].max_abs_difference:g}, "
                    f"rms={comparisons[-1].rms_difference:g}"
                )
    return comparisons


def compare_stable_headers(idl_file, python_file, *, ignored=()):
    """Compare non-provenance FITS cards across all matching HDUs."""
    from astropy.io import fits

    ignored = {
        "CHECKSUM", "DATASUM", "DATE", "HISTORY", "COMMENT", *ignored
    }
    with fits.open(idl_file) as idl_hdu, fits.open(python_file) as python_hdu:
        if len(idl_hdu) != len(python_hdu):
            raise AssertionError("HDU count differs")
        for extension, (idl_ext, python_ext) in enumerate(
            zip(idl_hdu, python_hdu)
        ):
            keys = (set(idl_ext.header) | set(python_ext.header)) - ignored
            for key in keys:
                if idl_ext.header.get(key) != python_ext.header.get(key):
                    raise AssertionError(
                        f"extension {extension}, {key}: "
                        f"{idl_ext.header.get(key)!r} != "
                        f"{python_ext.header.get(key)!r}"
                    )

