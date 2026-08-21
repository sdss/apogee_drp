"""Small numerical helpers shared by calibration builders."""

from contextlib import contextmanager
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter

from ...utils import lock
from .flatsmooth import flatsmooth
from .robust_slope import robust_slope

__all__ = [
    "calibration_lock",
    "file_build_lock",
    "flatsmooth",
    "product_build_lock",
    "robust_slope",
    "nan_uniform_filter",
    "safe_divide"
]

@contextmanager
def calibration_lock(filename, *, waittime=10, unlock=False):
    """Acquire a calibration lock and always clear it afterward."""
    filename = str(filename)

    lock.lock(
        filename,
        waittime=waittime,
        unlock=unlock,
    )
    lock.lock(filename, lock=True)

    try:
        yield
    finally:
        lock.lock(filename, clear=True)

@contextmanager
def file_build_lock(filename, *, clobber=False, unlock=False,
                    waittime=10, verbose=False):
    """Safely prepare one non-product file for construction."""
    path = Path(filename)

    def complete():
        return path.is_file() and path.stat().st_size > 0

    def report_existing():
        if verbose:
            print(f"File {path} already exists")

    # Fast path.
    if complete() and not clobber:
        report_existing()
        yield False
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with calibration_lock(
        path,
        waittime=waittime,
        unlock=unlock,
    ):
        # The file may have been created while waiting for the lock.
        if complete() and not clobber:
            report_existing()
            yield False
            return

        # Remove an incomplete file or a file being clobbered.
        if path.exists() or path.is_symlink():
            path.unlink()
        
        yield True
        
@contextmanager
def product_build_lock(load, product, name, *, clobber=False,
                       unlock=False, waittime=10, verbose=False):
    """Safely prepare a logical calibration product for construction.

    Yields
    ------
    build : bool
        Whether the caller should build the product.
    filenames : list of str
        Physical output files belonging to the product.
    """
    filenames = load.product_files(product, name)

    def report_existing():
        if verbose:
            print(f"{product} product {name} already exists")
    
    # Fast path: avoid acquiring a lock for an existing product.
    if load.product_exists(product, name) and not clobber:
        report_existing()
        yield False, filenames
        return

    lockfile = filenames[0]
    Path(lockfile).parent.mkdir(parents=True, exist_ok=True)

    with calibration_lock(lockfile, waittime=waittime, unlock=unlock):

        # The product may have been created while waiting.
        if load.product_exists(product, name) and not clobber:
            report_existing()
            yield False, filenames
            return

        # Remove an old complete product or partial leftovers.
        load.product_delete(product, name, verbose=verbose)
        
        yield True, filenames


def nan_uniform_filter(array, size):
    """Boxcar-smooth an array while ignoring nonfinite pixels."""
    array = np.asarray(array, dtype=float)
    finite = np.isfinite(array)

    values = uniform_filter(
        np.where(finite, array, 0.0),
        size=size,
        mode="nearest",
    )
    weights = uniform_filter(
        finite.astype(float),
        size=size,
        mode="nearest",
    )

    output = np.full(array.shape, np.nan, dtype=float)
    np.divide(values, weights, out=output, where=weights > 0)
    return output


def safe_divide(numerator, denominator):
    """Divide finite values, returning NaN for invalid divisions."""
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=float),
        np.asarray(denominator, dtype=float),
    )

    output = np.full(numerator.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0)
    )
    np.divide(numerator, denominator, out=output, where=valid)
    return output
