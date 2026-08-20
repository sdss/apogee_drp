"""Small numerical helpers shared by calibration builders."""

from contextlib import contextmanager
from pathlib import Path

from ...utils import lock
from .flatsmooth import flatsmooth
from .robust_slope import robust_slope

__all__ = [
    "calibration_lock",
    "file_build_lock",
    "flatsmooth",
    "product_build_lock",
    "robust_slope",
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
