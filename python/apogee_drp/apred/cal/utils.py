"""Small numerical helpers shared by calibration builders."""

from .flatsmooth import flatsmooth
from .robust_slope import robust_slope
from contextlib import contextmanager
from ...utils import lock

__all__ = ["calibration_lock","product_build_lock"]

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

    # Fast path: avoid acquiring a lock for an existing product.
    if load.product_exists(product, name) and not clobber:
        yield False, filenames
        return

    lockfile = filenames[0]
    lock.lock(
        lockfile,
        waittime=waittime,
        unlock=unlock,
    )
    lock.lock(lockfile, lock=True)

    try:
        # The product may have been created while waiting.
        if load.product_exists(product, name) and not clobber:
            yield False, filenames
            return

        # Remove an old complete product or partial leftovers.
        load.product_delete(
            product,
            name,
            verbose=verbose,
        )

        yield True, filenames

    finally:
        lock.lock(lockfile, clear=True)
