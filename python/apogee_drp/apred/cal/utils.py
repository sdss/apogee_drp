"""Small numerical helpers shared by calibration builders."""

from .flatsmooth import flatsmooth
from .robust_slope import robust_slope
from contextlib import contextmanager
from ...utils import lock

__all__ = ["calibration_lock","flatsmooth", "robust_slope"]

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
