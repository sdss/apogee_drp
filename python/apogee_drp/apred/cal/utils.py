"""Small numerical helpers shared by calibration builders."""

from contextlib import contextmanager
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter
import warnings

from ...utils import lock
from .flatsmooth import flatsmooth
from .robust_slope import robust_slope

__all__ = [
    "average_calibration_frames",
    "calibration_lock",
    "file_build_lock",
    "flatsmooth",
    "product_build_lock",
    "robust_slope",
    "nan_uniform_filter",
    "safe_divide",
    "robust_polyfit",
    "running_nanmedian",
    "interpolate_nonfinite",
    "planck"
]


def average_calibration_frames(frames):
    """Average compatible calibration frames without modifying the inputs."""
    frames = list(frames)
    if not frames:
        raise ValueError("frames must contain at least one frame")

    shape = np.asarray(frames[0]["flux"]).shape
    fluxes = []
    errors = []
    for frame in frames:
        flux = np.asarray(frame["flux"], dtype=float)
        error = np.asarray(frame["err"], dtype=float)
        mask = np.asarray(frame["mask"])
        if flux.shape != shape or error.shape != shape or mask.shape != shape:
            raise ValueError("calibration frames must have matching shapes")
        valid = np.isfinite(flux) & (mask == 0)
        fluxes.append(np.where(valid, flux, np.nan))
        errors.append(error)

    fluxes = np.stack(fluxes)
    errors = np.stack(errors)
    nvalid = np.sum(np.isfinite(fluxes), axis=0)
    flux = np.full(shape, np.nan, dtype=float)
    np.divide(np.nansum(fluxes, axis=0), nvalid, out=flux,
              where=nvalid > 0)

    # Preserve the historical calibration-combination convention.
    error = np.sqrt(np.nanmean(errors**2, axis=0))
    return {
        "flux": flux,
        "err": error,
        "mask": (nvalid == 0).astype(np.uint16),
        "header": frames[0]["header"].copy(),
    }

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


def robust_polyfit(x, y, degree, maxiter=5, clip=5.0):
    """Fit a polynomial while iteratively rejecting outliers.

    Coefficients are returned in increasing order, following
    ``numpy.polynomial.polynomial`` conventions.

    Parameters
    ----------
    x, y : array-like
        Coordinates and values to fit.
    degree : int
        Polynomial degree.
    maxiter : int, optional
        Maximum number of rejection iterations.
    clip : float, optional
        Rejection threshold in units of the median absolute deviation.

    Returns
    -------
    coefficients : numpy.ndarray
        Polynomial coefficients in increasing order.

    Raises
    ------
    ValueError
        If the inputs are invalid or too few finite points remain.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    if isinstance(degree, (bool, np.bool_)) or int(degree) != degree:
        raise ValueError("degree must be a non-negative integer")
    degree = int(degree)

    if degree < 0:
        raise ValueError("degree must be a non-negative integer")

    if isinstance(maxiter, (bool, np.bool_)) or int(maxiter) != maxiter:
        raise ValueError("maxiter must be a non-negative integer")
    maxiter = int(maxiter)

    if maxiter < 0:
        raise ValueError("maxiter must be a non-negative integer")

    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be positive and finite")

    good = np.isfinite(x) & np.isfinite(y)

    if np.count_nonzero(good) <= degree:
        raise ValueError("too few finite points for polynomial fit")

    for _ in range(maxiter):
        coefficients = np.polynomial.polynomial.polyfit(
            x[good], y[good], degree
        )
        model = np.polynomial.polynomial.polyval(x, coefficients)
        residual = y - model

        center = np.nanmedian(residual[good])
        scatter = np.nanmedian(np.abs(residual[good] - center))

        if not np.isfinite(scatter) or scatter == 0:
            break

        keep = good & (np.abs(residual - center) <= clip * scatter)

        if np.count_nonzero(keep) <= degree:
            break

        if np.array_equal(keep, good):
            break

        good = keep

    return np.polynomial.polynomial.polyfit(
        x[good], y[good], degree
    )


def running_nanmedian(values, width, axis=0):
    """Calculate a running NaN-aware median along one axis.

    The array is padded by replicating its edge values, so the returned
    array has the same shape as the input.

    Parameters
    ----------
    values : array-like
        Input data.
    width : int
        Width of the running window.
    axis : int, optional
        Axis along which to calculate the median.

    Returns
    -------
    result : numpy.ndarray
        Running median with the same shape as the input.
    """
    data = np.asarray(values, dtype=float)

    if data.ndim == 0:
        raise ValueError("values must have at least one dimension")

    if isinstance(width, (bool, np.bool_)) or int(width) != width:
        raise ValueError("width must be a positive integer")
    width = int(width)

    if width <= 0:
        raise ValueError("width must be a positive integer")

    axis = np.core.numeric.normalize_axis_index(axis, data.ndim)
    moved = np.moveaxis(data, axis, 0)

    before = width // 2
    after = width - 1 - before

    pad_width = [(0, 0)] * moved.ndim
    pad_width[0] = (before, after)
    padded = np.pad(moved, pad_width, mode="edge")

    shape = (moved.shape[0],) + moved.shape[1:] + (width,)
    strides = padded.strides + (padded.strides[0],)

    windows = np.lib.stride_tricks.as_strided(
        padded,
        shape=shape,
        strides=strides,
        writeable=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmedian(windows, axis=-1)

    return np.moveaxis(result, 0, axis)


def interpolate_nonfinite(values, axis=0):
    """Linearly interpolate nonfinite values along one axis.

    Finite edge values are extended outward. A vector containing no finite
    values is left unchanged.

    Parameters
    ----------
    values : array-like
        Input data.
    axis : int, optional
        Axis along which to interpolate.

    Returns
    -------
    result : numpy.ndarray
        A floating-point copy with nonfinite values interpolated.
    """
    result = np.asarray(values, dtype=float).copy()

    if result.ndim == 0:
        raise ValueError("values must have at least one dimension")

    axis = np.core.numeric.normalize_axis_index(axis, result.ndim)
    moved = np.moveaxis(result, axis, 0)

    original_shape = moved.shape
    columns = moved.reshape(original_shape[0], -1)
    coordinates = np.arange(original_shape[0])

    for column_index in range(columns.shape[1]):
        column = columns[:, column_index]
        good = np.isfinite(column)

        if good.any() and not good.all():
            column[~good] = np.interp(
                coordinates[~good],
                coordinates[good],
                column[good],
            )

    moved = columns.reshape(original_shape)
    return np.moveaxis(moved, 0, axis)


def planck(wavelength, temperature):
    """Return a Planck spectrum per unit wavelength, normalized arbitrarily."""
    wavelength = np.asarray(wavelength, float)
    if temperature <= 0 or np.any(wavelength <= 0):
        raise ValueError("wavelength and temperature must be positive")
    meters = wavelength * 1e-10
    exponent = 1.438776877e-2 / (meters * float(temperature))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        spectrum = 1.0 / (meters ** 5 * np.expm1(exponent))
    return spectrum
