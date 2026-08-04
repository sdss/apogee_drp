from astropy.time import Time
import numpy as np
import os
from scipy.ndimage import median_filter,generic_filter
from pathlib import Path
import subprocess
from typing import Optional

def getmjd5(dateobs):
    """ Convert a DATE-OBS string to 5-digit MJD number."""
    t = Time(dateobs)
    mjd = t.mjd
    # The Julian day starts at NOON, while MJD starts at midnight
    # For SDSS MJD we add 0.3 days
    mjd += 0.3
    # Truncate for MJD5 number
    mjd5 = int(mjd)
    return mjd5

def writelog(logfile,line):
    """ Append lines to a logfile."""
    # Convert to list
    if type(line) is list:
        lines = line
    elif type(line) is str:
        lines = [line]
    else:
        lines = [str(line)]
    # Make sure each line ends in newline
    lines = [l+'\n' if l.endswith('\n')==False else l for l in lines]
    # Append to the file
    with open(logfile,'a') as f:
        f.writelines(lines)

def localdir():
    """ Get local APOGEE directory."""
    local = os.environ['APOGEE_LOCALDIR']
    if local=='':
        return None
    else:
        return local+'/'

def smooth(y, box_pts,boundary='wrap'):
    """ Boxcar smooth a 1-D or 2-D array."""
    if y.ndim==1:
        kernel = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, kernel, mode='same')
    else:
        if np.array(box_pts).size == 0:
            kernel = np.ones([box_pts,box_pts])/box_pts**2
        elif np.array(box_pts).size == 1:
            kernel = np.ones([box_pts[0],box_pts[0]])/box_pts[0]**2	   
        else:
            kernel = np.ones(box_pts)/(box_pts[0]*box_pts[1])
            
        # scipy.signal.convolve2d() does nothing if one of the dimensions
        #  has size=1
        if kernel.shape[0]>1 and kernel.shape[1]>1:
            from scipy.signal import convolve2d
            y_smooth = convolve2d(y,kernel,mode='same',boundary=boundary)
        else:
            width = np.max(np.array(box_pts))
            kernel1 = np.ones(width) / width
            def convfunc(arr1d):
                return np.convolve(arr1d, kernel1, mode='same')
            if kernel.shape[0]==1:
                y_smooth = np.apply_along_axis(convfunc, axis=1, arr=y)
            else:
                y_smooth = np.apply_along_axis(convfunc, axis=0, arr=y)
                
    return y_smooth

def nanmedfilt(x,size,mode='reflect',check=True):
    out = None
    if mode=='edgecopy':
        edgecopy = True
        mode = 'reflect'
    else:
        edgecopy = False
    # 1D median filtering with NaN rejection
    if check:  # check if there are any NaNs in the data
        if np.sum(~np.isfinite(x))==0:
            out = median_filter(x, size, mode=mode)
    # Use nan-median filter
    if out is None:
        out = generic_filter(x, np.nanmedian, size=size, mode=mode)
    # "edgecopy" mode
    if edgecopy:
        # Copy the last "good" median value for the last and first size/2 pixels
        out[:size//2] = out[size//2]
        out[-size//2:] = out[-size//2-1]
    return out


def software_version() -> str:
    """Return the APOGEE DRP Git commit."""

    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / ".git").exists():
            try:
                completed = subprocess.run(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=5,
                )
            except (OSError, subprocess.SubprocessError):
                break
            commit = completed.stdout.strip()
            if commit:
                return commit
    try:
        from importlib.metadata import version

        return version("sdss-apogee-drp")
    except Exception:
        return "unknown"


def reduction_version() -> str:
    """Return the apogee_drp module version"""

    return (
        os.environ.get("APOGEE_DRP_VER")
        or os.environ.get("APRED")
	or "unknown"
    )

def idl_median(
    a: np.ndarray,
    axis: Optional[int] = None,
    *,
    even: bool = False,
) -> np.ndarray:
    """
    Compute an IDL-compatible median while ignoring NaNs.
    IDL returns the upper of the two middle values for an even number of
    samples unless the ``/EVEN`` keyword is supplied. NumPy instead averages
    the middle pair. This distinction matters for the 512-column reference
    output used by AP3D.

    Parameters
    ----------
    a 
        Input values. NaNs are ignored, matching IDL ``MEDIAN`` behavior.
    axis
        Axis along which to compute the median. ``None`` flattens the input.
    even
        Reproduce IDL's ``/EVEN`` keyword by averaging the middle pair.

    Returns
    -------
    numpy.ndarray
        Median values with the selected axis removed.
    """

    values = np.asarray(a)
    if even:
        return np.nanmedian(values, axis=axis)
    if axis is None:
        values = values.ravel()
        axis = 0
    axis = np.core.numeric.normalize_axis_index(axis, values.ndim)
    ordered = np.sort(values, axis=axis)
    finite_count = np.sum(~np.isnan(ordered), axis=axis)
    upper_index = finite_count // 2
    safe_index = np.minimum(
        upper_index, max(ordered.shape[axis] - 1, 0)
    )
    result = np.take_along_axis(
	ordered,
        np.expand_dims(safe_index, axis=axis),
        axis=axis,
    ).squeeze(axis=axis)
    if np.ndim(result) == 0:
        return np.asarray(np.nan if finite_count == 0 else result)
    return np.where(finite_count == 0, np.nan, result)


def idl_median_filter_1d(
    a: np.ndarray,
    width: int,
    *,
    edge_copy: bool,
) -> np.ndarray:
    """
    Apply the one-dimensional IDL median-filter placement convention.

    Filtering is along the final axis.  With ``edge_copy=True``, this matches
    ``MEDFILT1D(..., /EDGE_COPY)`` and ``MEDFILT2D(..., DIM=2,
    /EDGE_COPY)``.  Otherwise, incompletely supported edge values retain
    their unfiltered input values, as in IDL ``MEDIAN(array, width)``.
    """

    values = np.asarray(a)
    width = max(1, int(width))
    if width == 1:
        return values.copy()
    nvalue = values.shape[-1]
    if width > nvalue:
        raise ValueError("width must not exceed the length of the final axis")
    if width == nvalue:
        median = np.nanmedian(values, axis=-1, keepdims=True)
        if edge_copy:
            return np.broadcast_to(median, values.shape).copy()
        return values.astype(
            np.result_type(values.dtype, np.float32), copy=True
        )

    left = width // 2
    right = width - left
    result = values.astype(
        np.result_type(values.dtype, np.float32), copy=True
    )

    leading_size = int(np.prod(values.shape[:-1], dtype=np.int64))
    if nvalue <= 64 and leading_size > 128:
        # Batched masked-array sorting is slower for the CR use case:
        # thousands of pixels but only ~45 read differences. Preserve the
        # faster v23 loop when Numba is unavailable or deliberately bypassed.
        for index in range(left, nvalue - right):
            result[..., index] = np.nanmedian(
                values[..., index - left : index - left + width],
                axis=-1,
            )
    else:
        # Construct every complete window as a view, then reduce the entire
        # stack in one NumPy call. For the 2048-element reference profiles,
        # this removes thousands of small masked-array median calls.
        #
        # The strided view contains nvalue-width+1 windows, whereas MEDFILT1D
        # retains only nvalue-width filtered positions under the APOGEE
        # placement convention. Dropping the final window preserves that
        # historical off-by-one edge behavior exactly.
        window_shape = values.shape[:-1] + (nvalue - width + 1, width)
        window_strides = values.strides + (values.strides[-1],)
        windows = np.lib.stride_tricks.as_strided(
            values,
            shape=window_shape,
            strides=window_strides,
            writeable=False,
        )
        filtered = np.nanmedian(windows[..., :-1, :], axis=-1)
        result[..., left : nvalue - right] = filtered

    if edge_copy:
        result[..., :left] = result[..., left : left + 1]
        result[..., nvalue - right :] = result[
            ..., nvalue - right - 1 : nvalue - right
        ]
    return result

def idl_mean_float(
    a: np.ndarray,
    *,
    axis: Optional[int] = None,
    ignore_nonfinite: bool = False,
) -> np.ndarray:
    """
    Reproduce IDL ``MEAN`` without its ``/DOUBLE`` keyword.

    IDL performs these calculations in single precision for non-double
    inputs. NumPy promotes integer input to float64 in ``mean``/``nanmean``;
    casting that final result to float32 is not equivalent because the
    accumulation has already occurred at higher precision. This helper uses
    ordered float32 additions, matching the reference-pixel means in
    ``aprefcorr.pro``.
    """

    values = np.asarray(a, dtype=np.float32)
    if axis is None:
        values = values.ravel(order="C")
        axis = 0
    else:
        axis = int(axis)
        if axis < 0:
            axis += values.ndim
        if axis < 0 or axis >= values.ndim:
            raise np.AxisError(axis, ndim=values.ndim)
        values = np.moveaxis(values, axis, -1)
        axis = values.ndim - 1

    total = np.zeros(values.shape[:-1], dtype=np.float32)
    count = np.zeros(values.shape[:-1], dtype=np.int32)
    for index in range(values.shape[-1]):
        sample = values[..., index]
        if ignore_nonfinite:
            good = np.isfinite(sample)
            total += np.where(good, sample, np.float32(0.0))
            count += np.asarray(good, dtype=np.int32)
        else:
            total += sample
            count += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        result = total / count.astype(np.float32)
    return np.asarray(result, dtype=np.float32)


def idl_median_filter_2d(a: np.ndarray, size: int) -> np.ndarray:
    """
    Apply IDL's 2-D median filter, preserving unsupported edge pixels.

    IDL evaluates complete ``size`` by ``size`` neighborhoods in the image
    interior and leaves the input values unchanged where a complete
    neighborhood does not fit.  ``scipy.ndimage.median_filter`` supplies the
    interior values; the explicit edge copies reproduce IDL's boundary
    behavior.
    """

    result = median_filter(a, size=(size, size), mode="nearest")
    edge = size // 2
    if edge:
        result[:edge, :] = a[:edge, :]
        result[-edge:, :] = a[-edge:, :]
        result[:, :edge] = a[:, :edge]
        result[:, -edge:] = a[:, -edge:]
    return result


