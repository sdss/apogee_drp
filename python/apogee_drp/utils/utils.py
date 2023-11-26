from astropy.time import Time
import numpy as np
import os
from scipy.ndimage import median_filter,generic_filter

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

def median(data,axis=None,even=False,high=True,nan=False):
    """
    Return the median of the data.
    This is similar to the numpy version, but it
    does NOT average the central two values if there are
    an even number of elements.

    Parameters
    ----------
    data : numpy array
       The data array to take the median of.
    axis : int, optional
       Take the median along this axis.
    even : bool, optional
       Return the average of the two central values if there
         are an even number of elements.  Default is False.
    high : bool, optional
       If not averaging the two central values, then take
         the higher value.  Default high is True.
    nan : bool, optional
       Ignore NaNs.  Default is False.

    Returns
    -------
    med : float or numpy array
       The median of the data.

    Example
    -------

    med = median(data,axis=0)
    
    By D. Nidever  Nov 2023
    """

    # No axis
    if axis is None:
        iseven = data.size % 2 == 0
    # Along axis
    else:
        iseven = data.shape[axis] % 2 == 0
        
    # Even selected or odd number of elements
    #  use normal numpy median()
    if even or iseven==False:
        if nan:
            return np.nanmedian(data,axis=axis)
        else:
            return np.median(data,axis=axis)

    # Calculate median with no averaging of central
    # two elements.  Use argsort() to do this

    # Ignore the NaNs
    #  np.argsort() puts NaNs at the end of the list
    #  Use np.sum(np.isfinite()) to get the number of
    #  finite points and adjust the indexing accordingly
    
    # No axis
    if axis is None:
        si = np.argsort(data.ravel())        
        npts = len(si)
        if nan:
            npts = np.sum(np.isfinite(data))
        half = npts // 2
        # Pick low or high point of the two middle values        
        if high:
            midind = half
        else:
            midind = half-1
        index = si[midind]
        med = data.ravel()[index]

    # Along axis
    else:
        si = np.argsort(data,axis=axis)
        if nan:
            npts = np.sum(np.isfinite(data),axis=axis)
        else:
            npts = data.shape[axis]
        half = npts // 2
        # Pick low or high point of the two middle values
        if high:
            midind = half
        else:
            midind = half-1
        # Use slice object
        slc = [slice(None)]*data.ndim   # one slice object per dimension
        slc[axis] = midind
        slc = tuple(slc)
        index = si[slc]
        # Add dimension
        newshape = list(data.shape)
        newshape[axis] = 1
        index = index.reshape(newshape)
        med = np.take_along_axis(data,index,axis=axis)
        # Remove extra axis
        newshape = list(data.shape)
        del newshape[axis]
        med = med.reshape(newshape)

    return med
