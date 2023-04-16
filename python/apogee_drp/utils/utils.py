from astropy.time import Time
import numpy as np
import os

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
