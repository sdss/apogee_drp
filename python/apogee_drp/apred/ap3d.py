#!/usr/bin/env python

"""AP3D.PY - APOGEE software to process 3D data cubes.

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20200404'  # yyyymmdd

import os
import sys
from pwd import getpwuid
import numpy as np
import warnings
import time
from astropy.io import fits
from astropy.table import Table, Column
from astropy.time import Time
from astropy import modeling
from glob import glob
from scipy.signal import medfilt,medfilt2d
from scipy.ndimage.filters import median_filter,gaussian_filter1d
from scipy.optimize import curve_fit, least_squares
from scipy.special import erf
from scipy.interpolate import interp1d
#from numpy.polynomial import polynomial as poly
#from lmfit import Model
#from apogee.utils import yanny, apload
#from sdss_access.path import path
import traceback
from dlnpyutils import utils as dln,bindata
from ..utils import plan,apload,utils,apzip,bitmask,lock
from . import mjdcube

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

pixelmask = bitmask.PixelBitMask()
BADERR = 1.0000000e+10

def loaddata(filename,fitsdir=None,maxread=None,cleanuprawfile=True,
             verbose=False):
    """
    Load the input data file.

    Parameters
    ----------
    filename : str
       Data filename.
    fitsdir : str, optional
       Directory for the temporary, uncompressed FITS images.
    maxread : int, optional
       Maximum read to use.  Default is to use all reads.
    cleanuprawfile : boolean, optional
       If a compressed file is input and ap3dproc needs to
         Decompress the file (no decompressed file is on disk)
         then setting this keyword will delete the decompressed
         file at the very end of processing.  This is the default.
         Set cleanuprawfile=0 if you want to keep the decompressed
         file.  An input decompressed FITS file will always
         be kept.
    verbose : boolean, optional
       Verbose output to the screen.  Default is False.

    Returns
    -------
    cube : numpy array
       Input data cube.
    header : header
       Header for the data file.
    doapunzip : boolean
       Boolean flag whether the apz file was uncompressed.
    fitsfile : str
       Fits filename used.

    Examples
    --------

    cube,head,doapunzip,fitsfile = loaddata(filename)

    """
       
    # Check the file
    fdir = os.path.dirname(filename)
    base = os.path.basename(filename)
    nbase = len(base)
    basesplit = os.path.splitext(base)
    extension = basesplit[-1][1:]
    #if strmid(base,0,4) != 'apR-' or strmid(base,len-5,5) != '.fits':
    #  error = 'FILE must be of the form >>apR-a/b/c-XXXXXXXX.fits<<'
    #  if silent==False then print(error)
    #  return

    # Wrong extension
    if extension != 'fits' and extension != 'apz':
        error = 'FILE must have a ".fits" or ".apz" extension'
        if verbose:
            print(error)
        return None,None,False,''
  
    # Compressed file input
    if extension == 'apz':
        if verbose:
            print(filename,' is a COMPRESSED file')

        # Check if the decompressed file already exists
        nbase = len(base)
        if fitsdir is not None:
            fitsfile = os.path.join(fitsdir,base[:nbase-4]+'.fits')
        else:
            fitsfile = os.path.join(fdir,base[:nbase-4]+'.fits')
            fitsdir = None
  
        # Need to decompress
        if os.path.exists(fitsfile)==False:
            if verbose:
                print('Decompressing with APUNZIP')
            num = int(base[6:6+8])
            if num < 2490000:
                no_checksum = 1
            else:
                no_checksum = 0 
            print('no_checksum: ', no_checksum)
            try:
                apzip.unzip(filename,clobber=True,fitsdir=fitsdir,no_checksum=True)
            except:
                traceback.print_exc()
                print('ERROR in APUNZIP')
                return None,None,False,fitsfile
            print('')
            doapunzip = True     # we ran apunzip

        # Decompressed file already exists
        else:
            if verbose:
                print('The decompressed file already exists')
            doapunzip = False     # we didn't run apunzip

        # Cleanup by default
        if cleanuprawfile==False and extension=='apz' and doapunzip:   # remove recently decompressed file
            cleanuprawfile = True

    # Regular FITS file input
    else:
        fitsfile = filename
        doapunzip = False
 
    if verbose:
        if extension == 'apz' and cleanuprawfile and doapunzip == 1:
            print('Removing recently decompressed FITS file at end of processing')

    # Check that the file exists
    if os.path.exists(fitsfile)==False:
        error = 'FILE '+fitsfile+' NOT FOUND'
        if verbose:
            print('error')
        return None,None,False,fitsfile
 
    # Get header
    try:
        head = fits.getheader(fitsfile)
    except:
        error = 'There was an error loading the HEADER for '+fitsfile
        if verbose:
            print(error)
        return None,None,False,fitsfile
  
    # Check that this is a data CUBE
    naxis = head['NAXIS']
    try:
        dumim,dumhead = fits.getdata(fitsfile,1,header=True)
        readokay = True
    except:
        readokay = False
    if naxis != 3 and readokay==False:
        error = 'FILE must contain a 3D DATACUBE OR image extensions'
        if verbose:
            print(error)
        return None,None,False,fitsfile

    # Check that the file exists
    if os.path.exists(fitsfile)==False:
        error = ifile+' NOT FOUND'
        if verbose:
            print(error)
        return None,None,False,fitsfile

    # Read in the File
    #-------------------
    
    # DATACUBE
    if naxis==3:
        cube,head = fits.getdata(fitsfile,header=True)  # uint
    # Extensions
    else:
        head = fits.getheader(fitsfile)
        # Figure out how many reads/extensions there are
        #  the primary unit should be empty
        hdu = fits.open(fitsfile)
        nreads = len(hdu)-1
  
        # Only 1 read
        if nreads < 2:
            error = 'ONLY 1 read.  Need at least two'
            print(error)
            return None,None,False,fitsfile

        # allow user to specify maximum number of reads to use (e.g., in the
        #   case of calibration exposures that may be overexposed in some chip
        if maxread:
            if maxread < nreads:
                nreads = maxread
  
        # Initializing the cube
        im1 = hdu[1].data
        ny,nx = im1.shape
        cube = np.zeros((ny,nx,nreads),int)    # long is big enough and takes up less memory than float
  
        # Read in the extensions
        for k in np.arange(1,nreads+1):
            cube[:,:,k-1] = hdu[k].data
            # What do we do with the extension headers???
            # We could make a header structure or array
        hdu.close()

    return cube,head,doapunzip,fitsfile


def loaddetector(detcorr,verbose=False):
    """
    Load detector file (with gain, rednoise and linearity correction).

    Parameters
    ----------
    detcorr : str
       Filename of the detector file.
    verbose : bool, optional
       Verbose output to the screen.  Default is False.

    Returns
    -------
    rdnoiseim : numpy array
       Readnoise image/array.
    gainim : numpy array
       Gain image/array.
    lindata : numpy array
       Linearity data.

    Examples
    --------

    rdnoiseim,gainim,lindata = loaddetector(detcorr)

    """
    
    # DETCORR must be scalar string
    if (type(detcorr) != str) | (dln.size(detcorr) > 1):
        raise ValueError('DETCORR must be a scalar string with the filename of the DETECTOR file')
    # Check that the file exists
    if os.path.exists(detcorr) is False:
        raise ValueError('DETCORR file '+detcorr+' NOT FOUND')
    
    # Load the DET file
    #  This should be 2048x2048
    dethead = fits.getheader(detcorr,0)
    rdnoiseim,noisehead = fits.getdata(detcorr,1,header=True)
    gainim,gainhead = fits.getdata(detcorr,2,header=True)
    #  This should be 2048x2048x3 (each pixel) or 4x3 (each output),
    #  where the 2 is for a quadratic polynomial
    lindata,linhead = fits.getdata(detcorr,3,header=True)
    if verbose:
        print('DET file = '+detcorr)

  
    # Check that the file looks reasonable
    # Must be 2048x2048 or 4 and have be float
    if ((gainim.ndim==2) & (gainim.shape != (2048,2048))) | ((gainim.ndim==1) & (gainim.size != 4)) | \
        (isinstance(gainim[0], (np.floating, float))==False):
        raise ValueError('GAIN image must be 2048x2048 or 4 FLOAT image')
  
    # If Gain is 4-element then make it an array
    if gainim.size == 4:
        gainim0 = gainim.copy()
        gainim = np.zeros((2048,2048),float)
        for k in range(4):
            gainim[:,k*512:(k+1)*512] = gainim0[k]
  
    # Must be 2048x2048 or 4 and have be float
    if ((rdnoiseim.ndim==2) & (rdnoiseim.shape != (2048,2048))) | ((rdnoiseim.ndim==1) & (rdnoiseim.size != 4)) | \
        (isinstance(rdnoiseim[0], (np.floating, float))==False):
        raise ValueError('RDNOISE image must be 2048x2048 or 4 FLOAT image')
  
    # If rdnoise is 4-element then make it an array
    if rdnoiseim.size == 4:
        rdnoiseim0 = rdnoiseim.copy()
        rdnoiseim = np.zeros((2048,2048),float)
        for k in range(4):
            rdnoiseim[:,k*512:(k+1)*512] = rdnoiseim0[k]
        
                
        # Check that the file looks reasonable
        #  This should be 2048x2048x3 (each pixel) or 4x3 (each output),
        #  where the 3 is for a quadratic polynomial
        
        szlin = lindata.shape
        linokay = 0
        if (lindata.ndim == 2 and szlin[0] == 3 and szlin[1] == 4):
            linokay = 1
            lindata = lindata.T  # flip
        if (lindata.ndim == 3 and szlin[0] == 2048 and szlin[1] == 2048 and szlin[2] == 3):
            linokay = 1
        if linokay==0:
            raise ValueError('Linearity correction data must be 2048x2048x3 or 4x3')

    return rdnoiseim,gainim,lindata
  

def loadbpm(bpmcorr,verbose=False):
    """
    Load bad pixel mask (BPM) file.

    Parameters
    ----------
    bpmcorr : str
       Filename of the bad pixel mask file
    verbose : bool, optional
       Verbose output to the screen.  Default is False.

    Returns
    -------
    bpmim : numpy array
       Bad pixel mask image.
    bpmhead : Header
       Header for the bpm file.

    Examples
    --------

    bpmim,bpmhead = loadbpm(bpmcorr)

    """
    
    # BPMCORR must be scalar string
    if (type(bpmcorr) != str) | (dln.size(bpmcorr) > 1):
        raise ValueError('BPMCORR must be a scalar string with the filename of the BAD PIXEL MASK file')
    # Check that the file exists
    if os.path.exists(bpmcorr) is False:
        raise ValueError('BPMCORR file '+bpmcorr+' NOT FOUND')
    
    # Load the BPM file
    #  This should be 2048x2048
    bpmim,bpmhead = fits.getdata(bpmcorr,header=True)
    if verbose:
        print('BPM file = '+bpmcorr)
 
    # Check that the file looks reasonable
    #  must be 2048x2048 and have 0/1 values
    ny,nx = bpmim.shape
    bpmokay = 0
    if (bpmim.ndim != 2) | (nx != 2048) | (ny != 2048):
        raise ValueError('BAD PIXEL MASK must be 2048x2048 with 0/1 values')

    return bpmim,bpmhead


def loadlittrow(littrowcorr,verbose=False):
    """
    Load littrow mask file.

    Parameters
    ----------
    littrowcorr : str
       Filename of the littrow mask file
    verbose : bool, optional
       Verbose output to the screen.  Default is False.

    Returns
    -------
    littrowim : numpy array
       Littrow mask image.
    littrowhead : Header
       Header for the littrow mask file.

    Examples
    --------

    littrowim,littrowhead = loadlittrow(littrowcorr)

    """

    # LITTROWCORR must be scalar string
    if type(littrowcorr) is not str or np.atleast_1d(littrowcorr).size != 1:
        error = 'LITTROWCORR must be a scalar string with the filename of the LITTROW MASK file'
        raise ValueError(error)
  
    # Check that the file exists
    if os.path.exists(littrowcorr)==False:
        error = 'LITTROWCORR file '+littrowcorr+' NOT FOUND'
        raise ValueError(error)
    
    # Load the LITTROW file
    #  This should be 2048x2048
    littrowim,littrowhead = fits.getdata(littrowcorr,header=True)
  
    if verbose:
        print('LITTROW file = '+littrowcorr)
  
    # Check that the file looks reasonable
    #  must be 2048x2048 and have 0/1 values
    nyl,nxl = littrowim.shape
    nbad = np.sum((littrowim != 0) & (littrowim != 1))
    if littrowim.ndim != 2 or nxl != 2048 or nyl != 2048 or nbad > 0:
        error = 'LITTROW MASK must be 2048x2048 with 0/1 values'
        raise ValueError(error)

    return littrowim,littrowhead


def loadpersist(persistcorr,verbose=False):
    """
    Load persistence mask file.

    Parameters
    ----------
    persistcorr : str
       Filename of the persistence mask file
    verbose : bool, optional
       Verbose output to the screen.  Default is False.

    Returns
    -------
    persistim : numpy array
       Persistence mask image.
    persisthead : Header
       Header for the persistence mask file.

    Examples
    --------

    persistim,persisthead = loadpersist(persistcorr)

    """
  
    # PERSISTCORR must be scalar string
    if type(persistcorr) != str or dln.size(persistcorr) != 1:    
        error = 'PERSISTCORR must be a scalar string with the filename of the PERSIST MASK file'
        raise ValueError(error)
  
    # Check that the file exists
    if os.path.exists(persistcorr)==False:
        error = 'PERSISTCORR file '+persistcorr+' NOT FOUND'
        raise ValueError(error)
    
    # Load the PERSIST file
    #  This should be 2048x2048
    persistim,persisthead = fits.getdata(persistcorr,header=True)
  
    if verbose:
        print('PERSIST file = '+persistcorr)
  
    # Check that the file looks reasonable
    #  must be 2048x2048 and have 0/1 values
    szpersist = persistim.shape
    persistokay = 0
    if persistim.ndim != 2 or szpersist[0] != 2048 or szpersist[1] != 2048:
        error = 'PERSISTENCE MASK must be 2048x2048'
        raise ValueError(error)

    return persistim,persisthead


def loaddark(darkcorr,verbose=False):
    """
    Load dark correction file.

    Parameters
    ----------
    darkcorr : str
       Filename of the dark correction file.
    verbose : bool, optional
       Verbose output to the screen.  Default is False.

    Returns
    -------
    darkcube : numpy array
       Dark correction cube. [Ny,Nx,Nreads]
    darkhead : Header
       Header for the dark correction file.

    Examples
    --------

    darkcube,darkhead = loaddark(darkcorr)

    """

    # DARKCORR must be scalar string
    if type(darkcorr) != str or dln.size(darkcorr) != 1:    
        error = 'DARKCORR must be a scalar string with the filename of the dark correction file'
        raise ValueError(error)
  
    # Check that the file exists
    if os.path.exists(darkcorr)==False:
        error = 'DARKCORR file '+darkcorr+' NOT FOUND'
        raise ValueError(error)
  
    # Read header
    darkhead0 = fits.getheader(darkcorr,0)
    darkhead1 = fits.getheader(darkcorr,1)
  
    # Get number of reads
    if darkhead1['NAXIS'] == 3:
        nreads_dark = darkhead1['NAXIS3']
    else:
        # Extensions
        # Figure out how many reads/extensions there are
        #  the primary unit should be empty
        hdu = fits.open(darkcorr)
        nreads_dark = len(hdu)-1
        hdu.close()
    
    # Load the dark correction file
    #  This needs to be 2048x2048xNreads
    #  It's the dark counts for each pixel in counts

    # This always returns the cube as [Ny,Nx,Nreads]
    
    # Datacube
    if darkhead1['NAXIS'] == 3:
        darkcube = fits.getdata(darkcorr)
        darkcube = darkcube.T                 # [nreads,ny,nx] -> [nx,ny,nreads]
        darkcube = np.swapaxes(darkcube,0,1)  # [nx,ny,nreads] -> [ny,nx,nreads]
    # Extensions
    else:
        # Initializing the cube
        darkim,exthead = fits.getdata(darkcorr,1,header=True)
        ny,nx = darkim.shape
        darkcube = np.zeros((ny,nx,nreads_dark),float)
  
        # Read in the extensions
        hdu = fits.open(darkcorr)
        for k in np.arange(1,nreads_dark):
            darkcube[:,:,k-1] = hdu[k].data
        hdu.close()

    if verbose:
        print('Dark Correction file = '+darkcorr)
  
    # Check that the file looks reasonable
    szdark = darkcube.shape
    if (darkcube.ndim != 3 or szdark[0] < 2048 or szdark[1] != 2048):
        error = 'Dark correction data must a 2048x2048xNreads datacube of the dark counts per pixel'
        raise ValueError(error)
    
    return darkcube,darkhead1


def loadflat(flatcorr,verbose=False):
    """
    Load flat field correction file.

    Parameters
    ----------
    flatcorr : str
       Filename of the flat field correction file.
    verbose : bool, optional
       Verbose output to the screen.  Default is False.

    Returns
    -------
    flatim : numpy array
       Flat field correction image.
    flathead : Header
       Header for the flat field file.

    Examples
    --------

    flatim,flathead = loadflat(flatcorr)

    """
  
    # FLATCORR must be scalar string
    if type(flatcorr) != str or dln.size(flatcorr) != 1:    
        error = 'FLATCORR must be a scalar string with the filename of the flat correction file'
        raise ValueError(error)
  
    # Check that the file exists
    if os.path.exists(flatcorr)==False:
        error = 'FLATCORR file '+flatcorr+' NOT FOUND'
        raise ValueError(error)
    
    # Load the flat correction file
    #  This needs to be 2048x2048
    flatim,flathead = fits.getdata(flatcorr,header=True)
  
    if verbose:
        print('Flat Field Correction file = '+flatcorr)
  
    # Check that the file looks reasonable
    szflat = flatim.shape
    if (flatim.ndim != 2 or szflat[0] != 2048 or szflat[1] != 2048):
        error = 'Flat Field correction image must a 2048x2048 image'
        raise ValueError(error)

    return flatim,flathead


def checkbadreads(cube):
    """
    Check the cube for bad reads.

    Parameters
    ----------
    cube : numpy array
       The data cube.

    Returns
    -------
    badflag : boolean
       Flag indicating there are not enough good reads.
    bdreads : numpy array
       Array of bad read indices.

    Examples
    --------

    badflag,bdreads = checkbadreads(cube)

    """

    # Use the reference pixels and reference output for this

    shape = cube.shape
    ny,nx,nreads = shape
    
    if shape[1] == 2560:
        refout1 = np.median(cube[:,2048:,0:np.minimum(nreads,4)],axis=2)        
        sig_refout_arr = np.zeros(nreads,float)
        rms_refout_arr = np.zeros(nreads,float)

    refpix1 = np.hstack((np.median(cube[:4,:2048,0:np.minimum(nreads,4)],axis=2),
                         np.median(cube[:2048,:4,:np.minimum(nreads,4)],axis=2).T,
                         np.median(cube[:2048,2044:2048,0:np.minimum(nreads,4)],axis=2).T,
                         np.median(cube[2044:2048,:2048,0:np.minimum(nreads,4)],axis=2)))
    sig_refpix_arr = np.zeros(nreads,float)
    rms_refpix_arr = np.zeros(nreads,float)

    for k in range(nreads):
        refpix = np.hstack((cube[:4,:2048,k], cube[:2048,:4,k].T,
                            cube[:2048,2044:2048,k].T, cube[2044:2048,:2048,k]))
        refpix = refpix.astype(float)
  
        # The top reference pixels are normally bad
        diff_refpix = refpix - refpix1
        sig_refpix = dln.mad(diff_refpix[:,:12],zero=True)
        rms_refpix = np.sqrt(np.mean(diff_refpix[:,:12]**2))
        sig_refpix_arr[k] = sig_refpix
        rms_refpix_arr[k] = rms_refpix
  
        # Using reference pixel output (5th output)
        if shape[1] == 2560:
            refout = cube[:,2048:,k].astype(float)
            # The top and bottom are bad
            diff_refout = refout - refout1
            sig_refout = dln.mad(diff_refout[100:1951,:],zero=True)
            rms_refout = np.sqrt(np.mean(diff_refout[100:1951,:]**2))
            sig_refout_arr[k] = sig_refout
            rms_refout_arr[k] = rms_refout

    # Use reference output and pixels
    if shape[1] == 2560:
        if nreads>2:
            med_rms_refpix_arr = medfilt(rms_refpix_arr,np.minimum(nreads,11))
            med_rms_refout_arr = medfilt(rms_refout_arr,np.minimum(nreads,11))
        else:
            med_rms_refpix_arr = np.zeros(nreads,float)+np.median(rms_refpix_arr)
            med_rms_refout_arr = np.zeros(nreads,float)+np.median(rms_refout_arr)
        sig_rms_refpix_arr = np.maximum(dln.mad(rms_refpix_arr),1)
        sig_rms_refout_arr = np.maximum(dln.mad(rms_refout_arr),1)
        bdreads, = np.where( (rms_refout_arr-med_rms_refout_arr) > 10*sig_rms_refout_arr)
        nbdreads = len(bdreads)
  
    # Only use reference pixels
    else:
        if nreads > 2:
            med_rms_refpix_arr = medfilt(rms_refpix_arr,np.minimum(nreads,11))
        else:
            med_rms_refpix_arr = np.zeros(nreads,float)+np.median(rms_refpix_arr)
        sig_rms_refpix_arr = np.maximum(dln.mad(rms_refpix_arr),1)
        bdreads, = np.where( (rms_refpix_arr-med_rms_refpix_arr) > 10*sig_rms_refpix_arr)
        nbdreads = len(bdreads)
    
    if nbdreads == 0:
        bdreads = np.array([],int)
        
    # Too many bad reads
    badflag = False
    if nreads-nbdreads < 2:
        print('Warning: ONLY '+str(nreads-nbdreads)+' good reads.  Need at least 2.')
        badflag = True
        
    return badflag,bdreads


def lincorrect(slc_in,lindata):
    """
    Perform the linearity correction for a slice.

    The lindata array gives a quadratic polynomial either for
    each pixel or for each output (512 pixels)
    The polynomial should take the observed counts and convert
    them into corrected counts, i.e.
    counts_correct = poly(counts_obs,coef)

    Parameters
    ----------
    slc_in : numpy array
       Input slice 2D array.
    lindata : numpy array
       Linearity correction data.

    Returns
    -------
    slc_out : numpy array
       Linearity corrected output 2D slice array.

    Example
    -------

    slc_out = lincorrect(slc_in,lindata)

    """

    shape = slc_in.shape
    nreads = shape[1]
    shapelin = lindata.shape

    # A separate coefficient for each output (512 columns)
    corr = np.zeros((2048,nreads),float)
    npar = shapelin[1]
    
    # Loop over quadrants
    slc_out = slc_in.copy()
    for i in range(4):
        corr[512*i:512*i+512,:] = lindata[i,0]
        x = slc_in[512*i:512*i+512,:].copy()
        for j in np.arange(2,nreads):
            x[:,j] = (x[:,j]-x[:,1])*(j+1.0)/(j-1.0)
        x[~np.isfinite(x)] = 0.0
        term = x.copy()
        for j in np.arange(1,npar):
            corr[512*i:512*i+512,:] += lindata[i,j]*term
            term *= x
        # Set first read correction equal to second
        for j in range(2):
            corr[:,j] = corr[:,2]  # +(corr[*,2]-corr[*,3]) needs to work if only 3 reads!

    #slc_out[:2048,:] = slc_in[:2048,:]/corr      # correct code
    slc_out[:2048,0] = slc_in[:2048,0]/corr[:,0]  # this is what the IDL code does, BUG
    
    return slc_out


def darkcorrect(slc_in,darkslc):
    """
    Performs the dark correction for a slice.
    darkslc is a 2048xNreads array that gives the dark counts

    To get the dark current just multiply the dark count rate
    by the time for each read

    Parameters
    ----------
    slc_in : numpy array
       Input slice 2D array.
    darkdata : numpy array
       Dark correction data.

    Returns
    -------
    slc_out : numpy array
       dark corrected output 2D slice array.

    Examples
    --------

    slc_out = darkcorrect(slc_in,darkslc)

    """

    nreads = slc_in.shape[1]
    
    # Just subtract the darkslc
    slc_out = slc_in.copy()
    slc_out[:2048,:] -= darkslc[:,:nreads]
        
    return slc_out


def crcorrect(dCounts,satmask,sigthresh=10,onlythisread=None,noise=17.0,
              fix=False,verbose=False):
    """
    This subroutine fixes cosmic rays in a slice of the datacube.
    The last dimension in the slice should be the Nreads, i.e.
    [Npix,Nreads].

    Parameters
    ----------
    dCounts : numpy array
       The difference of neighboring pairs of reads.
    satmask : numpy array
       The saturation mask [Nx,3].  1-saturation mask,
         2-read # of first saturation, 3-number of saturated reads
    sigthresh : float, optional
       The Nsigma threshold for detecting a
         CR. Default is sigthresh=10.
    onlythisread : bool, optional
       Only accept CRs in this read index (+/-1 read).
         This is only used for the iterative CR rejection.
         Default is None.
    noise : float, optional
       The readnoise in ADU (for dCounts, not single reads).
         Default is 17.
    fix : bool, optional
       Actually fix the CR and not just detect it.  Default is False.
    verbose : bool, optional
       Verbose output to the screen.  Default is False.

    Returns
    -------
    crtab : table
       Table that gives information on the CRs.
    dCounts_fixed : numpy array
       The "fixed" dCounts with the CRs removed.
    med_dCounts : numpy array
       The median dCounts for each pixel
    variability : numpy array
       The fractional variability in each pixel
    mask : numpy array
       Bitmask with pixels with CRs marked.

    Examples
    --------

    crtab,dCounts_fixed,med_dCounts,variability,mask = crcorrect(dCounts,satmask)

    """

    nx,nreads = dCounts.shape
    # nreads is actually Nreads-1

    # Initializing dCounts_fixed
    dCounts_fixed = dCounts.copy()

    #-----------------------------------
    # Get median dCounts for each pixel
    #-----------------------------------
    med_dCounts = np.nanmedian(dCounts,axis=1)    # NANs are automatically ignored

    # Check if any medians are NANs
    #  would happen if all dCounts in a pixel are NAN
    med_dCounts[~np.isfinite(med_dCounts)] = 0.0    # set to zero
    
    # Number of non-saturated, "good" reads
    totgd = np.sum(np.isfinite(dCounts),axis=1)

    # If we only have 2 good Counts then we might need to use
    # the minimum dCounts in case there is a CR
    ind2, = np.where(totgd == 2)
    nind2 = len(ind2)
    for j in range(nind2):
        min_dCounts = np.min(dCounts[ind2[j],:],axis=1)
        max_dCounts = np.max(dCounts[ind2[j],:],axis=1)
        # If they are too different then use the lower value
        #  probably because of a CR
        if (max_dCounts-min_dCounts)/np.maximum(min_dCounts,1e-4) > 0.3:
            med_dCounts[ind2[j]] = np.maximum(min_dCounts,1e-4)

    med_dCounts2D = med_dCounts.repeat(nreads).reshape((nx,nreads))   # 2D version

    #-----------------------------------------------------
    # Get median smoothed/filtered dCounts for each pixel
    #-----------------------------------------------------
    #  this should help remove transparency variations
    smbin = np.minimum(11, nreads)    # in case Nreads is small
    if nreads > smbin:
        sm_dCounts = medfilt2d(dCounts,[1,smbin])
    else:
        sm_dCounts = med_dCounts2D

    # We need to deal with reads near saturated reads carefully
    # otherwise the median will be over less reads
    # For now this is okay, at worst the median will be over smbin/2 reads
        
    # If there are still some NAN then replace them with the global
    # median dCounts for that pixel.  These are probably saturated
    # so it probably doesn't matter
    bdnan = np.where(~np.isfinite(sm_dCounts))
    nbdnan = np.sum(bdnan)
    if nbdnan>0:
        sm_dCounts[bdnan] = med_dCounts2D[bdnan]

    #--------------------------------------
    # Variability from median (fractional)
    #--------------------------------------
    variability = dln.mad(dCounts-med_dCounts2D,axis=1,zero=True)
    variability = variability / np.maximum(med_dCounts, 0.001)  # make it a fractional variability
    bdvar = ~np.isfinite(variability)
    nbdvar = np.sum(bdvar)
    if nbdvar>0:
        variability[bdvar] = 0.0   # all NAN
    if nind2>0:
        variability[ind2] = 0.5    # high variability for only 2 good dCounts

    #----------------------------------
    # Get sigma dCounts for each pixel
    #----------------------------------
    #sig_dCounts = mad(dCounts,dim=2)
    # subtract smoothed version to remove any transparency variations
    # saturated reads (NAN in dCounts) are automatically ignored
    sig_dCounts = dln.mad(dCounts-sm_dCounts,axis=1,zero=True)
    sig_dCounts = np.maximum(sig_dCounts, noise)   # needs to be at least above the noise

    # Check if any sigma are NAN
    #  would happen if all dCounts in a pixel are NAN
    sig_dCounts[~np.isfinite(sig_dCounts)] = noise   # set to noise level

    # Pixels with only 2 good dCounts, set sigma to 30%
    for j in range(nind2):
        sig_dCounts[ind2[j]] = np.maximum(0.3*med_dCounts[ind2[j]], noise)

    sig_dCounts2D = sig_dCounts.repeat(nreads).reshape((nx,nreads))   # 2D version
    
    #-----------
    # Find CRs
    #-----------
    # threshold for number of sigma above (local) median
    if sigthresh is not None:
        nsig_thresh = sigthresh
    else:
        nsig_thresh = 10
    nsig_thresh = np.maximum(nsig_thresh, 3)    # 3 at a minimum

    # Saturated dCounts (NANs) are automatically ignored
    nsigma_slc = (dCounts-sm_dCounts)/sig_dCounts2D
    bdx,bdr = np.where( ( nsigma_slc > nsig_thresh ) &
                        ( dCounts > noise*nsig_thresh ))
    nbd1D = len(bdx)
    
    if verbose:
        print(str(nbd1D)+' CRs found')
    
    if nbd1D>0:
        # Initialize the crtab dictionary
        dtype = np.dtype([('x',int),('y',int),('read',int),('counts',float),('nsigma',float),
                          ('globalsigma',float),('fixed',bool),('localsigma',float),
                          ('fixerror',float),('neicheck',bool)])
        crtab = {'ncr':0,'data':np.zeros(nbd1D,dtype)}

    # CR loop
    #   correct the CRs and correct the pixels
    for j in range(nbd1D):
        ibdx = bdx[j]
        ibdr = bdr[j]
        dCounts_pixel = dCounts[ibdx,:].copy()

        # ONLYTHISREAD
        #  for checking neighboring pixels in the iterative part
        #--------------
        if onlythisread is not None:
            # onlythisread is the read index, while ibdr is a dCounts index
            # ibdr+1 is the read index for this CR
            if (ibdr+1) < onlythisread-1 or (ibdr+1) > onlythisread+1:
                break
            
        # Calculate Local Median and Local Sigma
        #----------------------------------------
        #   Use a local median/sigma so the affected CR dCount is not included
        # more than 2 good dCounts and Nreads>smbin
        if (totgd[ibdx] > 2) and (nreads > smbin):
            dCounts_pixel[ibdr] = np.nan        # don't use affected dCounts    
            maxind = nreads-1
            if satmask[ibdx,0] == 1:
                maxind = satmask[ibdx,1]-2      # don't include saturated reads (NANs)
            lor = np.maximum((ibdr-smbin//2),0)
            hir = np.minimum( (lor + smbin-1), maxind )
            if (hir == maxind):
                lor = np.maximum((hir-smbin+1),0)
    
            # -- Local median dCounts --
            #  make sure the indices make sense
            if (lor < 0 or hir < 0 or hir <= lor):
                local_med_dCounts = med_dCounts[ibdx]
            else:
                local_med_dCounts = np.median(dCounts_pixel[lor:hir])
    
            # If local median dCounts is NAN use all reads
            if ~np.isfinite(local_med_dCounts):
                local_med_dCounts = med_dCounts[ibdx]
            # If still NaN then set to 0.0
            if ~np.isfinite(local_med_dCounts):
                local_med_dCounts = 0.0
    
            # -- Local sigma dCounts --
            local_sigma = dln.mad(dCounts_pixel[lor:hir]-local_med_dCounts,zero=True)
    
            # If local sigma dCounts is NAN use all reads
            #   this should never actually happen
            if ~np.isfinite(local_sigma):
                local_sigma = sig_dCounts[ibdx]
            # If still NaN then set to noise
            if ~np.isfinite(local_sigma):
                local_sigma = noise
                
        # Only 2 good dCounts OR Nreads<smbin
        else:
            local_med_dCounts = med_dCounts[ibdx]
            local_sigma = sig_dCounts[ibdx]

        local_med_dCounts = med_dCounts[ibdx]
        local_sigma = sig_dCounts[ibdx]

        # Fix the CR
        #------------
        if fix:
            if verbose:
                print(' Fixing CR at Column '+str(ibdx)+' Read '+str(ibdr+1))
            # Replace with smoothed dCounts, i.e. median of neighbors
            dCounts_fixed[ibdx,ibdr] = local_med_dCounts       # fix CR dCounts
            # Error in the fix
            #   by taking median of smbin neighboring reads we reduce the error by ~1/sqrt(smbin)
            fixerror = local_sigma/np.sqrt(smbin-1)   # -1 because the CR read is in there
        
        # Add CR information to table
        #----------------------------
        crtab['data'][crtab['ncr']]['x'] = ibdx
        crtab['data'][crtab['ncr']]['read'] = ibdr+1  # ibdr is dCounts index, +1 to get read
        crtab['data'][crtab['ncr']]['counts'] = dCounts[ibdx,ibdr] - sm_dCounts[ibdx,ibdr]
        crtab['data'][crtab['ncr']]['nsigma'] = nsigma_slc[ibdx,ibdr]
        crtab['data'][crtab['ncr']]['globalsigma'] = sig_dCounts[ibdx]
        if fix:
            crtab['data'][crtab['ncr']]['fixed'] = True
        crtab['data'][crtab['ncr']]['localsigma'] = local_sigma
        if fix:
            crtab['data'][crtab['ncr']]['fixerror'] = fixerror
        crtab['ncr'] += 1
        
    #  Replace the dCounts with CRs with the median smoothed values
    #    other methods could be used to "fix" the affected read,
    #    e.g. polynomial fitting/interpolation, Gaussian smoothing, etc.

    # Now trim CRTAB
    if nbd1D>0:
        crtab['data'] = crtab['data'][:crtab['ncr']]
    else:
        crtab = {'ncr':0}   # blank table

    # Make CR mask
    mask = np.zeros(2048,int)
    if crtab['ncr'] > 0:
        # Add to MASK
        maskpix, = np.where(crtab['data']['x'] < 2048)
        nmaskpix = len(maskpix)
        if nmaskpix > 0:
            mask[crtab['data'][maskpix]['x']] = pixelmask.getval('CRPIX')
            
    return crtab, dCounts_fixed, med_dCounts, variability, mask


def refcorrect_sub(image,ref):
    """
    Subtracts the reference array from each quadrant with proper flipping. 
    """
    revref = np.flip(ref,axis=1)
    image[:,0:512] -= ref
    image[:,512:1024] -= revref
    image[:,1024:1536] -= ref
    image[:,1536:2048] -= revref
    return image


def refcorrect(cube,head,mask=None,indiv=3,vert=True,horz=True,cds=True,
               noflip=False,q3fix=False,keepref=False,verbose=True):
    """
    This corrects a raw APOGEE datacube for the reference pixels
    and reference output

    Parameters
    ----------
    cube : numpy array
       The raw APOGEE datacube with reference array.  This
         will be updated with the reference subtracted cube.
    head : Header
       The header for CUBE.
    mask : numpy array, optional
       Input bad pixel mask.
    indiv : int, optional
       Subtract the individual reference arrays after NxN median filter. If 
        If <0, subtract mean reference array. If ==0, no reference array subtraction
        Default is indiv=3.
    vert : bool, optional
       Use vertical ramp.  Default is True.
    horz : bool, optional
       Use horizontal ramp.  Default is True.
    cds : bool, optional
       Perform double-correlated sampling.  Default is True.
    noflip : bool, optional
       Do not flip the reference array.
    q3fix : bool, optional
       Fix issued with the third quadrant for MJD XXX.
    keepref : bool, optional
       Return the reference array in the output.
    verbose : bool, optional
       Verbose printing to the screen.  Default is True.

    Returns
    -------
    out : numpy array
       The reference-subtracted cube.
    mask : numpy array
       The flag mask array.
    readmask : numpy array
       Mask indicating if reads are bad (0-good, 1-bad).

    Example
    -------

    out,mask,readmask = refcorrect(cube,head)

    By J. Holtzman   2011
    Incorporated into ap3dproc.pro  D.Nidever May 2011
    Translated to Python  D.Nidever  Nov 2023
    """

    t0 = time.time()
    
    # refcorrect does the "bias" subtraction, using the reference array and
    #    the reference pixels. Subtract a mean reference array (or individual
    #    with /indiv), then subtract vertical ramps from each quadrant using
    #    reference pixels, then subtract smoothed horizontal ramps

    # Number of reads
    ny,nx,nread = cube.shape

    # Create long output
    out = np.zeros((2048,2048,nread),int)
    if keepref:
        refout = np.zeros((2048,512,nread),int)

    # Ignore reference array by default
    # Default is to do CDS, vertical, and horizontal correction
    print('in refcorrect, indiv: '+str(indiv))

    satval = 55000
    snmin = 10
    if indiv>0:
        hmax = 1e10
    else:
        hmax = 65530

    # Initalizing some output arrays
    if mask is None:
        mask = np.zeros((2048,2048),int)
    readmask = np.zeros(nread,int)

    # Calculate the mean reference array
    if verbose:
        print('Calculating mean reference')
    meanref = np.zeros((2048,512),float)
    nref = np.zeros((2048,512),int)
    for i in range(nread):
        ref = cube[:,2048:2560,i].astype(float)
        # Calculate the relevant statistics
        mn = np.mean(ref[128:2048-128,128:512-128])
        std = np.std(ref[128:2048-128,128:512-128])
        hm = np.max(ref[128:2048-128,128:512-128])
        ref[ref>=satval] = np.nan        
        # SLICE business is just for special fast handling, ignored if
        #   not in header
        card = 'SLICE%03d' % i
        iread = head.get(card)
        if iread is None:
            iread = i+1
        if verbose:
            print("\rreading ref: {:3d} {:3d}".format(i,iread), end='')
        # skip first read and any bad reads
        if (iread > 1) and (mn/std > snmin) and (hm < hmax):
            good = (np.isfinite(ref))
            meanref[good] += (ref[good]-mn)
            nref[good] += 1
            readmask[i] = 0
        else:
            if verbose:
                print('\nRejecting: ',i,mn,std,hm)
            readmask[i] = 1
            
    meanref /= nref
    
    if verbose:
        print('\nReference processing ')
        
    # Create vertical and horizontal ramp images
    rows = np.arange(2048,dtype=float)
    cols = np.ones(512,dtype=int)
    vramp = (rows.reshape(-1,1)*cols.reshape(1,-1))/2048
    vrramp = 1-vramp   # reverse
    cols = np.arange(2048,dtype=float)
    rows = np.ones(2048,dtype=int)
    hramp = (rows.reshape(-1,1)*cols.reshape(1,-1))/2048
    hrramp = 1-hramp
    clo = np.zeros(2048,float)
    chi = np.zeros(2048,float)

    if cds:
        cdsref = cube[:,:2048,1]
        
    # Loop over the reads
    lastgood = nread-1
    for iread in range(nread):
        # Do all operations as floats, then convert to int at the end
        
        # Subtract mean reference array
        im = cube[:,:2048,iread].astype(int)

        # Deal with saturated pixels
        sat = (im > satval)
        nsat = np.sum(sat)
        if nsat > 0:
            if iread == 0:
                nsat0 = nsat
            im[sat] = 65535
            mask[sat] = (mask[sat] | pixelmask.getval('SATPIX'))
            # If we have a lot of saturated pixels, note this read (but don't do anything)
            if nsat > nsat0+2000:
                if lastgood == nread-1:
                    lastgood = iread-1
        else:
            nsat0 = 0
            
        # Pixels that are identically zero are bad, see these in first few reads
        bad = (im == 0)
        nbad = np.sum(bad)
        if nbad > 0:
            mask[bad] |= pixelmask.getval('BADPIX')
        
        if verbose:
            print("\rRef processing: {:3d}  nsat: {:5d}".format(iread+1,nsat), end='')            

        # Skip this read
        if readmask[iread] > 0:
            im = np.nan
            out[:,:,iread] = int(-1e10)   # np.nan, int cannot be NaN
            if keepref:
                refout[:,:,iread] = 0
            continue
        
        # With cds keyword, subtract off first read before getting reference pixel values
        if cds:
            im -= cdsref.astype(int)

        # Use the reference array information
        ref = cube[:,2048:2560,iread].astype(float)
        # No reference array subtraction
        if indiv is None or indiv==0:
            pass
        # Subtract full reference array
        elif indiv==1:
            im = refcorrect_sub(im,ref)
            ref -= ref
        # Subtract median-filtered reference array
        elif indiv>1:
            mdref = medfilt2d(ref.astype(float),[indiv,indiv]).astype(int)
            im = refcorrect_sub(im,mdref)
            ref -= mdref
        # Subtract mean reference array
        elif indiv<0:
            im = refcorrect_sub(im,meanref)
            ref -= meanref
            
        # Subtract vertical ramp, using edges
        if vert:
            for j in range(4):
                rlo = np.nanmean(im[2:4,j*512:(j+1)*512])
                rhi = np.nanmean(im[2045:2047,j*512:(j+1)*512])
                im[:,j*512:(j+1)*512] = (im[:,j*512:(j+1)*512].astype(float) - rlo*vrramp).astype(int)
                im[:,j*512:(j+1)*512] = (im[:,j*512:(j+1)*512].astype(float) - rhi*vramp).astype(int)
                #im[:,j*512:(j+1)*512] -= rlo*vrramp
                #im[:,j*512:(j+1)*512] -= rhi*vramp                
                
        # Subtract horizontal ramp, using smoothed left/right edges
        if horz:
            clo = np.nanmean(im[:,1:4],axis=1)
            chi = np.nanmean(im[:,2044:2047],axis=1)
            sm = 7
            slo = utils.nanmedfilt(clo,sm,mode='edgecopy')
            shi = utils.nanmedfilt(chi,sm,mode='edgecopy')

            # in the IDL code, this step converts "im" from int to float
            if noflip:
                im = im.astype(float) - slo.reshape(-1,1)*hrramp
                im = im.astype(float) - shi.reshape(-1,1)*hramp
                #im -= slo.reshape(-1,1)*hrramp
                #im -= shi.reshape(-1,1)*hramp                
            else:
                #bias = (rows#slo)*hrramp+(rows#shi)*hramp
                # just use single bias value of minimum of left and right to avoid bad regions in one
                bias = np.min([slo,shi],axis=0).reshape(-1,1) * np.ones((1,2048))
                fbias = bias.copy()
                fbias[:,512:1024] = np.flip(bias[:,512:1024],axis=1)
                fbias[:,1536:2048] = np.flip(bias[:,1536:2048],axis=1)
                im = im.astype(float) - fbias
                #im -= fbias.astype(int)                
                
        # Fix quandrant 3 issue
        if q3fix:
            q2m = np.median(im[:,923:1024],axis=1)
            q3a = np.median(im[:,1024:1125],axis=1)
            q3b = np.median(im[:,1435:1536],axis=1)
            q4m = np.median(im[:,1536:1637],axis=1)
            q3offset = ((q2m-q3a)+(q4m-q3b))/2.
            im[:,1024:1536] += medfilt(q3offset,7).reshape(-1,1)*np.ones((1,512))
            
        # Make sure saturated pixels are set to 65535
        #  removing the reference values could have
        #  bumped them lower
        if nsat > 0:
            im[sat] = 65535

        # Stuff final values into our output arrays
        #   and convert form float to int
        out[:,:,iread] = im.astype(int)
        if keepref:
            refout[:,:,iread] = ref
            
    # Mask the reference pixels
    mask[0:4,:] = (mask[0:4,:] | pixelmask.getval('BADPIX'))
    mask[2044:2048,:] = (mask[2044:2048,:] | pixelmask.getval('BADPIX'))
    mask[:,0:4] = (mask[:,0:4] | pixelmask.getval('BADPIX'))
    mask[:,2044:2048] = (mask[:,2044:2048] | pixelmask.getval('BADPIX'))

    if verbose:
        print('')
        print('lastgood: ',lastgood)
        
    # Keep the reference array in the output
    if keepref:
        out = np.hstack((out,refout))
        
    return out,mask,readmask


def interpbadreads(cube,gdreads,bdreads):
    """
    Interpolate bad reads using neighboring reads.

    Parameters
    ----------
    cube : numpy array
       Input data cube.
    gdreads : numpy array
       Array of good read indices.
    bdreads : numpy array
       Array of bad read indices.

    Returns
    -------
    cube : numpy array
       Data cube with bad reads interpolated.

    Examples
    --------
    
    cube = interpbadreads(cube,gdreads,bdreads)

    """

    if len(bdreads)==0:
        return cube
    
    print('Read(s) '+', '.join(np.char.array(bdreads+1).astype(str))+' are bad.')
  
    # The bad reads are currently linearly interpolated using the
    # neighoring reads and used as if they were good.  The variance
    # needs to be corrected for this at the end.
    # This will give bad results for CRs
    
    # Use linear interpolation
    nbdreads = len(bdreads)
    for k in range(nbdreads):
        # Get good reads below
        gdbelow, = np.where(gdreads < bdreads[k])
        ngdbelow = len(gdbelow)
        # Get good reads above
        gdabove, = np.where(gdreads > bdreads[k])
        ngdabove = len(gdabove)
        
        if ngdbelow == 0:
            interp_type = 1                     # all above
        if ngdbelow > 0 and ngdabove > 0:
            interp_type = 2                     # below and above
        if ngdabove == 0:
            interp_type = 3                     # all below

        if interp_type==1:
            # all above
            gdlo = gdabove[0]
            gdhi = gdabove[1]
        elif interp_type==2:
            # below and above
            gdlo = gdbelow[-1]
            gdhi = gdabove[0]
        elif interp_type==3:
            # all below
            gdlo = gdbelow[ngdbelow-2]
            gdhi = gdbelow[ngdbelow-1]
        lo = gdreads[gdlo]
        hi = gdreads[gdhi]
  
        # Linear interpolation
        im1 = cube[:,:,lo].astype(float)
        im2 = cube[:,:,hi].astype(float)
        slope = (im2-im1)/float(hi-lo)            # slope, (y2-y1)/(x2-x1)
        zeropoint = (im1-slope*lo)                # zeropoint, zp = (y1-slp*x1)
        im0 = slope*bdreads[k] + zeropoint        # linear interpolation, y=mx+b
        
        # Stuff it in the cube
        cube[:,:,bdreads[k]] = np.round(im0)      # round to closest integer, LONG type
  
    return cube


def detectsatreads(slc,saturation):
    """
    Detect and flag saturated pixels/reads.

    Parameters
    ----------
    slc : numpy array
       Slice of datacube, [Ncols,Nreads].
    saturation : float
       Saturation level.

    Returns
    -------
    slc : numpy array
       Slice array.
    bdsat : numpy array
       2D boolean array of saturated pixels in slice.
    satmask : numpy array
       Saturated pixels array. [Npix,Nread,3]
         1st value is the mask, 2nd is the read at which it saturated,
         and the 3rd value is the number of saturated reads.

    Examples
    --------

    slc,bdsat,satmask = detectsatreads(slc,saturation)

    """

    nx,nreads = slc.shape
    bdsat = (slc > saturation)
    satmask = np.zeros((nx,3),int)   # sat mask
    mask = np.zeros(nx,int)
    
    if np.sum(bdsat)>0:
        # Flag saturated reads as NAN
        slc[bdsat] = np.nan
  
        # Get 2D indices
        bdsatx,bdsatr = np.where(bdsat)
        # bdsat is 1D array for slice(2D)
        # bdsatx is column index for 1D med_dCounts

        # Unique pixels
        ubdsatx = np.unique(bdsatx)
        nbdsatx = len(ubdsatx)
        
        # Figure out at which Read (NOT dCounts) each column saturated
        rindex = np.ones(nx,int).reshape(-1,1) * np.arange(nreads).reshape(1,-1)
        # each pixels read index
        satmask_slc = np.zeros((nx,nreads),int)   # sat mask
        satmask_slc[bdsatx,bdsatr] = 1            # set saturated reads to 1
        rindexsat = rindex*satmask_slc + (1-satmask_slc)*999999
        # okay pixels have 999999, sat pixels have their read index
        minsatread = np.min(rindexsat,axis=1)     # now find the minimum for each column
        nsatreads = np.sum(satmask_slc,axis=1)    # number of sat reads
  
        # Make sure that all subsequent reads to a saturated read are
        # considered "bad" and set to NAN
        for j in range(nbdsatx):
            slc[ubdsatx[j],minsatread[ubdsatx[j]]:] = np.nan
  
        # Update satmask
        satmask[ubdsatx,0] = 1                     # mask
        satmask[ubdsatx,1] = minsatread[ubdsatx]   # 1st saturated read, NOT dcounts
        satmask[ubdsatx,2] = nsatreads[ubdsatx]    # humber of saturated reads

    return slc,bdsat,satmask
            

def fixsatreads(dCounts,med_dCounts,bdsat,satmask,
                variability,satfix=True,rd3satfix=False):
    """
    Fix and flag saturated reads.

    Parameters
    ----------
    dCounts : numpy array
       Count differences. [Npix,Nreads]
    med_dCounts : numpy array
       Median count differences.
    bdsat : numpy array
       Array of bad read indices.
    satmask : numpy array
       Saturation mask for this slice.
    variability : numpy array
       Variabiity values for this slice.
    satfix : boolean, optional
       Fix saturated pixels.  This is done by default
         If satfix is False then saturated pixels are still detected
         and flagged in the mask file, but NOT corrected - 
         instead they are set to 0.  Saturated pixels that are
         not fixable ("unfixable", less than 3 unsaturated reads)
         are also set to 0.
    rd3satfix : boolean, optional
       Fix saturated pixels for 3 reads.  Default is False.

    Returns
    -------
    dCounts : numpy array
       Fixed count differences.
    mask : numpy array
       Saturated pixel bitmask for this slice.
    sat_extrap_error : numpy array
       Extra error for fixed saturated pixels.

    Examples
    --------

    dCounts,mask,sat_extrap_error = fixsatreads(dCounts,med_dCounts,
                                                bdsat,satmask,variability)

    """

    nx = dCounts.shape[0]
    nreads = dCounts.shape[1]+1
    sat_extrap_error = np.zeros(nx,float)
    mask = np.zeros(2048,int)
    ubdsatx, = np.where(satmask[:2048,0]==1)

    # No saturated pixels
    if np.sum(bdsat)==0:
        return dCounts,mask,sat_extrap_error
    
    # Only 2 reads, can't fix anything
    if (nreads <= 2):
        mask[ubdsatx] = pixelmask.getval('UNFIXABLE')
        # mask: 1-bad, 2-CR, 4-sat, 8-unfixable
        dCounts[bdsatx,:] = 0.0            # set saturated reads to zero
        return dCounts,mask,sat_extrap_error
        
    # Total number of good dCounts for each pixel
    totgd = np.sum(np.isfinite(dCounts),axis=1)
            
    # Unfixable pixels
    #------------------
    #  Need 2 good dCounts to be able to "safely" fix a saturated pixel
    thresh_dcounts = 2
    if rd3satfix and nreads==3:
        thresh_dcounts = 1  # fixing 3 reads                       
    unfixable, = np.where(totgd < thresh_dcounts)
    if len(unfixable) > 0:
        dCounts[unfixable,:] = 0.0
        mask[unfixable] = pixelmask.getval('UNFIXABLE')
  
    # Fixable Pixels
    #-----------------
    fixable, = np.where((totgd >= thresh_dcounts) & (satmask[:,0] == 1))
    nfixable = len(fixable)

    minsatread = satmask[:,1]
    
    # Loop through the fixable saturated pixels
    for j in range(nfixable):
        ibdsatx = fixable[j]

        # "Fix" the saturated pixels
        #----------------------------
        if satfix:  
            # Fix the pixels
            #   set dCounts to med_dCounts for that pixel
            #dCounts[bdsat] = med_dCounts[bdsatx]
            # if the first read is saturated then we start with
            #   the first dCounts
            dCounts[ibdsatx,np.maximum(minsatread[ibdsatx]-1,0):nreads-1] = med_dCounts[ibdsatx]
            # Saturation extrapolation error
            var_dCounts = variability[ibdsatx] * np.maximum(med_dCounts[ibdsatx],0.0001)
            # variability in dCounts
            sat_extrap_error[ibdsatx] = var_dCounts * satmask[ibdsatx,2]
            # Sigma of extrapolated counts, multipy by Nextrap
  
        # Do NOT fix the saturated pixels
        #---------------------------------
        else:
            dCounts[ibdsatx,minsatread[ibdsatx]-1:nreads-1] = 0.0
            # set saturated dCounts to zero
          
        # It might be better to use the last good value from sm_dCounts
        # rather than the straight median of all reads
        
    return dCounts,mask,sat_extrap_error


def process_slice(slc,mask,caldata,crfix=True,nocr=False,
                  satfix=False,rd3satfix=False,verbose=False,
                  debug=False):
    """
    Process a single row/slice of the datacube.

    Parameters
    ----------
    slc : numpy array
       Slice of the datacube, [Ncols,Nreads].
    mask : numpy array
       Bitmask array.
    crfix : boolean, optional
       Fix the cosmic rays.  Default is False.
    nocr : boolean, optional
       Do not perform CR detection or fixing.  Default is False.
    satfix : boolean, optional
       Fix saturated pixels.  This is done by default
    rd3satfix : boolean, optional
       Fix saturated pixels for 3 reads, and assume they don't
       have CRs.  Default is False.
    verbose : boolean, optional
       Verbose output to the screen.  Default is False.
    debug : boolean, optional
       Debugging mode with extra printing to the screen.
          Default is False.

    Returns
    -------
    slc_fixed : numpy array
       Updated slice of the datacube, [Ncols,Nreads].
    mask : numpy array
       Bitmask array.
    satmask : numpy array
       Saturation mask.
    crtab : dict
       Cosmic ray table.
    med_dCounts : numpy array
       Median count rate array.
    variability : numpy array
       Fractional variability for each pixel.
    sat_extrap_error : numpy array
       Extra error when fixing saturated pixels.

    Examples
    --------

    out = process_slice(slc,mask,caldata)
    slc_fixed,mask,satmask,crtab,med_dCounts,variability,sat_extrap_error = out

    """

    # Unpack the calibration data
    rdnoiseim,gainim,lindata = caldata['rdnoiseim'],caldata['gainim'],caldata['lindata']
    bpmim,littrowim,persistim = caldata['bpmim'],caldata['littrowim'],caldata['persistim']
    darkim,flatim,saturation = caldata['darkim'],caldata['flatim'],caldata['saturation']
    noise = caldata['noise']
    noise_dCounts = noise*np.sqrt(2)
    
    nx,nreads = slc.shape
    
    # Slice of datacube, [Ncol,Nread]
    #--------------------------------
 
    # Flag BAD pixels
    #----------------
    if bpmim is not None:
        bdpix, = np.where(bpmim > 0)
        nbdpix = len(bdpix)
        if nbdpix > 0:
            for j in range(nbdpix):
                slc[bdpix[j],:] = 0.0  # set them to zero
            mask[bdpix] |= bpmim[bdpix]
            
    # Flag LITTROW ghost pixels
    #   but don't change data values
    #---------------------------------
    if littrowim is not None:
        bdpix, = np.where(littrowim == 1)
        nbdpix = len(bdpix)
        if nbdpix > 0:
            mask[bdpix] |= pixelmask.getval('LITTROW_GHOST')
  
    # Flag persistence pixels
    #   but don't change data values
    #---------------------------------
    if persistim is not None:
        bdpix1, = np.where(persistim & 1)
        if len(bdpix1) > 0:
            mask[bdpix1] |= pixelmask.getval('PERSIST_HIGH')
        bdpix2, = np.where(persistim & 2)
        if len(bdpix2) > 0:
            mask[bdpix2] |= pixelmask.getval('PERSIST_MED')
        bdpix4, = np.where(persistim & 4)
        if len(bdpix4)>0:
            mask[bdpix4] |= pixelmask.getval('PERSIST_LOW')
            
    # Detect and Flag Saturated reads
    #---------------------------------
    #  The saturated pixels are detected in the reference subtraction
    #  step and fixed to 65535.
    slc,bdsat,satmask = detectsatreads(slc,saturation)
    mask[satmask[:2048,0]==1] |= pixelmask.getval('SATPIX')     # add to mask
    
    # Linearity correction
    #----------------------
    # This needs to be done BEFORE the pixels are "fixed" because
    # it needs to operate on the ORIGINAL counts, not the corrected
    # ones.
    if lindata is not None:
        slc = lincorrect(slc,lindata)

    # Dark correction
    #-----------------
    # Each read will have a different amount of dark counts in it
    if darkim is not None:
        slc = darkcorrect(slc,darkim)
  
    # Find difference of neighboring reads, dCounts
    #------------------------------------------------
    #  the difference between 1 or 2 NaNs will also be NaN
    dCounts = slc[:,1:nreads] - slc[:,:nreads-1]

    # Detect and Fix cosmic rays
    #----------------------------
    slc_prefix = slc.copy()
    if nocr==False and nreads>2:
        out = crcorrect(dCounts,satmask,noise=noise_dCounts,fix=crfix)
        crtab,dCounts,med_dCounts,variability,crmask = out
        # Add to the mask
        mask |= crmask   # add to the mask
    # Only 2 reads, Cannot detect or fix CRs
    else:
        crtab = {'ncr':0}
        med_dCounts = dCounts
        variability = np.zeros(nx,float)
        
    # Fix Saturated reads
    #----------------------
    #  do this after CR fixing, so we don't have to worry about CRs here
    #  set their dCounts to med_dCounts
    dCounts,smask,sat_extrap_error = fixsatreads(dCounts,med_dCounts,bdsat,
                                                 satmask,variability,
                                                 satfix=satfix,rd3satfix=rd3satfix)
    mask |= smask   # add to the mask
    
    # Reconstruct the SLICE from dCounts
    #------------------------------------
    slc0 = slc[:,0].copy()                # first read
    bdsat, = np.where(~np.isfinite(slc0))  # NAN in first read, set to 0.0
    if len(bdsat) > 0:
        slc0[bdsat] = 0.0
    # unfixable
    unfmask_slc = ((mask.astype(bool) & pixelmask.getval('UNFIXABLE')) == pixelmask.getval('UNFIXABLE'))
    unfmask_slc = unfmask_slc.astype(int)
    slc0[:2048] = slc0[:2048]*(1.0-unfmask_slc)    # set unfixable pixels to zero
    slc_fixed = slc0.reshape(-1,1) + np.zeros((nx,nreads),float)
    if nreads > 2:
        slc_fixed[:,1:] += np.cumsum(dCounts,axis=1)
    else:
        slc_fixed[:,0] = slc0
        slc_fixed[:,1] = slc0+dCounts
        
    # Final median of each "fixed" pixel
    #------------------------------------
    if nreads > 2:
        # Unfixable pixels are left at 0.0  
        # If NOT fixing saturated pixels, then we need to
        # temporarily set saturated reads to NAN
        #  Leave unfixable pixels at 0.0
        temp_dCounts = dCounts.copy()
        if satfix==False:
            bdsat, = np.where((satmask[:,0] == 1) & (unfmask_slc == 0))
            nbdsat = len(bdsat)
            for j in range(nbdsat):
                temp_dCounts[bdsat[j],satmask[bdsat[j],1]-1:] = np.nan
        fmed_dCounts = np.nanmedian(temp_dCounts,axis=1)    # NAN are automatically ignored
        bdnan, = np.where(~np.isfinite(fmed_dCounts))
        if len(bdnan) > 0:
            import pdb; pdb.set_trace()
        med_dCounts = fmed_dCounts
    # Only 2 reads
    else:
        med_dCounts = dCounts
        
    # Print some information
    if debug:
        nsatslc = np.sum(satmask[:,0])
        ncrslc = crtab['ncr']
        if nsatslc > 0 or ncrslc > 0:
            print('Nsat/NCR = {:}/{:} this row'.format(int(nsatslc),int(ncrslc)))
        
    return slc_fixed,mask,satmask,crtab,med_dCounts,variability,sat_extrap_error


def ap3dproc(files,outfile,apred,detcorr=None,bpmcorr=None,darkcorr=None,
             flatcorr=None,littrowcorr=None,persistcorr=None,persistmodelcorr=None,
             histcorr=None,crfix=True,satfix=True,rd3satfix=False,saturation=65000,
             nfowler=None,uptheramp=None,outelectrons=False,refonly=False,
             criter=False,clobber=False,outlong=None,cleanuprawfile=True,
             nocr=False,logfile=None,fitsdir=None,maxread=None,
             q3fix=False,usereference=False,seq=None,unlock=False,debug=False,
             verbose=False,**kwargs):
    """
    Process a single APOGEE 3D datacube.

    This is a general purpose program that takes a 3D APOGEE datacube
    and processes it in various ways.  It will return a 2D image.
    It can remove cosmic rays, correct saturated pixels, do linearity
    corrections, dark corrections, flat field correction, and reduce
    the read noise with Fowler or up-the-ramp sampling.
    A pixel mask and variance array are also created.

    For a 2048x2048x60 datacube it currently takes ~140 seconds to
    process on halo (minus read time) and ~200 seconds on stream.

    Parameters
    ----------
    files : str or list
       The filename(s) of an APOGEE chip file, e.g. apR-a-test3.fits
         A compressed file can also be input (e.g. apR-a-00000033.apz)
         and the file will be automatically uncompressed.
    outfile : str
       The output filename.  
    apred : str
       APOGEE DRP reduction version, e.g. "1.3".
    detcorr : str
       Filename of the detector file (containing gain,
         rdnoise, and linearity correction).
    bpmcorr : str
       Filename of the bad pixel mask file.
    darkcorr : str
       Filename of the dark correction file.
    flatcorr : str
       Filename of the flat field correction file.
    littrowcorr : str
       Filename of the Littrow ghost mask file.
    persistcorr : str
       Filename of the persistence mask file.
    persistmodelcorr : str
       Filename for the persistence model parameter file.
    histcorr : str
       Filename of the 2D exposure history cube for this night.
    crfix : boolean, optional
       Fix cosmic rays.  This is done by default.
         If crfix is False then cosmic rays are still detected and
         flagged in the mask file, but NOT corrected.  Default is crfix=True.
    satfix : boolean, optional
       Fix saturated pixels.  This is done by default
         If satfix is False then saturated pixels are still detected
         and flagged in the mask file, but NOT corrected - 
         instead they are set to 0.  Saturated pixels that are
         not fixable ("unfixable", less than 3 unsaturated reads)
         are also set to 0.  Default is True.
    rd3satfix : boolean, optional
       Fix saturated pixels for 3 reads, and assume they don't
       have CRs.  Default is False.
    saturation : int or float, optional
       The saturation level.  The default is 65000.  
    nfowler : int, optional
       The number of samples to use for the Fowler sampling.
         The default is 10
    uptheramp : boolean, optional
       Do up-the-ramp sampling instead of Fowler.  Currently
         this does NOT take throughput variations into account
         and is only meant for Darks and Flats.
    outelectrons : boolean, optional
       The output images should be in electrons instead of ADU.
         The default is ADU.
    refonly : boolean, optional
       Only do reference subtraction of the cube and return.
         This is used for creating superdarks.
    criter : boolean, optional
       Iterative CR detection.  Check neighbors of pixels
         with detected CRs for CRs as well using a lower
         threshold.  Default is False.
    clobber : boolean, optional
       Overwrite output file if it already exists.  Default is False.
    outlong : boolean, optional
       The output files should use LONG type intead of FLOAT.
         This actually takes up the same amount of space, but
         this can be losslessly compressed with FPACK.
    cleanuprawfile : boolean, optional
       If a compressed file is input and ap3dproc needs to
         Decompress the file (no decompressed file is on disk)
         then setting this keyword will delete the decompressed
         file at the very end of processing.  This is the default.
         Set cleanuprawfile=0 if you want to keep the decompressed
         file.  An input decompressed FITS file will always
         be kept.
    nocr : boolean, optional
       Do not perform CR detection or fixing.  Default is False.
    logfile : str, optional
       Name of file to write logging information to.
    fitsdir : str, optional
       Directory for the temporary, uncompressed FITS files.
    maxread : int, optional
       Maximum read to use.  Default is all reads.
    q3fix : boolean, optional
       Fix 3rd quadrant issue in blue detector.  Default is False.
    usereference : boolean, optional
       Subtract off the reference array to reduce/remove crosstalk.
         Default is False.
    seq : str, optional
       Sequence number input by ap3d driver program.
    unlock : boolean, optional
       Delete lock file and start fresh.  Default is False.
    debug : boolean, optional
       For debugging.  Will make a plot of every pixel with a
         CR or saturation showing hot it was corrected (if
         that option was set) and gives more verbose output.
    verbose : boolean, optional
       Verbose output to the screen.  Default is True.

    Returns
    -------
    The output is a [Nx,Ny,3] datacube written to "outfile".  The 3
    planes are:
     (1) The final Fowler (or Up-the-Ramp) sampled 2D image (in ADU) with (optional)
          linearity correction, dark current correction, cosmic ray fixing,
          aturated pixel fixing, and flat field correction.
     (2) The variance image in ADU.
     (3) A bitwise flag mask, with values meaning:  1-bad pixel, 2-CR,
            4-saturated, 8-unfixable
          where unfixable means that there are not enough unsaturated
          reads (need at least 3) to fix the saturation.
    cube : numpy array
       The "fixed" datacube
    head : header object
       The final header
    output : numpy array
       The final output data [Nx,Ny,3].
    crtab : table
       The Cosmic Ray structure.
    satmask : numpy array
       The saturation mask [Nx,Ny,3], where the 1st plane is
         the 0/1 mask for saturation or not, 2nd plane is
         the read # at which it saturated (starting with 0), 3rd
         plane is the # of saturated pixels.

    Example
    ------
    
    im = ap3dproc('apR-a-test3.fits','ap2D-a-test3.fits')

    SUBROUTINES:
    ap3dproc_lincorr   Does the linearity correction
    ap3dproc_darkcorr  Does the dark correction
    ap3dproc_crfix     Detects and fixes CRs
    ap3dproc_plotting  Plots original and fixed data for pixels
                        affected by CRs or saturation (for debugging
                        purposes)

    How this program should be run for various file types:
    DARK - detcorr, bpmcorr, crfix, satfix, uptheramp
    FLAT - detcorr, bpmcorr, darkcorr, crfix, satfix, uptheramp
    LAMP - detcorr, bpmcorr, darkcorr, flatcorr, crfix, satfix, nfowler
    SCIENCE - detcorr, bpmcorr, darkcorr, flatcorr, crfix, satfix, nfowler

    Potential future improvements:
    -use the neighboring flux rates as a function reads to get a better
    extrapolation/fixing of the saturated pixels.
    -improve the up-the-ramp sampling to take the variability into
    account.  Need a local "throughput" curve that basically says
    how the throughput changes over time.  Then fit the data with
    data = throughput*a*exptime + b
    where a is the count rate in counts/sec and b is the kTC noise
    (offset).  Throughput runs from 0 to 1.  It should be possible to
    calculate a/b directly and not have to actually "fit" anything.
    -add an /electrons keyword to output the image in electrons instead
    of ADU
    -when calculating the sigma of dCounts in ap3dproc_crfix.pro also
    try to use an anlytical value in addition to an empirical value.
    -maybe don't set unfixable pixels with 2 good reads to zero. just
    assume that there is no CR.
    -use reference pixels
    
    By D.Nidever  Jan 2010
    translated to python by D.Nidever 2021/2022
    """

    t00 = time.time()
    
    nfiles = np.char.array(files).size
    if type(files) is str:
        files = [files]
    if type(outfile) is str:
        outfile = [outfile]
        
    if outfile is None:
        raise ValueError('OUTFILE must have same number of elements as FILES')

    # Default parameters
    if (nfowler is None or nfowler==0) and (uptheramp is None or uptheramp==False):
        # number of reads to use at beg and end
        nfowler = 10
    if seq is None:
        seq = 'no seq'
        
    # Print out the parameters we are using
    if verbose:
        print('AP3DPROC Input Parameters:')
        print('Saturation = ',str(int(saturation)))
        if crfix:
            print('Fixing Cosmic Rays')
        else:
            print('NOT Fixing Cosmic Rays')
        if satfix:
            print('Fixing Saturated Pixels')
        else:
            print('NOT Fixing Saturated Pixels')
        if nfowler: print('Using FOWLER Sampling, Nfowler='+str(int(nfowler)))
        if nfowler==False: print('Using UP-THE-RAMP Sampling')
        if criter:
            print('Iterative CR detection ON')
        else:
            print('Iterative CR detection OFF')
        if clobber: print('Clobber ON')
        if outelectrons:
            print('Output will be in ELECTRONS')
        else:
            print('Output will be in ADU')
        print('')
    print(str(nfiles),' File(s) input')

    # File loop
    #------------
    outlist = nfiles*[None]
    for f in range(nfiles):
        t0 = time.time()
        ifile = files[f]

        if verbose:
            if f > 0:
                print('')
            print(str(f+1)+'/'+str(nfiles)+' filename = '+ifile)
            print('----------------------------------')

        # Test if the output file already exists
        if outfile:
            if os.path.exists(outfile[f]) and clobber==False:
                print('OUTFILE = '+outfile[f]+' ALREADY EXISTS.  Set clobber=True to overwrite.')
                continue
            
        # If another job is working on this file, wait
        if outfile:
            #if utils.localdir() is not None:
            #    lockfile = os.path.join(utils.localdir(),load.apred)+'/'+os.path.basename(outfile[f])
            #else:
            #    lockfile = outfile[f]
            lock.lock(outfile[f],unlock=unlock)

            # Lock the file
            lock.lock(outfile[f],lock=True)

        # Load the data
        cube,head,doapunzip,fitsfile = loaddata(ifile,fitsdir=fitsdir,maxread=maxread,
                                                cleanuprawfile=cleanuprawfile,verbose=verbose)
        if cube is None: continue

        # Dimensions of the cube
        ny,nx,nreads = cube.shape
        chip = head.get('CHIP')
        if chip is None:
            raise ValueError('CHIP not found in header')

        # File dimensions
        if verbose:
            print('Data file description:')
            print('Datacube size = '+str(int(nx))+' x '+str(int(ny))+' x '+str(int(nreads)))
            print('Nreads = '+str(int(nreads)))
            print('Chip = '+str(chip))
            print('')
  
        # Few reads
        if nreads == 2 and verbose:
            print('Warning: Only 2 READS. Cannot perform CR detection/fixing')
        if nreads == 2 and satfix and verbose:
            print('Warning: Only 2 READS. Cannot fix Saturated pixels')

        # Load the calibration files
        rdnoiseim,gainim,lindata = loaddetector(detcorr)
        if bpmcorr:
            bpmim,bpmhead = loadbpm(bpmcorr)
        else:
            bpmim = None
        if littrowcorr:
            littrowim,littrowhead = loadlittrow(littrowcorr)
        else:
            littrowim = None
        if persistcorr:
            persistim,persisthead = loadpersist(persistcorr)
        else:
            persistim = None
        darkim,darkhead = loaddark(darkcorr)
        flatim,flathead = loadflat(flatcorr)

        # Check that the dark has enough reads
        nreads_dark = darkim.shape[2]
        if nreads_dark < nreads:
            error = 'SUPERDARK file '+darkcorr+' does not have enough READS.'
            error += 'Have '+str(nreads_dark)+' but need '+str(nreads)
            raise ValueError(error)

        if detcorr or darkcorr or flatcorr and verbose: print('')
  
        #---------------------
        # Check for BAD READS
        #---------------------
        if verbose:
            print('Checking for bad reads')        
        badflag,bdreads = checkbadreads(cube)
        if badflag: continue
        
        # Reference pixel subtraction
        #----------------------------
        cube,mask,readmask = refcorrect(cube,head,q3fix=q3fix,keepref=usereference)        
        bdreads2, = np.where(readmask == 1)
        nbdreads2 = len(bdreads2)
        if nbdreads2 > 0:
            bdreads = np.hstack((bdreads,bdreads2))
        nbdreads = len(np.unique(bdreads))
        if nbdreads > (nreads-2):
            print('Error: Not enough good reads')
            continue
        gdreads = np.arange(nreads)
        if nbdreads>0:
            gdreads = np.delete(gdreads,bdreads)
        ngdreads = len(gdreads)
        
        # Interpolate bad reads
        if nbdreads > 0:
            cube = interpbadreads(cube,gdreads,bdreads)
            
        shape = cube.shape
        ny,nx = shape[:2]
        
        # Reference subtraction ONLY
        if refonly:
            if verbose:
                print('Reference subtraction only')
            #hdu = fits.HDUList()
            #hdu.append(fits.ImageHDU(ref))
            #outlist[f] = hdu
            #continue
            # goto,BOMB
                
        # READ NOISE
        #-----------
        if rdnoiseim is not None:
            noise = np.median(rdnoiseim)
        else:
            noise = 12.0  # default value
        noise_dCounts = noise*np.sqrt(2)  # noise in dcounts
  
  
        # Initialize some arrays
        #------------------------
        med_dCounts_im = np.zeros((ny,nx),float)     # the median dCounts for each pixel
        satmask = np.zeros((ny,nx,3),int)            # 1st plane is 0/1 mask, 2nd plane is which read
                                                     #   it saturated on, 3rd plane is # of
                                                     #   saturated reads
        variability_im = np.zeros((ny,nx),float)     # fractional variability for each pixel
        sat_extrap_error = np.zeros((ny,nx),float)   # saturation extrapolation error 
    
        #--------------------------------------------
        # PROCESS EACH SLICE (one row) ONE AT A TIME
        #--------------------------------------------
  
        # Here is the method.  For each pixel find the dCounts for
        # each neighboring pair of reads.  Then find the median dCounts
        # and the RMS (robustly).  If there are any dCounts that are
        # many +sigma then this is a cosmic ray.  Then use the median
        # dCounts around that read to correct the CR.

        if verbose:
            print('Processing the datacube')

        # Loop over the rows
        #-------------------
        crtab = None
        for i in range(ny):
            if debug:
                print('Scanning Row ',str(i+1))
            if debug==False:
                if (i+1) % 500 == 0:
                    print('{:4d}/{:4d}'.format(i+1,ny))
    
            # Getting calibration data for this slice
            caldata = {'rdnoiseim':rdnoiseim,'gainim':gainim,'lindata':lindata,
                       'bpmim':None,'littrowim':None,'persistim':None,
                       'darkim':None,'flatim':None,'noise':noise,
                       'saturation':saturation}
            for c in ['bpm','littrow','persist','dark','flat']:
                if locals()[c+'im'] is not None:
                    caldata[c+'im'] = locals()[c+'im'][i]  # get ith row
            if lindata.ndim == 3:
                caldata['lindata'] = lindata[i]
                
            # Process the slice
            out = process_slice(cube[i].astype(float),mask[i],caldata,
                                crfix=crfix,nocr=nocr,satfix=satfix,
                                rd3satfix=rd3satfix,verbose=debug)
            slcfixed,slcmask,slcsatmask,crtabslc,med_dCounts,variability,sat_xerror = out

            # Put information back into the full arrays
            cube[i,:,:] = (np.round( slcfixed )).astype(int)   # round to closest integer
            mask[i,:] = slcmask
            satmask[i,:,:] = slcsatmask
            med_dCounts_im[i,:] = med_dCounts
            variability_im[i,:] = variability
            sat_extrap_error[i,:] = sat_xerror
            
            # Add to main crtab table
            if crtabslc['ncr']>0:
                crtabslc['data']['y'] = i  # add the row information
                if crtab is None:
                    crtab = crtabslc
                else:
                    ncrtot = crtab['ncr'] + crtabslc['ncr']
                    crtab = {'ncr':ncrtot,'data':np.hstack((crtab['data'],crtabslc['data']))}
                    
        #--------------------------------
        # Measure "variability" of data
        #--------------------------------
        # Use pixels with decent count rates
        crmask = ((mask & pixelmask.getval('CRPIX')) == pixelmask.getval('CRPIX')).astype(int)
        highpix = np.where((satmask[:,:2048,0] == 0) & (crmask == 0) & (med_dCounts_im[:,:2048] > 40))
        nhighpix = np.sum(highpix)
        if nhighpix == 0:
            highpix = np.where((satmask[:,:2048,0] == 0) & (med_dCounts_im[:,:2048] > 20))
            nhighpix = np.sum(highpix)
        if nhighpix > 0:
            global_variability = np.median(variability_im[:,:2048][highpix])
        else:
            global_variability = -1

            
        #------------------------
        # COLLAPSE THE DATACUBE
        #------------------------
        
        # Fowler Sampling
        #------------------
        if uptheramp == False:  
            # Make sure that Nfowler isn't too large
            Nfowler_used = Nfowler
            if Nfowler > Nreads//2:
                Nfowler_used = Ngdreads//2
  
            # Use the mean of Nfowler reads

            # Beginning sample
            gd_beg = gdreads[:nfowler_used]
            if len(gd_beg) == 1:
                im_beg = cube[:,:,gd_beg]/float(Nfowler_used)
            else:
                im_beg = np.sum(cube[:,:,gd_beg],axis=2)/float(Nfowler_used)

            # End sample
            gd_end = gdreads[ngdreads-nfowler_used:ngdreads]
            if len(gd_end) == 1:
                im_end = cube[:,:,gd_end]/float(Nfowler_used)
            else:
                im_end = np.sum(cube[:,:,gd_end],axis=2)/float(Nfowler_used)

            # The middle read will be used twice for 3 reads

            # Subtract beginning from end
            im = im_end - im_beg
  
            # Noise contribution to the variance
            sample_noise = noise * np.sqrt(2.0/Nfowler_used)
  
        # Up-the-ramp sampling
        #---------------------
        else:
            # For now just fit a line to each pixel
            #  Use median dCounts for each pixel to get the flux rate, i.e. "med_dCounts_im"
            #  then multiply by the exposure time.
  
            # Fit a line to the reads for each pixel
            #   dCounts are noisier than the actual reads by sqrt(2)
            #   See Rauscher et al.(2007) Eqns.3
  
            # Calculating the slope for each pixel
            #  t is the exptime, s is the signal
            #  we will use the read index for t
            sumts = np.zeros((shape[0],shape[1]),float)   # SUM t*s
            sums = np.zeros((shape[0],shape[1]),float)    # SUM s
            sumn = np.zeros((shape[0],shape[1]),int)      # SUM s
            sumt = np.zeros((shape[0],shape[1]),float)    # SUM t*s
            sumt2 = np.zeros((shape[0],shape[1]),float)   # SUM t*s
            for k in range(ngdreads):
                slc = cube[:,:,gdreads[k]]
                if satfix==False:
                    good = ((satmask[:,:,0] == 0) | ((satmask[:,:,0] == 1) & (satmask[:,:,1] > i)))
                else:
                    good = np.isfinite(slc)
                sumts[good] += gdreads[k]*slc[good]
                sums[good] += slc[good]
                sumn[good] += 1
                sumt[good] += gdreads[k]
                sumt2[good] += gdreads[k]**2
            # The slope in Counts per read, similar to med_dCounts_im
            slope = (sumn*sumts - sumt*sums)/(sumn*sumt2 - sumt**2)
            # To get the total counts just multiply by nread
            im = slope * (ngdreads-1)
            # the first read doesn't really add any signal, just a zero-point
            
            # See Equation 1 in Rauscher et al.(2007), SPIE
            #  with m=1
            #  noise and image/flux should be in electrons, sample_noise is in electrons
            sample_noise = np.sqrt( 12*(ngdreads-1.)/(nreads*(ngdreads+1.))*noise**2 + \
                                    6.*(ngdreads**2+1)/(5.*ngdreads*(ngdreads+1))*im[:,:2048]*gainim )
            sample_noise /= gainim  # convert to ADU
            
        # With userference, subtract off the reference array to reduce/remove
        #   crosstalk. 
        if usereference:
            print('Subtracting reference array...')
            tmp = im[:2048,:2048].copy()
            ref = im[:2048,2048:2560].copy()
            # subtract smoothed horizontal structure
            smref = medfilt(np.median(ref,axis=1),7)
            ref -= smref.reshape(-1,1) + np.zeros((2048,512),float)
            im = refcorrect_sub(tmp,ref)
            nx = 2048

        #-----------------------------------
        # Apply the Persistence Correction
        #-----------------------------------
        pmodelim,ppar = None,None
        if persistmodelcorr and histcorr:
            # NOT TESTED YET!!!
            if verbose:
                print('PERSIST modelcorr file = '+persistmodelcorr)
            pmodelim,ppar = persistmodel(ifile,histcorr,persistmodelcorr,bpmfile=bpmcorr)
            im -= pmodelim

        #------------------------
        # Calculate the Variance
        #------------------------
        # the total variance is the sum of the rdnoise and poisson noise
        # poisson noise = sqrt(Counts) so variance is just Counts
        #   need to divide by gain^2 ??
        # gain is normally in e/count
        # is rdnoise normally in counts or e-??
        # do "help apvariance" in IRAF and look for the noise model
        #
        # variance(ADU) = N(ADU)/Gain + rdnoise(ADU)^2
  
  
        # Initialize varim
        varim = np.zeros((ny,nx),float)         # variance in ADU
  
        # 1. Poisson Noise from the image: note that the equation for UTR
        #    noise above already includes Poisson term
        if uptheramp==False:
            if pmodelim is not None:
                varim += np.maximum( (im+pmodelim)/gainim , 0)
            else:
                varim += np.maximum(im/gainim,0)
     
        # 2. Poisson Noise from dark current
        if darkcorr:
            darkim = darkim[:,:,nreads-1]
            varim += np.maximum(darkim/gainim,0)
  
        # 3. Sample/read noise
        varim += sample_noise**2         # rdnoise reduced by the sampling
        
        # 4. Saturation error
        #      We used median(dCounts) to extrapolate the saturated pixels
        #      Use the variability in dCounts to estimate the error of doing this
        if satfix:
            varim += sat_extrap_error[:,:2048]     # add saturation extrapolation error
        else:
            varim = varim*(1-satmask[:,:2048,0]) + satmask[:,:2048,0]*99999999.   # saturated pixels are bad!
        # Unfixable pixels
        unfmask = ((mask & pixelmask.getval('UNFIXABLE')) == pixelmask.getval('UNFIXABLE')).astype(int)  # unfixable
        varim = varim*(1-unfmask) + unfmask*99999999.         # unfixable pixels are bad!
  
        # 5. CR error
        #     We use median of neighboring dCounts to "fix" reads with CRs
        crmask = ((mask & pixelmask.getval('CRPIX')) == pixelmask.getval('CRPIX')).astype(int)
        if crfix and crtab is not None:
            # loop in case there are multiple CRs per pixel
            for i in range(crtab['ncr']):
                if crtab['data'][i]['x'] < 2048:
                    varim[crtab['data'][i]['y'],crtab['data'][i]['x']] += crtab['data'][i]['fixerror']
                else:
                    varim = varim*(1-crmask) + crmask*99999999.               # pixels with CRs are bad!
  
        # Bad pixels
        bpmmask = ((mask & pixelmask.getval('BADPIX')) == pixelmask.getval('BADPIX')).astype(int)
        varim = varim*(1-bpmmask) + bpmmask*99999999.               # bad pixels are bad!
        
        # Flat field correction  
        if flatcorr:
            varim /= flatim**2
            im /= flatim

        # Now convert to ELECTRONS
        if detcorr and outelectrons:
            varim *= gainim**2
            im *= gainim

        #----------------------------
        # Construct output array
        #  [image, error, mask]
        #----------------------------
        if pmodelim:
            output = np.zeros((ny,nx,4),float)
        else:
            output = np.zeros((ny,nx,3),float)
        output[:,:,0] = im
        output[:,:,1] = np.maximum(np.sqrt(varim),1)  # must be greater than zero
        output[:,:,2] = mask
        if pmodelim:
            output[:,:,3] = pmodelim  # persistence model in ADU
  
        #-----------------------------
        # Update header
        #-----------------------------
        leadstr = 'AP3D: '
        pyvers = sys.version.split()[0]
        head['V_APRED'] = plan.getgitvers(),'APOGEE software version' 
        head['APRED'] = apred,'APOGEE Reduction version'
        head['HISTORY'] = leadstr+time.asctime()  # need better time output
        import socket
        head['HISTORY'] = leadstr+getpwuid(os.getuid())[0]+' on '+socket.gethostname()
        import platform
        head['HISTORY'] = leadstr+'python '+pyvers+' '+platform.system()+' '+platform.release()+' '+platform.architecture()[0]
        head['HISTORY'] = leadstr+' APOGEE Reduction Pipeline Version: '+apred
        head['HISTORY'] = leadstr+'Output File:'
        if detcorr and outelectrons:
            head['HISTORY'] = leadstr+' HDU1 - image (electrons)'
            head['HISTORY'] = leadstr+' HDU2 - error (electrons)'
        else:
            head['HISTORY'] = leadstr+' HDU1 - image (ADU)'
            head['HISTORY'] = leadstr+' HDU2 - error (ADU)'
        head['HISTORY'] = leadstr+' HDU3 - flag mask'
        head['HISTORY'] = leadstr+'        1 - bad pixels'
        head['HISTORY'] = leadstr+'        2 - cosmic ray'
        head['HISTORY'] = leadstr+'        4 - saturated'
        head['HISTORY'] = leadstr+'        8 - unfixable'
        if pmodelim:
            head['HISTORY'] = leadstr+' HDU4 - persistence correction (ADU)'
        head['HISTORY'] = leadstr+'Global fractional variability = {:5.3f}'.format(global_variability)
        maxlen = 72-len(leadstr)
        # Bad pixel mask file
        if bpmcorr:
            line = 'BAD PIXEL MASK file="'+bpmcorr+'"'
            if len(line) > maxlen:
                line1 = line[:maxlen]
                line2 = line[maxlen:]
                head['HISTORY'] = leadstr+line1
                head['HISTORY'] = leadstr+line2
            else:
                head['HISTORY'] = leadstr+line
        # Detector file
        if detcorr:
            line = 'DETECTOR file="'+detcorr+'"'
            if len(line) > maxlen:
                line1 = line[:maxlen]
                line2 = line[maxlen:]
                head['HISTORY'] = leadstr+line1
                head['HISTORY'] = leadstr+line2
            else:
                head['HISTORY'] = leadstr+line
        # Dark Correction File
        if darkcorr:
            line = 'Dark Current Correction file="'+darkcorr+'"'
            if len(line) > maxlen:
                line1 = line[:maxlen]
                line2 = line[maxlen:]
                head['HISTORY'] = leadstr+line1
                head['HISTORY'] = leadstr+line2
            else:
                head['HISTORY'] = leadstr+line
        # Flat field Correction File
        if flatcorr:
            line = 'Flat Field Correction file="'+flatcorr+'"'
            if len(line) > maxlen:
                line1 = line[:maxlen]
                line2 = line[maxlen:]
                head['HISTORY'] = leadstr+line1
                head['HISTORY'] = leadstr+line2
            else:
                head['HISTORY'] = leadstr+line
        # Littrow ghost mask File
        if littrowcorr:
            line = 'Littrow ghost mask file="'+littrowcorr+'"'
            if len(line) > maxlen:
                line1 = line[:maxlen]
                line2 = line[maxlen:]
                head['HISTORY'] = leadstr+line1
                head['HISTORY'] = leadstr+line2
            else:
                head['HISTORY'] = leadstr+line
        # Persistence mask File
        if persistcorr:
            line = 'Persistence mask file="'+persistcorr+'"'
            if len(line) > maxlen:
                line1 = line[:maxlen]
                line2 = line[maxlen:]
                head['HISTORY'] = leadstr+line1
                head['HISTORY'] = leadstr+line2
            else:
                head['HISTORY'] = leadstr+line
        # Persistence model file
        if persistmodelcorr:
            line = 'Persistence model file="'+persistmodelcorr+'"'
            if len(line) > maxlen:
                line1 = line[:maxlen]
                line2 = line[maxlen:]
                head['HISTORY'] = leadstr+line1
                head['HISTORY'] = leadstr+line2
            else:
                head['HISTORY'] = leadstr+line
        # History file
        if histcorr:
            line = 'Exposure history file="'+histcorr+'"'
            if len(line) > maxlen:
                line1 = line[:maxlen]
                line2 = line[maxlen:]
                head['HISTORY'] = leadstr+line1
                head['HISTORY'] = leadstr+line2
            else:
                head['HISTORY'] = leadstr+line
        # Bad pixels 
        totbpm = np.sum((mask & pixelmask.getval('BADPIX')) == pixelmask.getval('BADPIX'))
        head['HISTORY'] = leadstr+str(int(totbpm))+' pixels are bad'
        # Cosmic Rays
        totcr = np.sum((mask & pixelmask.getval('CRPIX')) == pixelmask.getval('CRPIX'))
        if nreads > 2:
            head['HISTORY'] = leadstr+str(int(totcr))+' pixels have cosmic rays'
        if crfix and nreads>2:
            head['HISTORY'] = leadstr+'Cosmic Rays FIXED'
        # Saturated pixels
        totsat = np.sum((mask & pixelmask.getval('SATPIX')) == pixelmask.getval('SATPIX'))
        totunf = np.sum((mask & pixelmask.getval('UNFIXABLE')) == pixelmask.getval('UNFIXABLE'))
        totfix = totsat-totunf
        head['HISTORY'] = leadstr+str(int(totsat))+' pixels are saturated'
        if satfix and nreads>2:
            head['HISTORY'] = leadstr+str(int(totfix))+' saturated pixels FIXED'
        # Unfixable pixels
        head['HISTORY'] = leadstr+str(int(totunf))+' pixels are unfixable'
        # Sampling
        if uptheramp:
            head['HISTORY'] = leadstr+'UP-THE-RAMP Sampling'
        else:
            head['HISTORY'] = leadstr+'Fowler Sampling, Nfowler='+str(int(Nfowler_used))
        # Persistence correction factor
        if pmodelim is not None and ppar is not None:
            sppar = ['{:7.3g}'.format(p) for p in ppar]
            head['HISTORY'] = leadstr+'Persistence correction: '+' '.join(sppar)
  
        # Fix EXPTIME if necessary
        if head['NFRAMES'] != nreads:
            # NFRAMES is from ICC, NREAD is from bundler which should be correct
            exptime = nreads*10.647  # secs
            head['EXPTIME'] = exptime
            print('not halting, but NFRAMES does not match NREADS, NFRAMES: '+str(head['NFRAMES'])+' NREADS: '+str(nreads)+'  '+seq)

        # Add UT-MID/JD-MID to the header
        jd = Time(head['DATE-OBS'],format='fits').jd
        #jd = date2jd(head['DATE-OBS'])
        exptime = head['EXPTIME']
        jdmid = jd + (0.5*exptime)/24/3600
        utmid = Time(jd,format='jd').fits
        #utmid = jd2date(jdmid)
        head['UT-MID'] = utmid,' Date at midpoint of exposure'
        head['JD-MID'] = jdmid,' JD at midpoint of exposure'

        # remove CHECKSUM
        del head['CHECKSUM']
  
        #----------------------------------
        # Output the final image and mask
        #----------------------------------
        if outfile:
            ioutfile = outfile[f]  
            # Does the output directory exist?
            if os.path.exists(os.path.dirname(ioutfile))==False:
                print('Creating '+os.path.dirname(ioutfile))
                os.makedirs(os.path.dirname(ioutfile))
            # Test if the output file already exists
            filexists = os.path.exists(ioutfile)
            if verbose:
                print('')
            if filexists and clobber:
                print('OUTFILE = '+ioutfile+' ALREADY EXISTS.  OVERWRITING')
            if filexists and clobber==False:
                print('OUTFILE = '+ioutfile+' ALREADY EXISTS. ')

        # Create the HDUList
        #-------------------
        # HDU0 - header only
        hdu = fits.HDUList()
        hdu.append(fits.ImageHDU(header=head))
        # HDU1 - flux
        flux = output[:,:,0]
        # replace NaNs with zeros
        bad = np.where(np.isfinite(flux)==False)
        nbad = np.sum(bad)
        if nbad > 0:
            flux[bad] = 0.
        if outlong:
            flux = np.round(flux).astype(int)
        else:
            flux = flux.astype(np.float32)
        hdu.append(fits.ImageHDU(flux))
        hdu[1].header['CTYPE1'] = 'Pixel'
        hdu[1].header['CTYPE2'] = 'Pixel'
        if outelectrons:
            hdu[1].header['BUNIT'] = 'Flux (electrons)'
        else:
            hdu[1].header['BUNIT'] = 'Flux (ADU)'
        hdu[1].header['EXTNAME'] = 'FLUX'
        # HDU2 - error
        # errout sets the value to output for the error for bad pixels
        err = output[:,:,1]
        bd = np.where((err==BADERR) | (err <= 0) | (~np.isfinite(err)))
        if np.sum(bd)>0:
            err[bd] = BADERR
        if outlong:
            err = np.round(err).astype(int)
        else:
            err = err.astype(np.float32)
        hdu.append(fits.ImageHDU(err))
        hdu[2].header['CTYPE1'] = 'Pixel'
        hdu[2].header['CTYPE2'] = 'Pixel'
        if outelectrons:
            hdu[2].header['BUNIT'] = 'Error (electrons)'
        else:
            hdu[2].header['BUNIT'] = 'Error (ADU)'            
        hdu[2].header['EXTNAME'] = 'ERROR'
        # HDU3 - mask
        hdu.append(fits.ImageHDU(mask.astype(np.int16)))
        hdu[3].header['CTYPE1'] = 'Pixel'
        hdu[3].header['CTYPE2'] = 'Pixel'
        hdu[3].header['BUNIT'] = 'Flag Mask (bitwise)'
        hdu[3].header['HISTORY'] = 'Explanation of BITWISE flag mask'
        hdu[3].header['HISTORY'] = ' 1 - bad pixels'
        hdu[3].header['HISTORY'] = ' 2 - cosmic ray'
        hdu[3].header['HISTORY'] = ' 4 - saturated'
        hdu[3].header['HISTORY'] = ' 8 - unfixable'
        hdu[3].header['EXTNAME'] = 'MASK'
        # HDU4 - persistence model
        if pmodelim:
            hdu.append(fits.ImageHDU(pmodelim.astype(np.float32)))
            hdu[4].header['CTYPE1'] = 'Pixel'
            hdu[4].header['CTYPE2'] = 'Pixel'
            hdu[4].header['BUNIT'] = 'Persistence correction (ADU)'
            hdu[4].header['EXTNAME'] = 'PERSIST CORRECTION'
            
        # Write the file
        if filexists==False or clobber:        
            if verbose:
                print('Writing output to: '+ioutfile)
            if outlong:
                print('Saving FLUX/ERR as LONG instead of FLOAT')
            if os.path.exists(ioutfile): os.remove(ioutfile)
            hdu.writeto(ioutfile,overwrite=True)
            
        # Remove the recently Decompressed file
        if ifile.split('.')[-1] == 'apz' and cleanuprawfile and doapunzip:
            print('Deleting recently decompressed file ',fitsfile)
            if os.path.exists(fitsfile): os.remove(fitsfile)
    
        # Number of saturated and CR pixels
        print('')
        print('BAD/CR/Saturated Pixels:')
        print(str(int(totbpm))+' pixels are bad')
        print(str(int(totcr))+' pixels have cosmic rays')
        print(str(int(totsat))+' pixels are saturated')
        print(str(int(totunf))+' pixels are unfixable')
        print('')
            
        # Remove the lock file
        if outfile:
            lock.lock(outfile[f],clear=True)
  
        dt = time.time()-t0
        if verbose:
            print('dt = {:10.1f} sec'.format(dt))
        if logfile:
            fmt = '{:10.2f} {:8d} {:8d} {:8d} {:8d}'
            name = os.path.basename(ifile)+(fmt.format(dt,totbpm,totcr,totsat,totunf))
            utils.writelog(logfile,name)

        if nfiles > 1:
            dt = time.time()-t00
            if verbose:
                print('dt = {:10.1f} sec'.format(dt))

        # Add to the outlist list
        outlist[f] = hdu

    if nfiles==1: outlist=outlist[0]

    return outlist
                

def ap3d(planfiles,verbose=False,rogue=False,clobber=False,refonly=False,unlock=False):
    """
    This program processes all of the APOGEE RAW datacubes for
    a single night.

    Parameters
    ----------
    planfiles : str or list
       Input list of plate plan files
    verbose : boolean, optional
       Print a lot of information to the screen. Default is False.
    unlock : boolean, optional
       Delete lock file and start fresh.  Default is False
    clobber : boolean, optional
       Delete any existing output files.  Default is False.

    Returns
    -------
    The RAW APOGEE 3D datacube files are processed and 2D 
    images are output.

    Example
    -------

    ap3d(planfiles)

    Written by D.Nidever  Feb. 2010
    Modifications J. Holtzman 2011+
    Translated to python by D.Nidever  Jan 2022
    """

    t0 = time.time()

    if type(planfiles) is str:
        planfiles = [planfiles]
    nplanfiles = len(planfiles)

    print('')
    print('RUNNING AP3D')
    print('')
    print(str(nplanfiles)+' PLAN file(s)')

    chips = ['a','b','c']

    #--------------------------------------------
    # Loop through the unique PLATE Observations
    #--------------------------------------------
    for i in range(nplanfiles):
        t1 = time.time()

        planfile = planfiles[i]

        print('')
        print('=========================================================================')
        print(str(i+1)+'/'+str(nplanfiles)+'  Processing Plan file '+planfile)
        print('=========================================================================')
        
        # Load the plan file
        #--------------------
        print('')
        print('Plan file information:')
        planstr = plan.load(planfile,np=True,verbose=True)
        if 'apred_vers' not in planstr:
            print('apred_vers not found in planfile')
            continue
        if 'telescope' not in planstr:
            print('telescope not found in planfile')
            continue
        load = apload.ApLoad(apred=planstr['apred_vers'],telescope=planstr['telescope'])
        logfile = load.filename('Diag',plate=planstr['plateid'],mjd=planstr['mjd'])        

        # Check that we have all of the calibration IDs in the plan file
        ids = ['detid','darkid','flatid','bpmid','littrowid','persistid','persistmodelid']
        calid = ['Detector','Dark','Flat','BPM','Littrow','Persist','PersistModel']
        for id1,calid1 in zip(ids,calid):
            if id1 not in planstr:
                print(id1+' not in planstr')
                continue
            print('%-10s %s' % (str(id1)+':',planstr[id1]))
        
        # Try to make the required calibration files (if not already made)
        # Then check if the calibration files exist
        #--------------------------------------

        caltypes = ['det','dark','flat','bpm','littrow','persist','persistmodel']
        calnames = ['Detector','Dark','Flat','BPM','Littrow','Persist','PersistModel']
        for i in range(len(caltypes)):
            caltype = caltypes[i]
            calid = caltype+'id'
            calname = calnames[i]
            if planstr[calid] != 0:
                if load.exists(calname,num=planstr[calid]):
                    print(load.filename(calname,num=planstr[calid],chips=True)+' already exists')
                else:
                    out = subprocess.run(['makecal','--'+calname.lower(),planstr[calid]],shell=False)
                    if load.exists(calname,num=planstr[calid])==False:
                        raise ValueError(load.filename(calname,num=planstr[calid],chips=True)+' NOT FOUND')

        # apHist file
        if planstr['persistmodelid']>0:
            if load.exists('Hist',num=planstr['mjd']):
                print(load.filename('Hist',num=planstr['mjd'],chips=True)+' already exists')
            else:
                mjdcube.mjdcube(planstr['mjd'],dark=planstr['darkid'])
                if load.exists('Hist',num=planstr['mjd'])==False:
                    raise ValueError(load.filename('Hist',num=planstr['mjd'],chips=True)+' NOT FOUND')

  
        # Are there enough files
        if 'APEXP' not in planstr:
            print('APEXP not found in planstr')
            continue
        nframes = len(planstr['APEXP'])
        if nframes < 1:
            print('No frames to process')
            continue

        # Process each frame
        #-------------------
        for j in range(nframes):
            framenum = planstr['APEXP']['name'][j]
            rawfile = load.filename('R',num=planstr['APEXP']['name'][j],chips=True)
            chipfiles = [rawfile.replace('R-','R-'+ch+'-') for ch in chips]
            if load.exists('R',num=framenum)==False:
                print(rawfile+' NOT found')
                continue

            print('')
            print('--------------------------------------------------------')
            print(str(j+1),'/',str(nframes),'  Processing files for Frame Number >>',str(framenum),'<<')
            seq = str(j+1)+'/'+str(nframes)
            print('--------------------------------------------------------')
            
            # Determine file TYPE
            #----------------------
            # dark - should be processed with 
            # flat
            # lamps
            # object frame
            exptype = planstr['APEXP']['flavor'][j]
            if exptype == '' or exptype == '0':
                error = 'NO OBSTYPE keyword found for '+str(framenum)
                print(error)
                continue
            exptype = str(exptype).lower()

            # This is a DARK frame
            #----------------------
            if exptype == 'dark':
                print('This is a DARK frame.  This should be processed with APMKSUPERDARK.PRO')
                continue

            # Settings to use for each exposure type            
            kws = {'usedet':True,'usebpm':True,'usedark':True,'useflat':True,'uselittrow':True,
                   'usepersist':True,'dopersistcorr':False,'nocr':True,'crfix':False,'criter':False,
                   'satfix':True,'uptheramp':False,'nfowler':1,'rd3satfix':False}
            # Flat
            if exptype=='psf':
                print('This is a FLAT frame')
                # default settings
            # Lamp
            elif exptype=='lamp':
                print('This is a LAMP frame')
                kws['nocr'] = False
                kws['crfix'] = True
            # Wave
            elif exptype=='wave':
                print('This is a WAVE frame')
                kws['nocr'] = False
                kws['crfix'] = True
            # Object
            elif exptype=='object':
                print('This is an OBJECT frame')
                kws['dopersistcorr'] = True
                kws['nocr'] = False
                kws['crfix'] = True
                if planstr['platetype'] == 'single': kws['nocr']=True
                kws['uptheramp'] = True
                kws['nflower'] = 0
            # Flux
            elif exptype=='flux':
                print('This is an FLUX frame')
                # default settings
            else:
                print(exptype+' NOT SUPPORTED')
                continue

            #----------------------------------
            # Looping through the three chips
            #----------------------------------
            for k in range(3):
                chfile = chipfiles[k]

                # Check header
                head = fits.getheader(chfile)
                # Check that this is a data CUBE OR has extensions
                naxis = head.get('NAXIS')
                try:
                    dumim,dumhead = fits.getdata(chfile,1,header=True)
                    read_message = ''
                except:
                    read_message = 'problem'
                if naxis != 3 and read_message != '':
                    error = 'FILE must contain a 3D DATACUBE OR image extensions'
                    print(error)
                    continue

                # Chip specific calibration filenames
                kws['histcorr'] = None
                ids = ['detid','darkid','flatid','bpmid','littrowid','persistid','persistmodelid']
                calid = ['Detector','Dark','Flat','BPM','Littrow','Persist','PersistModel']
                calflag = ['usedet','usedark','useflat','usebpm','uselittrow','usepersist','dopersistcorr']
                gotcals = True
                for id1,calid1,calflag1 in zip(ids,calid,calflag):
                    caltype1 = id1[:-2]
                    calfile = None
                    if kws[calflag1] and planstr[id1]!=0:
                        if calid1=='Littrow' and chips[k]!='b':
                            continue
                        if calid1=='PersistModel' and chips[k]!='c':
                            continue
                        if calid1=='PersistModel' and chips[k]=='c':
                            kws['histcorr'] = histfiles[k]
                        calfile = load.filename(calid1,num=planstr[id1],chips=True)
                        calfile = calfile.replace(calid1+'-',calid1+'-'+chips[k]+'-')
                        # Does the file exist
                        if load.exists(calid1,num=planstr[id1])==False:
                            print(calfile+' NOT found')
                            gotcals = False
                            continue
                    kws[caltype1+'corr'] = calfile
                # Do not have all the cals
                if gotcals==False:
                    print('Do not have all the calibration files that we need')
                    continue

                # Do not have all the cals
                
                #if usedet and planstr['detid'] != 0:
                #      detcorr = detfiles[k]
                #if usebpm and planstr['bpmid'] != 0:
                #      bpmcorr = bpmfiles[k]
                #if usedark and planstr['darkid'] != 0:
                #      darkcorr = darkfiles[k]
                #if useflat and planstr['flatid'] != 0:
                #      flatcorr = flatfiles[k]
                #if uselittrow and planstr['littrowid'] != 0 and k == 1:
                #      littrowcorr = littrowfiles
                #if usepersist and planstr['persistid'] != 0:
                #      persistcorr = persistfiles[k]
                #if dopersistcorr and (planstr['persistmodelid'] != 0) and (chips[k] == 'c'):
                #    #        and (chiptag[k] == 'b' or (chiptag[k] == 'c' and planstr.mjd < 56860L)):
                #    persistmodelcorr = persistmodelfiles[k-1]
                #    histcorr = histfiles[k]
                #else:
                #    persistmodelcorr = None

                # note this q3fix still fails for apo1m flat/PSF processing, which calls ap3dproc directly
                q3fix = False
                if 'q3fix' in planstr:
                    if k==2 and planstr['q3fix']==1:
                        q3fix = True
                if k==2 and planstr['mjd'] > 56930 and planstr['mjd'] < 57600:
                    q3fix = True

                if 'usereference' in planstr:
                      usereference = bool(planstr['usereference'])
                else:
                      usereference = True
                if 'maxread' in planstr:
                      maxread = planstr['maxread']
                else:
                      maxread = None

                print('')
                print('-----------------------------------------')
                print(' Processing chip '+chips[k]+' - '+os.path.basename(chfile))
                print('-----------------------------------------')
                print('')

                # Output file
                outfile = load.filename('2D',num=framenum,mjd=planstr['mjd'],chips=True)
                outfile = outfile.replace('2D-','2D-'+chips[k]+'-')
                # Does the output directory exist?
                fitsdir = os.path.join(utils.localdir(),load.apred)
                if os.path.exists(os.path.dirname(outfile))==False:
                    os.makedirs(os.path.dirname(outfile))
                # Does the file exist already
                #if os.path.exists(outfile) and clobber==False:
                #    print(outfile,' exists already and clobber not set')
                #    continue
                
                # PROCESS the file
                #-------------------
                ap3dproc(chfile,outfile,load.apred,cleanuprawfile=True,verbose=verbose,
                         clobber=clobber,logfile=logfile,q3fix=q3fix,maxread=maxread,
                         fitsdir=fitsdir,usereference=usereference,refonly=refonly,
                         seq=seq,unlock=unlock,**kws)

    utils.writelog(logfile,'AP3D: '+os.path.basename(planfile)+('%8.2f' % time.time()))

    print('AP3D finished')
    dt = time.time()-t0
    print('dt = %.1f sec ' % dt)

            
