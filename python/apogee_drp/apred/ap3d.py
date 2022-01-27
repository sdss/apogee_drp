#!/usr/bin/env python


"""AP3D.PY - APOGEE software to process 3D data cubes.

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20200404'  # yyyymmdd                                                                                                                           

import os
import numpy as np
import warnings
from astropy.io import fits
from astropy.table import Table, Column
from astropy import modeling
from glob import glob
from scipy.signal import medfilt
from scipy.ndimage.filters import median_filter,gaussian_filter1d
from scipy.optimize import curve_fit, least_squares
from scipy.special import erf
from scipy.interpolate import interp1d
#from numpy.polynomial import polynomial as poly
#from lmfit import Model
#from apogee.utils import yanny, apload
#from sdss_access.path import path
import bindata

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")



def ap3dproc_lincorr(slice_in,lindata,linhead):
    """
    This subroutine does the Linearity Correction for a slice

    The lindata array gives a quadratic polynomial either for
    each pixel or for each output (512 pixels)
    The polynomial should take the observed counts and convert
    them into corrected counts, i.e.
    counts_correct = poly(counts_obs,coef)
    """

    readtime = 10.0   # the time between reads

    lny,lnx = slice_in.shape
    nreads = sz[1]

    szlin = lindata.shape

    # Each pixel separately
    #-----------------------
    if szlin[0]==2048:

        # Need to figure out the threshold for each pixel
        # Only want to correct data that need it and are
        # in the nonlinear regime
        #   y = c0 + c1*x + c2*x^2
        # Want to know where the deviation of this from linearity
        # is greater than some fraction
        #   y_lin = 0.0 + 1.0*x
        #   y-ylin = c0+c1*x+c2*x^2 - x > frac
        # Solve for x
        #   c2*x^2 + (c1-1)*x + c0 > frac
        # Solutions are
        #   x = -(c1-1)+/-sqrt( (c1-1)^2-4*c2*c0)/(2*c2)
        #linthresh = lindata
        
        # This takes you from OBSERVED COUNTS to CORRECTED COUNTS
        # x = observed counts
        # y = corrected counts

        coef0 = lindata[:,0].repeat(nreads).reshape(2048,nreads)
        coef1 = lindata[:,1].repeat(nreads).reshape(2048,nreads)
        coef2 = lindata[:,2].repeat(nreads).reshape(2048,nreads)        
        slice_out = coef0 + coef1*slice_in + coef2*slice_in**2

    # Each output separately
    #------------------------
    else:
        # a separate coefficient for each output (512 columns)
        coef0 = np.zeros((2048,nreads),float)
        coef1 = np.zeros((2048,nreads),float)
        coef2 = np.zeros((2048,nreads),float)
        corr = np.zeros((2048,nreads),float)
        npar = szlin[1]
        # loop over quadrants
        slice_out = slice_in.copy()
        for i in range(4):
            corr[512*i:512*i+512,:] = lindata[i,0]
            x = slice_in[512*i:512*i+512,:]
            for j in range(npar-1):
                corr[512*i:512*i+512,:] += lindata[i,j+1]*x
                x *= slice_in[512*i:512*i+512,:]
        slice_out = slice_in*corr
        for i in range(4):
            coef0[512*i:512*i+512,:] = lindata[i,0]
            coef1[512*i:512*i+512,:] = lindata[i,1]
            coef2[512*i:512*i+512,:] = lindata[i,2]
        slice_out = coef0 + coef1*slice_in + coef2*slice_in**2
    
    return slice_out



def ap3dproc_darkcorr(slice_in,darkslice,darkhead):
    """
    This subroutine does the Dark correction for a slice
    darkslice is a 2048xNreads array that gives the dark counts

    To get the dark current just multiply the dark count rate
    by the time for each read
    """

    nreads = slice_in.shape[1]

    # Just subtract the darkslice
    # subtract each read at a time in case we still have the reference pixels
    slice_out = slice_in.copy()
    for i in range(nreads):
        slice_out[0:2048,i] -= darkslice[:,i]

    return slice_out


def ap3dproc_crfix(dCounts,satmask,sigthresh=10,onlythisread=False,noise=17.0,crfix=False):
    """
    This subroutine fixes cosmic rays in a slice of the datacube.
    The last dimension in the slice should be the Nreads, i.e.
    [Npix,Nreads].

    Parameters:
    dCounts        The difference of neighboring pairs of reads.
    satmask        The saturation mask [Nx,3].  1-saturation mask,
                     2-read # of first saturation, 3-number of saturated reads
    =sigthresh     The Nsigma threshold for detecting a
                     CR. sigthresh=10 by default
    =onlythisread  Only accept CRs in this read index (+/-1 read).
                     This is only used for the iterative CR rejection.
    =noise         The readnoise in ADU (for dCounts, not single reads)
    =crfix         Actually fix the CR and not just detect it

    Returns:
    crstr          Structure that gives information on the CRs.
    dCounts_fixed  The "fixed" dCounts with the CRs removed.
    med_dCounts    The median dCounts for each pixel
    mask           An output mask for with the CRs marked
    crindex        At what read did the CR occur
    crnum          The number of CRs in each pixel of this slice
    variability   The fractional variability in each pixel

    """

    npix,nreads = dCounts.shape
    # nreads is actually Nreads-1

    # Initialize the CRSTR with 50 CRs, will trim later
    #  don't know Y right now
    dtype = np.dtype([('x',int),('y',int),('read',int),('counts',float),('nsigma',float),('globalsigma',float),
                      ('fixed',bool),('localsigma',float),('fixerror',float),('neicheck',bool)])
    crstr = np.zeros(100,dtype)
    #crstr_data_def = {X:0L,Y:0L,READ:0L,COUNTS:0.0,NSIGMA:0.0,GLOBALSIGMA:0.0,FIXED:0,LOCALSIGMA:0.0,FIXERROR:0.0,NEICHECK:0}
    #crstr = {NCR:0L,DATA:REPLICATE(crstr_data_def,50)}

    # Initializing dCounts_fixed
    dCounts_fixed = dCounts.copy()


    #-----------------------------------
    # Get median dCounts for each pixel
    #-----------------------------------
    med_dCounts = np.nanmedian(dCounts,axis=1)    # NAN are automatically ignored

    # Check if any medians are NAN
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

    med_dCounts2D = med_dCounts.repeat(nreads).reshape((npix,nreads))   # 2D version


    #-----------------------------------------------------
    # Get median smoothed/filtered dCounts for each pixel
    #-----------------------------------------------------
    #  this should help remove transparency variations

    smbin = np.minimum(11, nreads)    # in case Nreads is small
    if nreads > smbin:
        sm_dCounts = medfilt2d(dCounts,smbin,axis=2)
    else:
        sm_dCounts = med_dCounts2D

    # We need to deal with reads near saturated reads carefully
    # otherwise the median will be over less reads
    # For now this is okay, at worst the median will be over smbin/2 reads
        
    # If there are still some NAN then replace them with the global
    # median dCounts for that pixel.  These are probably saturated
    # so it probably doesn't matter
    bdnan, = np.where(np.finite(sm_dCounts) == False)
    nbdnan = len(bdnan)
    if nbdnan>0:
        sm_dCounts[bdnan] = med_dCounts2D[bdnan]

    #--------------------------------------
    # Variability from median (fractional)
    #--------------------------------------
    variability = dln.mad(dCounts-med_dCounts2D,axis=1,zero=True)
    variability = variability / np.maximum(med_dCounts, 0.001)  # make it a fractional variability
    bdvar, = np.where(np.finite(variability) == False)
    nbdvar = len(bdar)
    if nbdvar>0:
        variability[bdvar] = 0.0  # all NAN
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

    sig_dCounts2D = sig_dCounts.repeat(nreads).reshape((npix,nreads))   # 2D version
    
    #-----------
    # Find CRs
    #-----------
    # threshold for number of sigma above (local) median
    nsig_thresh = np.maximum(nsig_thresh, 3)    # 3 at a minimum

    # Saturated dCounts (NANs) are automatically ignored
    nsigma_slice = (dCounts-sm_dCounts)/sig_dCounts2D
    bd1D, = np.where( ( nsigma_slice > nsig_thresh ) &
                      ( dCounts > noise*nsig_thresh ))
    nbd1D = len(bd1D)
    
    if verbose: print(str(nbd1D)+' CRs found')

    ### DLN GOT TO HERE  8/6/2020

    
    # Some CRs found
    if nbd1D>0:
        bd2D = array_indices(dCounts,bd1D)
        bdx = bd2D[0,:]  # column
        bdr = bd2D[1,:]  # read

        # Correct the CRs and correct the pixels
        for j in range(nbd1D):
            ibdx = (bdx[j])[0]
            ibdr = (bdr[j])[0]

            dCounts_pixel = dCounts[ibdx,:]

            # ONLYTHISREAD
            #  for checking neighboring pixels in the iterative part
            #--------------
            if len(onlythisread) > 0:
                # onlthisread is the read index, while ibdr is a dCounts index
                # ibdr+1 is the read index for this CR
                if (ibdr+1) < onlythisread[0]-1 or (ibdr+1) > onlythisread[0]+1:
                    import pdb; pdb.set_trace()


            # Calculate Local Median and Local Sigma
            #----------------------------------------
            #   Use a local median/sigma so the affected CR dCount is not included
            # more than 2 good dCounts and Nreads>smbin
            if (totgd[ibdx] > 2) and (nreads > smbin):
                dCounts_pixel[ibdr] = np.nan  # don't use affected dCounts    
    
                maxind = nreads-1
                if satmask[ibdx,0] == 1:
                    maxind = satmask[ibdx,1]-2  # don't include saturated reads (NANs)
                lor = np.maximum((ibdr-smbin/2),0)
                hir = np.minumum( (lor + smbin-1), maxind )
                if (hir == maxind):
                    lor = np.maximum((hir-smbin+1),0)
    
                # -- Local median dCounts --
                #  make sure the indices make sense
                #if (lor < 0 or hir < 0 or hir <= lor) then import pdb; pdb.set_trace()
                if (lor < 0 or hir < 0 or hir <= lor):
                    local_med_dCounts = med_dCounts[ibdx]
                else:
                    local_med_dCounts = np.median(dCounts_pixel[lor:hir])
    
                # If local median dCounts is NAN use all reads
                if np.isfinite(local_med_dCounts) == False:
                    local_med_dCounts = med_dCounts[ibdx]
                # If still NaN then set to 0.0
                if np.isfinite(local_med_dCounts) == False:
                    local_med_dCounts = 0.0
    
                # -- Local sigma dCounts --
                local_sigma = dln.mad(dCounts_pixel[lor:hir]-local_med_dCounts,zero=True)
    
                # If local sigma dCounts is NAN use all reads
                #   this should never actually happen
                if np.isfinite(local_sigma) == False:
                    local_sigma = sig_dCounts[ibdx]
                # If still NaN then set to noise
                if np.isfinite(local_sigma) == False:
                    local_sigma = noise
                
            # Only 2 good dCounts OR Nreads<smbin
            else:
                local_med_dCounts = med_dCounts[ibdx]
                local_sigma = sig_dCounts[ibdx]


            local_med_dCounts = med_dCounts[ibdx]
            local_sigma = sig_dCounts[ibdx]

            # Fix the CR
            #------------
            if keyword_set(crfix):
                if keyword_set(verbose):
                    print(' Fixing CR at Column '+str(ibdx)+' Read '+str(ibdr+1))

                # Replace with smoothed dCounts, i.e. median of neighbors
                dCounts_fixed[ibdx,ibdr] = local_med_dCounts       # fix CR dCounts
                #dCounts_fixed[ibdx,ibdr] = sm_dCounts[ibdx,ibdr]   # fix CR dCounts

                # Error in the fix
                #   by taking median of smbin neighboring reads we reduce the error by ~1/sqrt(smbin)
                fixerror = local_sigma/sqrt(smbin-1)   # -1 because the CR read is in there


            # CRSTR stuff
            #--------------

            # Expand CR slots in CRSTR
            if crstr.ncr == len(crstr['data']):
                old_crstr = crstr
                nold = len(old_crstr['data'])
                crstr = {'ncr':0,'data':np.repeat(crstr_data_def,nold+50)}
                STRUCT_ASSIGN,old_crstr,crstr   # source, destination
                old_crstr = None

            # Add CR to CRSTR
            crstr['data'][crstr.ncr].x = ibdx
            crstr['data'][crstr.ncr].read = ibdr+1  # ibdr is dCounts index, +1 to get read
            crstr['data'][crstr.ncr].counts = dCounts[ibdx,ibdr] - sm_dCounts[ibdx,ibdr]
            crstr['data'][crstr.ncr].nsigma = nsigma_slice[ibdx,ibdr]
            crstr['data'][crstr.ncr].globalsigma = sig_dCounts[ibdx]
            if crfix:
                crstr['data'][crstr.ncr].fixed = 1
            crstr['data'][crstr.ncr].localsigma = local_sigma
            if crfix:
                crstr['data'][crstr.ncr].fixerror = fixerror
            crstr.ncr += 1

    #  Replace the dCounts with CRs with the median smoothed values
    #    other methods could be used to "fix" the affected read,
    #    e.g. polynomial fitting/interpolation, Gaussian smoothing, etc.

    # Now trim CRSTR
    if crstr.ncr>0:
        old_crstr = crstr
        crstr = {'ncr':old_crstr.ncr,'data':old_crstr['data'][0:old_crstr['ncr']]}
        old_crstr = None
    else:
        crstr = {'ncr':0}   # blank structure


    return crstr, dCounts_fixed, med_dCounts, mask, crindex, crnum, variability


def loaddetector(detcorr,silent=True):
    """ Load DETECTOR FILE (with gain, rdnoise and linearity correction). """
    

    # DETCORR must be scalar string
    if (type(detcorr) != str) | (dln.size(bpmcorr) > 1):
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
    if silent is False: print('DET file = '+detcorr)
  
    # Check that the file looks reasonable
    # Must be 2048x2048 or 4 and have be float
    if ((gainim.ndim==2) & (gainim.shape != (2048,2048))) | ((gainim.ndim==1) & (gainim.size != 4)) | (type(gainim) != np.float32):
        raise ValueError('GAIN image must be 2048x2048 or 4 FLOAT image')
  
    # If Gain is 4-element then make it an array
    if gainim.size == 4:
        gainim0 = gainim.copy()
        gainim = np.zeros((2048,2048),float)
        for k in range(4):
            gainim[:,k*512:(k+1)*512] = gainim0[k]
  
    # Must be 2048x2048 or 4 and have be float
    rny,rnx = rdnoiseim.shape
    if ((rdnoiseim.ndim==2) & (rdnoiseim.shape != (2048,2048)) | ((rdnoiseim.ndim==1) & (rdnoiseim.size != 4)) | (type(rdnoiseim) != np.float32)):
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
        
        szlin = size(lindata)
        linokay = 0
        if (szlin[0] == 2 and szlin[1] == 4 and szlin[2] == 3):
            linokay = 1
        if (szlin[0] == 3 and szlin[1] == 2048 and szlin[2] == 2048 and szlin[3] == 3):
            linokay = 1
        if linokay==0:
            raise ValueError('Linearity correction data must be 2048x2048x3 or 4x3')


    return XX,XX
  

def loadbpm(bpmcorr,silent=True):
    """ Load BAD PIXEL MASK (BPM) File """

    # BPMCORR must be scalar string
    if (type(bmpcorr) != str) | (dln.size(bpmcorr) > 1):
        raise ValueError('BPMCORR must be a scalar string with the filename of the BAD PIXEL MASK file')
    # Check that the file exists
    if os.path.exists(bpmcorr) is False:
        raise ValueError('BPMCORR file '+bpmcorr+' NOT FOUND')
    
    # Load the BPM file
    #  This should be 2048x2048
    bpmim,bpmhead = fits.getdata(bpmcorr,header=True)
    if silent is False:
        print('BPM file = '+bpmcorr)
 
    # Check that the file looks reasonable
    #  must be 2048x2048 and have 0/1 values
    ny,nx = bpmim.shape
    bpmokay = 0
    if (bpmim.ndim != 2) | (nx != 2048) | (ny != 2048):
        raise ValueError('BAD PIXEL MASK must be 2048x2048 with 0/1 values')

    return bpmim,bpmhead


def loadlittrow(littrowcorr,silent=True):
    """ Load LITRROW MASK File """

    # LITTROWCORR must be scalar string
    if type(littrowcorr) is not str or np.array(littrowcorr).shape != 1:
        error = 'LITTROWCORR must be a scalar string with the filename of the LITTROW MASK file'
        raise ValueError(error)
  
    # Check that the file exists
    if os.path.exists(littrowcorr)==False:
        error = 'LITTROWCORR file '+littrowcorr+' NOT FOUND'
        raise ValueError(error)
    
    # Load the LITTROW file
    #  This should be 2048x2048
    littrowim,littrowhead = fits.getdata(littrowcorr,header=True)
  
    if silent==False:
        print('LITTROW file = '+littrowcorr)
  
    # Check that the file looks reasonable
    #  must be 2048x2048 and have 0/1 values
    nyl,nxl = littrowim.shape
    littrowokay = 0
    dum, = np.where((littrowim != 0) and( littrowim != 1))
    nbad = len(dum)
    if littrowim.ndim != 2 or nxl != 2048 or nyl != 2048 or nbad > 0:
        error = 'LITTROW MASK must be 2048x2048 with 0/1 values'
        raise ValueError(error)

    return littrowim,littrowhead


def loadpersist(persistcorr,silent=True):
    """ Load PERSISTENCE MASK File """
  
    # PERSISTCORR must be scalar string
    if type(persistcorr) is not str or np.array(persistcorr).shape != 1:    
        error = 'PERSISTCORR must be a scalar string with the filename of the PERSIST MASK file'
        raise ValueError(error)
  
    # Check that the file exists
    if os.path.exists(persistcorr)==False:
        error = 'PERSISTCORR file '+persistcorr+' NOT FOUND'
        raise ValueError(error)
    
    # Load the PERSIST file
    #  This should be 2048x2048
    persistim,persisthead = fits.getdata(persistcorr,header=True)
  
    if silent==False:
        print('PERSIST file = '+persistcorr)
  
    # Check that the file looks reasonable
    #  must be 2048x2048 and have 0/1 values
    szpersist = size(persistim)
    persistokay = 0
    if szpersist[0] != 2 or szpersist[1] != 2048 or szpersist[2] != 2048:
        error = 'PERSISTENCE MASK must be 2048x2048'
        raise ValueError(error)

    return persistim,persisthead


def loaddark(darkcorr,silent=True):  
    """ Load DARK CORRECTION file """

    # DARKCORR must be scalar string
    if type(darkcorr) is not str or np.array(darkcorr).shape != 1:    
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
        nreads_dark = 0
        message = ''
        while (message == ''):
            nreads_dark += 1
            dum = fits.getheader(darkcorr,nreads_dark,errmsg=message)
        nreads_dark -= 1  # removing the last one
  

    # Check that it has enough reads
    #nreads_dark = sxpar(darkhead,'NAXIS3')
    if nreads_dark < nreads:
        error = 'SUPERDARK file '+darkcorr+' does not have enough READS. Have '+str(nreads_dark)+\
                ' but need '+str(nreads)
        raise ValueError(error)
    
    # Load the dark correction file
    #  This needs to be 2048x2048xNreads
    #  It's the dark counts for each pixel in counts
  
    # Datacube
    if darkhead['NAXIS'] == 3:
        darkcube = fits.getdata(darkcorr)

    # Extensions
    else:
  
        # Initializing the cube
        darkim,exthead = fits.getdata(darkcorr,1)
        ny,nx = darkim.shape
        darkcube = np.zeros((ny,nx,nreads_dark),float)
  
        # Read in the extensions
        for k in np.arange(1,nreads_dark):
            extim,exthead = fits.getdata(darkcorr,k)
            darkcube[:,:,k-1] = extim
  
    szdark = size(darkcube)
  
    if silent==False:
        print('Dark Correction file = '+darkcorr)
  
    # Check that the file looks reasonable
    szdark = size(darkcube)
    if (szdark[0] != 3 or szdark[1] < 2048 or szdark[2] != 2048):
        error = 'Dark correction data must a 2048x2048xNreads datacube of the dark counts per pixel'
        raise ValueError(error)
    
    return darkcube,darkhead


def loadflat(flatcorr,silent=True):
    """ Load FLAT FIELD CORRECTION file """
  
    # FLATCORR must be scalar string
    if type(flatcorr) is not str or np.array(flatcorr).shape != 1:    
        error = 'FLATCORR must be a scalar string with the filename of the flat correction file'
        raise ValueError(error)
  
    # Check that the file exists
    if os.path.exists(flatcorr)==False:
        error = 'FLATCORR file '+flatcorr+' NOT FOUND'
        raise ValueError(error)
    
    # Load the flat correction file
    #  This needs to be 2048x2048
    flatim,flathead = fits.getdata(flatcorr,header=True)
  
    if silent==False:
        print('Flat Field Correction file = '+flatcorr)
  
    # Check that the file looks reasonable
    szflat = size(flatim)
    if (szflat[0] != 2 or szflat[1] != 2048 or szflat[2] != 2048):
        error = 'Flat Field correction image must a 2048x2048 image'
        raise ValueError(error)

    return flatim,flathead


# refsub subtracts the reference array from each quadrant with proper flipping
def aprefcorr_sub(image,ref):
    revref = np.flip(ref,axis=1)
    image[:,0:512] -= ref
    image[:,512:1024] -= revref
    image[:,1024:1536] -= ref
    image[:,1536:2048] -= revref
    return image

def aprefcorr(cube,head,mask,indiv=3,vert=1,horz=1,noflip=False,silent=False,
              readmask=None,lastgood=None,cds=1,plot=False,fix=False,
              q3fix=False,keepref=False):
    """
    This corrects a raw APOGEE datacube for the reference pixels
    and reference output

    Parameters:
    cube       The raw APOGEE datacube with reference array.  This
                will be updated with the reference subtracted cube.
    head       The header for CUBE.
    indiv=n    Subtract the individual reference arrays after nxn median filter. If 
                If <0, subtract mean reference array. If ==0, no reference array subtraction
    /noflip    Do not flip the reference array.
    /silent    Don't print anything to the screen.

    Returns:
    cube is updated with the reference subtracted cube to save memory.
    mask       The flag mask.
    =readmask  Mask indicating if reads are bad (0-good, 1-bad)

    USAGE:
    >>>aprefcorr,cube,head,mask

    By J. Holtzman   2011
    Incorporated into ap3dproc.pro  D.Nidever May 2011
    """

    # refcorr does the "bias" subtraction, using the reference array and
    #    the reference pixels. Subtract a mean reference array (or individual
    #    with /indiv), then subtract vertical ramps from each quadrant using
    #    reference pixels, then subtract smoothed horizontal ramps

    # Number of reads
    nx,ny,nread = cube.shape

    # create long output
    out = np.zeros((2048,2048,nread),int)
    if keepref:
        refout = np.zeros((512,2048,nread),int)

    # Ignore reference array by default
    # Default is to do CDS, vertical, and horizontal correction
    print('in aprefcorr, indiv: '+str(indiv))

    satval = 55000

    snmin = 10
    if indiv>0:
        hmax = 1e10
    else:
        hmax = 65530

    if len(mask)<=1:
        mask = np.zeros((2048,2048),int)
    readmask = np.zeros(nread,int)
    if silent==False:
        print('Calculating mean reference')
    meanref = np.zeros((512,2048),float)
    nref = np.zeros((512,2048),int)
    for i in range(nread):
        ref = cube[2048:2560,:,i]

        m = np.mean(ref[128:512-128,128:2048-128].astype(np.float64))
        s = np.std(ref[128:512-128,128:2048-128].astype(np.float64))
        h = np.max(ref[128:512-128,128:2048-256])
        ref[ref>=sat] = np.nan        
        # SLICE business is just for special fast handling, ignored if
        #   not in header
        card = 'SLICE%03d' % i
        iread = head.getval(card)
        count = len(icard)
        if count==0:
            iread = i+1
        if silent==False:
            print('reading ref: %3d %3d\r' % (i,iread))
        # skip first read and any bad reads
        if (iread > 1) and (m/s > snmin) and (h < hmax):
            good, = np.where(np.isfinite(ref))
            meanref[good] += (ref[good]-m)
            nref[good] += 1
            readmask[i] = 0
        else:
            print('Rejecting: ',i,m,s,h)
            readmask[i] = 1

    meanref /= nref

    if silent == False:
        print('Reference processing ')


#### DLN got to here

        
    # Create vertical and horizontal ramp images
    rows = np.arange(2048,dtype=float)
    cols = np.ones(512,dtype=int)
    vramp = (cols.reshape(-1,1)*rows.reshape(1,-1))/2048
    vrramp = 1-vramp
    cols = np.arange(2048,dtype=float)
    rows = np.ones(2048,dtype=int)
    hramp = (cols.reshape(-1,1)*rows.reshape(1,-1))/2048
    hrramp = 1-hramp
    clo = np.zeros(2048,float)
    chi = np.zeros(2048,float)

    if cds:
        cdsref = cube[0:2048,:,1]

    # Loop over the reads
    lastgood = nread-1
    for iread in range(nread):

        # Subtract mean reference array
        red = cube[0:2048,:,iread].astype(int)

        ### I GOT TO HERE !!!!
    
        sat, = np.where(red > satval)
        nsat = len(sat)
        if nsat > 0:
            if iread == 0:
                nsat0 = nsat
            red[sat] = 65535
            mask[sat] = (mask[sat] or maskval('SATPIX'))
            # if we have a lot of saturated pixels, note this read (but don't do anything)
            if nsat > nsat0+2000:
                if lastgood == nread-1:
                    lastgood = iread-1
        else:
            nsat0 = 0
        # pixels that are identically zero are bad, see these in first few reads
        bad, = np.where(red == 0)
        nbad = len(bad)
        if nbad > 0:
            mask[bad] = (mask[bad] or maskval('BADPIX'))
        if silent==False:
            print(format='(%"Ref processing: %3d  nsat: %5d\r",$)',iread+1,len(sat))
        if readmask[iread] > 0:
            red = np.nan
            goto,nextread
  
            
        # with cds keyword, subtract off first read before getting reference pixel values
        if cds:
            red -= cdsref

        ref = cube[2048:2559,:,iread]
        if indiv==1:
            red = aprefcorr_sub(red,ref)
            ref -= ref
        elif indiv>1:
            ref = aprefcorr_sub(red,median(ref,indiv))
            ref -= np.median(ref,indiv)
        elif indiv<0:
            red = aprefcorr_sub(red,meanref)
            ref -= meanref
  
        if vert:
            # Subtract vertical ramp
            for j in range(4):
                rlo = np.nanmean(red[2:4,j*512:(j+1)*512])
                rhi = np.nanmean(red[2045:2048,j*512:(j+1)*512])
                red[:,j*512:(j+1)*512] -= rlo*vrramp
                red[:,j*512:(j+1)*512] -= rhi*vramp     
                #if keyword_set(plot):
                #  plot,rlo*vrramp[0,:]+rhi*vramp[0,:]
                #  print(j,rlo,rhi 
                #  atv,cube[0:2048,:,iread]-cube[0:2048,:,1]
                #  atv,red
                #  import pdb; pdb.set_trace()
                #

        # Subtract smoothed horizontal ramp
        if horz:
            clo = np.nanmean(im[:,1:4],axis=0)
            chi = np.nanmean(im[:,2044:2048],axis=0)
            #clo = total(red[1:3,:],1,/nan) / ( total(finite(red[1:3,:]),1) > 1)
            #chi = total(red[2044:2046,:],1,/nan) / ( total(finite(red[2044:2046,:]),1) > 1)

        sm = 7
        slo = medfilt(clo,sm)
        shi = medfilt(chi,sm)

        if noflip:
            red -= (rows.reshape(-1,1)*slo.reshape(1,-1))*hrramp
            red -= (rows.reshape(-1,1)*shi.reshape(1,-1))*hramp
            #red -= (rows#slo)*hrramp
            #red -= (rows#shi)*hramp
        else:
            #bias = (rows#slo)*hrramp+(rows#shi)*hramp
            # just use single bias value of minimum of left and right to avoid bad regions in one
            bias = rows.reshape(-1,1) * np.min([[slo],[shi]],axis=2).reshape(-1,1)
            fbias = bias
            fbias[512:1024,:] = reverse(bias[512:1024,:])
            fbias[1536:2048,:] = reverse(bias[1536:2048,:])
            red -= fbias

        if q3fix:
            #fix=red
            q3offset = np.zeros(2048,float)
            for irow in range(2048):
                q2m = np.median(red[923:1023,irow])
                q3a = np.median(red[1024:1124,irow])
                q3b = np.median(red[1435:1535,irow])
                q4m = np.median(red[1536:1636,irow])
                #fix[1024:1535,irow]+=((q2m-q3a)+(q4m-q3b))/2.
                q3offset[irow] = ((q2m-q3a)+(q4m-q3b))/2.
            #plot,q3offseta
            #oplot,medfilt1d(q3offset,7,/edge),color=2
            #red=fix
            red[1024:1535,:] += medfilt1d(q3offset,7)##(fltarr(512)+1))
            #atv,red,min=-200,max=200,/linear
            #import pdb; pdb.set_trace()

        # Make sure saturated pixels are set to 65535
        #  removing the reference values could have
        #  bumped them lower
        if nsat > 0:
            red[sat] = 65535

        #nextread:
        #reduced[:,:,iread] = red
        #cube[0:2048,:,iread] = red  # overwrite with the ref-subtracted image
        out[:,:,iread] = red
        if keepref:
            refout[:,:,iread] = ref

    # Trim off the reference array
    #cube = cube[0:2048,:,:]

    # mask the reference pixels
    mask[0:4,:] = (mask[0:4,:] or maskval('BADPIX'))
    mask[2044:2048,:] = (mask[2044:2048,:] or maskval('BADPIX'))
    mask[:,0:4] = (mask[:,0:4] or maskval('BADPIX'))
    mask[:,2044:2048] = (mask[:,2044:2048] or maskval('BADPIX'))

    if silent==False:
        print('')
        print('lastgood: ',lastgood)


    if keepref:
        return [out,refout]
    else:
        return out

def ap3dproc(files0,outfile,detcorr=None,bpmcorr=None,darkcorr=None,littrowcorr=None,
             persistcorr=None,persistmodelcorr=None,histcorr=None,
             flatcorr=None,crfix=True,satfix=True,rd3satfix=False,saturation=65000,
             nfowler=None,uptheramp=None,verbose=False,debug=False,silent=False,
             cube=None,head=None,output=None,crstr=None,satmask=None,criter=False,
             clobber=False,cleanuprawfile=True,outlong=False,refonly=False,
             outelectrons=False,nocr=False,logfile=None,fitsdir=None,maxread=None,
             q3fix=False,usereference=False,seq=None):
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
    files        The filename(s) of an APOGEE chip file, e.g. apR-a-test3.fits
                   A compressed file can also be input (e.g. apR-a-00000033.apz)
                   and the file will be automatically uncompressed.
    outfile      The output filename.  
    =detcorr     The filename of the detector file (containing gain,
                   rdnoise, and linearity correction).
    =bpmcorr     The filename of the bad pixel mask file.
    =darkcorr    The filaname of the dark correction file.
    =flatcorr    The filename of the flat field correction file.
    =littrowcorr The filename of the Littrow ghost mask file.
    =persistcorr   The filename of the persistence mask file.
    =persistmodelcorr   The filename for the persistence model parameter file.
    =histcorr    The filename of the 2D exposure history cube for this night.
    =crfix       Fix cosmic rays.  This is done by default.
                  If crfix=0 then cosmic rays are still detected and
                  flagged in the mask file, but NOT corrected.
    =satfix      Fix saturated pixels.  This is done by default
                  If satfix=0 then saturated pixels are still detected
                  and flagged in the mask file, but NOT corrected - 
                  instead they are set to 0.  Saturated pixels that are
                  not fixable ("unfixable", less than 3 unsaturated reads)
                  are also set to 0.
    =rd3satfix   Fix saturated pixels for 3 reads, and assume they don't
                  have CRs.
    =saturation  The saturation level.  The default is 65000
    =nfowler      The number of samples to use for the Fowler sampling.
                  The default is 10
    /uptheramp   Do up-the-ramp sampling instead of Fowler.  Currently
                  this does NOT taken throughput variations into account
                  and is only meant for Darks and Flats
    /outelectrons  The output images should be in electrons instead of ADU.
                    The default is ADU.
    /refonly     Only do reference subtraction of the cube and return.
                  This is used for creating superdarks.
    /criter      Iterative CR detection.  Check neighbors of pixels
                  with detected CRs for CRs as well using a lower
                  threshold.  This is the default.
    /clobber     Overwrite output file if it already exists.  clobber=0
                  by default.
    /outlong     The output files should use LONG type intead of FLOAT.
                   This actually takes up the same amount of space, but
                   this can be losslessly compressed with FPACK.
    /cleanuprawfile   If a compressed file is input and ap3dproc needs to
                      Decompress the file (no decompressed file is on disk)
                      then setting this keyword will delete the decompressed
                      file at the very end of processing.  This is the default.
                      Set cleanuprawfile=0 if you want to keep the decompressed
                      file.  An input decompressed FITS file will always
                      be kept.
    /debug       For debugging.  Will make a plot of every pixel with a
                  CR or saturation showing hot it was corrected (if
                  that option was set) and gives more verbose output.
    /verbose     Verbose output to the screen.
    /silent      Don't print anything to the screen

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
    =cube        The "fixed" datacube
    =head        The final header
    =output      The final output data [Nx,Ny,3].
    =crstr       The Cosmic Ray structure.
    =satmask     The saturation mask [Nx,Ny,3], where the 1st plane is
                  the 0/1 mask for saturation or not, 2nd plane is
                  the read # at which it saturated (starting with 0), 3rd
                  plane is the # of saturated pixels.

    USAGE:
    >>>im = ap3dproc('apR-a-test3.fits','ap2D-a-test3.fits')

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
    #apgundef,cube,head,output,crstr,satmask

    nfiles = len(files0)

    # No output requested
    #if len(outfile) == 0 and not arg_present(cube) and not arg_present(head) and 
    #    not arg_present(output) and not arg_present(crstr) and not arg_present(satmask):
    #    error = 'No output requested'
    #    print(error)
    #    return


    noutfile = len(outfile)
    if outfile is None:
        raise ValueError('OUTFILE must have same number of elements as FILES')

    # Default parameters
    if nfowler is None and uptheramp is None:      # number of reads to use at beg and end
        nfowler = 10
    if seq is None:
        seq = 'no seq'
  
    if silent==False:
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
    for f in range(nfiles):
        t0 = time.time()
        ifile = files0[f]

        if silent==False:
            if f > 0:
                print('')
            print(str(f+1),'/',str(nfiles),' Filename = ',ifile)
            print('----------------------------------')

        # if another job is working on this file, wait
        if len(outfile) > 0:
            if getlocaldir():
                
                lockfile = getlocaldir()+'/'+os.path.basename((outfile[f])+'.lock')
            else:
                lockfile = outfile[f]+'.lock'
            while os.path.exists(lockfile):
                apwait,lockfile,10

            # Test if the output file already exists
            if (os.path.exists(outfile[f]) or os.path.exists(outfile[f]+'.fz')) and clobber==False:
                if silent==False:
                    print('OUTFILE = ',outfile[f],' ALREADY EXISTS. Set clobber to overwrite')
                continue

            # set lock to notify other jobs that this file is being worked on
            #openw,lock,/get_lun,lockfile
            #free_lun,lock

        # Check the file
        fdir = os.path.dirname(ifile)
        base = os.path.basename(ifile)
        nbase = len(base)
        basesplit = os.path.splitext(base)
        extension = basesplit[-1]
        #if strmid(base,0,4) != 'apR-' or strmid(base,len-5,5) != '.fits':
        #  error = 'FILE must be of the form >>apR-a/b/c-XXXXXXXX.fits<<'
        #  if silent==False then print(error)
        #  return
        #
        if extension != 'fits' and extension != 'apz':
            error = 'FILE must have a ".fits" or ".apz" extension'
            if silent==False:
                print(error)
            continue
  
        # Compressed file input
        if extension == 'apz':
            if silent==False:
                print(ifile,' is a COMPRESSED file')

            # Check if the decompressed file already exists
            nbase = len(base)
            if fitsdir is not None:
                fitsfile = fitsdir+'/'+base[0,nbase-4]+'.fits' 
            else:
                fitsfile = fdir+'/'+base[0,nbase-4]+'.fits'
                fitsdir = 0
  
            # Need to decompress
            if os.path.exists(fitsfile)==False:
                if silent==False:
                    print('Decompressing with APUNZIP')
                num = int(base[6:6+8])
                if num < 2490000:
                    no_checksum = 1
                else:
                    no_checksum = 0 
                print('no_checksum: ', no_checksum)
                apzip.apunzip(ifile,clobber=True,fitsdir=fitsdir,no_checksum=True)
                print('')
                doapunzip = True     # we ran apunzip
  
                # An error occurred
                if len(errzip) > 0:
                    error = 'ERROR in APUNZIP '+errzip
                    if silent==False:
                        print('halt: '+error)
                    import pdb; pdb.set_trace()
                    continue

            # Decompressed file already exists
            else:
                if silent==False:
                    print('The decompressed file already exists')
                doapunzip = False     # we didn't run apunzip

            file = fitsfile  # using the decompressed FITS from now on

            # Cleanup by default
            if cleanuprawfile==False and extension=='apz' and doapunzip==1:   # remove recently decompressed file
                cleanuprawfile = True

        # Regular FITS file input
        else:
            file = ifile
            doapunzip = False
 
        if silent==False:
            if extension == 'apz' and cleanuprawfile and doapunzip == 1:
                print('Removing recently decompressed FITS file at end of processing')
  
        # Check that the file exists
        if os.path.exists(file)==False:
            error = 'FILE '+file+' NOT FOUND'
            if silent==False:
                print('halt: '+error)
            import pdb; pdb.set_trace()
            continue
 
        # Get header
        head = fits.getheader(file,errmsg=errmsg)
        if errmsg != '':
            error = 'There was an error loading the HEADER for '+file
            if silent==False:
                print('halt: '+error)
            import pdb; pdb.set_trace()
            continue
  
        # Check that this is a data CUBE
        naxis = head['NAXIS']
        dumim,dumhead = fits.getdata(file,1,header=True)
        if naxis != 3:
            error = 'FILE must contain a 3D DATACUBE OR image extensions'
            if silent==False:
                print('halt: '+error)
            import pdb; pdb.set_trace()
            continue

        # Test if the output file already exists
        if outfile is not None:
            if os.path.exists(outfile[f]) and clobber==False:
                print('OUTFILE = ',outfile[f],' ALREADY EXISTS.  Set /clobber to overwrite.')
                continue

        # Read in the File
        #-------------------
        if os.path.exists(file)==False:
            error = file+' NOT FOUND'
            if silent==False:
                print('halt: '+error)
            import pdb; pdb.set_trace()
            continue
        # DATACUBE
        if naxis==3:
            cube,head = fits.getdata(file,header=True)  # uint
        # Extensions
        else:
            head = fits.getheader(file)
            # Figure out how many reads/extensions there are
            #  the primary unit should be empty
            nreads = 0
            message = ''
            while (message == ''):
                nreads += 1
                dum = fits.getheader(file,nreads,errmsg=message)
            nreads -= 1  # removing the last one
  
            # Only 1 read
            if nreads < 2:
                error = 'ONLY 1 read.  Need at least two'
                raise ValueError(error)

            # allow user to specify maximum number of reads to use (e.g., in the
            #   case of calibration exposures that may be overexposed in some chip
            if maxread:
               if maxread < nreads:
                  nreads = maxread
  
            # Initializing the cube
            im1 = fits.getdata(file,1)
            ny,nx = im1.shape
            cube = np.zeros((ny,nx,nreads),int)    # long is big enough and takes up less memory than float
  
            # Read in the extensions
            for k in np.arange(1,nreads+1):
                extim,exthead = fits.getdata(file,k,header=True)  # uint
                cube[:,:,k] = extim
                # What do we do with the extension headers???
                # We could make a header structure or array
  
        # Dimensions of the cube
        ny,nx,nreads = cube.shape
        #type = size(cube,/type)  # UINT
        chip = head.get('CHIP')
        if chip is None:
            raise ValueError('CHIP not found in header')

        # File dimensions
        if silent==False:
            print('Data file description:')
            print('Datacube size = '+str(int(nx))+' x '+str(int(ny))+' x '+str(int(nreads)))
            print('Nreads = '+str(int(nreads)))
            print('Chip = '+str(chip))
            print('')
  
        # Few reads
        if nreads == 2 and silent==False:
            print('Only 2 READS. CANNOT do CR detection/fixing')
        if nreads == 2 and satfix and silent==False:
            print('Only 2 READS. CANNOT fix Saturated pixels')

        # Load the detector file
        XX,YY,ZZ = loaddetector(detcorr)
        # Load the bad pixel mask
        bpmim,bpmhead = loadbpm(bpmcorr)
        # Load the littrow mask file
        littrowim,littrowhead = loadlittrow(littrowcorr)
        # Load the persistence file
        persistim,persisthead = loadpersist(persistcorr)
        # Load the dark cube
        darkcube,darkhead = loaddark(darkcorr)
        # Load the flat image
        flatim,flathead = loadflat(flatcorr)

        
    ## I GOT TO HERE !!!!!

    if len(detcorr) > 0 or len(darkcorr) > 0 or len(flatcorr) > 0 and silent==False:
        print('')
  
  
    #---------------------
    # Check for BAD READS
    #---------------------
    if silent==False:
        print('Checking for bad reads')
  
    # Use the reference pixels and reference output for this
    
    if sz[1] == 2560:
        refout1 = median(cube[2048:,:,0:3<(nreads-1)],dim=3)
        sig_refout_arr = np.zeros(nreads,float)
        rms_refout_arr = np.zeros(nreads,float)
  
  
    refpix1 = [[ median(cube[0:2047,0:3,0:3<(nreads-1)],dim=3) ], 
               [transpose( median(cube[0:3,:,0:3<(nreads-1)],dim=3) ) ],
               [transpose( median(cube[2044:2047,:,0:3<(nreads-1)],dim=3) ) ],
               [ median(cube[0:2047,2044:2047,0:3<(nreads-1)],dim=3) ]]
    sig_refpix_arr = np.zeros(nreads,float)
    rms_refpix_arr = np.zeros(nreads,float)

    for k in range(nreads):
        refpix = [[cube[0:2047,0:3,k]], [transpose(cube[0:3,:,k])],
                  [transpose(cube[2044:2047,:,k])], [cube[0:2047,2044:2047,k]]]
        refpix = float(refpix)
  
        # The top reference pixels are normally bad
        diff_refpix = refpix - refpix1
        sig_refpix = dln.mad(diff_refpix[:,0:11],zero=True)
        rms_refpix = np.sqrt(np.mean(diff_refpix[:,0:11]**2))
  
        sig_refpix_arr[k] = sig_refpix
        rms_refpix_arr[k] = rms_refpix
  
        # Using reference pixel output (5th output)
        if sz[1] == 2560:
            refout = float(cube[2048:,:,k])
  
            # The top and bottom are bad
            diff_refout = refout - refout1
            sig_refout = dln.mad(diff_refout[:,100:1950],zero=True)
            rms_refout = np.sqrt(np.mean(diff_refout[:,100:1950]**2))
  
            sig_refout_arr[k] = sig_refout
            rms_refout_arr[k] = rms_refout

    # Use reference output and pixels
    if sz[1] == 2560:
        if nreads>2:
            med_rms_refpix_arr = medfilt(rms_refpix_arr,11<nreads)
            med_rms_refout_arr = medfilt(rms_refout_arr,11<nreads)
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
            med_rms_refpix_arr = medfilt(rms_refpix_arr,11<nreads)
        else:
            med_rms_refpix_arr = np.zeros(nreads,float)+np.median(rms_refpix_arr)
    
        sig_rms_refpix_arr = np.maximum(dln.mad(rms_refpix_arr),1)
        bdreads, = np.where( (rms_refpix_arr-med_rms_refpix_arr) > 10*sig_rms_refpix_arr)
        nbdreads = len(bdreads)
    
    if nbdreads == 0:
        bdreads = None
        
    # Too many bad reads
    if nreads-nbdreads < 2:
        raise ValueError('ONLY '+str(nreads-nbdreads)+' good reads.  Need at least 2.')

  
    # Reference pixel subtraction
    #----------------------------
    tmp = aprefcorr(cube,head,mask,readmask=readmask,q3fix=q3fix,keepref=usereference)
    cube = tmp

    bdreads2, = np.where(readmask == 1)
    nbdreads2 = len(bdreads2)
    if nbdreads2 > 0:
        bdreads = np.hstack((bdreads,bdreads))
    nbdreads = len(np.unique(bdreads))
  
    if nbdreads > (nreads-2):
        raise ValueError('Not enough good reads')

    gdreads = np.arange(nreads)
    REMOVE,bdreads,gdreads
    ngdreads = len(gdreads)

    # Interpolate bad reads
    if nbdreads > 0:
        print('Read(s) '+', '.join(str(bdreads+1))+' are bad.')
  
        # The bad reads are currently linearly interpolated using the
        # neighoring reads and used as if they were good.  The variance
        # needs to be corrected for this at the end.
        # This will give bad results for CRs

  
        # Use linear interpolation
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
            zeropoint = (im1*hi-im2*lo)/(hi-lo)       # zeropoint, (y1*x2-y2*x1)/(x2-x1)
            im0 = slope*bdreads[k] + zeropoint        # linear interpolation, y=mx+b
  
            # Stuff it in the cube
            cube[:,:,bdreads[k]] = np.round(im0)         # round to closest integer, LONG type
  
    ny,nx = cube.shape
  
    # Reference subtraction ONLY
    if refonly:
        if silent==False:
            print('Reference subtraction only')
            #goto,BOMB

        
    #-------------------------------------
    # INTER-PIXEL CAPACITANCE CORRECTION
    #-------------------------------------
    
    # Loop through each read
    # Make left, right, top, bottom images and
    #  use these to correct the flux for the inter-pixel capacitance
    # make sure to treat the edgs and reference pixels correctly
    # The "T" shape might indicate that we don't need to do the "top"
    #  portion
  
  
    # READ NOISE
    #-----------
    if rdnoiseim.shape>0:
        noise = np.median(rdnoiseim)
    else:
        noise = 12.0  # default value
    noise_dCounts = noise*np.sqrt(2)  # noise in dcounts
  
  
    # Initialize some arrays
    #------------------------
    #mask = intarr(nx,ny)               # CR and saturation mask
    #crindex = intarr(nx,ny,10)-1       # the CR read indices get put in here
    #crnum = intarr(nx,ny)              # # of CRs per pixel
    med_dCounts_im = np.zeros((ny,nx),float)     # the median dCounts for each pixel
    satmask = np.zeros((ny,nx,3),int)          # 1st plane is 0/1 mask, 2nd plane is which read
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

    if silent==False:
        print('Processing the datacube')

    # Loop through the rows
    for i in range(ny):
        if verbose or debug:
            print('Scanning Row ',str(i+1))
        if silent==False and verbose==False and debug==False:
            if (i+1) % 500 == 0:
                print(i+1,'/',ny,format='(I4,A1,I4)')
  
    # Slice of datacube, [Ncol,Nread]
    #--------------------------------
    slc = cube[:,i,:].astype(float)
    slc_orig = slc.copy()  # original slice
 
    #---------------------------------
    # Flag BAD pixels
    #---------------------------------
    if bpmim is not None:
        bdpix, = np.where(bpmim[:,i] > 0,nbdpix)
        nbdpix = len(bdpix)
        if nbdpix > 0:
            for j in range(nbdpix):
                slc[bdpix[j],:] = 0.0  # set them to zero
        mask[bdpix,i] = (mask[bdpix,i] | bpmim[bdpix,i])
  
    #---------------------------------
    # Flag LITTROW ghost pixels, but don't change data values
    #---------------------------------
    if littrowim is not None:
      bdpix, = np.where(littrowim[:,i] == 1)
      nbdpix = len(bdpix)
      if nbdpix > 0:
          mask[bdpix,i] = (mask[bdpix,i] | maskval('LITTROW_GHOST'))
  
    #---------------------------------
    # Flag persistence pixels, but don't change data values
    #---------------------------------
    if persistim is not None:
        bdpix, = np.where(persistim[:,i] and 1)
        if len(bdpix) > 0:
            mask[bdpix,i] = (mask[bdpix,i] | maskval('PERSIST_HIGH'))
        bdpix, = np.where(persistim[:,i] and 2)
        if len(bdpix) > 0:
            mask[bdpix,i] = (mask[bdpix,i] | maskval('PERSIST_MED'))
        bdpix, = np.where(persistim[:,i] and 4)
        if len(bdpix)>0:
            mask[bdpix,i] = (mask[bdpix,i] | maskval('PERSIST_LOW'))
  
    #---------------------------------
    # Detect and Flag Saturated reads
    #---------------------------------
    #  The saturated pixels are detected in the reference subtraction
    #  step and fixed to 65535.
    bdsat, = np.where(slc > saturation)
    if len(bdsat)>0:
        # Flag saturated reads as NAN
        slc[bdsat] = np.nan
  
        # Get 2D indices
        bdsat2d = array_indices(slc,bdsat)
        bdsatx = bdsat2d[0,:]      # X/column indices
        bdsatr = bdsat2d[1,:]      # read indices
        # bdsat is 1D array for slice(2D)
        # bdsatx is column index for 1D med_dCounts
  
        # Unique pixels
        uibdx = np.unique(bdsatx)
        ubdsatx = bdsatx[uibdx]
        nbdsatx = len(ubdsatx)
  
        # Figure out at which Read (NOT dCounts) each column saturated
        rindex = np.ones(nx,int).reshape(-1,1) * np.arange(nreads).reshape(1,-1)     # each pixels read index
        satmask_slice = np.zeros((nx,nreads),int)          # sat mask
        satmask_slice[bdsatx,bdsatr] = 1                   # set saturated reads to 1
        rindexsat = rindex*satmask_slice + \
                    (1-satmask_slice)*999999               # okay pixels ahve 999999, sat pixels have their read index
        minsatread = np.min(rindexsat,axis=2)              # now find the minimum for each column
        nsatreads = np.sum(satmask_slice,axis=2)           # number of sat reads
  
        # Make sure that all subsequent reads to a saturated read are
        # considered "bad" and set to NAN
        for j in range(nbdsatx):
            slc[ubdsatx[j],minsatread[ubdsatx[j]]:] = np.nan
  
        # Update satmask
        satmask[ubdsatx,i,0] = 1                     # mask
        satmask[ubdsatx,i,1] = minsatread[ubdsatx]   # 1st saturated read, NOT dcounts
        satmask[ubdsatx,i,2] = nsatreads[ubdsatx]    # # of saturated reads
  
        # Update mask
        mask[ubdsatx,i] = (mask[ubdsatx,i] | maskval('SATPIX'))     # mask: 1-bad, 2-CR, 4-sat, 8-unfixable
  
    #----------------------
    # Linearity correction
    #----------------------
    # This needs to be done BEFORE the pixels are "fixed" because
    # it needs to operate on the ORIGINAL counts, not the corrected
    # ones.
    if lindata is not None:
        if szlin[0] == 3:
            linslc = lindata[:,i,:]
        else:
            linslc = lindata
        slc_orig1 = slc           # temporary copy since we'll be overwriting it
        slc = aplincorr(slc_orig1,linslc)
  
    #-----------------
    # Dark correction
    #-----------------
    # Each read will have a different amount of dark counts in it
    if darkcube is not None:
        darkslc = darkcube[:,i,:]
        slc_orig2 = slc  # temporary copy since we'll be overwriting it
        slc = ap3dproc_darkcorr(slc_orig2,darkslc,darkhead)
  
    #------------------------------------------------
    # Find difference of neighboring reads, dCounts
    #------------------------------------------------
    #  a difference with 1 or 2 NaN will also be NAN
    dCounts = slc[:,1:sz[3]] - slc[:,0:sz[3]-1]
  
  
    # SHOULD I FIX BAD READS HERE?????
  
  
    #----------------------------
    # Detect and Fix cosmic rays
    #----------------------------
    slice_prefix = slc
    if nocr==False and nreads>2:
        dCounts_orig = dCounts  # temporary copy since we'll be overwriting it
        dCounts = None
        satmask_slc = satmask[:,i,:]
        out = ap3dproc_crfix(dCounts_orig,satmask_slc,noise=noise,crfix=crfix)
        crstr_slc, dCounts, med_dCounts, mask, crindex, crnum, variability_slice = out
  
        variability_im[:,i] = variability_slice
  
    # Only 2 reads, CANNOT detect or fix CRs
    else:
        med_dCounts = dCounts
        crstr_slice = {'ncr':0}
  
    # Some CRs detected, add to CRSTR structure
    if crstr_slice['ncr'] > 0:
  
        crstr_slice['data'].y = i  # add the row information
  
        # Add to MASK
        maskpix, = np.where(crstr_slice['data'].x < 2048)
        nmaskpix = len(maskpix)
        if nmaskpix > 0:
            mask[crstr_slice.data[maskpix].x,i] = (mask[crstr_slice.data[maskpix].x,i] | maskval('CRPIX'))
  
        # Starting global structure
        if len(crstr) == 0:
            crstr = crstr_slc
  
        # Add to global structure
        else:
            ncrtot = crstr['ncr'] + crstr_slice['ncr']
            old_crstr = crstr
            crstr = {'ncr':ncrtot,'data':[old_crstr.data, crstr_slice.data]}
            crstr = None
  
    #----------------------
    # Fix Saturated reads
    #----------------------
    #  do this after CR fixing, so we don't have to worry about CRs here
    #  set their dCounts to med_dCounts
  
    if nbdsat > 0:
  
        # Have enough reads (>2) to fix pixels
        if (nreads > 2):
  
            # Total number of good dCounts for each pixel
            totgd = np.sum(finite(dCounts),axis=2)
            
            # Unfixable pixels
            #------------------
            #  Need 2 good dCounts to be able to "safely" fix a saturated pixel
            thresh_dcounts = 2
            if rd3satfix and nreads==3:
                thresh_dcounts = 1  # fixing 3 reads                       
            unfixable, = np.where(totgd < thresh_dcounts)
            if len(unfixable) > 0:
                dCounts[unfixable,:] = 0.0
                mask[unfixable,i] = (mask[unfixable,i] | maskval('UNFIXABLE'))                   # mask: 1-bad, 2-CR, 4-sat, 8-unfixable
        
  
  
            # Fixable Pixels
            #-----------------
            fixable, = np.where((totgd >= thresh_dcounts) & (satmask[:,i,0] == 1))
            nfixable = len(fixable)
  
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
                    dCounts[ibdsatx,(minsatread[ibdsatx]-1)>0:nreads-2] = med_dCounts[ibdsatx]
  
                    # Saturation extrapolation error
                    var_dCounts = variability_im[ibdsatx,i] * (med_dCounts[ibdsatx]>0.0001)   # variability in dCounts
                    sat_extrap_error[ibdsatx,i] = var_dCounts * satmask[ibdsatx,i,2]          # Sigma of extrapolated counts, multipy by Nextrap
  
                # Do NOT fix the saturated pixels
                #---------------------------------
                else:
                    dCounts[ibdsatx,minsatread[ibdsatx]-1:nreads-2] = 0.0    # set saturated dCounts to zero
          
                
            # It might be better to use the last good value from sm_dCounts
            # rather than the straight median of all reads
  
        # Only 2 reads, can't fix anything
        else:
            mask[ubdsatx,i] = (mask[ubdsatx,i] | maskval('UNFIXABLE'))     # mask: 1-bad, 2-CR, 4-sat, 8-unfixable
            dCounts[bdsatx] = 0.0                        # set saturated reads to zero
   
    #------------------------------------
    # Reconstruct the SLICE from dCounts
    #------------------------------------
    slc0 = slc[:,0]  # first read
    bdsat, = np.where(finite(slc0) == 0)  # NAN in first read, set to 0.0
    if len(bdsat) > 0:
        slc0[bdsat] = 0.0
  
    unfmask_slc = int((int(mask[:,i]) & maskval('UNFIXABLE')) == maskval('UNFIXABLE'))  # unfixable
    slc0[0:2047] = slc0[0:2047]*(1.0-unfmask_slc)                 # set unfixable pixels to zero
  
    slc_fixed = slc0 # (fltarr(nreads)+1.0)
    if nreads > 2:
        slc_fixed[:,1:] += np.cumsum(dCounts,axis=2)
    else:
        slc_fixed[:,0] = slice0
        slc_fixed[:,1] = slice0+dCounts
    
  
    #--------------------------------
    # Put fixed slice back into cube
    #--------------------------------
    cube[:,i,:] = np.round( slc_fixed )        # round to closest integer, LONG type
  
    #------------------------------------
    # Final median of each "fixed" pixel
    #------------------------------------
    if nreads > 2:
        # Unfixable pixels are left at 0.0
  
        # If NOT fixing saturated pixels, then we need to
        # temporarily set saturated reads to NAN
        #  Leave unfixable pixels at 0.0
        temp_dCounts = dCounts
        if satfix==False:
            bdsat, = np.where((satmask[:,i,0] == 1) & (unfmask_slice == 0))
            nbdsat = len(bdsat)
            for j in range(bdsat):
                temp_dCounts[bdsat[j],satmask[bdsat[j],i,1]-1:] = np.nan
      
  
        fmed_dCounts = np.median(temp_dCounts,axis=2)    # NAN are automatically ignored
        bdnan, = np.where(finite(fmed_dCounts) == 0)
        if len(bdnan) > 0:
            import pdb; pdb.set_trace()
        med_dCounts_im[:,i] = fmed_dCounts
  
    # Only 2 reads
    else:
        med_dCounts_im[:,i] = dCounts
    
    if verbose or debug:
        nsatslice = np.sum(satmask[:,i,0])
        if len(crstr) == 0:
            ncrslice = 0
        else:
            dum, = np.where(crstr['data'].y == i)
            ncrslice = len(dum)
        print('Nsat/NCR = ',str(int(nsatslice)),'/',str(int(ncrslice)),' this row')
    
  
    #----------
    # Plotting
    #----------
    #debug = 1
    pltpix, = np.where(mask[:,i] > 1)
    npltpix = len(pltpix)
    if keyword_set(debueg) and npltpix > 0:
        ap3dproc_plotting(dcounts,slice_prefix,slice_fixed,mask,crstr,i,saturation)
  
    if len(crstr) == 0:
        crstr = {'ncr':0}

    
    #------------------------
    # Iterative CR Rejection
    #------------------------
    #   Check neighbors of CRs for CRs at the same read
    #   lower Nsigma threshold
    if criter:
        if silent==False:
            print('Checking neighbors for CRs')
  
        iterflag = 0
        niter = 0
        while (iterflag != 1) and (crstr.ncr > 0):
            newcr_thisloop = 0
  
            # CRs are left to check
            crtocheck, = np.where(crstr.data.neicheck == 0)
            ncrtocheck = len(crtocheck)
                    
            # Loop through the CRs
            for i in range(ncrtocheck):
                ix = crstr.data[crtocheck[i]].x
                iy = crstr.data[crtocheck[i]].y
                ir = crstr.data[crtocheck[i]].read
  
                # Look at neighboring pixels to the affected CR
                #   at the same read
                xlo = (ix-1) > 0
                xhi = (ix+1) < (nx-1)
                ylo = (iy-1) > 0
                yhi = (iy+1) < (ny-1)
                nnei = (xhi-xlo+1)*(yhi-ylo+1) - 1
  
                # Create a fake "slice" of the neighboring pixels
                nei_slice = np.zeros((nnei,nreads),float)
                nei_slice_orig = np.zeros((nnei,nreads),float)  # original slice if saturated
                nei_cols = np.zeros(nnei,float)  # x
                nei_rows = np.zeros(nnei,float)  # y
                nei_satmask = np.zeros((nnei,3),int)
                count = 0
                for j in np.arange(xlo,xhi+1):
                    for k in np.arange(ylo,yhi+1):
                        # Check that the read isn't saturated for this pixel and read
                        if satmask[j,k,0] == 1 and satmask[j,k,1] <= ir:
                            readsat = 1
                        else:
                            readsat = 0
                        # Only want neighbors
                        if (j != ix and k != iy) and (readsat == 0):
                            nei_slc[count,:] = cube[j,k,:].astype(float)
                            nei_slc_orig[count,:] = cube[j,k,:].astype(float)
                            nei_cols[count] = j
                            nei_rows[count] = k
                            nei_satmask[count,:] = satmask[j,k,:]
                            # if this pixel is saturated then we need to make the saturated
                            #   reads NAN again.  The "fixed" values are in nei_slice_orig
                            if satmask[j,k,0] == 1:
                                nei_slc[count,satmask[j,k,1]:nreads] = np.nan
                            count += 1
  
                # Difference of neighboring reads, dCounts
                nei_dCounts = nei_slc[:,1:nreads] - nei_slc[:,0:nreads-1]
  
                # Fix/detect cosmic rays
                ap3dproc_crfix(nei_dCounts,nei_satmask,nei_dCounts_fixed,med_nei_dCounts,nei_crstr,crfix=crfix,
                               noise=noise_dCounts,sigthresh=6,onlythisread=ir)
  
  
                # Some new CRs detected
                if nei_crstr.ncr > 0:
                    # Add the neighbor information
                    ind = nei_crstr.data.x             # index in the nei slice
                    nei_crstr.data.x = nei_cols[ind]   # actual column index
                    nei_crstr.data.y = nei_rows[ind]   # actual row index
  
                    # Replace NANs if there were saturated pixels
                    bdsat, = np.where(nei_satmask[:,0] == 1)
                    nbdsat = len(bdsat)
                    for j in range(nbdsat):
                        lr = nei_satmask[bdsat[j],1]-1
                        hr = nreads-2
                        nei_dCounts_orig = nei_slice_orig[:,1:nreads-1] - nei_slice_orig[:,0:nreads-2]  # get "original" dCounts
                        nei_dCounts_fixed[bdsat[j],lr:hr] = nei_dCounts_orig[bdsat[j],lr:hr]            # put the original fixed valued back in
                    
                    # Reconstruct the SLICE from dCounts
                    nei_slice0 = nei_slice[:,0]  # first read
                    bdsat, = np.where(finite(nei_slice0) == 0)  # set NAN in first read to 0.0
                    nbdsat = len(bdsat)
                    if nbdsat > 0:
                        nei_slice0[bdsat] = 0.0
                    nei_slice_fixed = nei_slice0 # (fltarr(nreads)+1.0)
                    if nreads > 2:
                        nei_slc_fixed[:,1:] += np.cumsum(nei_dCounts_fixed,axis=2)
                    else:
                        nei_slc_fixed[:,0] = nei_slc0
                        nei_slc_fixed[:,1] = nei_slc0+nei_dCounts_fixed
  
                    # Put fixed slice back into cube
                    #  only update new CR pixels
                    for j in range(nei_crstr.ncr):
                        cube[nei_crstr.data[j].x,nei_crstr.data[j].y,:] = np.round( nei_slc_fixed[ind[j],:] )  # round to integer
  
                    # Update med_dCounts_im ???
  
                    # Add to the total CRSTR
                    ncrtot = crstr.ncr + nei_crstr.ncr
                    old_crstr = crstr
                    crstr = {'ncr':ncrtot,'data':[old_crstr.data, nei_crstr.data]}
                    old_crstr = None
  
                # New CRs detected
                nnew = nei_crstr['ncr']
                newcr_thisloop += nnew
  
                # This CR has been checked
                crstr.data[crtocheck[i]].neicheck = 1  # checked!
  
            # Time to stop
            if (newcr_thisloop == 0) or (niter > 5) or (ncrtocheck == 0):
                iterflag = 1
  
            niter += 1
  
  
    #-------------------------------------
    # INTER-PIXEL CAPACITANCE CORRECTION
    #-------------------------------------
    
    # Loop through each read
    # Make left, right, top, bottom images and
    #  use these to correct the flux for the inter-pixel capacitance
    # make sure to treat the edgs and reference pixels correctly
    # The "T" shape might indicate that we don't need to do the "top"
    #  portion
    
    # THIS HAS BEEN MOVED TO JUST AFTER THE REFERENCE PIXEL SUBTRACTION!!!!
    # this needs to happen before dark subtraction!!
  
  
    #--------------------------------
    # Measure "variability" of data
    #--------------------------------
    # Use pixels with decent count rates
    crmask = int((int(mask) & maskval('CRPIX')) == maskval('CRPIX'))
    highpix, = np.where((satmask[:,:,0] == 0) & (crmask == 0) & (med_dCounts_im > 40))
    nhighpix = len(highpix)
    if nhighpix == 0:
        highpix, = np.where((satmask[:,:,0] == 0) & (med_dCounts_im > 20))
        nhighpix = len(highpix)
    if nhighpix > 0:
        global_variability = np.median(variability_im[highpix])
    else:
        global_variability = -1
  
  
  
    #-------------------------
    # FLAT FIELD CORRECTION
    #-------------------------
    #  if len(flatim) > 0:
    #  
    #    # Apply the flat field to each read.  Each pixel has a separate
    #    # kTC noise/offset, but applying the flat field before or after
    #    # this is removed gives IDENTICAL results.
    #  
    #    # Just divide the read image by the flat field iamge
    #    #for i=0,nreads-1 do cube[:,:,i] /= flatim
    #    for i=0,nreads-1 do cube[:,:,i] = round( cube[:,:,i] / flatim )  # keep cube LONG type
    #  
    #    # Hopefully this propogates everything to the variance
    #    # image properly.
    #  
    #  
  
  
    #------------------------
    # COLLAPSE THE DATACUBE
    #------------------------
    
    # No gain image, using gain=1.0 for all pixels
    # gain = Electrons/ADU
    if len(gainim)==0:
        print('NO gain image.  Using GAIN=1')
        gainim = np.ones((2048,2048),float)

    
    # Fowler Sampling
    #------------------
    if uptheramp == False:
  
        # Make sure that Nfowler isn't too large
        Nfowler_used = Nfowler
        if Nfowler > Nreads/2:
            Nfowler_used = Ngdreads/2
  
        # Use the mean of Nfowler reads

        # Beginning sample
        gd_beg = gdreads[0:nfowler_used]
        if len(gd_beg) == 1:
            im_beg = cube[:,:,gd_beg]/Nfowler_used).astype(float)
        else:
            im_beg = np.sum(cube[:,:,gd_beg],axis=3)/Nfowler_used.astype(float)

        # End sample
        gd_end = gdreads[ngdreads-nfowler_used:ngdreads-1]
        if len(gd_end) == 1:
            im_end = cube[:,:,gd_end]/Nfowler_used.astype(float)
        else:
            im_end = np.sum(cube[:,:,gd_end],axis=3)/Nfowler_used.astype(float)

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
        # THIS WILL NEED TO BE IMPROVED IN THE FUTURE TO TAKE THROUGHPUT VARIATIONS
        # INTO ACCOUNT.  FOR NOW THIS IS JUST FOR DARKS AND FLATS
        #im = med_dCounts_im * nreads
  
  
        # Fit a line to the reads for each pixel
        #   dCounts are noisier than the actual reads by sqrt(2)
        #   See Rauscher et al.(2007) Eqns.3
  
        # Calculating the slope for each pixel
        #  t is the exptime, s is the signal
        #  we will use the read index for t
        sumts = np.zeros((sz[1],sz[2]),float)   # SUM t*s
        sums = np.zeros((sz[1],sz[2]),float)    # SUM s
        sum = np.zeros((sz[1],sz[2]),int)    # SUM s
        sumt = np.zeros((sz[1],sz[2]),float)   # SUM t*s
        sumt2 = np.zeros((sz[1],sz[2]),float)   # SUM t*s
        for k in range(ngdreads):
            slc = cube[:,:,gdreads[k]]
            if satfix==False:
                good, = np.where((satmask[:,:,0] == 0) | ((satmask[:,:,0] == 1) & (satmask[:,:,1] > i)))
            else:
                good, = np.where(np.finite(slc))
        #good = np.where(finite(slice))
        sumts[good] += gdreads[k]*reform(slc[good])
        sums[good] += reform(slc[good])
        sum[good] += 1
        sumt[good] += gdreads[k]
        sumt2[good] += gdreads[k]**2
        #sumt = total(findgen(nread))     # SUM t
        #sumt2 = total(findgen(nread)^2)  # SUM t^2
        # The slope in Counts per read, similar to med_dCounts_im
        #slope = (nread*sumts - sumt*sums)/(nread*sumt2 - sumt^2)
        slope = (sum*sumts - sumt*sums)/(sum*sumt2 - sumt**2)
        # To get the total counts just multiply by nread
        im = slope * (ngdreads-1L)
        # the first read doesn't really add any signal, just a zero-point
  
        # See Equation 1 in Rauscher et al.(2007), SPIE
        #  with m=1
        #  noise and image/flux should be in electrons, sample_noise is in electrons
        #sample_noise = sqrt( 12*(ngdreads-1.)/(nreads*(ngdreads+1.))*(noise*gainim)^2 + 6.*(ngdreads^2+1)/(5.*ngdreads*(ngdreads+1))*im*gainim )
        sample_noise = np.sqrt( 12*(ngdreads-1.)/(nreads*(ngdreads+1.))*noise**2 + 6.*(ngdreads**2+1)/(5.*ngdreads*(ngdreads+1))*im[0:2048,:]*gainim )
        sample_noise /= gainim  # convert to ADU
  
        # Noise contribution to the variance
        #   this is only valid if the variability is zero
        #ngdreads = nreads - nbdreads
        #sample_noise = noise / sqrt(Ngdreads)

    # With userference, subtract off the reference array to reduce/remove
    #   crosstalk. 
    if usereference==True:
        print('subtracting reference array...')
        tmp = im[0:2048,0:2048]
        ref = im[2048:2560,0:2048]
        # subtract smoothed horizontal structure
        ref -= (np.zeros(512,float)+1)#medfilt1d(median(ref,dim=1),7)
        ref = aprefcorr_sub(tmp,ref)
        im = tmp
        nx = 2048

    #-----------------------------------
    # Apply the Persistence Correction
    #-----------------------------------
    if len(persistmodelcorr) > 0 and len(histcorr) > 0:
        if silent==False:
            print('PERSIST modelcorr file = '+persistmodelcorr)
        appersistmodel(ifile,histcorr,persistmodelcorr,pmodelim,ppar,bpmfile=bpmcorr,error=perror)
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
    varim = np.zeros((nx,ny),float)         # variance in ADU
  
    # 1. Poisson Noise from the image: note that the equation for UTR
    #    noise above already includes Poisson term
    if not keyword_set(uptheramp):
        if len(pmodelim) > 0:
            varim += np.maximum( (im+pmodelim)/gainim , 0)
        else:
            varim += np.maximum(im/gainim,0)
     
  # 2. Poisson Noise from dark current
  if len(darkcube) > 0:
      darkim = darkcube[:,:,nreads-1]
      varim += np.maximum(darkim/gainim,0)
  
  # 3. Sample/read noise
  varim += sample_noise**2         # rdnoise reduced by the sampling
  
  # 4. Saturation error
  #      We used median(dCounts) to extrapolate the saturated pixels
  #      Use the variability in dCounts to estimate the error of doing this
  if satfix:
      varim += sat_extrap_error     # add saturation extrapolation error
  else:
      varim = varim*(1-satmask[:,:,0]) + satmask[:,:,0]*99999999.   # saturated pixels are bad!
  # Unfixable pixels
  unfmask = int((int(mask) AND maskval('UNFIXABLE')) == maskval('UNFIXABLE'))  # unfixable
  varim = varim*(1-unfmask) + unfmask*99999999.         # unfixable pixels are bad!
  
  # 5. CR error
  #     We use median of neighboring dCounts to "fix" reads with CRs
  crmask = int((int(mask) & maskval('CRPIX')) == maskval('CRPIX'))
  if crfix == True:
      # loop in case there are multiple CRs per pixel
      for i in range(crstr.ncr):
          if crstr.data[i].x < 2048:
              varim[crstr.data[i].x,crstr.data[i].y]+=crstr.data[i].fixerror
          else:
              varim = varim*(1-crmask) + crmask*99999999.               # pixels with CRs are bad!
  
  # Bad pixels
  bpmmask = int(int(mask) & maskval('BADPIX')) == maskval('BADPIX'))
  varim = varim*(1-bpmmask) + bpmmask*99999999.               # bad pixels are bad!

  # Flat field  
  if len(flatim) > 0:
      varim /= flatim**2
      im /= flatim

  # Now convert to ELECTRONS
  if len(detcorr) > 0 and outelectrons==False:
      varim *= gainim**2
      im *= gainim

  #----------------------------
  # Construct output datacube
  #  [image, error, mask]
  #----------------------------
  if len(pmodelim) > 0:
      output = np.zeros((nx,ny,4),float)
  else:
      output = np.zeros((nx,ny,3),float)
  output[:,:,0] = im
  output[:,:,1] = np.maximum(np.sqrt(varim),1)  # must be greater than zero
  output[:,:,2] = mask
  if len(pmodelim)>0:
      output[:,:,3] = pmodelim  # persistence model in ADU
  
  #-----------------------------
  # Update header
  #-----------------------------
  leadstr = 'AP3D: '
  sxaddpar,head,'V_APRED',getvers()
  sxaddhist,leadstr+systime(0),head
  info = GET_LOGIN_INFO()
  sxaddhist,leadstr+info.user_name+' on '+info.machine_name,head
  sxaddhist,leadstr+'IDL '+!version.release+' '+!version.os+' '+!version.arch,head
  sxaddhist,leadstr+' APOGEE Reduction Pipeline Version: '+getvers(),head
  sxaddhist,leadstr+'Output File:',head
  if len(detcorr) > 0 and keyword_set(outelectrons):
      sxaddhist,leadstr+' HDU1 - image (electrons)',head
      sxaddhist,leadstr+' HDU2 - error (electrons)',head
  else:
      sxaddhist,leadstr+' HDU1 - image (ADU)',head
      sxaddhist,leadstr+' HDU2 - error (ADU)',head

  sxaddhist,leadstr+' HDU3 - flag mask',head
  sxaddhist,leadstr+'        1 - bad pixels',head
  sxaddhist,leadstr+'        2 - cosmic ray',head
  sxaddhist,leadstr+'        4 - saturated',head
  sxaddhist,leadstr+'        8 - unfixable',head
  if len(pmodelim) > 0:
      sxaddhist,leadstr+' HDU4 - persistence correction (ADU)',head
  sxaddhist,leadstr+'Global fractional variability = '+str(string(global_variability,format='(F5.3)')),head
  maxlen = 72-strlen(leadstr)
  # Bad pixel mask file
  if len(bpmim) > 0:
      line = 'BAD PIXEL MASK file="'+bpmcorr+'"'
      if strlen(line) > maxlen:
          line1 = strmid(line,0,maxlen)
          line2 = strmid(line,maxlen,100)
          sxaddhist,leadstr+line1,head
          sxaddhist,leadstr+line2,head
      else:
          sxaddhist,leadstr+line,head
  # Detector file
  if len(detcorr) > 0:
      line = 'DETECTOR file="'+detcorr+'"'
      if strlen(line) > maxlen:
          line1 = strmid(line,0,maxlen)
          line2 = strmid(line,maxlen,100)
          sxaddhist,leadstr+line1,head
          sxaddhist,leadstr+line2,head
      else:
          sxaddhist,leadstr+line,head
  # Dark Correction File
  if len(darkcube) > 0:
      line = 'Dark Current Correction file="'+darkcorr+'"'
      if strlen(line) > maxlen:
          line1 = strmid(line,0,maxlen)
          line2 = strmid(line,maxlen,100)
          sxaddhist,leadstr+line1,head
          sxaddhist,leadstr+line2,head
      else:
          sxaddhist,leadstr+line,head
  # Flat field Correction File
  if len(flatim) > 0:
      line = 'Flat Field Correction file="'+flatcorr+'"'
      if strlen(line) > maxlen:
          line1 = strmid(line,0,maxlen)
          line2 = strmid(line,maxlen,100)
          sxaddhist,leadstr+line1,head
          sxaddhist,leadstr+line2,head
      else:
          sxaddhist,leadstr+line,head
  # Littrow ghost mask File
  if len(littrowim) > 0:
      line = 'Littrow ghost mask file="'+littrowcorr+'"'
      if strlen(line) > maxlen:
          line1 = strmid(line,0,maxlen)
          line2 = strmid(line,maxlen,100)
          sxaddhist,leadstr+line1,head
          sxaddhist,leadstr+line2,head
      else:
          sxaddhist,leadstr+line,head
  # Persistence mask File
  if len(persistim) > 0:
      line = 'Persistence mask file="'+persistcorr+'"'
      if strlen(line) > maxlen:
          line1 = strmid(line,0,maxlen)
          line2 = strmid(line,maxlen,100)
          sxaddhist,leadstr+line1,head
          sxaddhist,leadstr+line2,head
      else:
          sxaddhist,leadstr+line,head
  # Persistence model file
  if len(persistmodelcorr) > 0:
      line = 'Persistence model file="'+persistmodelcorr+'"'
      if strlen(line) > maxlen:
          line1 = strmid(line,0,maxlen)
          line2 = strmid(line,maxlen,100)
          sxaddhist,leadstr+line1,head
          sxaddhist,leadstr+line2,head
      else:
          sxaddhist,leadstr+line,head
  # History file
  if len(histcorr) > 0:
      line = 'Exposure history file="'+histcorr+'"'
      if strlen(line) > maxlen:
          line1 = strmid(line,0,maxlen)
          line2 = strmid(line,maxlen,100)
          sxaddhist,leadstr+line1,head
          sxaddhist,leadstr+line2,head
      else:
          sxaddhist,leadstr+line,head
  # Bad pixels 
  bpmmask = int((int(mask) & maskval('BADPIX')) == maskval('BADPIX'))
  totbpm = total(bpmmask)
  sxaddhist,leadstr+str(int(totbpm))+' pixels are bad',head
  # Cosmic Rays
  crmask, = np.where(int(mask) & maskval('CRPIX'),totcr)
  if nreads > 2:
      sxaddhist,leadstr+str(int(totcr))+' pixels have cosmic rays',head
  if crfix and nreads>2:
      sxaddhist,leadstr+'Cosmic Rays FIXED',head
  # Saturated pixels
  satmask, = np.where(int(mask) & maskval('SATPIX'),totsat)
  unfmask, = np.where(int(mask) & maskval('UNFIXABLE'),totunf)
  totfix = totsat-totunf
  sxaddhist,leadstr+str(int(totsat))+' pixels are saturated',head
  if keyword_set(satfix) and nreads > 2 then sxaddhist,leadstr+str(int(totfix))+' saturated pixels FIXED',head
  # Unfixable pixels
  sxaddhist,leadstr+str(int(totunf))+' pixels are unfixable',head
  # Sampling
  if uptheramp:
      sxaddhist,leadstr+'UP-THE-RAMP Sampling',head
  else:
    sxaddhist,leadstr+'Fowler Sampling, Nfowler='+str(int(Nfowler_used)),head 
  # Persistence correction factor
  if len(pmodelim) > 0 and len(ppar) > 0:
    sxaddhist,leadstr+'Persistence correction: '+' '.join(str(string(ppar,format='(G7.3)'))),head
  
  
  # Fix EXPTIME if necessary
  if head['NFRAMES'] != nreads:
      # NFRAMES is from ICC, NREAD is from bundler which should be correct
      exptime = nreads*10.647  # secs
      sxaddpar,head,'EXPTIME',exptime
      print('not halting, but NFRAMES does not match NREADS, NFRAMES: ', head['NFRAMES'], ' NREADS: ',string(format='(i8)',nreads),'  ', seq)
      #print('halt: NFRAMES does not match NREADS, NFRAMES: ', sxpar(head,'NFRAMES'), ' NREADS: ',string(format='(i8)',nreads),'  ', seq

  # Add UT-MID/JD-MID to the header
  jd = date2jd(head['DATE-OBS'])
  exptime = head['EXPTIME']
  jdmid = jd + (0.5*exptime)/24./3600.d0
  utmid = jd2date(jdmid)
  sxaddpar,head,'UT-MID',utmid,' Date at midpoint of exposure'
  sxaddpar,head,'JD-MID',jdmid,' JD at midpoint of exposure'

  # remove CHECKSUM
  sxdelpar,head,'CHECKSUM'
  
  #----------------------------------
  # Output the final image and mask
  #----------------------------------
  if len(outfile) > 0:
    ioutfile = outfile[f]
  
    # Does the output directory exist?
    if os.path.exists(os.path.dirname(ioutfile),/directory)==False:
      print('Creating ',os.path.dirname(ioutfile))
      os.makedirs(os.path.dirname(ioutfile))
  
    # Test if the output file already exists
    test = os.path.exists(ioutfile)

    if silent==False:
        print('')
    if test == 1 and clobber:
        print('OUTFILE = ',ioutfile,' ALREADY EXISTS.  OVERWRITING')
    if test == 1 and clobber==False:
        print('OUTFILE = ',ioutfile,' ALREADY EXISTS. ')
    
    # Writing file
    if test == 0 or clobber:
        if silent==False:
            print('Writing output to: ',ioutfile)
        if outlong:
            print('Saving FLUX/ERR as LONG instead of FLOAT'))
      # HDU0 - header only
      FITS_WRITE,ioutfile,0,head,/no_abort,message=write_error    
      # HDU1 - flux
      flux = output[:,:,0]
      # replace NaNs with zeros
      bad, = np.where(finite(flux) == 0)
      nbad = len(bad)
      if nbad > 0:
          flux[bad] = 0.
      if outlong:
          flux = np.round(flux)
      MKHDR,head1,flux,/image
      sxaddpar,head1,'CTYPE1','Pixel'
      sxaddpar,head1,'CTYPE2','Pixel'
      sxaddpar,head1,'BUNIT','Flux (ADU)'
      MWRFITS,flux,ioutfile,head1,/silent

      # HDU2 - error
      #err = sqrt(reform(output[:,:,1])) > 1  # must be greater than zero
      err = errout(output[:,:,1])
      if outlong:
          err = np.round(err)
      MKHDR,head2,err,/image
      sxaddpar,head2,'CTYPE1','Pixel'
      sxaddpar,head2,'CTYPE2','Pixel'
      sxaddpar,head2,'BUNIT','Error (ADU)'
      MWRFITS,err,ioutfile,head2,/silent


      # HDU3 - mask
      #flagmask = fix(reform(output[:,:,2]))
      # don't go through conversion to float and back!
      flagmask = mask.astype(int)
      MKHDR,head3,flagmask,/image
      sxaddpar,head3,'CTYPE1','Pixel'
      sxaddpar,head3,'CTYPE2','Pixel'
      sxaddpar,head3,'BUNIT','Flag Mask (bitwise)'
      sxaddhist,'Explanation of BITWISE flag mask',head3
      sxaddhist,' 1 - bad pixels',head3
      sxaddhist,' 2 - cosmic ray',head3
      sxaddhist,' 4 - saturated',head3
      sxaddhist,' 8 - unfixable',head3
      MWRFITS,flagmask,ioutfile,head3,/silent
      #FITS_WRITE,ioutfile,output,head,/no_abort,message=write_error
      #if write_error != '' then print('Error writing file '+write_error)

      # HDU4 - persistence model
      if len(pmodelim) > 0:
          MKHDR,head4,pmodelim,/image
          sxaddpar,head4,'CTYPE1','Pixel'
          sxaddpar,head4,'CTYPE2','Pixel'
          sxaddpar,head4,'BUNIT','Persistence correction (ADU)'
          MWRFITS,pmodelim,ioutfile,head4,/silent
  
  # Remove the recently Decompressed file
  if extension == 'apz' and keyword_set(cleanuprawfile) and doapunzip == 1:
      print('Deleting recently decompressed file ',file)
      if os.path.exists(file): os.remove(file)
    
  # Number of saturated and CR pixels
  if silent==False:
      print('')
      print('BAD/CR/Saturated Pixels:')
      print(str(int(totbpm)),' pixels are bad')
      print(str(int(totcr)),' pixels have cosmic rays')
      print(str(int(totsat)),' pixels are saturated')
      print(str(int(totunf)),' pixels are unfixable')
      print('')

  os.remove(lockfile)
  
  dt = systime(1)-t0
  if silent==False:
      print('dt = ',str(string(dt,format='(F10.1)')),' sec')
  if logfile is not None:
      writelog,logfile,os.path.basename((file)+string(format='(f10.2,1x,i8,1x,i8,1x,i8,i8)',dt,totbpm,totcr,totsat,totunf)

if nfiles > 1:
  dt = systime(1)-t00
  if silent==False:
    print('dt = ',str(string(dt,format='(F10.1)')),' sec')
