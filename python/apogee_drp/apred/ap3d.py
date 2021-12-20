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

    sz = slice_in.shape
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

    nreads = slice_in.shape[]1

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
        sm_dCounts = MEDFILT2D(dCounts,smbin,dim=2,/edge_copy,/even)
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
        bdx = reform(bd2D[0,*])  # column
        bdr = reform(bd2D[1,*])  # read

        # Correct the CRs and correct the pixels
        for j in range(nbd1D):
            ibdx = (bdx[j])[0]
            ibdr = (bdr[j])[0]

            dCounts_pixel = reform(dCounts[ibdx,:])

            # ONLYTHISREAD
            #  for checking neighboring pixels in the iterative part
            #--------------
            if n_elements(onlythisread) gt 0:
                # onlthisread is the read index, while ibdr is a dCounts index
                # ibdr+1 is the read index for this CR
                if (ibdr+1) lt onlythisread[0]-1 or (ibdr+1) gt onlythisread[0]+1 then goto,BOMB

            # Calculate Local Median and Local Sigma
            #----------------------------------------
            #   Use a local median/sigma so the affected CR dCount is not included
            # more than 2 good dCounts and Nreads>smbin
            if (totgd[ibdx] gt 2) and (nreads gt smbin):
                dCounts_pixel[ibdr] = !values.f_nan  # don't use affected dCounts    
    
                maxind = nreads-1
                if satmask[ibdx,0] eq 1 then maxind=satmask[ibdx,1]-2  # don't include saturated reads (NANs)
                lor = (ibdr-smbin/2) > 0
                hir = (lor + smbin-1) < maxind
                if (hir eq maxind) then lor=(hir-smbin+1) > 0
    
                # -- Local median dCounts --
                #  make sure the indices make sense
                #if (lor lt 0 or hir lt 0 or hir le lor) then stop
                if (lor lt 0 or hir lt 0 or hir le lor) then local_med_dCounts=med_dCounts[ibdx] else $
                local_med_dCounts = median(dCounts_pixel[lor:hir],/even)
    
                # If local median dCounts is NAN use all reads
                if finite(local_med_dCounts) eq 0 then local_med_dCounts=med_dCounts[ibdx]
                # If still NaN then set to 0.0
                if finite(local_med_dCounts) eq 0 then local_med_dCounts=0.0
    
                # -- Local sigma dCounts --
                local_sigma = MAD(dCounts_pixel[lor:hir]-local_med_dCounts,/zero)
    
                # If local sigma dCounts is NAN use all reads
                #   this should never actually happen
                if finite(local_sigma) eq 0 then local_sigma=sig_dCounts[ibdx]
                # If still NaN then set to noise
                if finite(local_sigma) eq 0 then local_sigma=noise
                
            # Only 2 good dCounts OR Nreads<smbin
            else:
                local_med_dCounts = med_dCounts[ibdx]
                local_sigma = sig_dCounts[ibdx]


            local_med_dCounts = med_dCounts[ibdx]
            local_sigma = sig_dCounts[ibdx]

            # Fix the CR
            #------------
            if keyword_set(crfix):
                if keyword_set(verbose) then $
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
            if crstr.ncr eq n_elements(crstr.data):
                old_crstr = crstr
                nold = n_elements(old_crstr.data)
                crstr = {NCR:0L,DATA:REPLICATE(crstr_data_def,nold+50L)}
                STRUCT_ASSIGN,old_crstr,crstr   # source, destination
                apgundef,old_crstr

            # Add CR to CRSTR
            crstr.data[crstr.ncr].x = ibdx
            crstr.data[crstr.ncr].read = ibdr+1  # ibdr is dCounts index, +1 to get read
            crstr.data[crstr.ncr].counts = dCounts[ibdx,ibdr] - sm_dCounts[ibdx,ibdr]
            crstr.data[crstr.ncr].nsigma = nsigma_slice[ibdx,ibdr]
            crstr.data[crstr.ncr].globalsigma = sig_dCounts[ibdx]
            if keyword_set(crfix) then crstr.data[crstr.ncr].fixed = 1
            crstr.data[crstr.ncr].localsigma = local_sigma
            if keyword_set(crfix) then crstr.data[crstr.ncr].fixerror = fixerror
            crstr.ncr++

    #  Replace the dCounts with CRs with the median smoothed values
    #    other methods could be used to "fix" the affected read,
    #    e.g. polynomial fitting/interpolation, Gaussian smoothing, etc.

    # Now trim CRSTR
    if crstr.ncr>0:
        old_crstr = crstr
        crstr = {NCR:old_crstr.ncr,DATA:old_crstr.data[0:old_crstr.ncr-1]}
        apgundef,old_crstr
    else:
        crstr = {NCR:0L}   # blank structure


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
    if ((rdnoiseim.ndim==2) & (rdnoiseim.shape != (2048,2048)) | ((rdnoiseim.ndim==1) & (rdnoiseim.size != 4)) | (type(rdnoiseim) != np.float32):
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
        if (szlin[0] eq 2 and szlin[1] eq 4 and szlin[2] eq 3) then linokay=1
        if (szlin[0] eq 3 and szlin[1] eq 2048 and szlin[2] eq 2048 and szlin[3] eq 3) then linokay=1
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
    if silent is False: print('BPM file = '+bpmcorr)
 
    # Check that the file looks reasonable
    #  must be 2048x2048 and have 0/1 values
    ny,nx = bpmim.shape
    bpmokay = 0
    if (bpmim.ndim != 2) | (nx != 2048) | (ny != 2048):
        raise ValueError('BAD PIXEL MASK must be 2048x2048 with 0/1 values')

    return bpmim,bpmhead


def loadlittrow(littrowcorr,silent=True):
  #--------------------------------
  # Load LITRROW MASK File
  #--------------------------------.
  if n_elements(littrowcorr) gt 0 and n_elements(littrowim) eq 0 then begin
  
    # LITTROWCORR must be scalar string
    if size(littrowcorr,/type) ne 7 or n_elements(littrowcorr) ne 1 then begin
      error = 'LITTROWCORR must be a scalar string with the filename of the LITTROW MASK file'
      if not keyword_set(silent) then print('halt: '+error)
      stop
      return
    endif
  
    # Check that the file exists
    if file_test(littrowcorr) eq 0 then begin
      error = 'LITTROWCORR file '+littrowcorr+' NOT FOUND'
      if not keyword_set(silent) then print('halt: '+error)
      stop
      return
    endif
    
    # Load the LITTROW file
    #  This should be 2048x2048
    FITS_READ,littrowcorr,littrowim,littrowhead,message=message,/no_abort
  
    # Error opening file
    if message ne '' then begin
      error = message
      if not keyword_set(silent) then print('halt: '+error)
      stop
      return
    endif
  
    if not keyword_set(silent) then print('LITTROW file = '+littrowcorr)
  
    # Check that the file looks reasonable
    #  must be 2048x2048 and have 0/1 values
    szlittrow = size(littrowim)
    littrowokay = 0
    dum = where(littrowim ne 0 and littrowim ne 1,nbad)
    if szlittrow[0] ne 2 or szlittrow[1] ne 2048 or szlittrow[2] ne 2048 or nbad gt 0 then begin
      error = 'LITTROW MASK must be 2048x2048 with 0/1 values'
      if not keyword_set(silent) then print('halt: '+error)
      stop
      return
    endif
  endif # loading littrow file

  return littrowim,littrowhead


def loadpersist(persistcorr,silent=True):
  #--------------------------------
  # Load PERSISTENCE MASK File
  #--------------------------------.
  if n_elements(persistcorr) gt 0 and n_elements(persistim) eq 0 then begin
  
    # PERSISTCORR must be scalar string
    if size(persistcorr,/type) ne 7 or n_elements(persistcorr) ne 1 then begin
      error = 'PERSISTCORR must be a scalar string with the filename of the PERSIST MASK file'
      if not keyword_set(silent) then print('halt: '+error)
      stop
      return
    endif
  
    # Check that the file exists
    if file_test(persistcorr) eq 0 then begin
      error = 'PERSISTCORR file '+persistcorr+' NOT FOUND'
      if not keyword_set(silent) then print('halt: '+error)
      stop
      return
    endif
    
    # Load the PERSIST file
    #  This should be 2048x2048
    FITS_READ,persistcorr,persistim,persisthead,message=message,/no_abort
  
    # Error opening file
    if message ne '' then begin
      error = message
      if not keyword_set(silent) then print('halt: '+error)
      stop
      return
    endif
  
    if not keyword_set(silent) then print('PERSIST file = '+persistcorr)
  
    # Check that the file looks reasonable
    #  must be 2048x2048 and have 0/1 values
    szpersist = size(persistim)
    persistokay = 0
    if szpersist[0] ne 2 or szpersist[1] ne 2048 or szpersist[2] ne 2048 then begin
      error = 'PERSISTENCE MASK must be 2048x2048'
      if not keyword_set(silent) then print('halt: '+error)
      stop
      return
    endif
  endif # loading persistence file

  return persistim,persisthead


def loaddark(darkcorr,silent=True):  
  #----------------------------
  # Load DARK CORRECTION file
  #----------------------------.
  if n_elements(darkcorr) gt 0 and n_elements(darkcube) eq 0 then begin
  
    # DARKCORR must be scalar string
    if size(darkcorr,/type) ne 7 or n_elements(darkcorr) ne 1 then begin
      error = 'DARKCORR must be a scalar string with the filename of the dark correction file'
      if not keyword_set(silent) then print('halt: '+error)
      stop
      return
    endif
  
    # Check that the file exists
    if file_test(darkcorr) eq 0 then begin
      error = 'DARKCORR file '+darkcorr+' NOT FOUND'
      if not keyword_set(silent) then print('halt: '+error)
      stop
      return
    endif
  
    # Read header
    darkhead0 = HEADFITS(darkcorr,errmsg=errmsg0,exten=0)
    darkhead1 = HEADFITS(darkcorr,errmsg=errmsg1,exten=1)
    # Error reading header
    if errmsg0 ne '' or errmsg1 ne '' then begin
      error = errmsg0+' '+errmsg1
      if not keyword_set(silent) then print('halt: '+error)
      stop
      return
    endif
  
    # Get number of reads
    if sxpar(darkhead1,'NAXIS') eq 3 then begin
      nreads_dark = sxpar(darkhead1,'NAXIS3')
    endif else begin
      # Extensions
      # Figure out how many reads/extensions there are
      #  the primary unit should be empty
      nreads_dark = 0
      message = ''
      while (message eq '') do begin
        nreads_dark++
        dum = HEADFITS(darkcorr,exten=nreads_dark,errmsg=message)
      end
      nreads_dark--  # removing the last one
  
    endelse
  
  
    # Check that it has enough reads
    #nreads_dark = sxpar(darkhead,'NAXIS3')
    if nreads_dark lt nreads then begin
      error = 'SUPERDARK file '+darkcorr+' does not have enough READS. Have '+strtrim(nreads_dark,2)+$
              ' but need '+strtrim(nreads,2)
      if not keyword_set(silent) then print('halt: '+error)
      stop
      return
    endif
  
    
    # Load the dark correction file
    #  This needs to be 2048x2048xNreads
    #  It's the dark counts for each pixel in counts
  
    # Datacube
    if sxpar(darkhead1,'NAXIS') eq 3 then begin
  
      FITS_READ,darkcorr,darkcube,/no_abort

    # Extensions
    endif else begin
  
      # Initializing the cube
      FITS_READ,darkcorr,darkim,exthead,exten_no=1,message=message,/no_abort
      szim = size(darkim)
      darkcube = np.zeros((szim[1],szim[2],nreads_dark),float)
  
      # Read in the extensions
      For k=1,nreads_dark do begin
        FITS_READ,darkcorr,extim,exthead,exten_no=k,message=message,/no_abort 
        darkcube[*,*,k-1] = extim
  
    endelse # extensions
  
    szdark = size(darkcube)
  
    if not keyword_set(silent) then print('Dark Correction file = '+darkcorr)
  
    # Check that the file looks reasonable
    szdark = size(darkcube)
    if (szdark[0] ne 3 or szdark[1] lt 2048 or szdark[2] ne 2048) then begin
      error = 'Dark correction data must a 2048x2048xNreads datacube of the dark counts per pixel'
      if not keyword_set(silent) then print('halt: '+error)
      stop
      return
    endif
  
  endif # loading dark correction file

  return darkcube,darkhead


def loadflat(flatcorr,silent=True):
  
  #----------------------------------
  # Load FLAT FIELD CORRECTION file
  #----------------------------------.
  if n_elements(flatcorr) gt 0 and n_elements(flatim) eq 0 then begin
  
    # FLATCORR must be scalar string
    if size(flatcorr,/type) ne 7 or n_elements(flatcorr) ne 1 then begin
      error = 'FLATCORR must be a scalar string with the filename of the flat correction file'
      if not keyword_set(silent) then print('halt: '+error)
      stop
      return
    endif
  
    # Check that the file exists
    if file_test(flatcorr) eq 0 then begin
      error = 'FLATCORR file '+flatcorr+' NOT FOUND'
      if not keyword_set(silent) then print('halt: '+error)
      stop
      return
    endif
    
    # Load the flat correction file
    #  This needs to be 2048x2048
    FITS_READ,flatcorr,flatim,flathead,message=message,/no_abort
  
    # Error opening file
    if message ne '' then begin
      error = message
      if not keyword_set(silent) then print('halt: '+error)
      stop
      return
    endif
  
    if not keyword_set(silent) then print('Flat Field Correction file = '+flatcorr)
  
    # Check that the file looks reasonable
    szflat = size(flatim)
    if (szflat[0] ne 2 or szflat[1] ne 2048 or szflat[2] ne 2048) then begin
      error = 'Flat Field correction image must a 2048x2048 image'
      if not keyword_set(silent) then print('halt: '+error)
      stop
      return
    endif
  
  endif # loading flat field correction file

  return flatim,flathead


# refsub subtracts the reference array from each quadrant with proper flipping
def aprefcorr_sub(image,ref):
    revref = np.flip(ref,axis=1)
    image[:,0:512] -= ref
    image[:,512:1024] -= revref
    image[:,1024:1536] -= ref
    image[:,1536:2048] -= revref
    return image

def aprefcorr(cube,head,mask,indiv=3,vert=1,horz=1,noflip=noflip,silent=silent,
              readmask=readmask,lastgood=lastgood,cds=1,plot=plot,fix=fix,q3fix=q3fix,keepref=False):
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
    if keepref: refout = np.zeros(512,2048,nread),int)

    # Ignore reference array by default
    # Default is to do CDS, vertical, and horizontal correction
    print('in aprefcorr, indiv: '+str(indiv))

    satval = 55000

    snmin = 10
    if indiv>0:
        hmax = 1e10
    else:
        hmax = 65530

    if len(mask)<=1: mask=np.zeros((2048,2048),int)
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
        if count==0: iread=i+1
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

    if cds: cdsref = cube[0:2048,:,1]

    # Loop over the reads
    lastgood = nread-1
    for iread in range(nread):

        # Subtract mean reference array
        red = cube[0:2048,:,iread].astype(int)

### I GOT TO HERE !!!!
             
  sat = where(red gt satval,nsat)
  if nsat gt 0 then begin
    if iread eq 0 then nsat0=nsat
    red[sat] = 65535
    mask[sat] = (mask[sat] or maskval('SATPIX'))
    # if we have a lot of saturated pixels, note this read (but don't do anything)
    if nsat gt nsat0+2000 then begin
      if lastgood eq nread-1 then lastgood=iread-1
    endif
  endif else nsat0=0
  # pixels that are identically zero are bad, see these in first few reads
  bad = where(red eq 0,nbad)
  if nbad gt 0 then mask[bad] = (mask[bad] or maskval('BADPIX'))
  if not keyword_set(silent) then $
    print,format='(%"Ref processing: %3d  nsat: %5d\r",$)',iread+1,n_elements(sat)
  if readmask[iread] gt 0 then begin
    red = !values.f_nan
    goto,nextread
  endif

  # with cds keyword, subtract off first read before getting reference pixel values
  if keyword_set(cds) then red-=cdsref

  ref = cube[2048:2559,*,iread]
  if indiv==1:
    red = aprefcorr_sub(red,ref)
    ref -= ref
  elif indiv>1:
    APREFCORR_SUB,red,median(ref,indiv)
    ref-=median(ref,indiv)
  endif else if indiv lt 0 then begin
    APREFCORR_SUB,red,meanref
    ref-=meanref
  endif

  if vert:
    # Subtract vertical ramp
    for j in range(4):
      rlo = np.nanmean(red[2:4,j*512:(j+1)*512])                                                                                                      
      rhi = np.nanmean(red[2045:2047,j*512:(j+1)*512])                                                                                                
      red[:,j*512:(j+1)*512] -= rlo*vrramp                                                                                                            
      red[:,j*512:(j+1)*512] -= rhi*vramp     
      #if keyword_set(plot) then begin
      #  plot,rlo*vrramp[0,*]+rhi*vramp[0,*]
      #  print,j,rlo,rhi 
      #  atv,cube[0:2047,*,iread]-cube[0:2047,*,1]
      #  atv,red
      #  stop
      #endif

  # Subtract smoothed horizontal ramp
  if horz:
    clo = np.nanmean(im[:,1:4],axis=0)                                                                                                                 
    chi = np.nanmean(im[:,2044:2047],axis=0)                                                                                                           
   #clo = total(red[1:3,*],1,/nan) / ( total(finite(red[1:3,*]),1) > 1)
   #chi = total(red[2044:2046,*],1,/nan) / ( total(finite(red[2044:2046,*]),1) > 1)

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
    bias = rows#min([[slo],[shi]],dim=2)
    fbias = bias
    fbias[512:1023,*] = reverse(bias[512:1023,*])
    fbias[1536:2047,*] = reverse(bias[1536:2047,*])
    red -= fbias

  endif
  if keyword_set(q3fix) then begin
    #fix=red
    q3offset = np.zeros(2048,float)
    for irow=0,2047 do begin
      q2m=median(red[923:1023,irow])
      q3a=median(red[1024:1124,irow])
      q3b=median(red[1435:1535,irow])
      q4m=median(red[1536:1636,irow])
      #fix[1024:1535,irow]+=((q2m-q3a)+(q4m-q3b))/2.
      q3offset[irow]=((q2m-q3a)+(q4m-q3b))/2.
    endfor
    #plot,q3offset
    #oplot,medfilt1d(q3offset,7,/edge),color=2
    #red=fix
    red[1024:1535,*]+=(medfilt1d(q3offset,7,/edge)##(fltarr(512)+1))
    #atv,red,min=-200,max=200,/linear
    #stop
  endif

  # Make sure saturated pixels are set to 65535
  #  removing the reference values could have
  #  bumped them lower
  if nsat gt 0 then red[sat]=65535

  nextread:
  #reduced[*,*,iread] = red
  #cube[0:2047,*,iread] = red  # overwrite with the ref-subtracted image
  out[*,*,iread] = red
  if keyword_set(keepref) then refout[*,*,iread] = ref

endfor # read loop

# Trim off the reference array
#cube = cube[0:2047,*,*]

# mask the reference pixels
mask[0:3,*] = (mask[0:3,*] or maskval('BADPIX'))
mask[2044:2047,*] = (mask[2044:2047,*] or maskval('BADPIX'))
mask[*,0:3] = (mask[*,0:3] or maskval('BADPIX'))
mask[*,2044:2047] = (mask[*,2044:2047] or maskval('BADPIX'))

if not keyword_set(silent) then begin
  print,''
  print,'lastgood: ',lastgood
endif

if keyword_set(keepref) then return,[out,refout] else return,out

    return cube,mask,readmask



def ap3dproc(files0,outfile,detcorr=detcorr,bpmcorr=bpmcorr,darkcorr=darkcorr,littrowcorr=littrowcorr,
             persistcorr=persistcorr,persistmodelcorr=persistmodelcorr,histcorr=histcorr,
             flatcorr=flatcorr,crfix=True,satfix=True,rd3satfix=rd3satfix,saturation=65000,
             nfowler=nfowler,uptheramp=uptheramp,verbose=verbose,debug=debug,error=error,silent=silent,
             cube=cube,head=head,output=output,crstr=crstr,satmask=satmask,criter=False,
             clobber=clobber,stp=stp,cleanuprawfile=cleanuprawfile,outlong=False,refonly=refonly,
             outelectrons=False,nocr=nocr,logfile=logfile,fitsdir=fitsdir,maxread=maxread,q3fix=q3fix,
             usereference=usereference,seq=seq):
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

    Parameters:
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

    Returns:
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
    >>>ap3dproc,'apR-a-test3.fits','ap2D-a-test3.fits'

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
    """

    
    t00 = systime(1)
    apgundef,cube,head,output,crstr,satmask

    nfiles = n_elements(files0)


    if keyword_set(debug):
        setdisp,/silent
        psym8


    # No output requested
    if n_elements(outfile) eq 0 and not arg_present(cube) and not arg_present(head) and 
    not arg_present(output) and not arg_present(crstr) and not arg_present(satmask):
        error = 'No output requested'
        print,error
           return


    noutfile = n_elements(outfile)
    if n_elements(outfile) gt 0 then if noutfile ne nfiles:
        error = 'OUTFILE must have same number of elements as FILES'
        if not keyword_set(silent) then print,error
        return

    # Default parameters
    if n_elements(nfowler) eq 0 and n_elements(uptheramp) eq 0 then nfowler = 10    # number of reads to use at beg and end
    if n_elements(seq) eq 0 then seq='no seq'
  
    if not keyword_set(silent):
        print,'AP3DPROC Input Parameters:'
        print,'Saturation = ',strtrim(long(saturation),2)
        if keyword_set(crfix) then print,'Fixing Cosmic Rays' else print,'NOT Fixing Cosmic Rays'
        if keyword_set(satfix) then print,'Fixing Saturated Pixels' else print,'NOT Fixing Saturated Pixels'
        if keyword_set(nfowler) then print,'Using FOWLER Sampling, Nfowler='+strtrim(long(nfowler),2)
        if not keyword_set(nfowler) then print,'Using UP-THE-RAMP Sampling'
        if keyword_set(criter) then print,'Iterative CR detection ON' else print,'Iterative CR detection OFF'
        if keyword_set(clobber) then print,'Clobber ON'
        if keyword_set(outelectrons) then print,'Output will be in ELECTRONS' else print,'Output will be in ADU'
        print,''
    print,strtrim(nfiles,2),' File(s) input'


    # File loop
    #------------
    for f in range(nfiles):
        t0 = time.time()
        ifile = files0[f]

        if not keyword_set(silent):
            if f gt 0 then print,''
            print,strtrim(f+1,2),'/',strtrim(nfiles,2),' Filename = ',ifile
            print,'----------------------------------'

        # if another job is working on this file, wait
        if n_elements(outfile) gt 0 then:
            if getlocaldir():
                lockfile=getlocaldir()+'/'+file_basename(outfile[f])+'.lock'
            else:
                lockfile=outfile[f]+'.lock'
            while file_test(lockfile) do apwait,lockfile,10

            # Test if the output file already exists
            #if n_elements(outfile) gt 0 then begin
            if (file_test(outfile[f]) eq 1 or file_test(outfile[f]+'.fz') eq 1) and not keyword_set(clobber):
                if not keyword_set(silent) then print,'OUTFILE = ',outfile[f],' ALREADY EXISTS. Set /clobber to overwrite'
                continue

            # set lock to notify other jobs that this file is being worked on
            openw,lock,/get_lun,lockfile
            free_lun,lock

        # Check the file
        dir = file_dirname(ifile)
        base = file_basename(ifile)
        len = strlen(base)
        basesplit = strsplit(base,'.',/extract)
        extension = first_el(basesplit,/last)
        #if strmid(base,0,4) ne 'apR-' or strmid(base,len-5,5) ne '.fits' then begin
        #  error = 'FILE must be of the form >>apR-a/b/c-XXXXXXXX.fits<<'
        #  if not keyword_set(silent) then print,error
        #  return
        #endif
        if extension ne 'fits' and extension ne 'apz':
            error = 'FILE must have a ".fits" or ".apz" extension'
            if not keyword_set(silent) then print,error
            continue
  
        # Compressed file input
        if extension eq 'apz':
            if not keyword_set(silent) then print,ifile,' is a COMPRESSED file'

            # Check if the decompressed file already exists
            len = strlen(base)
            if keyword_set(fitsdir):
                fitsfile = fitsdir+'/'+strmid(base,0,len-4)+'.fits' 
            else:
                fitsfile = dir+'/'+strmid(base,0,len-4)+'.fits'
                fitsdir=0
  
            # Need to decompress
            if file_test(fitsfile) == 0:
                if not keyword_set(silent) then print,'Decompressing with APUNZIP'
                id=0L
                reads,strmid(base,6,8),id
                if id lt 02490000L then no_checksum=1 else no_checksum=0 
                print,'no_checksum: ', no_checksum
                APUNZIP,ifile,/clobber,error=errzip,fitsdir=fitsdir,no_checksum=no_checksum  # /silent
                print,''
                doapunzip = 1     # we ran apunzip
  
                # An error occurred
                if n_elements(errzip) gt 0:
                    error = 'ERROR in APUNZIP '+errzip
                    if not keyword_set(silent) then print,'halt: '+error
                    stop
                    continue

            # Decompressed file already exists
            else:
                if not keyword_set(silent) then $
                print,'The decompressed file already exists'
                doapunzip = 0     # we didn't run apunzip

            file = fitsfile  # using the decompressed FITS from now on

            # Cleanup by default
            if n_elements(cleanuprawfile) eq 0 and extension eq 'apz' and doapunzip eq 1 then cleanuprawfile=1     # remove recently decompressed file

        # Regular FITS file input
        else:
            file = ifile
            doapunzip = -1
 
        if not keyword_set(silent):
            if extension eq 'apz' and keyword_set(cleanuprawfile) and doapunzip eq 1 then $
            print,'Removing recently decompressed FITS file at end of processing'
  
        # Check that the file exists
        if file_test(file) eq 0:
            error = 'FILE '+file+' NOT FOUND'
            if not keyword_set(silent) then print,'halt: '+error
            stop
            continue
 
        # Get header
        head = headfits(file,errmsg=errmsg)
        if errmsg ne '':
            error = 'There was an error loading the HEADER for '+file
            if not keyword_set(silent) then print,'halt: '+error
            stop
            continue
  
        # Check that this is a data CUBE
        naxis = sxpar(head,'NAXIS')
        FITS_READ,file,dumim,dumhead,exten_no=1,message=read_message,/no_abort
        if naxis ne 3 and read_message ne '':
            error = 'FILE must contain a 3D DATACUBE OR image extensions'
            if not keyword_set(silent) then print,'halt: '+error
            stop
            continue


        # Test if the output file already exists
        if n_elements(outfile) gt 0:
            test = file_test(outfile[f])
            if test eq 1 and not keyword_set(clobber):
                print,'OUTFILE = ',outfile[f],' ALREADY EXISTS.  Set /clobber to overwrite.'
                continue

        # Read in the File
        #-------------------
        test = file_test(file)
        if file_test(file) eq 0:
            error = file+' NOT FOUND'
            if not keyword_set(silent) then print,'halt: '+error
            stop
            continue
        # DATACUBE
        if naxis==3:
            FITS_READ,file,cube,head,message=message,/no_abort   # UINT
            # Error opening file
            if message ne '':
                error = message
                if not keyword_set(silent) then print,'halt: '+error
                stop
                goto,BOMB
        # Extensions
        else:
            head = headfits(file)
            # Figure out how many reads/extensions there are
            #  the primary unit should be empty
            nreads = 0
            message = ''
            while (message eq ''):
                nreads++
                dum = headfits(file,exten=nreads,errmsg=message)
            nreads--  # removing the last one
  
            # Only 1 read
            if nreads lt 2:
                error = 'ONLY 1 read.  Need at least two'
                if not keyword_set(silent) then print,'halt: '+error
                stop
                goto,BOMB

            # allow user to specify maximum number of reads to use (e.g., in the
            #   case of calibration exposures that may be overexposed in some chip
            if keyword_set(maxread) then if maxread lt nreads then nreads=maxread
  
            # Initializing the cube
            FITS_READ,file,im1,exten_no=1
            sz = size(im1)
            #cube = uintarr(sz[1],sz[2],nreads)
            cube = lonarr(sz[1],sz[2],nreads)    # long is big enough and takes up less memory than float
  
            # Read in the extensions
            for k=1,nreads:
                FITS_READ,file,extim,exthead,exten_no=k,message=message,/no_abort   # UINT
                cube[*,*,k-1] = extim
                # What do we do with the extension headers???
                # We could make a header structure or array
  
        # Dimensions of the cube
        sz = size(cube)
        type = size(cube,/type)  # UINT
        nx = sz[1]
        ny = sz[2]
        nreads = sz[3]
        chip = strtrim(sxpar(head,'CHIP'),2)
        if chip eq '0':
            raise ValueError('CHIP not found in header')

        # File dimensions
        if silent is False:
            print('Data file description:')
            print('Datacube size = '+str(int(nx))+' x '+str(int(ny))+' x '+str(int(nreads)))
            print('Nreads = '+str(int(nreads)))
            print('Chip = '+str(chip))
            print('')
  
        # Few reads
        if nreads eq 2 and not keyword_set(silent) then print,'Only 2 READS. CANNOT do CR detection/fixing'
        if nreads eq 2 and keyword_set(satfix) and not keyword_set(silent):
            print,'Only 2 READS. CANNOT fix Saturated pixels'

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

  if n_elements(detcorr) gt 0 or n_elements(darkcorr) gt 0 or n_elements(flatcorr) gt 0 and $
     not keyword_set(silent) then print,''
  
  
  #---------------------
  # Check for BAD READS
  #---------------------
  if not keyword_set(silent) then $
    print,'Checking for bad reads'
  
  # Use the reference pixels and reference output for this
  
  if sz[1] eq 2560 then begin
    refout1 = median(cube[2048:*,*,0:3<(nreads-1)],dim=3)
    sig_refout_arr = np.zeros(nreads,float)
    rms_refout_arr = np.zeros(nreads,float)
  endif
  
  refpix1 = [[ median(cube[0:2047,0:3,0:3<(nreads-1)],dim=3) ], $
             [transpose( median(cube[0:3,*,0:3<(nreads-1)],dim=3) ) ],$
             [transpose( median(cube[2044:2047,*,0:3<(nreads-1)],dim=3) ) ],$
             [ median(cube[0:2047,2044:2047,0:3<(nreads-1)],dim=3) ]]
  sig_refpix_arr = np.zeros(nreads,float)
  rms_refpix_arr = np.zeros(nreads,float)
  
  for k=0,nreads-1 do begin
  
    refpix = [[cube[0:2047,0:3,k]], [transpose(cube[0:3,*,k])],$
              [transpose(cube[2044:2047,*,k])], [cube[0:2047,2044:2047,k]]]
    refpix = float(refpix)
  
    # The top reference pixels are normally bad
    diff_refpix = refpix - refpix1
    sig_refpix = MAD(diff_refpix[*,0:11],/zero)
    rms_refpix = sqrt(mean(diff_refpix[*,0:11]^2))
  
    sig_refpix_arr[k] = sig_refpix
    rms_refpix_arr[k] = rms_refpix
  
    # Using reference pixel output (5th output)
    if sz[1] eq 2560 then begin
  
      refout = float(cube[2048:*,*,k])
  
      # The top and bottom are bad
      diff_refout = refout - refout1
      sig_refout = MAD(diff_refout[*,100:1950],/zero)
      rms_refout = sqrt(mean(diff_refout[*,100:1950]^2))
  
      sig_refout_arr[k] = sig_refout
      rms_refout_arr[k] = rms_refout
    endif
  
  end


  # Use reference output and pixels
  if sz[1] eq 2560:
  
    if nreads>2:
      med_rms_refpix_arr = MEDFILT1D(rms_refpix_arr,11<nreads,/edge)
      med_rms_refout_arr = MEDFILT1D(rms_refout_arr,11<nreads,/edge)
    else:
      med_rms_refpix_arr = np.zeros(nreads,float)+np.median(rms_refpix_arr)
      med_rms_refout_arr = np.zeros(nreads,float)+np.median(rms_refout_arr)

    sig_rms_refpix_arr = mad(rms_refpix_arr) > 1
    sig_rms_refout_arr = mad(rms_refout_arr) > 1
    bdreads = where( (rms_refout_arr-med_rms_refout_arr) gt 10*sig_rms_refout_arr,nbdreads)
  
  # Only use reference pixels
  else:
    if nreads gt 2 then begin
      med_rms_refpix_arr = MEDFILT1D(rms_refpix_arr,11<nreads,/edge)
    endif else begin
      med_rms_refpix_arr = np.zeros(nreads,float)+np.median(rms_refpix_arr)
    endelse
    sig_rms_refpix_arr = mad(rms_refpix_arr) > 1
    bdreads = where( (rms_refpix_arr-med_rms_refpix_arr) gt 10*sig_rms_refpix_arr,nbdreads)
  
  if nbdreads eq 0 then apgundef,bdreads
  
  # Too many bad reads
  if nreads-nbdreads lt 2:
    raise ValueError('ONLY '+str(nreads-nbdreads)+' good reads.  Need at least 2.')

  
  # Reference pixel subtraction
  #----------------------------
  tmp=aprefcorr(cube,head,mask,readmask=readmask,q3fix=q3fix,keepref=usereference)
  cube=tmp

  bdreads2 = where(readmask eq 1,nbdreads2)
  if nbdreads2 gt 0 then PUSH,bdreads,bdreads2
  nbdreads = n_elements(uniq(bdreads,sort(bdreads)))
  
  if nbdreads gt (nreads-2):
    raise ValueError('Not enough good reads')

  gdreads = lindgen(nreads)
  REMOVE,bdreads,gdreads
  ngdreads = n_elements(gdreads)

  # Interpolate bad reads
  if nbdreads gt 0:
    print('Read(s) '+strjoin(str(bdreads+1),', ')+' are bad.')
  
    # The bad reads are currently linearly interpolated using the
    # neighoring reads and used as if they were good.  The variance
    # needs to be corrected for this at the end.
    # This will give bad results for CRs

  
    # Use linear interpolation
    for k=0,nbdreads-1:
  
      # Get good reads below
      gdbelow = where(gdreads lt bdreads[k],ngdbelow)
      # Get good reads above
      gdabove = where(gdreads gt bdreads[k],ngdabove)
  
      if ngdbelow eq 0 then interp_type=1                     # all above
      if ngdbelow gt 0 and ngdabove gt 0 then interp_type=2   # below and above
      if ngdabove eq 0 then interp_type=3                     # all below
  
      case interp_type of
        # all above
        1: begin
          gdlo = gdabove[0]
          gdhi = gdabove[1]
        end
        # below and above
        2: begin
          gdlo = first_el(gdbelow,/last)
          gdhi = gdabove[0]
        end
        # all below
        3: begin
          gdlo = gdbelow[ngdbelow-2]
          gdhi = gdbelow[ngdbelow-1]
        end
      endcase
      lo = gdreads[gdlo]
      hi = gdreads[gdhi]
  
      # Linear interpolation
      im1 = float(cube[*,*,lo])
      im2 = float(cube[*,*,hi])
      slope = (im2-im1)/float(hi-lo)            # slope, (y2-y1)/(x2-x1)
      zeropoint = (im1*hi-im2*lo)/(hi-lo)       # zeropoint, (y1*x2-y2*x1)/(x2-x1)
      im0 = slope*bdreads[k] + zeropoint        # linear interpolation, y=mx+b
  
      # Stuff it in the cube
      cube[*,*,bdreads[k]] = np.round(im0)         # round to closest integer, LONG type
  
    end
  
  endif

  ny,nx = cube.shape
  
  # Reference subtraction ONLY
  if keyword_set(refonly):
    if not keyword_set(silent) then print,'Reference subtraction only'
    goto,BOMB

                       
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
  
  if not keyword_set(silent) then print,'Processing the datacube'

  # Loop through the rows
  for i in range(ny):
  
    if keyword_set(verbose) or keyword_set(debug) then $
      print,'Scanning Row ',strtrim(i+1,2)
    if not keyword_set(silent) and not keyword_set(verbose) and not keyword_set(debug) then $
      if (i+1) mod 500 eq 0 then print,i+1,'/',ny,format='(I4,A1,I4)'
  
    # Slice of datacube, [Ncol,Nread]
    #--------------------------------
    slice = np.float(cube[:,i,:].flatten())
    slice_orig = slice.copy()  # original slice
 
    #---------------------------------
    # Flag BAD pixels
    #---------------------------------
    if bpmim is not None:
      bdpix = where(bpmim[*,i] gt 0,nbdpix)
      if nbdpix gt 0:
        for j=0,nbdpix-1 do slice[bdpix[j],*] = 0.0   # set them to zero
        #mask[bdpix,i] = (mask[bdpix,i] OR maskval('BADPIX'))
        mask[bdpix,i] = (mask[bdpix,i] OR bpmim[bdpix,i])
  
    #---------------------------------
    # Flag LITTROW ghost pixels, but don't change data values
    #---------------------------------
    if littrowim is not None:
      bdpix = where(littrowim[*,i] eq 1,nbdpix)
      if nbdpix gt 0 then mask[bdpix,i] = (mask[bdpix,i] OR maskval('LITTROW_GHOST'))
  
    #---------------------------------
    # Flag persistence pixels, but don't change data values
    #---------------------------------
    if persistim is not None:
      bdpix = where(persistim[*,i] and 1,nbdpix)
      if nbdpix gt 0 then mask[bdpix,i] = (mask[bdpix,i] OR maskval('PERSIST_HIGH'))
      bdpix = where(persistim[*,i] and 2,nbdpix)
      if nbdpix gt 0 then mask[bdpix,i] = (mask[bdpix,i] OR maskval('PERSIST_MED'))
      bdpix = where(persistim[*,i] and 4,nbdpix)
      if nbdpix gt 0 then mask[bdpix,i] = (mask[bdpix,i] OR maskval('PERSIST_LOW'))
  
    #---------------------------------
    # Detect and Flag Saturated reads
    #---------------------------------
    #  The saturated pixels are detected in the reference subtraction
    #  step and fixed to 65535.
    bdsat = where(slice gt saturation,nbdsat)
    if nbdsat>0:
  
      # Flag saturated reads as NAN
      slice[bdsat] = np.nan
  
      # Get 2D indices
      bdsat2d = array_indices(slice,bdsat)
      bdsatx = reform(bdsat2d[0,*])      # X/column indices
      bdsatr = reform(bdsat2d[1,*])      # read indices
      # bdsat is 1D array for slice(2D)
      # bdsatx is column index for 1D med_dCounts
  
      # Unique pixels
      uibdx = uniq(bdsatx,sort(bdsatx))
      ubdsatx = bdsatx[uibdx]
      nbdsatx = n_elements(ubdsatx)
  
      # Figure out at which Read (NOT dCounts) each column saturated
      rindex = (lonarr(nx)+1L)#lindgen(nreads)     # each pixels read index
      satmask_slice = lonarr(nx,nreads)            # sat mask
      satmask_slice[bdsatx,bdsatr] = 1             # set saturated reads to 1
      rindexsat = rindex*satmask_slice + $         # okay pixels have 999999
                  (1-satmask_slice)*999999L        #   sat pixels have their read index
      minsatread = MIN(rindexsat,dim=2)            # now find the minimum for each column
      nsatreads = total(satmask_slice,2)           # number of sat reads
  
      # Make sure that all subsequent reads to a saturated read are
      # considered "bad" and set to NAN
      for j=0,nbdsatx-1 do slice[ubdsatx[j],minsatread[ubdsatx[j]]:*]=!values.f_nan
  
      # Update satmask
      satmask[ubdsatx,i,0] = 1                     # mask
      satmask[ubdsatx,i,1] = minsatread[ubdsatx]   # 1st saturated read, NOT dcounts
      satmask[ubdsatx,i,2] = nsatreads[ubdsatx]    # # of saturated reads
  
      # Update mask
      mask[ubdsatx,i] = (mask[ubdsatx,i] OR maskval('SATPIX'))     # mask: 1-bad, 2-CR, 4-sat, 8-unfixable
  
    #----------------------
    # Linearity correction
    #----------------------
    # This needs to be done BEFORE the pixels are "fixed" because
    # it needs to operate on the ORIGINAL counts, not the corrected
    # ones.
    if lindata is not None:
      if szlin[0] eq 3 then linslice = reform(lindata[*,i,*]) else linslice=lindata
      slice_orig1 = slice  # temporary copy since we'll be overwriting it
      slice = aplincorr(slice_orig1,linslice)
  
    #-----------------
    # Dark correction
    #-----------------
    # Each read will have a different amount of dark counts in it
    if darkcube is not None:
      darkslice = reform(darkcube[*,i,*])
      slice_orig2 = slice  # temporary copy since we'll be overwriting it
      slice = ap3dproc_darkcorr(slice_orig2,darkslice,darkhead)
  
    #------------------------------------------------
    # Find difference of neighboring reads, dCounts
    #------------------------------------------------
    #  a difference with 1 or 2 NaN will also be NAN
    dCounts = slice[*,1:sz[3]-1] - slice[*,0:sz[3]-2]
  
  
    # SHOULD I FIX BAD READS HERE?????
  
  
    #----------------------------
    # Detect and Fix cosmic rays
    #----------------------------
    slice_prefix = slice
    if not keyword_set(nocr) and nreads gt 2:
      dCounts_orig = dCounts  # temporary copy since we'll be overwriting it
      apgundef,dCounts
      satmask_slice = reform(satmask[*,i,*])
      crstr_slice, dCounts, med_dCounts, mask, crindex, crnum, variability_slice = 
                       ap3dproc_crfix(dCounts_orig,satmask_slice,noise=noise,crfix=crfix):
                       
      #AP3DPROC_CRFIX,dCounts_orig,satmask_slice,dCounts,med_dCounts,crstr_slice,$
      #               crfix=crfix,noise=noise_dCounts,variability=variability_slice
  
      variability_im[:,i] = variability_slice
  
    # Only 2 reads, CANNOT detect or fix CRs
    else:
      med_dCounts = dCounts
      crstr_slice = {NCR:0L}
  
    # Some CRs detected, add to CRSTR structure
    if crstr_slice.ncr gt 0:
  
      crstr_slice.data.y = i  # add the row information
  
      # Add to MASK
      maskpix=where(crstr_slice.data.x lt 2048,nmaskpix)
      if nmaskpix gt 0 then mask[crstr_slice.data[maskpix].x,i] = $
        (mask[crstr_slice.data[maskpix].x,i] OR maskval('CRPIX'))
  
      # Starting global structure
      if n_elements(crstr) eq 0:
        crstr = crstr_slice
  
      # Add to global structure
      else:
        ncrtot = crstr.ncr + crstr_slice.ncr
        old_crstr = crstr
        crstr = {NCR:ncrtot,DATA:[old_crstr.data, crstr_slice.data]}
        apgundef,old_crstr
  
  
    #----------------------
    # Fix Saturated reads
    #----------------------
    #  do this after CR fixing, so we don't have to worry about CRs here
    #  set their dCounts to med_dCounts
  
    if nbdsat gt 0:
  
      # Have enough reads (>2) to fix pixels
      if (nreads gt 2) then begin
  
        # Total number of good dCounts for each pixel
        totgd = total(finite(dCounts),2)
  
        # Unfixable pixels
        #------------------
        #  Need 2 good dCounts to be able to "safely" fix a saturated pixel
        thresh_dcounts = 2
        if keyword_set(rd3satfix) and nreads eq 3 then thresh_dcounts=1  # fixing 3 reads
        unfixable = where(totgd lt thresh_dcounts,nunfixable)
        if nunfixable gt 0 then begin
          dCounts[unfixable,*] = 0.0
          mask[unfixable,i] = (mask[unfixable,i] OR maskval('UNFIXABLE'))                   # mask: 1-bad, 2-CR, 4-sat, 8-unfixable
        endif
  
  
        # Fixable Pixels
        #-----------------
        fixable = where(totgd ge thresh_dcounts and satmask[*,i,0] eq 1,nfixable)
  
        # Loop through the fixable saturated pixels
        for j=0,nfixable-1 do begin
  
          ibdsatx = fixable[j]
  
          # "Fix" the saturated pixels
          #----------------------------
          if keyword_set(satfix) then begin
  
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
          endif else begin
            dCounts[ibdsatx,minsatread[ibdsatx]-1:nreads-2] = 0.0    # set saturated dCounts to zero
          endelse
  
        endfor # loop through the fixable saturated pixels
  
        # It might be better to use the last good value from sm_dCounts
        # rather than the straight median of all reads
  
      # Only 2 reads, can't fix anything
      endif else begin
        mask[ubdsatx,i] = (mask[ubdsatx,i] OR maskval('UNFIXABLE'))     # mask: 1-bad, 2-CR, 4-sat, 8-unfixable
        dCounts[bdsatx] = 0.0                        # set saturated reads to zero
      endelse
  
    endif  # some saturated pixels
  
   
    #------------------------------------
    # Reconstruct the SLICE from dCounts
    #------------------------------------
    slice0 = slice[*,0]  # first read
    bdsat = where(finite(slice0) eq 0,nbdsat)  # NAN in first read, set to 0.0
    if nbdsat gt 0 then slice0[bdsat] = 0.0
  
    unfmask_slice = long((long(mask[*,i]) AND maskval('UNFIXABLE')) eq maskval('UNFIXABLE'))  # unfixable
    slice0[0:2047] = slice0[0:2047]*(1.0-unfmask_slice)                 # set unfixable pixels to zero
  
    slice_fixed = slice0 # (fltarr(nreads)+1.0)
    if nreads gt 2 then begin
      slice_fixed[*,1:*] += TOTAL(dCounts,2,/cumulative)
    endif else begin
      slice_fixed[*,0] = slice0
      slice_fixed[*,1] = slice0+dCounts
    endelse
  
    #--------------------------------
    # Put fixed slice back into cube
    #--------------------------------
    cube[*,i,*] = round( slice_fixed )        # round to closest integer, LONG type
  
    #------------------------------------
    # Final median of each "fixed" pixel
    #------------------------------------
    if nreads gt 2 then begin
      # Unfixable pixels are left at 0.0
  
      # If NOT fixing saturated pixels, then we need to
      # temporarily set saturated reads to NAN
      #  Leave unfixable pixels at 0.0
      temp_dCounts = dCounts
      if not keyword_set(satfix) then begin
        bdsat = where(satmask[*,i,0] eq 1 and unfmask_slice eq 0,nbdsat)
        for j=0,nbdsat-1 do $
          temp_dCounts[bdsat[j],satmask[bdsat[j],i,1]-1:*] = !values.f_nan
      endif
  
      fmed_dCounts = median(temp_dCounts,dim=2,/even)    # NAN are automatically ignored
      bdnan = where(finite(fmed_dCounts) eq 0,nbdnan)
      if nbdnan gt 0 then stop
      med_dCounts_im[*,i] = fmed_dCounts
  
    # Only 2 reads
    endif else begin
      med_dCounts_im[*,i] = dCounts
    endelse
  
    if keyword_set(verbose) or keyword_set(debug) then begin
      nsatslice = total(satmask[*,i,0])
      if n_elements(crstr) eq 0 then ncrslice=0 else dum=where(crstr.data.y eq i,ncrslice)
      print,'Nsat/NCR = ',strtrim(long(nsatslice),2),'/',strtrim(long(ncrslice),2),' this row'
    endif
  
    #----------
    # Plotting
    #----------
    #debug = 1
    pltpix = where(mask[*,i] gt 1,npltpix)
    if keyword_set(debug) and npltpix gt 0 then $
      AP3DPROC_PLOTTING,dcounts,slice_prefix,slice_fixed,mask,crstr,i,saturation
  
    #if i eq 623 then stop
  
    #stop
  
  END  # loop through the rows
  
  if n_elements(crstr) eq 0 then crstr={ncr:0L}

    
  #------------------------
  # Iterative CR Rejection
  #------------------------
  #   Check neighbors of CRs for CRs at the same read
  #   lower Nsigma threshold
  if keyword_set(criter) then begin
    if not keyword_set(silent) then print,'Checking neighbors for CRs'
  
    iterflag = 0
    niter = 0
    WHILE (iterflag ne 1) and (crstr.ncr gt 0) do begin
  
      newcr_thisloop = 0
  
      # CRs are left to check
      crtocheck = where(crstr.data.neicheck eq 0,ncrtocheck)
  
      # Loop through the CRs
      for i=0L,ncrtocheck-1 do begin
  
        ix = crstr.data[crtocheck[i]].x
        iy = crstr.data[crtocheck[i]].y
        ir = crstr.data[crtocheck[i]].read
  
        # Look at neighboring pixels to the affected CR
        #   at the same read
        xlo = (ix-1) > 0L
        xhi = (ix+1) < (nx-1L)
        ylo = (iy-1) > 0L
        yhi = (iy+1) < (ny-1L)
        nnei = (xhi-xlo+1)*(yhi-ylo+1) - 1
  
        # Create a fake "slice" of the neighboring pixels
        nei_slice = np.zeros((nnei,nreads),float)
        nei_slice_orig = np.zeros((nnei,nreads),float)  # original slice if saturated
        nei_cols = np.zeros(nnei,float)  # x
        nei_rows = np.zeros(nnei,float)  # y
        nei_satmask = np.zeros((nnei,3),int)
        count = 0
        for j=xlo,xhi do begin
          for k=ylo,yhi do begin
            # Check that the read isn't saturated for this pixel and read
            if satmask[j,k,0] eq 1 and satmask[j,k,1] le ir then readsat=1 else readsat=0
            # Only want neighbors
            if (j ne ix and k ne iy) and (readsat eq 0) then begin
              nei_slice[count,*] = float( cube[j,k,*] )
              nei_slice_orig[count,*] = float( cube[j,k,*] )
              nei_cols[count] = j
              nei_rows[count] = k
              nei_satmask[count,*] = satmask[j,k,*]
              # if this pixel is saturated then we need to make the saturated
              #   reads NAN again.  The "fixed" values are in nei_slice_orig
              if satmask[j,k,0] eq 1 then $
                nei_slice[count,satmask[j,k,1]:nreads-1] = !values.f_nan
              count++
            end
          end
        end
  
        # Difference of neighboring reads, dCounts
        nei_dCounts = nei_slice[*,1:nreads-1] - nei_slice[*,0:nreads-2]
  
        # Fix/detect cosmic rays
        AP3DPROC_CRFIX,nei_dCounts,nei_satmask,nei_dCounts_fixed,med_nei_dCounts,nei_crstr,crfix=crfix,$
                       noise=noise_dCounts,sigthresh=6,onlythisread=ir
  
  
        # Some new CRs detected
        if nei_crstr.ncr gt 0 then begin
  
          # Add the neighbor information
          ind = nei_crstr.data.x             # index in the nei slice
          nei_crstr.data.x = nei_cols[ind]   # actual column index
          nei_crstr.data.y = nei_rows[ind]   # actual row index
  
          # Replace NANs if there were saturated pixels
          bdsat = where(nei_satmask[*,0] eq 1,nbdsat)
          for j=0,nbdsat-1 do begin
            lr = nei_satmask[bdsat[j],1]-1 & hr=nreads-2
            nei_dCounts_orig = nei_slice_orig[*,1:nreads-1] - nei_slice_orig[*,0:nreads-2]  # get "original" dCounts
            nei_dCounts_fixed[bdsat[j],lr:hr] = nei_dCounts_orig[bdsat[j],lr:hr]            # put the original fixed valued back in
          end
  
          # Reconstruct the SLICE from dCounts
          nei_slice0 = nei_slice[*,0]  # first read
          bdsat = where(finite(nei_slice0) eq 0,nbdsat)  # set NAN in first read to 0.0
          if nbdsat gt 0 then nei_slice0[bdsat] = 0.0
          nei_slice_fixed = nei_slice0 # (fltarr(nreads)+1.0)
          if nreads gt 2 then begin
            nei_slice_fixed[*,1:*] += TOTAL(nei_dCounts_fixed,2,/cumulative)
          endif else begin
            nei_slice_fixed[*,0] = nei_slice0
            nei_slice_fixed[*,1] = nei_slice0+nei_dCounts_fixed
          endelse
  
          # Put fixed slice back into cube
          #  only update new CR pixels
          for j=0,nei_crstr.ncr-1 do $
            cube[nei_crstr.data[j].x,nei_crstr.data[j].y,*] = round( nei_slice_fixed[ind[j],*] )  # round to integer
  
          # Update med_dCounts_im ???
  
          # Add to the total CRSTR
          ncrtot = crstr.ncr + nei_crstr.ncr
          old_crstr = crstr
          crstr = {NCR:ncrtot,DATA:[old_crstr.data, nei_crstr.data]}
          apgundef,old_crstr
  
        endif # some new CRs detected
  
        # New CRs detected
        nnew = nei_crstr.ncr
        newcr_thisloop += nnew
  
        # This CR has been checked
        crstr.data[crtocheck[i]].neicheck = 1  # checked!
  
      endfor  # CR loop
  
      # Time to stop
      if (newcr_thisloop eq 0) or (niter gt 5) or (ncrtocheck eq 0) then iterflag=1
  
      niter++
  
    ENDWHILE  # CR iteration loop
  
  end # check CR neighbors
  
  #print,'Neighbors done'
  
  
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
  crmask = long((long(mask) AND maskval('CRPIX')) eq maskval('CRPIX'))
  highpix = where(satmask[*,*,0] eq 0 and crmask eq 0 and med_dCounts_im gt 40,nhighpix)
  if nhighpix eq 0 then $
    highpix = where(satmask[*,*,0] eq 0 and med_dCounts_im gt 20,nhighpix)
  if nhighpix gt 0 then begin
    global_variability = median(variability_im[highpix],/even)
  endif else begin
    global_variability = -1
  endelse
  
  
  #-------------------------
  # FLAT FIELD CORRECTION
  #-------------------------
#  if n_elements(flatim) gt 0 then begin
#  
#    # Apply the flat field to each read.  Each pixel has a separate
#    # kTC noise/offset, but applying the flat field before or after
#    # this is removed gives IDENTICAL results.
#  
#    # Just divide the read image by the flat field iamge
#    #for i=0,nreads-1 do cube[*,*,i] /= flatim
#    for i=0,nreads-1 do cube[*,*,i] = round( cube[*,*,i] / flatim )  # keep cube LONG type
#  
#    # Hopefully this propogates everything to the variance
#    # image properly.
#  
#  endif
  
  
  #------------------------
  # COLLAPSE THE DATACUBE
  #------------------------
  
  # No gain image, using gain=1.0 for all pixels
  # gain = Electrons/ADU
  if len(gainim)==0:
    print('NO gain image.  Using GAIN=1')
    gainim = np.zeros((2048,2048),float)+1.0

    
  # Fowler Sampling
  #------------------
  if uptheramp == False:
  
    # Make sure that Nfowler isn't too large
    Nfowler_used = Nfowler
    if Nfowler gt Nreads/2 then Nfowler_used=Ngdreads/2
  
    # Use the mean of Nfowler reads

    # Beginning sample
    gd_beg = gdreads[0:nfowler_used-1]
    if n_elements(gd_beg) eq 1 then  $
    im_beg = cube[*,*,gd_beg]/float(Nfowler_used) else $
    im_beg = total(cube[*,*,gd_beg],3)/float(Nfowler_used)

    # End sample
    gd_end = gdreads[ngdreads-nfowler_used:ngdreads-1]
    if n_elements(gd_end) eq 1 then  $
    im_end = cube[*,*,gd_end]/float(Nfowler_used) else $
    im_end = total(cube[*,*,gd_end],3)/float(Nfowler_used)

    # The middle read will be used twice for 3 reads

    # Subtract beginning from end
    im = im_end - im_beg
  
    # Noise contribution to the variance
    sample_noise = noise * sqrt(2.0/Nfowler_used)
  
  
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
    for k=0L,ngdreads-1:
      slice = cube[*,*,gdreads[k]]
      if not keyword_set(satfix):
        good = where(satmask[*,*,0] eq 0 or (satmask[*,*,0] eq 1 and satmask[*,*,1] gt i),ngood)
      else:
        good = where(finite(slice))
      #good = where(finite(slice))
      sumts[good] += gdreads[k]*reform(slice[good])
      sums[good] += reform(slice[good])
      sum[good] += 1
      sumt[good] += gdreads[k]
      sumt2[good] += gdreads[k]^2
    #sumt = total(findgen(nread))     # SUM t
    #sumt2 = total(findgen(nread)^2)  # SUM t^2
    # The slope in Counts per read, similar to med_dCounts_im
    #slope = (nread*sumts - sumt*sums)/(nread*sumt2 - sumt^2)
    slope = (sum*sumts - sumt*sums)/(sum*sumt2 - sumt^2)
    # To get the total counts just multiply by nread
    im = slope * (ngdreads-1L)
    # the first read doesn't really add any signal, just a zero-point
  
    # See Equation 1 in Rauscher et al.(2007), SPIE
    #  with m=1
    #  noise and image/flux should be in electrons, sample_noise is in electrons
    #sample_noise = sqrt( 12*(ngdreads-1.)/(nreads*(ngdreads+1.))*(noise*gainim)^2 + 6.*(ngdreads^2+1)/(5.*ngdreads*(ngdreads+1))*im*gainim )
    sample_noise = sqrt( 12*(ngdreads-1.)/(nreads*(ngdreads+1.))*noise^2 + 6.*(ngdreads^2+1)/(5.*ngdreads*(ngdreads+1))*im[0:2047,*]*gainim )
    sample_noise /= gainim  # convert to ADU
  
    # Noise contribution to the variance
    #   this is only valid if the variability is zero
    #ngdreads = nreads - nbdreads
    #sample_noise = noise / sqrt(Ngdreads)

  # With userference, subtract off the reference array to reduce/remove
  #   crosstalk. 
  if keyword_set(usereference):
    print,'subtracting reference array...'
    tmp=im[0:2047,0:2047]
    ref=im[2048:2559,0:2047]
    # subtract smoothed horizontal structure
    ref-=(np.zeros(512,float)+1)#medfilt1d(median(ref,dim=1),7)
    aprefcorr_sub,tmp,ref
    im=tmp
    nx = 2048

  #-----------------------------------
  # Apply the Persistence Correction
  #-----------------------------------
  if n_elements(persistmodelcorr) gt 0 and n_elements(histcorr) gt 0:
    if not keyword_set(silent) then print,'PERSIST modelcorr file = '+persistmodelcorr
    APPERSISTMODEL,ifile,histcorr,persistmodelcorr,pmodelim,ppar,bpmfile=bpmcorr,error=perror
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
    if n_elements(pmodelim) gt 0 then varim+=(im+pmodelim)/gainim > 0 else  varim += im/gainim > 0
     
  # 2. Poisson Noise from dark current
  if n_elements(darkcube) gt 0:
    darkim = reform(darkcube[*,*,nreads-1])
    varim += darkim/gainim > 0
  
  # 3. Sample/read noise
  varim += sample_noise^2         # rdnoise reduced by the sampling
  
  # 4. Saturation error
  #      We used median(dCounts) to extrapolate the saturated pixels
  #      Use the variability in dCounts to estimate the error of doing this
  if keyword_set(satfix):
    varim += sat_extrap_error     # add saturation extrapolation error
  else:
    varim = varim*(1-satmask[*,*,0]) + satmask[*,*,0]*99999999.   # saturated pixels are bad!
  # Unfixable pixels
  unfmask = long((long(mask) AND maskval('UNFIXABLE')) eq maskval('UNFIXABLE'))  # unfixable
  varim = varim*(1-unfmask) + unfmask*99999999.         # unfixable pixels are bad!
  
  # 5. CR error
  #     We use median of neighboring dCounts to "fix" reads with CRs
  crmask = long((long(mask) AND maskval('CRPIX')) eq maskval('CRPIX'))
  if crfix is True:
    # loop in case there are multiple CRs per pixel
    for i=0LL,crstr.ncr-1 do if crstr.data[i].x lt 2048 then $
      varim[crstr.data[i].x,crstr.data[i].y]+=crstr.data[i].fixerror
  else:
    varim = varim*(1-crmask) + crmask*99999999.               # pixels with CRs are bad!
  
  # Bad pixels
  bpmmask = long((long(mask) AND maskval('BADPIX')) eq maskval('BADPIX'))
  varim = varim*(1-bpmmask) + bpmmask*99999999.               # bad pixels are bad!

  # Flat field  
  if n_elements(flatim) gt 0:
    varim /= flatim^2
    im /= flatim

  # Now convert to ELECTRONS
  if n_elements(detcorr) gt 0 and keyword_set(outelectrons):
    varim *= gainim^2
    im *= gainim

  #----------------------------
  # Construct output datacube
  #  [image, error, mask]
  #----------------------------
  if n_elements(pmodelim) gt 0 then output = np.zeros((nx,ny,4),float) else  output = np.zeros((nx,ny,3),float)
  output[:,:,0] = im
  output[:,:,1] = np.maximum(np.sqrt(varim),1)  # must be greater than zero
  output[:,:,2] = mask
  if len(pmodelim)>0:
    output[:,:,3]=pmodelim  # persistence model in ADU
  
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
  if n_elements(detcorr) gt 0 and keyword_set(outelectrons):
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
  if n_elements(pmodelim) gt 0:
    sxaddhist,leadstr+' HDU4 - persistence correction (ADU)',head
  sxaddhist,leadstr+'Global fractional variability = '+strtrim(string(global_variability,format='(F5.3)'),2),head
  maxlen = 72-strlen(leadstr)
  # Bad pixel mask file
  if n_elements(bpmim) gt 0 then begin
    line = 'BAD PIXEL MASK file="'+bpmcorr+'"'
    if strlen(line) gt maxlen then begin
      line1 = strmid(line,0,maxlen)
      line2 = strmid(line,maxlen,100)
      sxaddhist,leadstr+line1,head
      sxaddhist,leadstr+line2,head
    endif else sxaddhist,leadstr+line,head
  end
  # Detector file
  if n_elements(detcorr) gt 0 then begin
    line = 'DETECTOR file="'+detcorr+'"'
    if strlen(line) gt maxlen then begin
      line1 = strmid(line,0,maxlen)
      line2 = strmid(line,maxlen,100)
      sxaddhist,leadstr+line1,head
      sxaddhist,leadstr+line2,head
    endif else sxaddhist,leadstr+line,head
  end
  # Dark Correction File
  if n_elements(darkcube) gt 0 then begin
    line = 'Dark Current Correction file="'+darkcorr+'"'
    if strlen(line) gt maxlen then begin
      line1 = strmid(line,0,maxlen)
      line2 = strmid(line,maxlen,100)
      sxaddhist,leadstr+line1,head
      sxaddhist,leadstr+line2,head
    endif else sxaddhist,leadstr+line,head
  endif
  # Flat field Correction File
  if n_elements(flatim) gt 0 then begin
    line = 'Flat Field Correction file="'+flatcorr+'"'
    if strlen(line) gt maxlen then begin
      line1 = strmid(line,0,maxlen)
      line2 = strmid(line,maxlen,100)
      sxaddhist,leadstr+line1,head
      sxaddhist,leadstr+line2,head
    endif else sxaddhist,leadstr+line,head
  endif
  # Littrow ghost mask File
  if n_elements(littrowim) gt 0 then begin
    line = 'Littrow ghost mask file="'+littrowcorr+'"'
    if strlen(line) gt maxlen then begin
      line1 = strmid(line,0,maxlen)
      line2 = strmid(line,maxlen,100)
      sxaddhist,leadstr+line1,head
      sxaddhist,leadstr+line2,head
    endif else sxaddhist,leadstr+line,head
  endif
  # Persistence mask File
  if n_elements(persistim) gt 0 then begin
    line = 'Persistence mask file="'+persistcorr+'"'
    if strlen(line) gt maxlen then begin
      line1 = strmid(line,0,maxlen)
      line2 = strmid(line,maxlen,100)
      sxaddhist,leadstr+line1,head
      sxaddhist,leadstr+line2,head
    endif else sxaddhist,leadstr+line,head
  endif
  # Persistence model file
  if n_elements(persistmodelcorr) gt 0 then begin
    line = 'Persistence model file="'+persistmodelcorr+'"'
    if strlen(line) gt maxlen then begin
      line1 = strmid(line,0,maxlen)
      line2 = strmid(line,maxlen,100)
      sxaddhist,leadstr+line1,head
      sxaddhist,leadstr+line2,head
    endif else sxaddhist,leadstr+line,head
  endif
  # History file
  if n_elements(histcorr) gt 0 then begin
    line = 'Exposure history file="'+histcorr+'"'
    if strlen(line) gt maxlen then begin
      line1 = strmid(line,0,maxlen)
      line2 = strmid(line,maxlen,100)
      sxaddhist,leadstr+line1,head
      sxaddhist,leadstr+line2,head
    endif else sxaddhist,leadstr+line,head
  endif
  # Bad pixels 
  bpmmask = long((long(mask) AND maskval('BADPIX')) eq maskval('BADPIX'))
  totbpm = total(bpmmask)
  sxaddhist,leadstr+strtrim(long(totbpm),2)+' pixels are bad',head
  # Cosmic Rays
  crmask = where(long(mask) AND maskval('CRPIX'),totcr)
  if nreads gt 2 then sxaddhist,leadstr+strtrim(long(totcr),2)+' pixels have cosmic rays',head
  if keyword_set(crfix) and nreads gt 2 then sxaddhist,leadstr+'Cosmic Rays FIXED',head
  # Saturated pixels
  satmask = where(long(mask) AND maskval('SATPIX'),totsat)
  unfmask = where(long(mask) AND maskval('UNFIXABLE'),totunf)
  totfix = totsat-totunf
  sxaddhist,leadstr+strtrim(long(totsat),2)+' pixels are saturated',head
  if keyword_set(satfix) and nreads gt 2 then sxaddhist,leadstr+strtrim(long(totfix),2)+' saturated pixels FIXED',head
  # Unfixable pixels
  sxaddhist,leadstr+strtrim(long(totunf),2)+' pixels are unfixable',head
  # Sampling
  if keyword_set(uptheramp) then sxaddhist,leadstr+'UP-THE-RAMP Sampling',head else $
    sxaddhist,leadstr+'Fowler Sampling, Nfowler='+strtrim(long(Nfowler_used),2),head 
  # Persistence correction factor
  if n_elements(pmodelim) gt 0 and n_elements(ppar) gt 0 then begin
    sxaddhist,leadstr+'Persistence correction: '+strjoin(strtrim(string(ppar,format='(G7.3)'),2),' '),head
  endif
  
  
  # Fix EXPTIME if necessary
  if sxpar(head,'NFRAMES') ne nreads then begin
    # NFRAMES is from ICC, NREAD is from bundler which should be correct
    exptime = nreads*10.647  # secs
    sxaddpar,head,'EXPTIME',exptime
    print,'not halting, but NFRAMES does not match NREADS, NFRAMES: ', sxpar(head,'NFRAMES'), ' NREADS: ',string(format='(i8)',nreads),'  ', seq
    #print,'halt: NFRAMES does not match NREADS, NFRAMES: ', sxpar(head,'NFRAMES'), ' NREADS: ',string(format='(i8)',nreads),'  ', seq
    #stop
  endif

  # Add UT-MID/JD-MID to the header
  jd = date2jd(sxpar(head,'DATE-OBS'))
  exptime = sxpar(head,'EXPTIME')
  jdmid = jd + (0.5*exptime)/24./3600.d0
  utmid = jd2date(jdmid)
  sxaddpar,head,'UT-MID',utmid,' Date at midpoint of exposure'
  sxaddpar,head,'JD-MID',jdmid,' JD at midpoint of exposure'

  # remove CHECKSUM
  sxdelpar,head,'CHECKSUM'
  
  #----------------------------------
  # Output the final image and mask
  #----------------------------------
  if n_elements(outfile) gt 0 then begin
    ioutfile = outfile[f]
  
    # Does the output directory exist?
    if file_test(file_dirname(ioutfile),/directory) eq 0 then begin
      print,'Creating ',file_dirname(ioutfile)
      FILE_MKDIR,file_dirname(ioutfile)
    endif
  
    # Test if the output file already exists
    test = file_test(ioutfile)
  
    if not keyword_set(silent) then print,''
    if test eq 1 and keyword_set(clobber) then print,'OUTFILE = ',ioutfile,' ALREADY EXISTS.  OVERWRITING'
    if test eq 1 and not keyword_set(clobber) then print,'OUTFILE = ',ioutfile,' ALREADY EXISTS. '
    
    # Writing file
    if test eq 0 or keyword_set(clobber) then begin
      if not keyword_set(silent) then print,'Writing output to: ',ioutfile
      if keyword_set(outlong) then print,'Saving FLUX/ERR as LONG instead of FLOAT'
      # HDU0 - header only
      FITS_WRITE,ioutfile,0,head,/no_abort,message=write_error    
      # HDU1 - flux
      flux = reform(output[*,*,0])
      # replace NaNs with zeros
      bad=where(finite(flux) eq 0,nbad) 
      if nbad gt 0 then flux[bad]=0.
      if keyword_set(outlong) then flux=round(flux)
      MKHDR,head1,flux,/image
      sxaddpar,head1,'CTYPE1','Pixel'
      sxaddpar,head1,'CTYPE2','Pixel'
      sxaddpar,head1,'BUNIT','Flux (ADU)'
      MWRFITS,flux,ioutfile,head1,/silent

      # HDU2 - error
      #err = sqrt(reform(output[*,*,1])) > 1  # must be greater than zero
      err = errout(reform(output[*,*,1]))
      if keyword_set(outlong) then err=round(err)
      MKHDR,head2,err,/image
      sxaddpar,head2,'CTYPE1','Pixel'
      sxaddpar,head2,'CTYPE2','Pixel'
      sxaddpar,head2,'BUNIT','Error (ADU)'
      MWRFITS,err,ioutfile,head2,/silent


      # HDU3 - mask
      #flagmask = fix(reform(output[*,*,2]))
      # don't go through conversion to float and back!
      flagmask = fix(mask)
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
      #if write_error ne '' then print,'Error writing file '+write_error

      # HDU4 - persistence model
      if n_elements(pmodelim) gt 0 then begin
        MKHDR,head4,pmodelim,/image
        sxaddpar,head4,'CTYPE1','Pixel'
        sxaddpar,head4,'CTYPE2','Pixel'
        sxaddpar,head4,'BUNIT','Persistence correction (ADU)'
        MWRFITS,pmodelim,ioutfile,head4,/silent
      endif
        
    endif
  
  endif
  
  
  # Remove the recently Decompressed file
  if extension eq 'apz' and keyword_set(cleanuprawfile) and doapunzip eq 1 then begin
    print,'Deleting recently decompressed file ',file
    FILE_DELETE,file,/allow,/quiet
  end
  
  
  # Number of saturated and CR pixels
  if not keyword_set(silent) then begin
    print,''
    print,'BAD/CR/Saturated Pixels:'
    print,strtrim(long(totbpm),2),' pixels are bad'
    print,strtrim(long(totcr),2),' pixels have cosmic rays'
    print,strtrim(long(totsat),2),' pixels are saturated'
    print,strtrim(long(totunf),2),' pixels are unfixable'
    print,''
  endif

  file_delete,lockfile
  
  dt = systime(1)-t0
  if not keyword_set(silent) then print,'dt = ',strtrim(string(dt,format='(F10.1)'),2),' sec'
  if keyword_set(logfile) then writelog,logfile,$
    file_basename(file)+string(format='(f10.2,1x,i8,1x,i8,1x,i8,i8)',dt,totbpm,totcr,totsat,totunf)

  BOMB:

ENDFOR  # file loop

if nfiles gt 1 then begin
  dt = systime(1)-t00
  if not keyword_set(silent) then print,'dt = ',strtrim(string(dt,format='(F10.1)'),2),' sec'
endif
