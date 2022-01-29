#!/usr/bin/env python

"""APZIP.PY - APOGEE raw datacube compression software"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
#__version__ = '20180922'  # yyyymmdd

import os
import numpy as np
import warnings
import time
from astropy.io import fits
from astropy.table import Table, Column
from glob import glob
from . import config  # get loaded config values
from . import __version__
import subprocess
import traceback
import tempfile
import shutil

from pydl.pydlutils import yanny


def apzip(files,delete=True,verbose=True):
    """
    This program compresses the raw APOGEE files
    using various techniques.

    If the output compressed file already exists then
    it is automatically overwritten!

    This program is specificially designed to compress
    ONLY raw APOGEE data.  It must be in this EXACT format:
    HDU0: header but NO data
    HDU1: header, read1 image  as UNSIGNED INTEGERS (BITPIX=16 or UINT)
    HDU2: header, read2 image  as UNSIGNED INTEGERS (BITPIX=16 or UINT)
    and so on for all the reads.

    Parameters
    ----------
    files : list
       A list of input raw bundled APOGEE fits files.
    delete : boolean, optional
       Delete the original file after successfully compressing.
    verbose : boolean, optional
       Print information to the screen.  Default is True.
    
    Returns
    -------
    The files are compressed and have filenames with
    extensions of ".apz".

    Example
    -------
    apzip('apR-a-00000085.fits')

    By D.Nidever  August 2010
    Translated by D.Nidever from IDL to python June 2021
    """

    t0 = time.time()

    if isinstance(files,str):
        nfiles = 1
    else:
        nfiles = len(files)

    # More than one file input
    if nfiles>1:
        if verbose: print(str(nfiles),' files input')
        for i in range(nfiles):
            if verbose:
                print(str(i+1),'/',str(nfiles))
                print(' ')
            apzip(files[i],delete=delete,verbose=verbose)
        return

    if verbose:
        print('Compressing >>%s<< (%.2f MB)' % (files,float(os.path.getsize(files))/1e6))

    # Does the file exist
    if os.path.exists(files) is False:
        print(files,' NOT FOUND')
        return

    # Check that "fpack" is available
    #out = subprocess.run(['which','fpack'],shell=False,capture_output=True)  # python 3
    #if out.stdout.strip().decode()=='':
    out = subprocess.check_output(['which','fpack'],shell=False)   # python 2
    if out.strip()=='':
        print('FPACK not found')
        return
    #print('KLUDGE. SKIPPING FPACK CHECK FOR NOW!!!')

    # Check that the extension is ".fits"
    fdir = os.path.dirname(files)+'/'
    fil = os.path.basename(files)
    base,ext = os.path.splitext(fil)
    if ext != '.fits':
        print('Extension must be .fits')
        return

    # Test that we can read the file
    try:
        testhead = fits.getheader(files,0)
    except:
        print('Error reading ',files)
        traceback.print_exc()
        return

    # Temporary directory
    tempdir = fdir


    # Check format and get number of reads
    #--------------------------------------
    # Check primary header, no data allowed
    head0 = fits.getheader(files,0)
    naxis0 = head0['NAXIS']
    if naxis0 != 0:
        print('Primary HDU has data in it.  This is not allowed!')
        return

    # Open input file and get number of reads
    print('Verifying the CHECKSUMs')
    inhdul = fits.open(files,checksum=True)
    #  this checks the CHECKSUm values for all HDUs
    nreads = len(inhdul)-1

    # Checking data format
    for i in np.arange(1,nreads+1):
        bitpix = inhdul[i].header.get('BITPIX')
        if bitpix != 16:
            print('Error: BITPIX='+str(bitpix)+'  READS must be UNSIGNED INTEGERS (BITPIX=16)')
            inhdul.close()
            return

    if verbose: print('Nreads = ',str(nreads))

    # There is data to compress, Nreads>0
    #-------------------------------------
    if nreads>=1:

        # Load first read
        hdu = inhdul[1]
        im = hdu.data.copy()
        shape1 = im.shape
        npix = shape1[0]
        im1 = im.astype(int)  # the first image

        # Step I: Make dCounts temporary file
        #------------------------------------
        if verbose: print('Step I: Making dCounts temporary file')

        tid,tfile = tempfile.mkstemp(prefix="apzip",dir=tempdir)
        dcounts_tempfile = tfile

        # Initialize the dCounts temporary file
        head0 = inhdul[0].header.copy()
        head0['SIMPLE'] = 'T'
        torem = ['CHECKSUM','DATASUM']
        for name in torem:
            head0.remove(name,ignore_missing=True)
        #FITS_ADD_CHECKSUM, head0, /no_timestamp
        #MWRFITS,0,dcounts_tempfile,head0,/create,/no_comment  # exten=0 is blank
        #fits.writeto(dcounts_tempfile,None,head0,checksum=True,overwrite=True)
        temphdul = fits.HDUList()
        temphdul.append(fits.PrimaryHDU(None,head0))
        
        # Loop through the reads
        #  start with 2nd read
        lastim = im1
        tot_dcounts = im1.copy() * 0.0
        for i in np.arange(2,nreads+1):

            # Load the next READ
            hdu = inhdul[i]
            im = hdu.data.copy().astype(int)
            shape = im.shape
            head = hdu.header.copy()
            
            # Check that the image dimension is correct
            if shape != shape1:
                print('Images dimensions of READ1 (in exten=1) and READ'+str(i)+' (in exten='+str(i)+') do NOT MATCH')
                if os.path.exists(dcounts_tempfile): os.remove(dcounts_tempfile)
                os.close(tid)
                inhdul.close()
                return

            # Make dCounts
            dcounts = im - lastim

            # Fix the header
            head['BITPIX'] = 32  # needs to be LONG
            head['BZERO'] = 0
            torem = ['SIMPLE','CHECKSUM','DATASUM']
            # delete SIMPLE if present, only allowed in PDU
            for name in torem:
                head.remove(name,ignore_missing=True)                
            #FITS_ADD_CHECKSUM, head, dcounts, /no_timestamp

            # Write to the temporary dCounts file
            #MWRFITS,dcounts,dcounts_tempfile,head,/silent
            temphdul.append(fits.PrimaryHDU(dcounts,head))
            #fits.append(dcounts_tempfile,dcounts,head,checksum=True)

            # Save last read
            lastim = im

            tot_dcounts += dcounts  # add to the sum of all dCounts

        # Close dcounts_tempfile
        temphdul.writeto(dcounts_tempfile,overwrite=True,checksum=True)
        temphdul.close()
        os.close(tid)

        # Calculate average dCounts
        avg_dcounts = np.round( tot_dcounts/(nreads-1) ).astype(np.int32)  # must be an integer
        # shoud this be int or uint16??

        # Initialize the final (pre-compressed) file
        #--------------------------------------------
        tid2,tfile2 = tempfile.mkstemp(prefix="apzip",dir=tempdir)
        outfile_precmp = tfile2

        # Put Average dCounts in HDU0 with the original header
        head0 = fits.getheader(files,0)
        head0['BITPIX'] = 32   # needs to be LONG
        head0.set('NAXIS',avg_dcounts.ndim,'Dimensionality',after='BITPIX')
        head0.set('NAXIS1',len(avg_dcounts[:,0]),after='NAXIS')
        head0.set('NAXIS2',len(avg_dcounts[0,:]),after='NAXIS1')
        head0.set('BZERO',0,after='NAXIS2')
        head0.set('BSCALE',1,after='BZERO')
        torem = ['CHECKSUM','DATASUM']
        for name in torem:
            head0.remove(name,ignore_missing=True)
        #FITS_ADD_CHECKSUM, head0, avg_dcounts, /no_timestamp
        #MWRFITS,avg_dcounts,outfile_precmp,head0,/create,/no_comment
        outhdul = fits.HDUList()
        outhdul.append(fits.PrimaryHDU(avg_dcounts,head0))
        #fits.writeto(outfile_precmp,avg_dcounts,head0,checksum=True)
         
        # Put first read in exten=1
        read0 = im1.astype(np.uint16)
        head1 = fits.getheader(files,extend=1)
        head1['BITPIX'] = 16  # leave as UINT
        head1['BZERO'] = 32768
        # delete SIMPLE if present, only allowed in PDU        
        torem = ['SIMPLE','CHECKSUM','DATASUM']
        for name in torem:
            head1.remove(name,ignore_missing=True)                            
        #FITS_ADD_CHECKSUM, head1, read0, /no_timestamp
        #MWRFITS,read0,outfile_precmp,head1,/silent
        outhdul.append(fits.PrimaryHDU(read0,head1))
        #fits.append(outfile_precmp,read0,head1,checksum=True)

        # Step II: Load in dCounts and subtract AVG dCounts
        #---------------------------------------------------
        if verbose: print('Step II: Subtracting average dCounts')
        temphdul = fits.open(dcounts_tempfile,cheksum=True)
        for i in np.arange(1,nreads):

            # Load dCounts image (use mrdfits to keep header intact for checksum)
            #dcounts = MRDFITS(dcounts_tempfile,i,head,/silent)
            hdu = temphdul[i]
            #if hdu._checksum_valid != 1:
            #    # the checksum doesn't match -> send a warning and keep going
            #    print('BAD checksum for file (ext='+str(i)+') '+dcounts_tempfile)
            #    return
            dcounts = hdu.data.copy().astype(np.int32)
            
            # Subtract the average dcounts
            resid = dcounts - avg_dcounts

            # Get the header for 2nd read of this pair
            #  read=2 for first dcounts
            head = inhdul[i+1].header.copy()

            # Difference images minus Mean count rate
            head['BITPIX'] = 32  # needs to be LONG
            head['BZERO'] = 0
            # delete SIMPLE if present, only allowed in PDU        
            torem = ['SIMPLE','CHECKSUM','DATASUM']
            for name in torem:
                head.remove(name,ignore_missing=True)                                            
            #FITS_ADD_CHECKSUM, head, resid, /no_timestamp
            #MWRFITS,resid,outfile_precmp,head,/silent
            outhdul.append(fits.PrimaryHDU(resid,head))
            #fits.append(outfile_precmp,resid,head,checksum=True)

        # Write and close the file
        outhdul.writeto(outfile_precmp,overwrite=True,checksum=True)
        outhdul.close()

        # Delete temporary file
        if os.path.exists(dcounts_tempfile): os.remove(dcounts_tempfile)

    # No data to compress, Nreads=0
    #-------------------------------
    else:
        # Making pre-compressed temporary filename
        tid2,tfile2 = tempfile.mkstemp(prefix="apzip",dir=tempdir)
        outfile_precmp = tfile2
        shutil.filecopy(files,outfile_precmp)

    # Close the input file
    inhdul.close()


    # Step III: Compress the file with fpack
    #--------------------------------------
    if verbose: print('Step III: Compressing with fpack')
    if os.path.exists(outfile_precmp+'.fz'): os.remove(outfile_precmp+'.fz')
    try:
        # -C suppresses checksum update
        #out = subprocess.run(['fpack','-C',outfile_precmp],shell=False,capture_output=True)  # python 2
        out = subprocess.check_output(['fpack','-C',outfile_precmp],shell=False)  # python 2
        if out.strip() != '':
            print('fpack error')
            print(out.stdout.strip().decode())
            if os.path.exists(outfile_precmp): os.remove(outfile_precmp)
            return
    except:
        print('Error fpack compressing ',outfile_precmp)
        traceback.print_exc()
        if os.path.exists(outfile_precmp): os.remove(outfile_precmp)
        return

    # Make final output filename
    outdir = os.path.dirname(files)+'/'
    outbase,dum = os.path.splitext(os.path.basename(files))
    finalfile = outdir+outbase+'.apz'

    # Rename the compressed file
    if os.path.exists(finalfile) and verbose:
        print('Overwriting ',finalfile)
    if os.path.exists(finalfile): os.remove(finalfile)
    shutil.move(outfile_precmp+'.fz',finalfile)

    # Final compression
    if verbose:
        insize = os.path.getsize(files)
        outsize = os.path.getsize(finalfile)
        print('Input file size = ',str(insize),' bytes')
        print('Output file size = ',str(outsize),' bytes')
        print('Compression ratio = %.3f' % (float(insize)/outsize))

    # Delete temporary files
    if os.path.exists(outfile_precmp): os.remove(outfile_precmp)  # delete temporary file
    os.close(tid2)

    # Delete original file
    if delete:
        if verbose: print('Deleting Original file ',files)
        os.remove(files)

    # Time elapsed
    dt = time.time()-t0
    if verbose: print('dt = ',dt,' sec')
