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
#from . import config  # get loaded config values
#from . import __version__
from dlnpyutils import utils as dln
import subprocess
import traceback
import tempfile
import shutil

from . import yanny


def zip(files,delete=True,verbose=True):
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
        os.close(tid)
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

        # Calculate average dCounts
        avg_dcounts = np.round( tot_dcounts/(nreads-1) ).astype(np.int32)  # must be an integer
        # shoud this be int or uint16??

        # Initialize the final (pre-compressed) file
        #--------------------------------------------
        tid2,tfile2 = tempfile.mkstemp(prefix="apzip",dir=tempdir)
        os.close(tid2)
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
        os.close(tid2)
        outfile_precmp = tfile2
        shutil.copyfile(files, outfile_precmp)
        
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

    # Delete original file
    if delete:
        if verbose: print('Deleting Original file ',files)
        os.remove(files)

    # Time elapsed
    dt = time.time()-t0
    if verbose: print('dt = ',dt,' sec')


def unzip(input,clobber=False,delete=False,silent=False,no_checksum=True,fitsdir=None,
          nohalt=True,unlock=False):
    """
    This program uncompresses the raw APOGEE files
    that were compressed with APZIP

    This program is specificially designed to compress
    ONLY raw APOGEE data.  It assumes that the data is
    in this format:
     HDU0: header but NO data
     HDU1: header, read1 image  as UNSIGNED INTEGERS (BITPIX=16 or UINT)
     HDU2: header, read2 image  as UNSIGNED INTEGERS (BITPIX=16 or UINT)
     and so on for all the reads.
    The uncompression process returns the data to this exact format.

    Parameters
    ----------
    input : str or list
       A list of input compressed raw bundled APOGEE fits files
         with endings of .apz.
    clobber : boolean, optional
       If output file exists then overwrite it.  Default is False.
    delete : boolean, optional
       Delete compressed file after successfully uncompressing
    silent : boolean, optional
       Don't print anything to the screen.  Default is False.
    no_checksum : boolean, optional
       If specified, will skip the checksum validation
    fitsdir : str, optional
       The output directory.
    unlock : boolean, optional
       Delete any lock file and start fresh.  Default is False.

    Returns
    -------
    The files are uncompressed and have filenames with
    extensions of ".apz".

    Example
    -------

    unzip('apR-a-00000085.apz')

    By D.Nidever  August 2010
    S.Beland  Aug 2011 - Added the checksum
    Translated to python by D.Nidever, Jan 2022
    """

    t0 = time.time()

    # Get the inputs
    files = dln.loadinput(input)
    nfiles = len(files)

    # More than one file input
    if nfiles > 1:
        for i in range(nfiles):
            unzip(files[i],clobber=clobber,delete=delete,silent=silent,error=error,no_checksum=no_checksum)
        return
    if type(files) is list:
        files = files[0]

    # Does file exist
    if os.path.exists(files)==False:
        error = files+' NOT FOUND'
        if silent==False:
            print(error)
        return

    # Check that "funpack" is available
    try:
        out = subprocess.check_output(['which','funpack'],shell=False)
        out = out.decode().split('\n')
    except:
        raise ValueError('FUNPACK not found')

    # Check that the extension is ".apz"
    fdir = os.path.dirname(files)+'/'
    fil = os.path.basename(files)
    dum = fil.split('.')
    ext = dum[len(dum)-1]
    if ext != 'apz':
        error = 'Extension must be .apz'
        if silent==False:
            print(error)
        return
    base = os.path.basename(files)[0:-4]

    # Temporary directory
    #  use /tmp/ if possible otherwise the directory that the file is in
    #tempdir = '/tmp/'
    #if FILE_TEST(tempdir,/directory) eq 0 then
    if fitsdir is not None:
        tempdir = fitsdir
    else:
        tempdir = os.path.dirname(files)

    # Getting file info
    filesize = os.path.getsize(files)
    if silent==False:
        print('Uncompressing >>'+files+'<< (%.2f MB)' % (filesize/1e6))

    # Final output filename
    if fitsdir is not None:
        finalfile = os.path.join(fitsdir,base+'.fits')
    else:
        finalfile = os.path.join(fdir,base+'.fits')

    # if another process is working already on this file, wait until done,
    #    then return
    lockfile = finalfile+'.lock'
    if not unlock and not clobber:
        while os.path.exists(lockfile):
            print('Waiting for lockfile '+lockfile)
            time.sleep(10)    
    else: 
        if os.path.exists(lockfile): 
            os.remove(lockfile)

    if os.path.exists(os.path.dirname(lockfile))==False:
        os.makedirs(os.path.dirname(lockfile),exist_ok=True)
    open(lockfile,'w').close()

    if os.path.exists(finalfile) and clobber==False:
        if silent==False:
            print('Overwriting ',finalfile)
        if os.path.exists(finalfile): os.remove(finalfile)
    if os.path.exists(finalfile) and clobber==False:
        if silent==False:
            print(finalfile,' exists already.  Writing compressed file to ',finalfile+'.1')
        finalfile = finalfile+'.1'


    # Uncompress the input file to a temporary file
    # get a unique filename (and delete the created empty file)
    tid2, outfile_uncmp = tempfile.mkstemp(prefix="apzip", dir=tempdir)
    os.close(tid2)
    if os.path.exists(outfile_uncmp): os.remove(outfile_uncmp)
    

    # Step I: Uncompress the file with funpack
    #-------------------------------------------
    if silent==False:
        print('Step I: Uncompress with funpack')

    try:
        # -C suppresses checksum update
        subprocess.run(["funpack", "-O", outfile_uncmp, "-C", files],
                       shell=False, check=True)
    except (OSError, subprocess.CalledProcessError):
        if not silent:
            print("funpack failed")
        raise

    # Now read the fits file
    with fits.open(outfile_uncmp, memmap=False, uint=True) as packed:
        nreads = len(packed) - 1

        if not silent:
            print(f"        Nreads = {nreads}")

        if nreads >= 1:
            if not silent:
                print("Step II: Reconstructing the original reads")

            # average dCounts image
            avg_dcounts = np.array(packed[0].data, copy=True)
            head0 = packed[0].header.copy()

            # Load read=1 (first one)
            read1 = np.array(packed[1].data, copy=True)
            head1 = packed[1].header.copy()

            if avg_dcounts.shape != read1.shape:
                raise ValueError(
                    "AVERAGE DCOUNTS and READ1 dimensions do not match: "
                    f"{avg_dcounts.shape} != {read1.shape}"
                )

            output_hdus = fits.HDUList()

            # Prepare the output primary header
            head0["SIMPLE"] = True
            head0["BITPIX"] = 16
            head0["NAXIS"] = 0
            for key in ("NAXIS1", "NAXIS2", "PCOUNT", "GCOUNT",
                        "CHECKSUM", "DATASUM", "BZERO", "BSCALE"):
                head0.remove(key, ignore_missing=True)
            output_hdus.append(fits.PrimaryHDU(header=head0))

            # First reconstructed read
            head1["XTENSION"] = "IMAGE"
            head1["NAXIS"] = 2
            head1["NAXIS1"] = read1.shape[1]
            head1["NAXIS2"] = read1.shape[0]
            head1["PCOUNT"] = 0
            head1["GCOUNT"] = 1
            for key in ("SIMPLE", "CHECKSUM", "DATASUM"):
                head1.remove(key, ignore_missing=True)
            output_hdus.append(fits.ImageHDU(read1, head1))

            # Loop through extensions and add them together
            last_image = read1
            for index in range(2, nreads + 1):
                resid = np.array(packed[index].data, dtype=np.int32, copy=True)
                header = packed[index].header.copy()

                if resid.shape != read1.shape:
                    raise ValueError(
                        f"READ1 and residual {index - 1} dimensions do not "
                        f"match: {read1.shape} != {resid.shape}"
                    )

                # Re-construct the original counts
                #----------------------------------
                #  This is how the dcounts/resid were created:
                #    dcounts[i] = read[i+1]-read[i]
                #    resid[i] = dcounts[i] - avg_dcounts
                #  So, adding avg_dcounts to resid gives back dcounts
                #  and you just keep adding dCounts to the last read to
                #  reconstruct all of the reads.
                
                original = (last_image.astype(np.int32) + resid +
                            avg_dcounts ).astype(np.uint32)

                header["BITPIX"] = 16
                header["XTENSION"] = "IMAGE"
                header["NAXIS"] = 2
                header["NAXIS1"] = original.shape[1]
                header["NAXIS2"] = original.shape[0]
                header["PCOUNT"] = 0
                header["GCOUNT"] = 1
                header["BZERO"] = 32768
                header["BSCALE"] = 1
                for key in ("SIMPLE", "CHECKSUM", "DATASUM"):
                    header.remove(key, ignore_missing=True)

                output_hdus.append(fits.ImageHDU(original, header))
                last_image = original

            output_hdus.writeto(finalfile, overwrite=True, checksum=True)
            output_hdus.close()

        # No data to uncompress, Nreads=0
        else:
            # just copy the file
            shutil.copyfile(outfile_uncmp, finalfile)

    # Delete temporary file
    if os.path.exists(outfile_uncmp):
        os.remove(outfile_uncmp)

    if silent==False:
        print('Writing to '+finalfile)

    # Delete original file
    if delete:
        if silent==False:
            print('Deleting Original file ',files)
        if os.path.exists(files): os.remove(files)

    # Remove lock file
    if os.path.exists(finalfile+'.lock'): os.remove(finalfile+'.lock')

    # Time elapsed
    dt = time.time()-t0
    if silent==False:
        print('dt = %.1f sec' % dt)
