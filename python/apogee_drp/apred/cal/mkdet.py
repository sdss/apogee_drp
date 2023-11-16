import os
import numpy as np
from astropy.io import fits
from ..utils import apload,lock

def mkdet(detid,apred='daily',telescope='apo25m',linid=None,
          unlock=False,clobber=False):
    """
    Make an APOGEE detector calibration product.

    Parameters
    ----------
    detid : int
       ID8 number for the detector file.
    apred : str, optional
       APOGEE reduction version.  Default is 'daily'.
    telescope : str, optional
       Telescope name, 'apo25m' or 'lco25m'.  Default is 'apo25m'.
    linid : str, optional
       ID8 number for the linearity file.
    unlock : bool, optional
       Delete lock file and start fresh.  Default is False.
    clobber : bool, optional
       Overwrite any existing files.  Default is False.

    Returns
    -------
    Detector files (e.g. apDetector-a-ID8.fits) are created in the
    appropriate directory using the SDSS/APOGEE tree structure. 

    Example
    -------

    mkdet(detid,linid,apred='daily',telescope='apo25m')

    By J. Holtzman, 2011?
    Added doc strings, updates to use data model  D. Nidever, Sep 2020
    Translated to Python  D. Nidever, Nov 2023
    """
    
    load = apload.ApLoad(apred=apred,telescope=telescope)
    detfile = load.filename('Detector', num=detid, chips=True)

    # If another process is already making this file, wait!
    lock.lock(detfile, waittime=10, unlock=unlock)

    # Does the product already exist?
    print('Testing detector file:', detfile)
    # check all three chip files
    sdetid = '{:08d}'.format(detid)
    chips = ['a', 'b', 'c']
    detdir = os.path.dirname(load.filename('Detector', num=detid, chips=True))
    allfiles = [detdir+load.prefix + 'Detector-{:s}-{:s}.fits'.format(chip,sdetid) for chip in chips]
    if all([os.path.exists(f) for f in allfiles]) and not clobber:
        print('Detector file:', detfile, 'already made')
        return
    # Delete any existing files to start fresh
    for f in allfiles:
        if os.path.exists(f): os.remove(f)

    print('Making Detector:', detid)
    # open .lock file
    lock.lock(detfile,lock=True)
    
    lincorr = np.zeros((4, 3), dtype=float)
    for iquad in range(4):
        lincorr[iquad, :] = [1.0, 0.0, 0.0]

    if linid is not None and linid > 0:
        linpar = mklinearity(linid)
        for iquad in range(4):
            lincorr[iquad, :] = linpar

    if telescope[0:3] == 'apo':
        g = 1.9
        # These are supposed to be CDS DN!
        r = [20.,11.,16.]/np.sqrt(2.)
        # 10/17 analysis get 12, 9, 10 single read in DN
        r = [12, 8, 8]    # Single read in DN (achieved CDS)
        r = [13, 11, 10]  # Equivalent single read in UTR analysis
    else:
        g = 3.0
        # These are supposed to be CDS DN!
        r = [15.,15.,15.]/np.sqrt(2)
        # 10/17, get 6, 8, 4.5 single read in DN
        r = [4, 5, 3]  # Single read in DN (achieved CDS)
        r = [7, 8, 4]  # Equivalent single read in UTR analysis
        # JCW 2/28/17 email
        # Our current measurements are (blue, green, red):
        #
        # gain (e-/ADU): 3.0, 3.3, 2.7
        # read noise (e-): 9.3, 15.2, 8.6
        # dark current (e-/sec): 0.011, 0.014, 0.008

        
    for ichip in range(3):
        gain = [g, g, g, g]
        rn = [r[ichip] * gain[i] for i in range(4)]
        outfile = load.filename('Detector',num=detid,chips=True)
        outfile = outfile.replace('Detector-','Detector-'+chips[ichip]+'-')
        #file = apogee_filename('Detector', num=detid, chip=chips[ichip])
        # Create FITS headers and write the data to files
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU())
        hdulist.append(fits.PrimaryHDU(rn))
        hdulist[1].header['EXTNAME'] = 'READNOISE'
        hdulist.append(fits.ImageHDU(gain))
        hdustli[2].header['EXTNAME'] = 'GAIN'
        hdulist.append(fits.ImageHDU(lincorr))
        hdustli[3].header['EXTNAME'] = 'LINEARITY CORRECTION'
        hdulist.writeto(outfile,overwrite=True)
        
        #head = mkhdr(0)
        #MWRFITS(0, file, head, create=True)
        #head1 = MKHDR(rn)
        #sxaddpar(head1, 'EXTNAME', 'READNOISE')
        #MWRFITS(rn, file, head1)
        #head2 = MKHDR(gain)
        #sxaddpar(head2, 'EXTNAME', 'GAIN')
        #MWRFITS(gain, file, head2)
        #head3 = MKHDR(lincorr)
        #sxaddpar(head3, 'EXTNAME', 'LINEARITY CORRECTION')
        #MWRFITS(lincorr, file, head3)

    # Clear the lock file
    lock.lock(detfile, clear=True)
