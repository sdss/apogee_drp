import os
import numpy as np

def mkdet(detid, linid=None, unlock=False):
    """
    Make an APOGEE detector calibration product.

    Parameters
    ----------
    detid    ID8 number for the detector file.
    linid    ID8 number for the linearity file.
    /unlock  Delete lock file and start fresh 

    Returns
    -------
    Detector files (e.g. apDetector-a-ID8.fits) are created in the
    appropriate directory using the SDSS/APOGEE tree structure. 

    Example
    -------

    mkdet,detid,linid

    By J. Holtzman, 2011?
    Added doc strings, updates to use data model  D. Nidever, Sep 2020
    """
    
    dirs = getdir()
    caldir = dirs.caldir
    detfile = apogee_filename('Detector', num=detid, chip='c')

    # If another process is already making this file, wait!
    aplock(detfile, waittime=10, unlock=unlock)

    # Does the product already exist?
    print('Testing detector file:', detfile)
    # check all three chip files
    sdetid = str(detid).zfill(8)
    chips = ['a', 'b', 'c']
    detdir = apogee_filename('Detector', num=detid, chip='c', _dir=True)
    allfiles = [os.path.join(detdir, dirs.prefix + f'Detector-{chip}-{sdetid}.fits') for chip in chips]
    if all([os.path.exists(file) for file in allfiles]) and not clobber:
        print('Detector file:', detfile, 'already made')
        return
    # Delete any existing files to start fresh
    for file in allfiles:
        if os.path.exists(file):
            os.remove(file)

    print('Making Detector:', detid)
    # open .lock file
    aplock(detfile,lock=True)
    
    lincorr = np.zeros((4, 3), dtype=float)
    for iquad in range(4):
        lincorr[iquad, :] = [1.0, 0.0, 0.0]

    if linid is not None and linid > 0:
        linpar = mklinearity(linid)
        for iquad in range(4):
            lincorr[iquad, :] = linpar

    if dirs.instrument == 'apogee-n':
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
        file = apogee_filename('Detector', num=detid, chip=chips[ichip])
        # Create FITS headers and write the data to files
        head = mkhdr(0)
        MWRFITS(0, file, head, create=True)
        head1 = MKHDR(rn)
        sxaddpar(head1, 'EXTNAME', 'READNOISE')
        MWRFITS(rn, file, head1)
        head2 = MKHDR(gain)
        sxaddpar(head2, 'EXTNAME', 'GAIN')
        MWRFITS(gain, file, head2)
        head3 = MKHDR(lincorr)
        sxaddpar(head3, 'EXTNAME', 'LINEARITY CORRECTION')
        MWRFITS(lincorr, file, head3)

    # Clear the lock file
    aplock(detfile, clear=True)
