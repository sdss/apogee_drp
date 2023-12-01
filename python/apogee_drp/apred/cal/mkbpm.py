import os
import numpy as np
from . import plan
from .utils import lock,apload

def mkbpm(bpmid, apred='daily', telescope='apo25m', darkid=None, flatid=None, badrow=None,
          clobber=False, unlock=False):
    """
    Create an APOGEE bad pixel mask calibration file.

    Parameters
    ----------
    bpmid : int
       ID8 number for this bad pixel mask.
    apred : str, optional
       APOGEE reduction version.  Default is 'daily'.
    telescope : str, optional
       Telescope name, 'apo25m' or 'lco25m'.  Default is 'apo25m'.
    darkid : int, optional
       ID8 number for the dark to use.
    flatid : int, optional
       ID8 number for the flat to use.
    badrow : list, optional
       List/array of known bad rows.
    clobber : bool, optional
       Overwrite existing files.  Default is False.
    unlock : bool, optional
       Delete lock file and start fresh.   Default is False.

    Returns
    -------
    BPM files (e.g. apBPM-a-ID8.fits) are created in the appropriate
    directory using the SDSS/APOGEE tree structure.

    Example
    -------

    mkbpm(12345678,'daily','apo25m',darkid=00110012,flatid=00110013)

    By J. Holtzman, 2011?
    Added doc strings and cleanup.  D. Nidever, Sep 2020
    Translated to python. D. Nidever, Nov 2023
    """

    load = apload.ApLoad(apred=apred,telescope=telescope)
    filename = load.filename('BPM',num=bpmid, chips=True)
    lockfile = filename.replace('BPM-','BPM-c-')
    
    #dirs = getdir()
    #file = apogee_filename('BPM', num=bpmid, chip='c')
    #lockfile = file + '.lock'

    # If another process is already making this file, wait!
    lock.lock(lockfile, waittime=10, unlock=unlock)

    # Does the product already exist?
    # Check all three chip files
    sbpmid = '{:08d}'.format(bpmid)
    chips = ['a', 'b', 'c']
    bpmdir = os.path.dirname(load.filename('BPM', num=bpmid, chips=True))
    allfiles = [bpmdir+load.prefix + 'BPM-{:s}-{:s}.fits'.format(chip,sbpmid) for chip in chips]
    if all([os.path.exists(f) for f in allfiles]) and not clobber:
        print('BPM file:', filename, 'already made')
        return
    # Delete any existing files to start fresh
    for f in allfiles:
        if os.path.exists(f): os.remove(f)

    print('Making BPM:', bpmid)

    # Open .lock file
    lock.lock(lockfile, lock=True)

    for ichip in range(3):
        chip = chips[ichip]
        mask = np.zeros((2048, 2048), dtype=int)

        # Bad pixels from dark frame
        outfile = load.filename("Dark", num=darkid, chips=True).replace('Dark-','Dark-'+chip+'-')
        
        darkmask = mrdfits(file, 3)
        bad = np.where(darkmask > 0)
        mask[bad] = mask[bad] | maskval('BADDARK')

        # Bad pixels from flat frame
        if flatid is not None:
            file = load.filename("Flat", chip=chip, num=flatid)
            flatmask = mrdfits(file, 3)
            bad = np.where(flatmask > 0)
            mask[bad] = mask[bad] | maskval('BADFLAT')
        else:
            flatmask = darkmask * 0

        # Flag them both as bad pixel (for historical compatibility?)
        bad = np.where((darkmask | flatmask) > 0)
        mask[bad] = mask[bad] | maskval('BADPIX')

        if badrow is not None:
            for row in badrow:
                if row.chip == ichip:
                    mask[:, row.row] = mask[:, row.row] | maskval('BADPIX')

        outfile = load.filename('BPM', num=bpmidf, chips=True)
        outfile = outfile.replace('BPM-','BPM-'+chip+'-')
        hdu = fits.HDUList()
        hdu.append(fits.PrimaryHDU(mask))
        hdu[0].header['EXTNAME'] = 'BPM'
        hdu[0].header['V_APRED'] = plan.getgitvers(),'APOGEE software version' 
        hdu[0].header['APRED'] = load.apred,'APOGEE Reduction version' 
        hdu.writeto(outfile,overwrite=True)
        
    lock.unlock(lockfile, clear=True)
