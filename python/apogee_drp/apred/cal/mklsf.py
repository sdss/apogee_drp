import os
import subprocess
from astropy.io import fits
import numpy as np

def mklsf(lsfid, waveid, darkid=None, flatid=None, psfid=None, fiberid=None, clobber=False,
          full=False, newwave=False, pl=False, fibers=None, nowait=False, unlock=False):
    """
    Procedure to make an APOGEE LSF calibration file.  This is a wrapper
    around APLSF.PRO but this ensures that the necessary calibration
    files and basic processing steps have already been performed.

    Parameters
    ----------
    lsfid       The ID8 number of the exposure to use.
    waveid      ID8 number of the wave calibration to use. 
    =darkid     ID8 number of the dark calibration to use.
    =flatid     ID8 number of the flat calibration to use.
    =psfid      ID8 number of the psf calibration to use.
    =fiberid    ID8 number for the ETrace calibration file to use. 
    =fibers     An array if fibers to fit.  By default all 300 are fit.
    /clobber    Overwrite any existing files.
    /full       Perform full Gauss-Hermite fitting, otherwise just
                Gaussian fitting is performed by default.
    /pl         Make plots.
    /nowait     If LSF file is already being made then don't wait
                just return.
    /newwave    Depricated parameter.
    /unlock     Delete the lock file and start fresh.

    Returns
    -------
    A set of apLSF-[abc]-ID8.fits files in the appropriate location
    determined by the SDSS/APOGEE tree directory structure.

    Example
    -------

    mklsf,ims,waveid,darkid=darkid,flatid=flatid,psfid=psfid,fiberid=fiberid,/full,/clobber,/pl

    By J. Holtzman, 2011
    Added doc strings, updates to use data model  D. Nidever, Sep 2020 
    """
    
    
    if not newwave:
        newwave = False

    dirs = getdir(apodir, caldir, spectrodir, vers)
    caldir = dirs.caldir
    file = apogee_filename('LSF', num=lsfid[0], nochip=True)
    lsffile = os.path.dirname(file) + '/' + os.path.splitext(os.path.basename(file))[0]

    # If another process is alreadying make this file, wait!
    aplock(lsffile, waittime=10, unlock=unlock)

    # Does product already exist?
    # check all three chip files and .sav file exist
    slsfid = str(lsfid[0]).zfill(8)
    chips = ['a', 'b', 'c']
    lsfdir = apogee_filename('LSF', num=lsfid[0], chip='c', dir=True)
    allfiles = [lsfdir + dirs.prefix + 'LSF-' + chip + '-' + slsfid + '.fits' for chip in chips]
    allfiles += [lsfdir + dirs.prefix + 'LSF-' + slsfid + '.sav']
    if np.sum([os.path.exists(file) for file in allfiles]) == 4 and not clobber:
        print('LSF file:', file + '.sav', 'already made')
        return

    # Delete any existing files to start fresh
    for file in allfiles:
        if os.path.exists(file):
            os.remove(file)

    # Open .lock file
    aplock(lsffile, lock=True)

    cmjd = getcmjd(psfid)

    lsffile = apogee_filename('1D', num=lsfid[0], chip='c')
    clobber = 0
    mkpsf(psfid, darkid=darkid, flatid=flatid, fiberid=fiberid, clobber=clobber, unlock=True)

    w = approcess(lsfid, dark=darkid, flat=flatid, psf=psfid, flux=0,
                  doproc=True, skywave=True, clobber=clobber)

    cmd = ['apskywavecal', 'dummy', '--frameid', str(lsfid), '--waveid', str(waveid),
           '--apred', dirs.apred, '--telescope', dirs.telescope]
    subprocess.Popen(cmd)

    lsffile = os.path.dirname(lsffile) + '/' + str(lsfid[0]).zfill(8)
    if isinstance(waveid, str) and len(waveid) == 7:
        wavefile = caldir + 'wave/' + waveid
    else:
        wavefile = caldir + 'wave/' + str(waveid[0]).zfill(8)

    psffile = caldir + '/psf/' + str(psfid[0]).zfill(8)
    aplsf(lsffile, wavefile, psf=psffile, gauss=not full, pl=pl)
    if full:
        aplsf(lsffile, wavefile, psf=psffile, clobber=True, pl=pl, fibers=fibers)

    aplock(lsffile, clear=True)
