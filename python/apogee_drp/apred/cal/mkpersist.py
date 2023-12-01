import os
import numpy as np
from astropy.io import fits
from ..utils import apload,lock,plan

def mkpersist(persistid, apred='daily', telescope='apo25m', dark=None,
              flat=None, mjd=None, darkid=None, flatid=None,
              sparseid=None, fiberid=None, clobber=False, thresh=0.1, unlock=False):
    """
    Procedure to make an APOGEE persistence calibration file from
    a dark and flat exposure.

    Parameters
    ----------
    persistid   The frame name for the output apPersist file.
    dark        The dark frame to use to derive the persistence.
    flat        The flat frame to use to derive the persistence.
    =cmjd       The MJD directory to put the ap2D/ap1D files in.
    =darkid     Dark frame to be used if images are reduced.
    =flatid     Flat frame to be used if images are reduced.
    =sparseid   Sparse frame to be used if images are reduced.
    =fiberid    ETrace frame to be used if images are reduced.
    =thresh     Threshold to use for persistence.  Default is 0.1.
    /clobber    Overwrite existing files.
    /unlock     Delete lockfile and start fresh.

    Returns
    -------
    A set of apPersist-[abc]-ID8.fits files in the appropriate location
    determined by the SDSS/APOGEE tree directory structure.

    Example
    -------

    mkpersist,persist,darkid,flatid,thresh=thresh,cmjd=cmjd,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid,/clobber

    By J. Holtzman, 2011
    Added doc strings, updates to use data model  D. Nidever, Sep 2020 
    """
    
    if thresh is None:
        thresh = 0.1

    chips = ['a','b','c']        
    load = apload.ApLoad(apred=apred,telescope=telescope)
    #dirs = getdir()

    perdir = load.filename('Persist', num=persistid, chip='c', dir=True)
    file = load.filename('Persist', num=persistid, chip='c', base=True)
    lockfile = os.path.join(perdir, file)

    # If another process is alreadying making this file, wait!
    lock.lock(lockfile, waittime=10, unlock=unlock)

    # Does product already exist?
    # check all three chip files
    spersistid = str(persistid).zfill(8)
    chip = ['a', 'b', 'c']
    allfiles = [os.path.join(perdir, dirs.prefix + 'Persist-' + c + '-' + spersistid + '.fits') for c in chip]

    if sum([os.path.exists(file) for file in allfiles]) == 3 and not clobber:
        print('persist file:', os.path.join(perdir, file), 'already made')
        return

    # Delete any existing files to start fresh
    for file in allfiles:
        if os.path.exists(file):
            os.remove(file)

    # Open .lock file
    lock.lock(lockfile, lock=True)

    if mjd is not None:
        d = approcess([dark, flat], cmjd=cmjd, darkid=darkid, flatid=flatid, psfid=psfid, nfs=1,
                      doap3dproc=True, unlock=unlock)
    else:
        d = approcess([dark, flat], darkid=darkid, flatid=flatid, psfid=psfid, nfs=1,
                      doap3dproc=True, unlock=unlock)

    d = apread('2D', num=dark)
    f = apread('2D', num=flat)

    # Write out an integer mask
    for ichip in range(3):
        chip = chips[ichip]
        persist = np.zeros((2048, 2048), dtype=int)
        r = d[ichip].flux / f[ichip].flux
        bad = np.where((d[ichip].mask & badmask()) | (f[ichip].mask & badmask()))
        r[bad] = 0.0
        rz = zap(r, [10, 10])
        print(np.median(rz))
        bad = np.where(rz > thresh / 4.0)
        persist[bad] = 4
        bad = np.where(rz > thresh / 2.0)
        persist[bad] = 2
        bad = np.where(rz > thresh)
        persist[bad] = 1
        outfile = apogee_filename('Persist', num=persistid, chips=True)
        outfile = outfile.replace('Persis-','Persist-'+chip+'-')
        hdu = fits.HDUList()
        hdu.append(fits.PrimaryHDU())
        hdu[0].header['V_APRED'] = plan.getgitvers(),'APOGEE software version' 
        hdu[0].header['APRED'] = load.apred,'APOGEE Reduction version' 
        hdu.append(fits.ImageHDU(persist))
        hdu[1].header['EXTNAME'] = 'PERSIST'
        hdu.append(fits.ImageHDU(rz))
        hdu[1].header['EXTNAME'] = 'PERSRATE'
        hdu.writeto(outfile,overwrite=True)
        #MWRFITS(persist, file, create=True)
        #MWRFITS(rz, file)

    lock.lock(lockfile, clear=True)
