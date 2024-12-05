import os
import numpy as np

def mkpersist(persistid, dark, flat, cmjd=None, darkid=None, flatid=None,
              sparseid=None, fiberid=None, clobber=False, thresh=0.1, unlock=False):
    """
    Procedure to make an APOGEE persistence calibration file from
    a dark and flat exposure.

    Parameters
    ----------
    persistid : int
       The frame name for the output apPersist file.
    dark : int
       The dark frame to use to derive the persistence.
    flat : int
       The flat frame to use to derive the persistence.
    cmjd : int
       The MJD directory to put the ap2D/ap1D files in.
    darkid : int
       Dark frame to be used if images are reduced.
    flatid : int
       Flat frame to be used if images are reduced.
    sparseid : int
       Sparse frame to be used if images are reduced.
    fiberid
       ETrace frame to be used if images are reduced.
    thresh : 
       Threshold to use for persistence.  Default is 0.1.
    clobber : bool, optional
       Overwrite existing files.  Default is False.
    unlock : bool, optional
       Delete lockfile and start fresh.

    Returns
    -------
    A set of apPersist-[abc]-ID8.fits files in the appropriate location
    determined by the SDSS/APOGEE tree directory structure.

    Example
    -------

    mkpersist(persist,darkid,flatid,thresh=thresh,cmjd=cmjd,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid,clobber=True)

    By J. Holtzman, 2011
    Added doc strings, updates to use data model  D. Nidever, Sep 2020 
    Translated to Python  D. Nidever  2023/2024
    """

    load = apload.ApLoad(apred=apred,telescope=telescope)
    perdir = os.path.dirname(load.filename('Persist', num=persistid, chips=True))
    perfile = load.filename('Persist', num=persistid, chips=True)

    # If another process is alreadying making this file, wait!
    lock.lock(perfile, waittime=10, unlock=unlock)

    # Does product already exist?
    # check all three chip files
    chips = ['a', 'b', 'c']
    allfiles = [os.path.join(perdir, load.prefix+'Persist-{:s}-{:08d}.fits'.format(c,persistid) for c in chips]

    if np.sum([os.path.exists(fil) for fil in allfiles]) == 3 and clobber==False:
        print('persist file:',perfile, 'already made')
        return

    # Delete any existing files to start fresh
    for fil in allfiles:
        if os.path.exists(fil): os.remove(fil)

    # Open .lock file
    lock.lock(perfile, lock=True)

    if cmjd is not None:
        d = approcess([dark, flat], cmjd=cmjd, darkid=darkid, flatid=flatid, psfid=psfid, nfs=1,
                      doap3dproc=True, unlock=unlock)
    else:
        d = approcess([dark, flat], darkid=darkid, flatid=flatid, psfid=psfid, nfs=1,
                      doap3dproc=True, unlock=unlock)

    d = apread('2D', num=dark)
    f = apread('2D', num=flat)

    # Write out an integer mask
    for ichip in range(3):
        persist = np.zeros((2048, 2048), dtype=int)
        r = d[ichip]['flux'] / f[ichip]['flux']
        bad = np.where((d[ichip]['mask'] & badmask()) | (f[ichip]['mask'] & badmask()))
        r[bad] = 0.0
        rz = zap(r, [10, 10])
        print(np.median(rz))
        bad, = np.where(rz > thresh / 4.0)
        persist[bad] = 4
        bad, = np.where(rz > thresh / 2.0)
        persist[bad] = 2
        bad, = np.where(rz > thresh)
        persist[bad] = 1

	leadstr = 'MKPERSIST: '
        head = fits.Header()
	head['HISTORY'] = leadstr+time.asctime()
	import socket
	head['HISTORY'] = leadstr+getpass.getuser()+' on '+socket.gethostname()
        import platform
        head['HISTORY'] = leadstr+'Python '+pyvers+' '+platform.system()+' '+platform.release()+' '+platform.architecture()\
[0]
        # add reduction pipeline version to the header         
        head['HISTORY'] = leadstr+' APOGEE Reduction Pipeline Version: '+load.apred
        outfile = load.filename('Persist', num=persistid, chips=True)
        outfile = outfile.replace('Persist-','Persist-{:s}'.format(chips[ichip]))
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(header=head))
	hdulist.append(fits.ImageHDU(persist))
        hdulist[1].header['EXTNAME'] = 'PERSIST'
	hdulist.append(fits.ImageHDU(rz))
        hdulist[2].header['EXTNAME'] = 'PERSIST_RATE'
        hdulist.writeto(outfile,overwrite=True)

    lock.lock(perfile, clear=True)
