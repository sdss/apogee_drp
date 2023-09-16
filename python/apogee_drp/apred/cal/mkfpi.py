import os
import subprocess


def mkfpi(fpiid, name=None, darkid=None, flatid=None, psfid=None, modelpsf=None,
          fiberid=None, clobber=False, unlock=False, psflibrary=None):
    """
    Procedure to make an APOGEE FPI wavelength calibration file from
    FPI arc lamp exposures.  This is a wrapper around the python
    apmultiwavecal program.

    Parameters
    ----------
    fpiid       The ID8 numbers of the FPI arc lamp exposures to use.
    =name       Output filename base.  By default fpiid is used.
    =darkid     Dark frame to be used if images are reduced.
    =flatid     Flat frame to be used if images are reduced.
    =psfid      PSF frame to be used if images are reduced.
    =modelpsf   Model PSF calibration frame to use.
    =fiberid    ETrace frame to be used if images are reduced.
    /clobber    Overwrite existing files.
    /unlock     Delete the lock file and start fresh.
    /psflibrary   Use PSF library to get PSF cal for images.

    Returns
    -------
    A set of apWaveFPI-[abc]-ID8.fits files in the appropriate location
    determined by the SDSS/APOGEE tree directory structure.

    Example
    -------

    mkfpi,ims,name=name,darkid=darkid,flatid=flatid,psfid=psfid,fiberid=fiberid,/clobber

    By D. Nidever, 2021
    copied from mkwave.pro
    """

    # Common variables
    obs = telescop[:3]

    if name is None:
        name = str(fpiid[0])
    
    dirs = getdir(apodir, caldir, spectrodir, vers)
    wavedir = apogee_filename('Wave', num=name, chip='a', dir=True)
    file = dirs.prefix + f"WaveFPI-{name:08d}"
    fpifile = os.path.join(wavedir, file)

    # If another process is already making this file, wait!
    aplock(fpifile, waittime=10, unlock=unlock)

    # Does the product already exist?
    sfpiid = [str(x).zfill(8) for x in fpiid]
    cmjd = getcmjd(fpiid)
    mjd = int(cmjd)
    chips = ['a', 'b', 'c']
    allfiles = [os.path.join(wavedir, dirs.prefix + f'WaveFPI-{chip}-{cmjd}-{sfpiid}.fits') for chip in chips]
    if all([os.path.exists(file) for file in allfiles]) and not clobber:
        print('Wavecal file:', os.path.join(wavedir, file + '.fits'), 'already made')
        return
    # Delete any existing files to start fresh
    for file in allfiles:
        if os.path.exists(file):
            os.remove(file)

    print('Making fpi:', fpiid)

    # Open .lock file
    aplock(fpifile, lock=True)

    # Make sure we have the PSF cal product
    if psfid is not None:
        MKPSF(psfid, darkid=darkid, flatid=flatid, fiberid=fiberid, unlock=unlock)

    # Get all FPI frames for this night and process them
    query = f"select * from apogee_drp.exposure where mjd={mjd} and observatory='{obs}' and exptype='FPI'"
    expinfo = dbquery(query)
    expinfo.exptype = [x.strip() for x in expinfo.exptype]
    gdfpi = np.where(expinfo.exptype == 'FPI')[0]
    if len(gdfpi) == 0:
        print(f'No FPI exposures for MJD {mjd}')
        aplock(fpifile, clear=True)
        return
    allfpinum = expinfo[gdfpi].num
    print(f'Found {len(allfpinum)} FPI exposures for MJD {mjd}')

    # Process the frames
    for num in allfpinum:
        w = approcess(num, dark=darkid, flat=flatid, psf=psfid, modelpsf=modelpsf, flux=0, doproc=True, unlock=unlock)

    # Make sure the dailywave file is there; it uses modelpsf by default now
    makecal(dailywave=mjd, unlock=unlock, librarypsf=psflibrary)

    # New Python version!
    cmd = ['mkfpi', str(cmjd), dirs.apred, obs, '--num', sfpiid, '--verbose']
    print('Running:', cmd)
    subprocess.run(cmd)

    # Check that the calibration file was successfully created
    outfile = os.path.join(wavedir, file.replace(dirs.prefix + 'WaveFPI-', dirs.prefix + 'WaveFPI-a-'))
    if os.path.exists(outfile):
        with open(outfile + '.dat', 'w') as lock:
            pass

    aplock(fpifile, clear=True)
