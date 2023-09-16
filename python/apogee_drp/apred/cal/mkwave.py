import os
import subprocess
import numpy as np
from astropy.io import fits
from scipy.signal import medfilt2d

def mkwave(waveid, name=None, darkid=None, flatid=None, psfid=None,
           modelpsf=None, fiberid=None, clobber=False, nowait=False,
           nofit=False, unlock=False, plot=False):
    """
    Procedure to make an APOGEE wavelength calibration file from
    arc lamp exposures.  This is a wrapper around the python
    apmultiwavecal program.

    Parameters
    ----------
    waveid      The ID8 numbers of the arc lamp exposures to use.
    =name       Output filename base.  By default waveid[0] is used.
    =darkid     Dark frame to be used if images are reduced.
    =flatid     Flat frame to be used if images are reduced.
    =psfid      PSF frame to be used if images are reduced.
    =modelpsf   Model PSF calibration frame to use.
    =fiberid    ETrace frame to be used if images are reduced.
    /nowait     If file is already being made then don't wait
                just return.
    /clobber    Overwrite existing files.
    /nofit      Skip fit (find lines only).
    /unlock     Delete the lock file and start fresh.

    Returns
    -------
    A set of apWave-[abc]-ID8.fits files in the appropriate location
    determined by the SDSS/APOGEE tree directory structure.

    Example
    -------

    mkwave,ims,name=name,darkid=darkid,flatid=flatid,psfid=psfid,fiberid=fiberid,/clobber

    By J. Holtzman, 2011
      Added doc strings, updates to use data model  D. Nidever, Sep 2020 
    """


    if name is None:
        name = str(waveid[0])

    dirs = getdir(apodir, caldir, spectrodir, vers)
    wavedir = apogee_filename('Wave', num=name, chip='a', dir=True)
    file = os.path.join(dirs.prefix, f"Wave-{name:08d}")
    wavefile = os.path.join(wavedir, file)

    # If another process is alreadying make this file, wait!
    aplock(wavefile, waittime=10, unlock=unlock)

    # Does product already exist?
    # check all three chips and .dat file
    chips = ['a', 'b', 'c']
    swaveid = f"{waveid[0]:08d}"
    allfiles = [os.path.join(wavedir, dirs.prefix, f"Wave-{chip}-{swaveid}.fits") for chip in chips]
    if all(np.array([os.path.exists(file) for file in allfiles])) and not clobber:
        print(f"Wavecal file: {os.path.join(wavedir, file)}.dat already exists")
        return

    # Delete existing files to start fresh
    for file in allfiles:
        if os.path.exists(file):
            os.remove(file)

    print('Making wave:', waveid)

    # Process the frame if necessary
    chipfiles = apogee_filename('1D', num=waveid, chip=['a', 'b', 'c'])
    if not all([os.path.exists(file) for file in chipfiles]):
        if psfid is not None:
            cmjd = getcmjd(psfid)
            mkpsf(psfid, darkid=darkid, flatid=flatid, fiberid=fiberid, unlock=unlock)
        w = approcess(waveid, dark=darkid, flat=flatid, psf=psfid, modelpsf=modelpsf, flux=0,
                      doproc=True, unlock=unlock)

    # Check that the data is okay
    chfile = apogee_filename('2D', num=waveid, chip='b')
    if not os.path.exists(chfile):
        print(f"{chfile} NOT FOUND")
        aplock(wavefile, clear=True)
        return

    head0 = fits.getheader(chfile, ext=0)
    im1, head1 = fits.getdata(chfile, header=True)

    # UNE, bright line at X=1452
    if 'LAMPUNE' in head0:
        sub = im1[1452 - 100:1452 + 100, :]
        thresh = 40
    # THARNE, bright line at X=1566 
    elif 'LAMPTHAR' in head0:
        sub = im1[1566 - 100:1566 + 100, :]
        thresh = 1000
    else:
        sub = im1[900:1100, :]
        thresh = 10
    smsub = medfilt2d(sub, kernel_size=7, mode='constant')  # smooth in spectral axis
    resmsub = np.repeat(smsub[:, :2048//8], 8, axis=1)      # rebin in spatial axis
    peakflux = np.max(resmsub, axis=1)                      # peak flux feature in spectral dim.
    avgpeakflux = np.median(peakflux)

    # Check the line flux
    if avgpeakflux / head0['nread'] < thresh:
        print(f"Not enough flux in {chfile}")
        aplock(wavefile, clear=True)
        return

    # Call external Python script using subprocess
    cmd = ['apmultiwavecal', '--name', name.strip(), '--vers', dirs.apred]
    if nofit:
        cmd.append('--nofit')
    if plot:
        cmd.extend(['--plot', '--hard'])
    if clobber:
        cmd.append('--clobber')
    cmd.extend(['--inst', dirs.instrument, '--verbose'])
    cmd.extend(map(str, waveid))
    subprocess.run(cmd)

    # Check if the calibration file was successfully created
    outfile = os.path.join(wavedir, file.replace('apWave-', 'apWave-a-'))
    if os.path.exists(outfile):
        with open(wavedir + file + '.dat', 'w') as lock:
            pass

    aplock(wavefile, clear=True)

