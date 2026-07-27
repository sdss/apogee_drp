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
    waveid : list or array
       The ID8 numbers of the arc lamp exposures to use.
    name : str, optional
       Output filename base.  By default waveid[0] is used.
    darkid : int
       Dark frame to be used if images are reduced.
    flatid : int
       Flat frame to be used if images are reduced.
    psfid : int
       PSF frame to be used if images are reduced.
    modelpsf : int
       Model PSF calibration frame to use.
    fiberid : int
       ETrace frame to be used if images are reduced.
    nowait : bool, optional
       If file is already being made then don't wait just return.
          Default is False.
    clobber : bool, optional
       Overwrite existing files.  Default is False.
    nofit : bool, optional
       Skip fit (find lines only).  Default is False.
    unlock : bool, optional
       Delete the lock file and start fresh.  Default is False.

    Returns
    -------
    A set of apWave-[abc]-ID8.fits files in the appropriate location
    determined by the SDSS/APOGEE tree directory structure.

    Example
    -------

    mkwave(ims,name=name,darkid=darkid,flatid=flatid,psfid=psfid,fiberid=fiberid,clobber=True)

    By J. Holtzman, 2011
    Added doc strings, updates to use data model  D. Nidever, Sep 2020 
    Translated to Python  D. Nidever  2023/2024

    """

    images = np.atleast_1d(waveid)
    if name is None:
        name = str(images[0])

    load = apload.ApLoad(apred=apred,telescope=telescope)
    wavedir = os.path.dirname(load.filename('Wave',num=name, chips=True))
    wavefile = load.filename('Wave',num=name, chips=True)

    # Does product already exist?
    # check all three chips and .dat file
    chips = ['a', 'b', 'c']
    chipfiles = [wavefile.replace('Wave-','Wave-'+c) for c in chips]
    allfiles = chipfiles
    if all(np.array([os.path.exists(fil) for fil in allfiles])) and clobber==False:
        print('Wavecal file:',wavefile,'already exists')
        return

    # If another process is alreadying make this file, wait!
    lock.lock(wavefile, waittime=10, unlock=unlock)
    
    # Delete existing files to start fresh
    for fil in allfiles:
        if os.path.exists(fil): os.remove(fil)

    print('Making wave:', name)

    # Process the frame if necessary
    if not all([os.path.exists(fil) for fil in chipfiles]):
        if psfid is not None:
            cmjd = getcmjd(psfid)
            mkpsf(psfid, darkid=darkid, flatid=flatid, fiberid=fiberid, unlock=unlock)
        w = approcess(waveid, dark=darkid, flat=flatid, psf=psfid, modelpsf=modelpsf, flux=0,
                      doproc=True, unlock=unlock)

    # Check that the data is okay
    chfile = load.filename('2D', num=waveid, chip='b')
    if os.path.exists(chfile)==False:
        print(chfile,'NOT FOUND')
        lock.lock(wavefile, clear=True)
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
        print('Not enough flux in',chfile)
        lock.lock(wavefile, clear=True)
        return

    # Call external Python script using subprocess
    cmd = ['apmultiwavecal', '--name', name.strip(), '--vers', load.apred]
    if nofit:
        cmd += ['--nofit']
    if plot:
        cmd += ['--plot', '--hard']
    if clobber:
        cmd += ['--clobber']
    cmd += ['--inst', load.instrument, '--verbose']
    cmd += [str(value) for value in images]
    res = subprocess.run(cmd,capture_output=True,shell=False)
    stdout = res.stdout.decode()
    stderr = res.stderr.decode()
    if res.returncode != 0:
        print('subprocess failed:')
        print(stdout)
        print(stderr)
        lock.lock(wavefile, clear=True)
        return
    
    # Check if the calibration file was successfully created
    if all(np.array([os.path.exists(fil) for fil in allfiles])):
        open(wavefile.replace('.fits', '.dat'), 'a').close()

    lock.lock(wavefile, clear=True)
