import os
import time
import numpy as np
from astropy.io import fits
from ..utils import plan

def mkflux(ims, apred, telescope, cmjd=None, darkid=None, flatid=None, psfid=None, modelpsf=None,
           waveid=None, littrowid=None, persistid=None, clobber=False,
           onedclobber=False, bbtemp=None, plate=0, plugid=None, holtz=False,
           temp=None, unlock=False):
    """
    Makes APOGEE flux calibration file.

    Parameters
    ----------
    ims: list of image numbers to include in flux calibration file.
    cmjd=cmjd : (optional,obsolete) gives MJD directory name if not encoded in file number
    darkid=darkid : dark frame to be used if images are reduced
    flatid=flatid : flat frame to be used if images are reduced
    psfid=psfid : psf frame to be used if images are reduced
    modelpsf=modelpsf : model psf calibration frame to use for extraction
    waveid=waveid : wave frame to be used if images are reduced
    littrowid=littrowid : littrow frame to be used if images are reduced
    persistid=persistid : persist frame to be used if images are reduced
    /clobber : rereduce images even if they exist
    /onedclobber : overwrite the 1D files
    /unlock : delete lock file and start fresh

    Returns
    -------
    A set of apFlux-[abc]-ID8.fits files in the appropriate location
    determined by the SDSS/APOGEE tree directory structure.

    Example
    -------

    mkflux,ims,cmjd=cmjd,darkid=darkid,flatid=flatid,/clobber

    By J. Holtzman, 2011?
      Added doc strings, updates to use data model  D. Nidever, Sep 2020 
    """

    load = apload.ApLoad(apred=apred,telescope=telescope)
    dirs = getdir(apodir, caldir, spectrodir, vers)
    caldir = dirs.caldir

    file = apogee_filename('Flux', num=ims[0], chip='c', base=True)
    fluxdir = apogee_filename('Flux', num=ims[0], chip='c', dir=True)
    if not os.path.exists(fluxdir):
        os.makedirs(fluxdir)
    fluxfile = os.path.join(fluxdir, file)
    
    # If another process is alreadying making this file, wait!
    aplock(fluxfile, waittime=10, unlock=unlock)

    # Does product already exist?
    # check all three chip files
    sfluxid = str(ims[0]).zfill(8)
    chips = ['a', 'b', 'c']
    fluxdir = apogee_filename('Flux', num=ims[0], chip='c', dir=True)
    allfiles = [fluxdir + dirs.prefix + f'Flux-{chip}-{sfluxid}.fits' for chip in chips]
    
    if all([os.path.exists(file) for file in allfiles]) and not clobber:
        print(f'Flux file: {fluxdir + file} already made')
        if temp is not None:
            return
    # Delete any existing files to start fresh
    for file in allfiles:
        if os.path.exists(file):
            os.remove(file)

    # Open .lock file
    aplock(fluxfile, lock=True)

    if plate is None:
        plate = 0

    # Need to make sure extraction is done without flux calibration
    i1 = ims[0]
    files = apogee_filename('1D', num=i1, chip=['a', 'b', 'c'])
    if np.sum([os.path.exists(file) for file in files], dtype=int) > 0:
        for file in files:
            if os.path.exists(file):
                os.remove(file)

    if cmjd is not None:
        d = approcess(ims, cmjd=cmjd, darkid=darkid, flatid=flatid, psfid=psfid,
                      littrowid=littrowid, persistid=persistid, modelpsf=modelpsf,
                      fluxid=0, nocr=True, nfs=1, doproc=True, unlock=unlock)
    else:
        d = approcess(ims, darkid=darkid, flatid=flatid, psfid=psfid,
                      littrowid=littrowid, persistid=persistid, modelpsf=modelpsf,
                      fluxid=0, nocr=True, nfs=1, doproc=True, unlock=unlock)

    cmjd = getcmjd(i1)
    inpfile = os.path.join(apogee_filename('1D', num=i1, chip='a', dir=True), f'{i1:08d}')
    apmkfluxcal(inpfile, outdir=fluxdir, clobber=True, unlock=unlock)
    
    # Holtz's flux calibration method
    if holtz:
        nframes = len(ims)
        flux = np.zeros((2048, 300, 3), dtype=float)
        head0 = fits.getheader(os.path.join(fluxdir, apogee_filename('Flux', num=i1, chip='c')))
        norm = None

        for ii in range(nframes):
            i = ims[ii]
            frame = fits.getdata(apogee_filename('1D', num=i), ext=3)
            
            if ii == 0:
                sz = frame.shape
                flux = np.zeros((sz[1], sz[2], 3), dtype=float)
            
            for ichip in range(3):
                flux[:, :, ichip] += frame[ichip]

        bad = -1
        
        if plate is not None:
            if cmjd is None:
                cmjd = getcmjd(ims[0])
            fiber = getfiber(plate, cmjd, plugid=plugid)
            bad = np.where(fiber.fiberid < 0)[0]

        sz = flux.shape
        chips = ['a', 'b', 'c']
        resp = np.zeros((2048, 300, 3), dtype=float)

        for ichip in range(3):
            if bbtemp is not None:
                wavefile1 = apogee_filename('Wave', num=waveid, chip=chips[ichip], dir=True)
                file = apogee_filename('Wave', num=waveid, chip=chips[ichip], base=True)
                # We are often using dailywave with MJD names now
                if waveid < 1e7:
                    file = os.path.join(wavefile1, dirs.prefix + f'Wave-{chips[ichip]}-{str(waveid).strip()}.fits')
                wavetab = fits.getdata(file, ext=1)
                refspec = np.zeros(sz[1], dtype=float)
                
                for ifiber in range(sz[1]):
                    refspec[ifiber] = planck(wavetab[ifiber], bbtemp)
            else:
                refflux = flux[:, 150, ichip].reshape((sz[1], 1))
                refspec = refflux / refflux
            
            rows = np.arange(sz[2] + 1)
            refimg = rows[:, np.newaxis] ** refspec
            tmp = zap(flux[:, :, ichip], [100, 1])
            if ichip == 1:
                norm = tmp[1024, 150]
            resp[:, :, ichip] = refimg / tmp
            
            if bad[0] >= 0:
                for i in range(len(bad)):
                    resp[:, bad[i]] = 0.0

        # Normalize to center of green chip
        for ichip in range(3):
            resp[:, :, ichip] *= norm
            file = apogee_filename('Flux', num=i1, chip=chips[ichip], base=True)
            MWRFITS(resp[:, :, ichip], os.path.join(fluxdir, file), head0, create=True)

    # Delete lock file
    aplock(fluxfile, clear=True)

    # Extra block if we are calculating response function 
    if temp is not None:
        file = apogee_filename('Response', num=ims[0], chip='c', base=True, nochip=True)
        responsefile = os.path.join(fluxdir, file)
        # If another process is alreadying making this file, wait!        
        aplock(responsefile, waittime=10)
        if os.path.exists(responsefile) and not clobber:
            print(f'Flux file: {responsefile} already made')
            return
        # Open .lock file
        aplock(responsefile, lock=True)

        chips = ['a', 'b', 'c']
        if waveid < 1e7:
            wavefile1 = os.path.join(apogee_filename('Wave', num=waveid, chip=chips[1], dir=True),
                                     dirs.prefix + f'Wave-{chips[1]}-{str(waveid).strip()}.fits')
        
        else:
            wavefile1 = apogee_filename('Wave', num=waveid, chip=chips[1])

        wave = fits.getdata(wavefile1, ext=2)
        flux = fits.getdata(apogee_filename('Flux', num=ims[0], chip=chips[1]), ext=3)
        bbnorm = flux[1024] / planck(wave[1024, 150], temp)
        
        for i in range(3):
            if waveid < 1e7:
                wavefile1 = os.path.join(apogee_filename('Wave', num=waveid, chip=chips[i], dir=True),
                                         dirs.prefix + f'Wave-{chips[1]}-{str(waveid).strip()}.fits')
            else:
                wavefile1 = apogee_filename('Wave', num=waveid, chip=chips[i])
            
            wave = fits.getdata(wavefile1, ext=2)
            flux = fits.getdata(apogee_filename('Flux', num=ims[0], chip=chips[i]), ext=3)
            bbflux = planck(wave[:, 150], temp) * bbnorm
            head = make_header(head0, bbflux / flux)
            leadstr = 'APMKFLAT: '
            head['HISTORY'] = leadstr+time.asctime()
            import socket
            head['HISTORY'] = leadstr+getpass.getuser()+' on '+socket.gethostname()
            import platform
            head['HISTORY'] = leadstr+'Python '+pyvers+' '+platform.system()+' '+platform.release()+' '+platform.architecture()[0]
            # add reduction pipeline version to the header
            head['HISTORY'] = leadstr+' APOGEE Reduction Pipeline Version: '+load.apred
            head['V_APRED'] = plan.getgitvers(),'APOGEE software version'
	    head['APRED'] = load.apred,'APOGEE Reduction version'
            outfile = apogee_filename('Response', num=ims[0], chip=chipos[i], base=True, nochip=True)
            MWRFITS(bbflux / flux, os.path.join(fluxdir, file), head)

        # Remove lock file
        aplock(responsefile, clear=True)


