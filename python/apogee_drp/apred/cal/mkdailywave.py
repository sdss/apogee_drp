import os
import subprocess
import numpy as np

def mkdailywave(mjd, darkid=None, flatid=None, psfid=None, modelpsf=None, fiberid=None,
                clobber=False, nowait=False, nofit=False, unlock=False, psflibrary=None):
    """
    Procedure to make an APOGEE daily wavelength calibration file
    from 8 nearby days of arclamp exposures.  This is a wrapper
    around the python apmultiwavecal program.

    Parameters
    ----------
    mjd          MJD of the night for which to make the daily
                 wavelength solution.
    =darkid      Dark frame to be used if images are reduced.
    =flatid      Flat frame to be used if images are reduced.
    =psfid       PSF frame to be used if images are reduced.
    =modelpsf    Model PSF calibration frame to use.
    =fiberid     ETrace frame to be used if images are reduced.
    /nowait      If file is already being made then don't wait
                 just return.
    /clobber     Overwrite existing files.
    /nofit       Skip fit (find lines only).
    /unlock      Delete the lock file and start fresh.
    /psflibrary  Use PSF library for extraction.

    Returns
    -------
    A set of apWave-[abc]-ID8.fits files in the appropriate location
    determined by the SDSS/APOGEE tree directory structure.

    Example
    -------

    mkdailywave,mjd,darkid=darkid,flatid=flatid,psfid=psfid,fiberid=fiberid,/clobber

    Made from mkwave.pro by D.Nidever, March 2022
    """
    
    name = str(int(mjd)).zfill(2)
    dirs = getdir(apodir, caldir, spectrodir, vers)
    wavedir = apogee_filename('Wave', num=0, chip='a', dir=True)
    wavebase = dirs.prefix + 'Wave-' + name
    wavefile = os.path.join(wavedir, wavebase)

    # If another process is already making this file, wait!
    aplock(wavefile,waittime=10,unlock=unlock)

    # Does the product already exist?
    # Check all three chips and .dat file
    chips = ['a', 'b', 'c']
    allfiles = [os.path.join(wavedir, dirs.prefix + f'Wave-{chip}-{name}.fits') for chip in chips]
    if all([os.path.exists(file) for file in allfiles]) and not clobber:
        print('Wavecal file:', wavefile, 'already made')
        return
    for file in allfiles:
        if os.path.exists(file):
            os.remove(file)

    print('Making dailywave:', name)
    # Open .lock file
    aplock(wavefile,lock=True)
    
    # Get the arclamps that we need for the daily cal
    query = f"select * from apogee_drp.exposure where mjd>={int(mjd) - 10} and mjd<={int(mjd) + 10} " \
            f"and exptype='ARCLAMP' and observatory='{dirs.telescope[:3]}'"
    expinfo = dbquery(query)
    
    for i in range(len(expinfo)):
        # arctype info is missing in the db for early SDSS-V dates
        if expinfo[i].arctype == '':
            fil = apogee_filename('R', num=expinfo[i].num, chip='a')
            expinfo2 = apfileinfo(fil, silent=True)
            if expinfo2.lampune == 1:
                expinfo[i].arctype = 'UNE'
            if expinfo2.lampthar == 1:
                expinfo[i].arctype = 'THAR'

    expinfo.arctype = [arc.strip() for arc in expinfo.arctype]
    gdarc = np.where((expinfo.arctype == 'UNE') | (expinfo.arctype == 'THAR'))[0]
    if len(gdarc) == 0:
        print('No arclamps for these nights')
        return
    arcinfo = expinfo[gdarc]
    
    # Figure out which nights to use
    ui = np.unique(arcinfo.mjd)
    mjds = np.sort(arcinfo[ui].mjd)
    nmjds = len(mjds)
    si = np.argsort(np.abs(mjds - mjd))
    keep = si[:(nmjds - 1) < 7]
    mjds = mjds[keep]
    mjds = np.sort(mjds)
    
    # Only keep the arclamps for these nights
    keep = []
    for i in range(len(mjds)):
        gd = np.where(arcinfo.mjd == mjds[i])[0]
        if len(gd) > 0:
            keep.extend(gd)
    
    if len(keep) == 0:
        print('No arclamps for these nights')
        return
    
    arcinfo = arcinfo[keep]
    waveid = np.array(arcinfo.num)
    print(len(arcinfo), 'arclamps')

    print('\n***** Processing the frames and finding the lines *****\n')

    for i in range(len(waveid)):
        print('')
        print(f'--- Frame {i + 1}: {waveid[i]} ---')

        # Check if it exists already
        file1 = apogee_filename('Wave', num=waveid[i], chip='c')
        wavedir1 = os.path.dirname(file1)
        swaveid1 = str(waveid[i]).zfill(8)
        allfiles1 = [os.path.join(wavedir1, dirs.prefix + f'Wave-{chip}-{swaveid1}.fits') for chip in chips] + \
                    [os.path.join(wavedir1, dirs.prefix + f'Wave-{swaveid1}.dat')]
        if all([os.path.exists(file) for file in allfiles1]):
            print(f'wave file: {dirs.prefix}Wave-{swaveid1} already made')
            continue

        # Check that the data is okay
        chfile = apogee_filename('2D', num=waveid[i], chip='b')
        if not os.path.exists(chfile):
            print(chfile, 'NOT FOUND')
            continue

        head0 = headfits(chfile, exten=0)
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

        smsub = medfilt2d(sub, 7, axis=1)                                          # smooth in spectral axis
        resmsub = np.repeat(np.mean(smsub.reshape(201, -1, 8), axis=2), 8, axis=1)  # rebin in spatial axis
        peakflux = np.max(resmsub, axis=1)                                      # peak flux feature in spectral dim.
        avgpeakflux = np.median(peakflux)

        # Check the line flux
        if avgpeakflux / head0['nread'] < thresh:
            print(f'Not enough flux in {chfile}')
            continue

        makecal(wave=waveid[i], file=os.path.join(dirs.libdir, 'cal', dirs.instrument + '-wave.par'),
                nofit=True, unlock=unlock, librarypsf=psflibrary, modelpsf=modelpsf)

    # New Python version!
    cmd = ['apdailywavecal', '--apred', dirs.apred]
    if clobber:
        cmd.append('--clobber')
    cmd.extend(['--observatory', dirs.telescope[:3], '--verbose', name])
    subprocess.Popen(cmd)

    # Check that the calibration file was successfully created
    outfile = os.path.join(wavedir, wavefile.replace('apWave-', 'apWave-a-'))
    if os.path.exists(outfile):
        with open(outfile + '.dat', 'w') as lock:
            pass

    aplock(wavefile, clear=True)
