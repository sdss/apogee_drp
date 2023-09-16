import os
import subprocess

def mkmultiwave(waveid, name=None, clobber=False, nowait=False, calfile=None, unlock=False, psflibrary=None):
    """
    Procedure to make an APOGEE wavelength calibration file from
    multiple arc lamp exposures from different nights.  This is
    a wrapper around the python apmultiwavecal program.

    Parameters
    ----------
    waveid      Array of ID8 numbers of the arc lamp exposures to use.
    =name       Output filename base.  By default waveid[0] is used.
    /nowait     If file is already being made then don't wait
                just return.
    /clobber    Overwrite existing files.
    =file       Depricated parameter.
    /unlock     Delete the lockfile and start fresh.
    /psflibrary   Use PSF library to get PSF cal for images.

    Returns
    -------
    A set of apWave-[abc]-ID8.fits files in the appropriate location
    determined by the SDSS/APOGEE tree directory structure.

    Example
    -------

    mkmultiwave,ims,name=name,/clobber

    By J. Holtzman, 2011
    Added doc strings, updates to use data model  D. Nidever, Sep 2020 
    """
    
    if name is None:
        name = str(waveid[0])

    dirs = getdir(apodir, caldir, spectrodir, vers)
    wavedir = apogee_filename('Wave', num=name, chip='a', dir=True)
    if not os.path.exists(wavedir):
        os.makedirs(wavedir)
    file = dirs.prefix + "Wave-" + str(name).zfill(8)
    wavefile = os.path.join(wavedir, file)
    
    # If another process is alreadying make this file, wait!
    aplock(wavefile, waittime=10, unlock=unlock)

    # Does product already exist?
    chips = ['a', 'b', 'c']
    swaveid = str(name).zfill(8)
    allfiles = [os.path.join(wavedir, dirs.prefix + 'Wave-' + chip + '-' + swaveid + '.fits') for chip in chips]
    allfiles.append(os.path.join(wavedir, dirs.prefix + 'Wave-' + swaveid + 'py.dat'))

    if sum([os.path.exists(file) for file in allfiles]) == 4 and not clobber:
        print('Wavecal file:', os.path.join(wavedir, file + 'py.dat'), 'already made')
        return

    for file in allfiles:
        if os.path.exists(file):
            os.remove(file)

    print('Making wave:', waveid)
    # Open .lock file
    aplock(wavefile, lock=True)

    # Process the frames and find lines
    print('')
    print('***** Processing the frames and finding the lines *****')
    print('')

    for i in range(0, len(waveid), 2):
        print('')
        print('--- Frame ', str(i + 1).zfill(2), ':  ', str(waveid[i]).zfill(2), ' ---')
        makecal.makecal(wave=waveid[i], file=os.path.join(dirs.libdir, 'cal', dirs.instrument + '-wave.par'),
                        nofit=True, unlock=unlock, librarypsf=psflibrary)

    # New Python version!
    print('Running apmultiwavecal')
    cmd = ['apmultiwavecal', '--name', name, '--vers', dirs.apred,
           '--hard', '--inst', dirs.instrument, '--verbose']
    for w in waveid:
        cmd.append(str(w))
    subprocess.Popen(cmd)
    
    # Check that all three chip files exist
    swaveid = str(name).zfill(8)
    allfiles = [os.path.join(wavedir, dirs.prefix + 'Wave-' + chip + '-' + swaveid + '.fits') for chip in chips]
    
    if sum([os.path.exists(file) for file in allfiles]) == 3:
        with open(os.path.join(wavedir, file + 'py.dat'), 'w') as f:
            pass
    else:
        raise Exception('Failed to make wavecal')

    aplock(wavefile, clear=True)
