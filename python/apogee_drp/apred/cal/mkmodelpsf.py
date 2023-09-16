import os
import subprocess

def mkmodelpsf(modelpsf, sparseid=None, psfid=None, clobber=False, unlock=False):
    """
    Procedure to make an APOGEE model PSF master calibration file.
    This is a wrapper around the python mkmodelpsf program.

    Parameters
    ----------
    modelpsf    The ID8 numbers of the model PSF.
    =sparseid   Sparse frame to use.
    =psfid      PSF frame to use.
    /clobber    Overwrite existing files.
    /unlock     Delete the lock file and start fresh.

    Returns
    -------
    A set of apPSFModel-[abc]-ID8.fits files in the appropriate location
    determined by the SDSS/APOGEE tree directory structure.

    Example
    -------

    mkmodelpsf,modelpsf,sparseid=sparseid,psfid=psfid,/clobber

    By D. Nidever, 2022
    copied from mfpi.pro
    """

    
    if not isinstance(name, str):
        name = str(modelpsf[0])

    dirs = getdir(apodir, caldir, spectrodir, vers)
    psfdir = apogee_filename('PSFModel', num=name, chip='a', dir=True)
    file = dirs.prefix + "PSFModel-" + str(name).zfill(8)
    psffile = os.path.join(psfdir, file)

    # If another process is alreadying make this file, wait!
    aplock(psffile, waittime=10, unlock=unlock)

    # Does product already exist?
    # check all three chip files
    smodelpsf = str(modelpsf).zfill(8)
    cmjd = getcmjd(modelpsf)
    mjd = int(cmjd)
    chips = ['a', 'b', 'c']
    allfiles = [os.path.join(psfdir, dirs.prefix + 'PSFModel-' + chip + '-' + str(cmjd) + '-' + smodelpsf + '.fits') for chip in chips]

    if sum([os.path.exists(file) for file in allfiles]) == 3 and not clobber:
        print('modelpsf file:', os.path.join(psfdir, file + '.fits'), 'already made')
        return
    
    # Delete any existing files to start fresh
    for file in allfiles:
        if os.path.exists(file):
            os.remove(file)

    print('Making modelpsf:', modelpsf)
    # Open .lock file
    aplock(psffile, lock=True)

    # New Python version! 
    cmd = ['mkmodelpsf',str(modelpsf).strip(),str(sparseid).strip(),
           str(psfid).strip(),dirs.apred,dirs.telescope,'--verbose']
    print('Running:', cmd)
    subprocess.Popen(cmd)

    # Check that the calibration file was successfully created
    outfile = os.path.join(psfdir, file.replace(dirs.prefix + 'PSFModel-', dirs.prefix + 'PSFModel-a-') + '.dat')
    if os.path.exists(outfile):
        openw, lock, lun = True, False, 0  # Assuming these are functions you've defined elsewhere
        free_lun(lock)

    aplock(psffile, clear=True)
