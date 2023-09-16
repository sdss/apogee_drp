import os
import numpy as np
from astropy.io import fits
from astropy.io.fits import Header
from scipy.ndimage import median_filter

def mklittrow(littrowid, darkid=None, flatid=None, sparseid=None, fiberid=None, clobber=False,
              cmjd=None, unlock=None):
    """
    Procedure to derive the APOGEE Littrow calibration file.

    Parameters
    ----------
    littrowid   The ID8 number of the exposure to use.
    =darkid     ID8 number of the dark calibration to use.
    =flatid     ID8 number of the flat calibration to use.
    =sparseid   ID8 number of the sparse calibration to use.
    =fiberid    ID8 number for the ETrace calibration file to use. 
    /clobber    Overwrite any existing files.
    =cmjd       Depricated parameter.
    /unlock     Delete lock file and start fresh 

    Returns
    -------
    A set of apLittrow-[abc]-ID8.fits files in the appropriate location
    determined by the SDSS/APOGEE tree directory structure.

    Example
    -------

    mklittrow,littrowid

    By J. Holtzman, 2011
    Added doc strings, updates to use data model  D. Nidever, Sep 2020 
    """

    dirs = getdir()
    caldir = dirs.caldir

    litdir = apogee_filename('Littrow', num=littrowid, chip='b', dir=True)
    if not os.path.exists(litdir):
        os.makedirs(litdir)
    litfile = apogee_filename('Littrow', num=littrowid, chip='b', base=True)

    # If another process is already making this file, wait!
    aplock(litfile, waittime=10, unlock=unlock)

    # Does product already exist? We only use the 'b' detector file
    if os.path.exists(litfile) and not clobber:
        print('littrow file:', litfile, 'already made')
        return
    allfiles = [litfile]
    # Delete any existing files to start fresh
    for file_path in allfiles:
        if os.path.exists(file_path):
            os.remove(file_path)
    # Open .lock file
    aplock(litfile, lock=True)

    # Make empirical PSF with broader smoothing in columns so that Littrow ghost is not incorporated as much
    mkpsf(littrowid, darkid=darkid, flatid=flatid, sparseid=sparseid, fiberid=fiberid,
          average=200, clobber=True, unlock=unlock)

    # Process the frame with this PSF to get a model that does not have Littrow ghost
    psfdir = apogee_filename('PSF', num=littrowid, chip='b', dir=True)
    wavefile = 0
    indir = apogee_filename('2D', num=littrowid, chip='b', dir=True)
    outdir = apogee_filename('2D', num=littrowid, chip='b', dir=True)
    outbase = apogee_filename('2D', num=littrowid, chip='b', base=True)

    ap2dproc(indir + '/' + str(littrowid).zfill(8), outdir + '/' + str(littrowid).zfill(8), 4,
             wavefile=wavefile, clobber=True)

    # Read in the 2D file and the model, and use them to find the Littrow ghost
    im2 = apread('2D', num=littrowid, chip='b')
    im2mod = apread('2Dmodel', num=littrowid, chip='b')
    im = im2['flux']
    immask = im2['mask']
    scat_remove(im, scat=1)
    immod = im2mod['flux']
    bad = np.where((immask & badmask()) > 0)
    im[bad] = np.nan
    l = np.where(np.median(im[1200:1500, :] - immod[1200:1500, :], 20) > 10)[0]

    # Write out an integer mask
    litt = np.zeros((2048, 2048), dtype=int)
    tmp = np.zeros(im[1200:1500, :].shape)
    tmp[l] = 1
    litt[1250:1450, :] = tmp[50:250, :]

    file_path = apogee_filename('Littrow', num=littrowid, chip='b', base=True)
    head = Header()
    head['EXTNAME'] = 'LITTROW MASK'
    leadstr = 'MKLITTROW: '
    sxaddhist(leadstr + systime(0), head)
    info = GET_LOGIN_INFO()
    sxaddhist(leadstr + info['user_name'] + ' on ' + info['machine_name'], head)
    sxaddhist(leadstr + 'Python ' + sys.version, head)
    sxaddhist(leadstr + 'APOGEE Reduction Pipeline Version: ' + getvers(), head)

    hdu = fits.PrimaryHDU(data=litt, header=head)
    hdu.writeto(file_path, overwrite=True)

    # Move PSFs to the littrow directory since they are not a standard PSF
    outdir = litdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    psf_files = file_search(psfdir + '/*' + str(littrowid).zfill(8) + '*.fits')
    for psf_file in psf_files:
        new_file_path = os.path.join(outdir, os.path.basename(psf_file))
        os.rename(psf_file, new_file_path)

    files = file_search(apogee_filename('1D', num=littrowid, chip='b', dir=True) + '/*1D*' + str(littrowid).zfill(8) + '*.fits')
    for file_path in files:
        new_file_path = os.path.join(outdir, os.path.basename(file_path))
        os.rename(file_path, new_file_path)

    files = file_search(apogee_filename('2Dmodel', num=littrowid, chip='b', dir=True) + '/*2Dmodel*' + str(littrowid).zfill(8) + '*.fits')
    for file_path in files:
        new_file_path = os.path.join(outdir, os.path.basename(file_path))
        os.rename(file_path, new_file_path)

    file_path = apogee_filename('Littrow', num=littrowid, chip='b', base=True, nochip=True)
    # Remove lock file
    aplock(litfile, clear=True)
