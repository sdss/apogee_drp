import os
import numpy as np
from astropy.io import fits
from ..utils import apload,plan

def mktelluric(tellid, apred='daily', telescope='apo25m',
               clobber=False, nowait=False, unlock=False):
    """
    Procedure to make an APOGEE daily telluric calibration file.

    Parameters
    ----------
    tellid       Telluric ID, WAVEID-LSFID.
    /nowait      If file is already being made then don't wait
                 just return.
    /clobber     Overwrite existing files.
    /unlock      Delete the lock file and start fresh.

    Returns
    -------
    A set of apTelluric-[abc]-WAVEID-LSFID.fits files in the appropriate location
    determined by the SDSS/APOGEE tree directory structure.

    Example
    -------

    mktelluric,tellid,/clobber

    Made from mkwave.pro by D.Nidever, March 2022
    """

    load = apload.ApLoad(apred=apred,telescope=telescope)
    name = str(tellid).zfill(2)
    dirs = getdir(apodir, caldir, spectrodir, vers)
    telldir = apogee_filename('Telluric', num=0, chip='a', dir=True)
    file = dirs.prefix + 'Telluric-' + name
    tellfile = os.path.join(telldir, file)
        
    # If another process is already making this file, wait!
    aplock(tellfile, waittime=10, unlock=unlock)
    
    # Check if the product already exists
    chips = ['a', 'b', 'c']
    allfiles = telldir + dirs.prefix + 'Telluric-' + '-'.join(chips) + '-' + name + '.fits'
    if sum(file_test(allfiles)) == 3 and not clobber:
        print('Telluric file:', telldir + file, 'already made')
        return
    
    file_delete(allfiles, allow=True)
    
    print('Making telluric:', name)


    # aptelluric_convolve will return the array of LSF-convolved telluric spectra appropriate
    #   for the specific wavelength solutions of this frame
    # There are 3 species, and there may be models computed with different "scale" factor, i.e.
    #   columns and precipitable water values. If this is the case, we fit each star not
    #   only for a scaling factor of the model spectrum, but also for which of the models
    #   is the best fit. For self-consistency, we adopt the model for each species that provides
    #   the best fit for the largest number of stars, and then go back and refit all stars
    #   with this model
    # The telluric array is thus 4D: [npixels,nfibers,nspecies,nmodels]
    
    # Construct fake frame with wavefile, lsffile, flux, and lsfcoef
    flux = np.zeros((2048, 300))
    FITS_READ(lsffiles[0], lsfcoef1)
    aframe = {'wavefile': wavefiles[0], 'lsffile': lsffiles[0], 'lsfcoef': lsfcoef1, 'flux': flux.copy()}
    FITS_READ(lsffiles[1], lsfcoef2)
    bframe = {'wavefile': wavefiles[1], 'lsffile': lsffiles[1], 'lsfcoef': lsfcoef2, 'flux': flux.copy()}
    FITS_READ(lsffiles[2], lsfcoef3)
    cframe = {'wavefile': wavefiles[2], 'lsffile': lsffiles[2], 'lsfcoef': lsfcoef3, 'flux': flux.copy()}
    frame = {'chipa': aframe, 'chipb': bframe, 'chipc': cframe}
    
    # Make the telluric calibration file
    convolved_telluric = APTELLURIC_CONVOLVE(frame, fiber=np.arange(300), convonly=True, unlock=unlock, clobber=clobber)
    
    # Check that the calibration file was successfully created
    tellfiles = apogee_filename('Telluric', num=tellid, chip=chips)
    if sum(file_test(outfile)) == 3:
        print('Telluric file', dirs.prefix + 'Telluric-' + str(tellid).zfill(2), 'completely successfully')
        openw, lock, _ = get_lun(telldir + file + '.dat')
        free_lun(lock)
    else:
        print('PROBLEMS with apTelluric-' + str(tellid).zfill(2) + ' files')

    aplock(tellfile,clear=True)
