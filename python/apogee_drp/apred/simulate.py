import os
import numpy as np
from dlnpyutils import utils as dln
from astropy.io import fits
from . import psf,ap2d


def add_detector_noise(
    flux,
    gain=1.0,
    readnoise=10.0,
    flux_unit="electrons",
    seed=None,
):
    """
    Add Poisson noise and Gaussian read noise to a noiseless 2D image.

    Parameters
    ----------
    flux : array-like
        Expected noiseless counts per pixel.
    gain : float or array-like
        Detector gain in electrons/ADU. Only used when flux_unit='adu'.
    readnoise : float or array-like
        Effective read noise in electrons per pixel.
    flux_unit : {'electrons', 'adu'}
        Units of the input and returned image.
    seed : int, optional
        Random seed.

    Returns
    -------
    noisy_flux : ndarray
        Simulated noisy image in the same units as `flux`.
    error : ndarray
        Expected 1-sigma uncertainty in the same units as `flux`.
    """
    rng = np.random.default_rng(seed)
    flux = np.asarray(flux, dtype=float)

    if flux_unit == "adu":
        expected_electrons = flux * gain
    elif flux_unit == "electrons":
        expected_electrons = flux
    else:
        raise ValueError("flux_unit must be 'electrons' or 'adu'")

    # Poisson means cannot be negative.
    poisson_mean = np.clip(expected_electrons, 0.0, None)

    noisy_electrons = rng.poisson(poisson_mean).astype(float)
    noisy_electrons += rng.normal(
        loc=0.0,
        scale=readnoise,
        size=flux.shape,
    )

    error_electrons = np.sqrt(poisson_mean + readnoise**2)

    if flux_unit == "adu":
        return noisy_electrons / gain, error_electrons / gain

    return noisy_electrons, error_electrons


def simulate(modelepsffile,tracefile,wavefile,spectra,lsffile=None,nreads=42,
             noreadnoise=False):
    """
    Simulate 2D APOGEE image using input data.
    """

    exptime = nreads*10.65

    # input flux needs to be in e/sec
    
    #psfhdu = fits.open(epsffile)
    epsf = psf.PSF.read(modelepsffile)
    traceim = fits.getdata(tracefile)
    wavelength = fits.getdata(wavefile,2)
    #wavehdu = fits.open(wavefile)
    #wavelength = wavehdu[2].data
    #wavehdu.close()
    #nfibers = len(psfhdu)-1
    nfibers = traceim.shape[0]

    sky_flux_e_per_sec = 0.19
    sky_flux_e_per_exposure = sky_flux_e_per_sec * exptime
    
    # Loop over the spectra:
    fluxim = np.zeros((2048,2048),float)
    fluxerr = np.zeros((2048,2048),float)
    trueflux = []
    fullepsf = []
    #print('ONLY DOING 50 FIBERS!!')
    for i in range(nfibers):
    #for i in range(50):
        if i % 50 == 0: print(i+1,'/',nfibers)
        #epsf1 = psfhdu[i+1].data
        #epsfim1 = epsf1['IMG'][0]
        #epsflo = epsf1['LO'][0]
        #epsfhi = epsf1['HI'][0]

        spec1 = spectra[i]
        wave1 = wavelength[i,:]
        # Interpolate spectrum to our pixels
        object_flux_e_per_sec = dln.interp(spec1.wave,spec1.flux,wave1)
        # scale by exptime
        object_flux_e_per_exposure = object_flux_e_per_sec * exptime
        fiber_flux_e_per_exposure = object_flux_e_per_exposure + sky_flux_e_per_exposure
        
        # model psf goes from -14.95 to + 14.95 in the y profile
        ycen = traceim[i,:]
        ylo = int(np.min(np.round(ycen)))-20
        ylo = np.maximum(ylo,0)
        yhi = int(np.max(np.round(ycen)))+20
        yhi = np.minimum(yhi,2047)
        ny = yhi-ylo+1
        y = np.arange(ny)+ylo
        epsfimg = np.zeros((ny,2048),float)
        # Column loop                                                                                                       
        for j in range(2048):
            try:
                m1 = epsf([j,ycen[j]],y=y,ycen=ycen[j])
            except:
                print('problem')
                import pdb; pdb.set_trace()
            m1 /= np.sum(m1)
            epsfimg[:,j] = m1

        fiberimg = epsfimg * fiber_flux_e_per_exposure.reshape(1,-1)
        fluxim[ylo:yhi+1,:] += fiberimg

        trueflux1 = {'lo':ylo,'hi':yhi,'ycen':ycen,'img':fiberimg,
                     'objflux':object_flux_e_per_exposure,
                     'skyflux':sky_flux_e_per_exposure,
                     'flux':fiber_flux_e_per_exposure}
        trueflux.append(trueflux1)

        fullepsf.append({'fiber':i,'lo':ylo,'hi':yhi,'img':epsfimg,'ycen':ycen})
        
        # how about the wings??


        
        #import pdb; pdb.set_trace()
        


        
    #import pdb; pdb.set_trace()

    # Add background / scattered light
    # it's quite low at LCO
    fluxim += 1
    
    # Make the observed image
    # add poisson and rdnoise

    # gain is 1.9 at APO and 3.0 at LCO
    # rdnoise is XX e at APO and XX e at LCO
    gain = 3.0
    #rdnoise_adu = 3.4    # for a 42-read exposure
    rdnoise_adu = 2.43   # for 84-read exposure
    rdnoise_e = rdnoise_adu*gain

    if noreadnoise:
        rdnoise_e = 0.0
    
    print('gain =',gain)
    print('rdnoise (e)=',rdnoise_e)

    noisy_im, err_im = add_detector_noise(
        fluxim,
        gain=gain,          # electrons/ADU
        readnoise=rdnoise_e,    # effective electrons/pixel
        flux_unit="electrons",
        seed=12345,
    )

    # Convert from electrons to ADUs
    observed_im = noisy_im / gain
    observed_err = err_im / gain
    
    
    return observed_im,observed_err,fluxim,trueflux,fullepsf
