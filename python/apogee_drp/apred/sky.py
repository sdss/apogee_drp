# Software to use to fit and correct sky lines and telluric absorption
# in the APOGEE spectra

# D. Nidever, May 2022

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from glob import glob
import pdb
import time
from astropy.io import ascii, fits
from scipy.optimize import curve_fit
#from ..utils import apload, yanny, plan, peakfit, info
#from ..plan import mkplan
#from ..database import apogeedb
#from . import wave
from astropy.table import Table,hstack,vstack
from dlnpyutils import utils as dln, robust, coords
import doppler
import thecannon as tc
from scipy.interpolate import BSpline


# Load the data

# Load the Cannon telluric models

# Loop over the stars to fit with Cannon synthetic spectrum model and telluric model

#  -determine Teff for star using Gaia+2MASS photometry and color-teff relations
#  -convolve Cannon telluric model with LSF for this fiber
#  -fit stellar parameters, wavelength offset, telluric parameters simultaneously
#    stellar parameters:  Teff (constrained), logg, and [Fe/H]
#    wavelength offset: maybe just a constant offset?
#    telluric parameters: airmass (known?), pwscale, scale (for three species)
#  -also fit the bright airglow lines

def colorteff(tab):

    tab['BPMAG'] = 16.9790
    tab['RPMAG'] = 14.8709
    tab['GMAG'] = 15.9482
    tab['JMAG'] = 13.3710
    tab['HMAG'] = 12.5060
    tab['KSMAG'] = 12.2760

    
    colors = np.array([tab['BPMAG'],tab['RPMAG'],tab['JMAG'],tab['HMAG'],tab['KSMAG']])-tab['GMAG']
    
    # Load the color-teff information
    filename = '/Users/nidever/sdss5/mwm/apogee/colorteff/colorteff_gaiaedr3_2mass.fits'
    hdu = fits.open(filename)
    young = {}
    for i in np.arange(1,6):
        head = hdu[i].header
        data = hdu[i].data
        nord = head['NORD']
        band = head['BAND']        
        spl = BSpline(data['t'],data['c'],nord)
        young[band] = spl
    old = {}
    for i in np.arange(6,11):
        head = hdu[i].header
        data = hdu[i].data
        nord = head['NORD']
        band = head['BAND']        
        spl = BSpline(data['t'],data['c'],nord)
        old[band] = spl        

    # Now find the best Teff and extinction for the input star
    # X is 5000.0/Teff(K), Y is BAND-GMAG
        
    import pdb; pdb.set_trace()
        
    
def skycorr(spec,tab):
    """
    Correct an APOGEE exposure for sky lines and telluric absorption

    Parameters
    ----------
    spec
       Spectrum object with flux, error, wavelength and mask.
    tab : table
       Table of information on all the objects.

    """

    codedir = os.environ['APOGEE_DRP_DIR']
    
    # Load the Doppler Cannon models
    smodels = doppler.models
    
    # Load the telluric Cannon models
    tfiles = glob(codedir+'data/telluric/telluric_cannon_???.pkl')
    tmodels = []
    for f in tfiles:
        model1 = tc.CannonModel.read(f)
        # Generate the model ranges
        ranges = np.zeros([2,2])
        for i in range(2):
            ranges[i,:] = dln.minmax(model1._training_set_labels[:,i])
        model1.ranges = ranges
        # Rename _fwhm to fwhm and _wavevac to wavevac
        if hasattr(model1,'_fwhm'):
            setattr(model1,'fwhm',model1._fwhm)
            delattr(model1,'_fwhm')
        model1.wavevac = True
        tmodels.append(model1)
        
    # Loop over the objects to fit
    for i in range(len(tab)):
        # spec must be Spec1D object with wavelength and LSF
        out = specfit(spec,tab[i],smodel,tmodel)
    

def specfit(spec,tab,smodel,tmodel):
    """
    Fit a single spectrum

    Parameters
    ----------
    spec : Spec1D object
      The spectrum to fit.
    tab : table
      Table with information on the object including Gaia+2MASS photometry.
    smodel : Cannon model
      The Doppler Cannon models.
    tmodel : Cannon model
      The Telluric Cannon models.
    
    """
    pass
