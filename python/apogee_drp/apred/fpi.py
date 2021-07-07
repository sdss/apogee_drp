
# Software to use the arclamp and FPI data to get improved
# wavelength solution and accurate wavelengths for the FPI lines

# D. Nidever, July 2021

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

#import copy
import numpy as np
#import matplotlib.pyplot as plt
import os
import glob
#import pdb
#from functools import wraps
from astropy.io import ascii, fits
#from scipy import signal, interpolate
#from scipy.optimize import curve_fit
#from scipy.special import erf, erfc
#from scipy.signal import medfilt, convolve, boxcar
from ..utils import apload, yanny, plan
from ..plan import mkplan
from . import wave
#from holtztools import plots, html
from astropy.table import Table


def dailyfpiwave(mjd5,observatory='apo',apred='daily',verbose=True):
    """
    Function to run daily that generates a wavelength solution using a week's worth of
    arclamp data simultaneously fit with "apmultiwavecal" then we use the FPI full-frame exposure
    to get the FPI wavelengths and redo the wavelength solution.
    """
    
    dailydir = os.environ['APOGEE_REDUX']+'/'+apred+'/log/'+observatory+'/'
    instrument = {'apo':'apogee-n','lco':'apogee-s'}[observatory]

    # Check if multiple new MJDs
    datadir = {'apo':os.environ['APOGEE_DATA_N'],'lco':os.environ['APOGEE_DATA_S']}[observatory]
    mjdlist = os.listdir(datadir)
    mjds = [mjd for mjd in mjdlist if int(mjd) >= (int(mjd5)-7) and int(mjd)<=int(mjd5) and mjd.isdigit()]
    if verbose:
        print(len(mjds), ' nights: ',','.join(mjds))

    # Get exposure information
    if verbose:
        print('Getting exposure information')
    for i,m in enumerate(mjds):
        expinfo1 = mkplan.getexpinfo(observatory,m)
        nexp = len(expinfo1)
        if verbose:
            print(m,' ',nexp,' exposures')
        if i==0:
            expinfo = expinfo1
        else:
            expinfo = np.hstack((expinfo,expinfo1))


    # Step 1: Find the arclamp frames for the last week
    #--------------------------------------------------

    # Get arclamp exposures
    arc, = np.where((expinfo['exptype']=='ARCLAMP') & ((expinfo['arctype']=='UNE') | (expinfo['arctype']=='THAR')))
    narc = len(arc)
    if narc==0:
        print('No arclamp exposures for these nights')
        return
    arcframes = expinfo['num'][arc]

    # Get full frame FPI exposure to use
    fpi, = np.where(expinfo['exptype']=='FPI')
    fpiframe = 38310023
    print('KLUDGE!!!  Hardcoding FPI full-frame exposure number')
    print('FPI full-frame exposure ',fpiframe)


    # Step 2: Fit wavelength solutions simultaneously
    #------------------------------------------------
    # This is what apmultiwavecal does
    if verbose:
        print('Solving wavelength solutions simultaneously using all arclamp exposures')
    # The previously measured lines in the apLines files will be reused if they exist
    wave.wavecal(arcframes,rows=np.arange(300),name='apFPIWave',npoly=4,inst=instrument,verbose=verbose,vers=apred)

    import pdb; pdb.set_trace()


    

    # Step 3: Fit peaks to the full-frame FPI data

    # Step 4: Determine median wavelength per FPI lines

    # Step 5: Refit wavelength solutions using FPI lines


    # Save the results
    # table of FPI lines data: chip, gauss center, Gaussian parameters, wavelength, flux
    # wavelength coefficients
    # wavelength array??




    # make a little python function that generates a wavelength solution using a week's worth of
    # arclamp data simultaneously fit with "apmultiwavecal" and then we use the FPI full-frame exposure
    # to get the FPI wavelengths and redo the wavelength solution
    # -find the arclamp frames for the last week
    # -run apmultiwavecal on them
    # -fit peaks in FPI data
    # -define median wavelengths per FPI line
    # -refit wavelength solution with FPI lines, maybe holding higher-order coefficients fixed



