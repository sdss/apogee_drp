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

def loadfiles():

    codedir = os.environ['APOGEE_DRP_DIR']
    
    # Load the color-teff information
    filename = codedir+'/data/colorteff/colorteff_gaiaedr3_2mass.fits'
    hdu = fits.open(filename)
    young = {}
    for i in np.arange(1,6):
        head = hdu[i].header
        data = hdu[i].data
        nord = head['NORD']
        band = head['BAND']
        xmin = head['XMIN']
        xmax = head['XMAX']         
        spl = BSpline(data['t'],data['c'],nord)
        young[band] = spl
        young['XMIN'] = xmin
        young['XMAX'] = xmax
        young['TYPE'] = 'young'
    old = {}
    for i in np.arange(6,11):
        head = hdu[i].header
        data = hdu[i].data
        nord = head['NORD']
        band = head['BAND']
        xmin = head['XMIN']
        xmax = head['XMAX']                 
        spl = BSpline(data['t'],data['c'],nord)
        old[band] = spl
        old['XMIN'] = xmin
        old['XMAX'] = xmax        
        old['TYPE'] = 'old'
        
    return young,old

def bestslope(x,y,sigma=None,axis=None):
    """
    Determine the best slope using a linear equation
    y = m*x
    """

    if sigma is None:
        sigma = np.ones(x.shape,float)

    # y = mx*
    # slp = (W*XY - X*Y)/(W*X^2-X*X)
    # where
    # W = Sum(1/sigma^2)
    # X = Sum(x/sigma^2)
    # Y = Sum(y/sigma^2)
    # XY = Sum(x*y/sigma^2)    
    # X^2 = Sum(x^2/sigma^2)
    wsum = np.sum(1/sigma**2,axis=axis)
    xsum = np.sum(x/sigma**2,axis=axis)
    ysum = np.sum(y/sigma**2,axis=axis)
    xysum = np.sum(x*y/sigma**2,axis=axis)    
    xsqsum = np.sum(x**2/sigma**2,axis=axis)
    slp = (wsum*xysum-xsum*ysum)/(wsum*xsqsum-xsum**2)

    return slp

def bestextinction(obscolor,modelcolor,redav,obscolorerr=None):
    """
    Determine best A(V) extinction given observed colors,
    observed color errors, model intrinsic colors, and
    reddening values (E(color)/A(V)).
    """

    if obscolorerr is None:
        obscolorerr = np.ones(obscolor.shape,float)
    
    # For each Teff point, calculate the best-fitting extinction value
    # observed color = intrinsic color + Av * reddening
    # Calculate weighted mean slope to get Av
    # x = redav, y = obscolor - intrinsic colors
    if obscolor.ndim==1 and modelcolor.ndim==2:
        ngrid = modelcolor.shape[1]
        diffcolor = obscolor.reshape(-1,1) - modelcolor
        obscolorerr2d = obscolorerr.reshape(-1,1) + np.zeros(ngrid).reshape(1,-1)
        redav2d = redav.reshape(-1,1) + np.zeros(ngrid).reshape(1,-1)
        av = bestslope(redav2d,diffcolor,obscolorerr2d,axis=0)    
    else:
        diffcolors = obscolor-modelcolor
        av = bestslope(redav,diffcolors,obscolorerr)

    return av

def bestmodelcolor(obscolor,modelcolor,redav,obscolorerr=None):
    """
    Find the best model color given observed color, a grid of
    model colors, and reddening values.
    """

    if obscolorerr is None:
        obscolorerr = np.ones(obscolor.shape,float)

    ngrid = modelcolor.shape[1]
        
    # For each Teff point, calculate the best-fitting extinction value
    # observed color = intrinsic color + Av * reddening
    # Calculate weighted mean slope to get Av
    avest = bestextinction(obscolor,modelcolor,redav,obscolorerr=obscolorerr)
    av = np.maximum(avest,0)   # extinction must be non-negative
    # Now calculate chi-squared
    diffcolors = obscolor.reshape(-1,1) - modelcolor
    obscolorerr2d = obscolorerr.reshape(-1,1) + np.zeros(ngrid).reshape(1,-1)
    chisq = np.sum((diffcolors-redav.reshape(-1,1)*av.reshape(1,-1))**2/obscolorerr2d**2,axis=0)
    minind = np.argmin(chisq)

    return minind,chisq,av


def solve(tab):
    """
    Calculate Teff using colors and color-teff relationships.

    Parameters
    ----------
    tab : table
      Table of photometry values.  It must contain Gaia EDR3
       GMAG, BPMAG, RPMAG and 2MASS JMAG, HMAG, KSMAG and their
       associated uncertainties.

    Returns
    -------
    teff : float
       Effective temperature of the star.
    av : float
       A(V) extinction of the star.

    Example
    -------

    teff,av = solve(tab)

    """

    bands = ['GMAG','BPMAG','RPMAG','JMAG','HMAG','KSMAG']
    
    # Extinction from Parsec CMD website, A(lambda)/A(V)
    # Cardelli+1989 extinction curve
    alav = {}
    alav['GMAG'] = 0.83627
    alav['BPMAG'] = 1.08337
    alav['RPMAG'] = 0.63439
    alav['JMAG'] = 0.28665
    alav['HMAG'] = 0.18082
    alav['KSMAG'] = 0.11675

    # Reddening for our colors
    redav = np.array([alav['BPMAG'],alav['RPMAG'],alav['JMAG'],alav['HMAG'],alav['KSMAG']])-alav['GMAG']

    # Observed magnitudes/colors
    obsmag = np.zeros(len(bands),float)
    for i,b in enumerate(bands):
        import pdb; pdb.set_trace()
        obsmag[i] = tab[b]
    obsmagerr = np.zeros(len(bands),float)
    for i,b in enumerate(bands):
        if b+'_ERR' in tab.colnames:
            obsmagerr[i] = tab[b+'_ERR']
        else:
            obsmagerr[i] = 0.01
    # GMAG is the reference magnitude for all the colors
    obscolor = obsmag[1:]-obsmag[0]
    obscolorerr = np.sqrt(obsmagerr[1:]**2+obsmagerr[0]**2)
    ncolors = len(obscolor)
    
    # Load the intrinsic isochrone color-teff data
    young,old = loadfiles()

    # Now find the best Teff and extinction for the input star
    # X is 5000.0/Teff(K), Y is BAND-GMAG

    # --- Try OLD, MORE EVOLVED BSpline colors ---
    
    # Make grid of color points at various temperatures
    oldxgrid = np.linspace(old['XMIN'],old['XMAX'],50)
    oldcolgrid = np.zeros((ncolors,50),float)
    for i,b in enumerate(['BPMAG','RPMAG','JMAG','HMAG','KSMAG']):
        oldcolgrid[i,:] = old[b](oldxgrid)
    # Find the best model color and extinction
    oldminind,oldchisq,oldav = bestmodelcolor(obscolor,oldcolgrid,redav,obscolorerr=obscolorerr)

    # --- Try YOUNG, MAIN-SEQUENCE BSpline colors ---
    
    # Make grid of color points at various temperatures
    yngxgrid = np.linspace(young['XMIN'],young['XMAX'],50)
    yngcolgrid = np.zeros((ncolors,50),float)
    for i,b in enumerate(['BPMAG','RPMAG','JMAG','HMAG','KSMAG']):
        yngcolgrid[i,:] = young[b](yngxgrid)
    # Find the best model color and extinction
    yngminind,yngchisq,yngav = bestmodelcolor(obscolor,yngcolgrid,redav,obscolorerr=obscolorerr)

    # Which group is better
    if np.min(oldchisq) <= np.min(yngchisq):
        xgrid = oldxgrid
        minind = oldminind
        bspl = old
        btype = 'old'
    else:
        xgrid = yngxgrid
        minind = yngminind
        bspl = young
        btype = 'young'

    # YOUNG ONLY!
    xgrid = yngxgrid
    minind = yngminind
    bspl = young
    btype = 'young'
        
    # Finer grid around that point
    if minind<2:
        minx = xgrid[0]
    else:
        minx = xgrid[minind-2]
    if minind>(len(xgrid)-3):
        maxx = xgrid[-1]
    else:
        maxx = xgrid[minind+2]
    xgrid2 = np.arange(minx,maxx,(xgrid[1]-xgrid[0])*0.1)
    ngrid2 = len(xgrid2)
    colgrid2 = np.zeros((ncolors,ngrid2),float)
    for i,b in enumerate(['BPMAG','RPMAG','JMAG','HMAG','KSMAG']):
        colgrid2[i,:] = bspl[b](xgrid2)

    # Find the best model color and extinction
    minind2,chisq2,av2 = bestmodelcolor(obscolor,colgrid2,redav,obscolorerr=obscolorerr)
  
    # Final values
    bestx = xgrid2[minind2]
    bestav = av2[minind2]
    bestteff = 5000.0/bestx
    
    return bestteff,bestav,btype
 
