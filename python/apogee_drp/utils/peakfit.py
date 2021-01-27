# encoding: utf-8
#
# @Author: Jon Holtzman, some routines based off of IDL routines of David Nidever
# @Date: October 2018
# @Filename: wave.py
# @License: BSD 3-Clause
# @Copyright: Jon Holtzman

# Routines for peak fitting in spectra

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.special import erf, erfc
from scipy.signal import medfilt, convolve, boxcar, argrelextrema
from scipy.ndimage import median_filter
from dlnpyutils import utils as dln

def gauss(x,a,x0,sig) :
    """ Evaluate Gaussian function 
    """
    return a/np.sqrt(2*np.pi)/sig*np.exp(-(x-x0)**2/2./sig**2)

def gaussbin(x,a,x0,sig,yoffset=0.0) :
    """ Evaluate integrated Gaussian function 
    """
    # bin width
    xbin = 1.
    t1 = (x-x0-xbin/2.)/(np.sqrt(2.)*sig)
    t2 = (x-x0+xbin/2.)/(np.sqrt(2.)*sig)
    y = a*np.sqrt(2)*sig * np.sqrt(np.pi)/2 * (erf(t2)-erf(t1)) + yoffset
    return y

def gausspeakfit(spec,pix0=None,estsig=5,sigma=None,func=gaussbin) :
    """ Return integrated-Gaussian centers near input pixel center
    
    Parameters
    ----------
    spec : float array
       Data spectrum array.
    pix0 : float or integer (scalar or array)
       Initial pixel guess.
    estsig : float
       Initial guess for window width=5*estsig (default=5).
    sigma : float array, optional
       Uncertainty array (default=None).
    func : function, optional
       User-supplied function to use to fit (default=gaussbin).

    Returns
    -------
    pars : float array
       Array of best-fitting parameters (height, center, sigma, yoffset)
    perr : float array
       Array of uncertainties of the parameters.

    Examples
    --------
    pars,perr = gausspeakfit(spec,10,sigma=specerr)

    """
    # No initial guess input, use maximum value
    if pix0 is None:
        pix0 = spec.argmax()

    medspec = np.median(spec)
    sigspec = dln.mad(spec-medspec)

    dx = 1
    x = np.arange(len(spec))
    xwid = 5
    xlo0 = pix0-xwid
    xhi0 = pix0+xwid+1
    xx0 = x[xlo0:xhi0]
    
    # Get quantitative estimates of height, center, sigma
    flux = spec[xlo0:xhi0]-medspec
    flux -= np.median(flux)            # put the median at zero
    flux = np.maximum(0,flux)          # don't want negative pixels
    ht0 = np.max(flux)
    totflux = np.sum(flux)
    #  Gaussian area is A = ht*wid*sqrt(2*pi)
    sigma0 = np.maximum( (totflux*dx)/(ht0*np.sqrt(2*np.pi)) , 0.01)
    cen0 = np.sum(flux*xx0)/totflux
    cen0 = np.minimum(np.maximum((pix0-dx*0.7), cen0), (pix0+dx*0.7))   # constrain the center
    # Use linear-least squares to calculate height and sigma
    psf1 = np.exp(-0.5*(xx0-cen0)**2/sigma0**2)          # normalized Gaussian
    wtht1 = np.sum(flux*psf1)/np.sum(psf1*psf1)          # linear least squares
    # Second iteration
    sigma1 = (totflux*dx)/(wtht1*np.sqrt(2*np.pi))
    psf2 = np.exp(-0.5*(xx0-cen0)**2/sigma1**2)          # normalized Gaussian
    wtht2 = np.sum(flux*psf2)/np.sum(psf2*psf2)

    # Now get more pixels to fit if necessary
    npixwide = int(np.maximum(np.ceil((2*sigma1)/dx), 5))
    xlo = int(np.round(cen0))-npixwide
    xhi = int(np.round(cen0))+npixwide+1
    xx = x[xlo:xhi]
    y = spec[xlo:xhi]
    if sigma is not None:
        yerr = sigma[xlo:xhi]
    else:
        yerr = y*0+1

    # Bounds
    initpar = [wtht2, cen0, sigma1, medspec]
    lbounds = [0.5*ht0, initpar[1]-1, 0.2, initpar[3]-np.maximum(3*sigspec,0.3*np.abs(initpar[3]))]
    ubounds = [3.0*ht0, initpar[1]+1, 5.0, initpar[3]+np.maximum(3*sigspec,0.3*np.abs(initpar[3]))]
    bounds = (lbounds,ubounds)
    # Sometimes curve_fit hits the maximum number fo function evaluations and crashes
    #try:
    #    pars,cov = curve_fit(func,xx,y,p0=initpar,sigma=yerr,bounds=bounds,maxfev=1000)            
    #    perr = np.sqrt(np.diag(cov))
    #except:
    #    return None,None
    
    return pars,perr


def test() :
    """ test routine for peakfit """
    spec = np.zeros([200])
    specbin = np.zeros([200])
    spec[50:151] = gauss(np.arange(50,151),100.,99.5,0.78)
    specbin[50:151] = gaussbin(np.arange(50,151),100.,99.5,0.78)
    plt.plot(spec)
    plt.plot(specbin)
    plt.show()
    plt.draw()
    pdb.set_trace()
    peakfit(spec,[95,99,102,107])

def peakfit(spec,sigma=None,pix0=None):
    """
    Find lines in a spectrum fit Gaussians to them.

    Parameters
    ----------
    spec : float array
       Data spectrum array.
    sigma : float array, optional
       Uncertainty array (default=None).
    pix0 : float or integer (scalar or array), optional
       Initial pixel guess.

    Returns
    -------
    pars : numpy structured array
       Table of best-fitting parameters and uncertainties.

    Examples
    --------
    pars = peakfit(spec,sigma=specerr)

    """    

    if np.ndim(spec)>1:
        raise ValueError('Spec must be 1-D')

    # X array
    npix = len(spec)
    x = np.arange(npix)

    smspec = median_filter(spec,101,mode='nearest')
    sigspec = dln.mad(spec-smspec)

    # Find the peaks
    if pix0 is None:
        maxind, = argrelextrema(spec-smspec, np.greater)  # maxima
        # sigma cut on the flux
        gd, = np.where((spec-smspec)[maxind] > 4*sigspec)
        if len(gd)==0:
            print('No peaks found')
            return
        pix0 = maxind[gd]
    pix0 = np.atleast_1d(pix0)
    npeaks = len(pix0)

    # Initialize the output table
    dtype = np.dtype([('num',int),('pix0',int),('pars',np.float64,4),('perr',np.float64,4),('success',bool)])
    out = np.zeros(npeaks,dtype=dtype)
    out['num'] = np.arange(npeaks)
    out['pix0'] = pix0

    # Initialize the residuals spectrum
    resid = spec.copy()

    # Loop over the peaks and fit them
    for i in range(npeaks):
        # Run gausspeakfit() on the residual
        pars,perr = gausspeakfit(resid,pix0=pix0[i],sigma=sigma)
        if pars is not None:
            # Get model and subtract from residuals
            xlo = np.maximum(0,int(pars[1]-5*pars[2]))
            xhi = np.minimum(npix,int(pars[1]+5*pars[2]))
            peakmodel = gaussbin(x[xlo:xhi],pars[0],pars[1],pars[2])  # leave yoffset in
            resid[xlo:xhi] -= peakmodel

            # Stuff results in output table
            out['pars'][i] = pars
            out['perr'][i] = perr
            out['success'][i] = True
        else:
            out['pars'][i] = np.nan
            out['perr'][i] = np.nan

    return out


def findlines(frame,rows,waves,lines,out=None,verbose=False,estsig=2) :
    """ Determine positions of lines from input file in input frame for specified rows

    Args:
        frame (dict) : dictionary with ['a','b','c'] keys for each chip containing HDULists with flux, error, and mask
        rows (list) : list of rows to look for lines in
        waves (list)  : list of wavelength arrays to be used to get initial pixel guess for input lines
        lines :  table with desired lines, must have at least CHIPNUM and WAVE tags
        out= (str) : optional name of output ASCII file for lines (default=None)

    Returns :
        structure with identified lines, with tags chip, row, wave, peak, pixrel, dpixel, frameid
    """
    num = int(os.path.basename(frame['a'][0].header['FILENAME']).split('-')[1])
    nlines = len(lines)
    nrows = len(rows)
    linestr = np.zeros(nlines*nrows,dtype=[
                       ('chip','i4'), ('row','i4'), ('wave','f4'), ('peak','f4'), ('pixel','f4'),
                       ('dpixel','f4'), ('wave_found','f4'), ('frameid','i4')
                       ])
    nline=0
    for ichip,chip in enumerate(['a','b','c']) :
        # Use median offset of previous row for starting guess
        # Add a dummy first row to get starting guess offset for the first row
        dpixel_median = 0.
        for irow,row in enumerate(np.append([rows[0]],rows)) :
            # subtract off median-filtered spectrum to remove background
            medspec = frame[chip][1].data[row,:]-medfilt(frame[chip][1].data[row,:],101)
            j = np.where(lines['CHIPNUM'] == ichip+1)[0]
            dpixel=[]
            # for dummy row, open up the search window by a factor of two
            if irow == 0 : estsig0=2*estsig
            else : estsig0=estsig
            for iline in j :
                wave = lines['WAVE'][iline]
                try :
                    pix0 = wave2pix(wave,waves[chip][row,:])+dpixel_median
                    # find peak in median-filtered subtracted spectrum
                    pars = peakfit(medspec,pix0,estsig=estsig0,
                                   sigma=frame[chip][2].data[row,:],mask=frame[chip][3].data[row,:])
                    if lines['USEWAVE'][iline] == 1 : dpixel.append(pars[1]-pix0)
                    if irow > 0 :
                        linestr['chip'][nline] = ichip+1
                        linestr['row'][nline] = row
                        linestr['wave'][nline] = wave
                        linestr['peak'][nline] = pars[0]
                        linestr['pixel'][nline] = pars[1]
                        linestr['dpixel'][nline] = pars[1]-pix0
                        linestr['wave_found'][nline] = pix2wave(pars[1],waves[chip][row,:])
                        linestr['frameid'][nline] = num
                        nline+=1
                    if out is not None :
                        out.write('{:5d}{:5d}{:12.3f}{:12.3f}{:12.3f}{:12.3f}{:12d}\n'.format(
                                  ichip+1,row,wave,pars[0],pars[1],pars[1]-pix0,num))
                    elif verbose :
                        print('{:5d}{:5d}{:12.3f}{:12.3f}{:12.3f}{:12.3f}{:12d}'.format(
                              ichip+1,row,wave,pars[0],pars[1],pars[1]-pix0,num))
                except :
                    if verbose : print('failed: ',num,row,chip,wave)
            if len(dpixel) > 10 : dpixel_median = np.median(np.array(dpixel))
            if verbose: print('median offset: ',row,chip,dpixel_median)

    return linestr[0:nline]

