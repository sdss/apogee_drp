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
from dlnpyutils import utils as dln

def gauss(x,a,x0,sig) :
    """ Evaluate Gaussian function 
    """
    return a/np.sqrt(2*np.pi)/sig*np.exp(-(x-x0)**2/2./sig**2)

def myerf(t) :
    """ Evaluate function that integrates Gaussian from -inf to t
    """
    neg = np.where(t<0.)[0]
    pos = np.where(t>=0.)[0]
    out = t*0.
    out[neg] = erfc(abs(t[neg]))/2.
    out[pos] = 0.5+erf(abs(t[pos]))/2.
    return out

def gaussbin(x,a,x0,sig,yoffset=0.0) :
    """ Evaluate integrated Gaussian function 
    """
    # bin width
    xbin = 1.
    t1 = (x -x0-xbin/2.)/np.sqrt(2.)/sig
    t2 = (x-x0+xbin/2.)/np.sqrt(2.)/sig
    y = (myerf(t2)-myerf(t1))/xbin + yoffset
    return a*y

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

    x = np.arange(len(spec))
    cen = int(round(pix0))
    sig = estsig
    for iter in range(3) :
        # window width to search
        xwid = int(round(5*sig))
        if xwid < 3 : xwid=3
        y = spec[cen-xwid:cen+xwid+1]
        yerr = sigma[cen-xwid:cen+xwid+1]
        x0 = y.argmax()+(cen-xwid)
        peak = y.max()
        sig = np.sqrt(y.sum()**2/peak**2/(2*np.pi))
        sig = np.maximum(0.51,sig)
        yoffset = np.min(y)
        lbounds = [0.0,cen-xwid-2,0.5,np.min(y)-(np.max(y)-np.min(y))]
        ubounds = [1.5*(np.max(y)-np.min(y)),cen+xwid+2,len(y)/2.0,np.max(y)+(np.max(y)-np.min(y))]
        bounds = (lbounds,ubounds)
        initpar = [peak/sig/np.sqrt(2*np.pi),x0,sig,yoffset]
        print(str(iter)+' '+str(initpar))
        print(lbounds)
        print(ubounds)
        # Sometimes curve_fit htis the maximum number fo function evaluations and crashes
        try:
            pars,cov = curve_fit(func,x[cen-xwid:cen+xwid+1],y,p0=initpar,sigma=yerr,bounds=bounds,maxfev=1000)
            perr = np.sqrt(np.diag(cov))
        except:
            return None,None
        # iterate unless new array range is the same
        if int(round(5*pars[2])) == xwid and int(round(pars[1])) == cen : break
        cen = int(round(pars[1]))
        sig = pars[2]

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
    estsig : float
       Initial guess for window width=5*estsig (default=5).

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


    # Find the peaks
    if pix0 is None:
        maxind, = argrelextrema(spec, np.greater)  # maxima
        # sigma cut on the flux
        sigspec = dln.mad(spec)
        medspec = np.median(spec)
        gd, = np.where(spec[maxind] > (3*sigspec+medspec))
        if len(gd)==0:
            print('No peaks found')
            return
        pix0 = maxind[gd]
    pix0 = np.atleast_1d(pix0)
    npeaks = len(pix0)

    # Initialize the output table
    dtype = np.dtype([('num',int),('pix0',int),('pars',np.float64,4),('perr',np.float64,4)])
    out = np.zeros(npeaks,dtype=dtype)
    out['num'] = np.arange(npeaks)
    out['pix0'] = pix0

    # Initialize the residuals spectrum
    resid = spec.copy()

    # Loop over the peaks and fit them
    for i in range(npeaks):
        print(i)
        # Run gausspeakfit() on the residual
        pars,perr = gausspeakfit(resid,pix0=pix0[i],sigma=sigma)
        if pars is not None:
            # Get model and subtract from residuals
            xlo = np.minimum(0,int(pars[1]-3*pars[2]))
            xhi = np.maximum(npix,int(pars[1]+3*pars[2]))
            peakmodel = gaussbin(x[xlo:xhi],pars[0],pars[1],pars[2])  # leave yoffset in
            resid[xlo:xhi] -= peakmodel

            # Stuff results in output table
            out['pars'][i] = pars
            out['perr'][i] = perr
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

