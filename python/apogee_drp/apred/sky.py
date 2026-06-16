# Software to use to fit and correct sky lines and telluric absorption
# in the APOGEE spectra

# D. Nidever, May 2022

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import copy
import numpy as np
import yaml
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
from apogee_drp.utils import yanny  #,apload
from astropy.table import Table,hstack,vstack
from astropy.time import Time
from dlnpyutils import utils as dln, robust, coords
import doppler
import thecannon as tc
from scipy.interpolate import BSpline
from numba import njit,types
from thecannon.model import CannonModel

chips = ['a','b','c']

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


def loaddata(planfile,verbose=False):

    # Load the planfile
    if planfile.endswith('.yaml'):
        with open(planfile) as file:
            plandata = yaml.full_load(file)
    else:
        plandata = yanny.yanny(planfile,np=True)

    #load = apload.ApLoad(apred=plandata['apred'],telescope=plandata['telescope'])
    #prefix = load.prefix
    basedir = '/Users/nidever/as5/hge/SEXTANS/'
    prefix = 'as'
    
    # Loop over the exposures
    edata = {}
    frames = []
    for i in range(len(plandata['APEXP'])):
        #base = load.filename('1D',num=plandata['APEXP']['name'][i],chips=True)
        num = '{:08d}'.format(int(plandata['APEXP']['name'][i]))
        base = os.path.join(basedir,prefix+'1D-'+num+'.fits')
        if verbose:
            print('Loading',base)
        exp = {'num':plandata['APEXP']['name'][i],'filename':base,'head':[]}
        for c in chips:
            filename = base.replace('-','-'+c+'-')
            hdu = fits.open(filename)
            exp[c] = {'head':hdu[0].header,'flux':hdu[1].data,'error':hdu[2].data,'mask':hdu[3].data,'wave':hdu[4].data}
            hdu.close()
        exp['head'] = exp['b']['head'].copy()
        exp['date-obs'] = exp['head']['date-obs']
        exp['jd'] = Time(exp['head']['date-obs']).jd
        exp['nread'] = exp['head']['nread']
        exp['exptime'] = exp['head']['exptime']
        exp['airmass'] = exp['head']['armass']
        exp['plate'] = exp['head']['plateid']
        exp['ra'] = exp['head']['ra']
        exp['dec'] = exp['head']['dec']
        exp['ha'] = exp['head']['ha']
        exp['seeing'] = exp['head']['seeing']
        exp['dithpix'] = exp['head']['dithpix']
        exp['mjd'] = plandata['mjd']
        edata[num] = exp
        frames.append(num)
        
    # Load the plate data
    if planfile.endswith('.yaml'):
        platedatafile = planfile.replace('Plan','PlateData').replace('.yaml','.fits')
    else:
        platedatafile = planfile.replace('Plan','PlateData').replace('.par','.fits')
    platedata = Table.read(platedatafile)

    # Put it all together
    data = plandata
    data['frames'] = frames
    data['nframes'] = len(frames)
    data['spec'] = edata
    data['fiberdata'] = platedata
    
    return data

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

#@njit
def xcorr(x,xerr,xmask,y,yerr,ymask,nlag=21,nomean=False,covariance=False):
    """ Cross-correlated x with y """
    # mostly copied from doppler ccorrelate()

    nx = len(x)
    npix = nx
    
    xd = x.copy()
    xderr = xerr.copy()
    yd = y.copy()
    yderr = yerr.copy()
    
    # Remove the means
    if nomean is False:
        xd -= np.nanmean(xd)
        yd -= np.nanmean(yd)
    
    # Set NaNs or Infs to 0.0, mask bad pixels
    fx = (np.isfinite(xd) & np.isfinite(xerr) & (xerr<1e10) & (xmask==0))
    ngdx = np.sum(fx)
    nbdx = np.sum(fx==False)
    if nbdx>0:
        xd[~fx] = 0
        xderr[~fx] = 0
    fy = (np.isfinite(yd) & np.isfinite(yerr) & (yerr<1e10) & (ymask==0))
    ngdy = np.sum(fy)
    nbdy = np.sum((fy==False))
    if nbdy>0:
        yd[~fy] = 0
        yderr[~fy] = 0

    lag = np.arange(nlag)-nlag//2

    cross = np.zeros(nlag,float)
    cross_error = np.zeros(nlag,float)
    num = np.zeros(nlag,int)  # number of "good" points at this lag  
    
    for k in range(nlag):
        # Note the reversal of the variables for negative lags.
        if lag[k]>0:
            cross[k] = np.sum(xd[0:nx-lag[k]] * yd[lag[k]:])
            num[k] = np.sum(fx[0:nx-lag[k]] * fy[lag[k]:]) 
            cross_error[k] = np.sum( (xd[0:nx-lag[k]] * yderr[lag[k]:])**2 ) + np.sum( (xderr[0:nx-lag[k]] * yd[lag[k]:])**2 )
        else:
            cross[k] =  np.sum(yd[0:nx+lag[k]] * xd[-lag[k]:])
            num[k] = np.sum(fy[0:nx+lag[k]] * fx[-lag[k]:])
            cross_error[k] = np.sum( (yderr[0:nx+lag[k]] * xd[-lag[k]:])**2 ) + np.sum( (yd[0:nx+lag[k]] * xderr[-lag[k]:])**2 )

    rmsx = np.sqrt(np.sum((xd*fx)**2))
    if rmsx==0.0: rmsx=1.0
    rmsy = np.sqrt(np.sum((yd*fy)**2))
    if rmsy==0.0: rmsy=1.0

    # Normalize by number of "good" points
    cross *= np.max(num)
    pnum = (num>0)
    cross[pnum] /= num[pnum]  # normalize by number of "good" points
    # Take sqrt to finish adding errors in quadrature
    cross_error = np.sqrt(cross_error)
    # normalize
    cross_error *= np.max(num)
    cross_error[pnum] /= num[pnum]

    # Divide by N for covariance, or divide by variance for correlation.
    if covariance is True:
        cross /= npix
        cross_error /= npix
    else:
        cross /= rmsx*rmsy
        cross_error /= rmsx*rmsy

    return cross,cross_error
    

def ccf_peak_near_zero_with_error(ccf, ccferr=None, lags=None,
                                  max_abs_shift=1.0):
    ccf = np.asarray(ccf, dtype=float)

    if lags is None:
        i0 = len(ccf) // 2
        lag0 = 0.0
        lag_step = 1.0
    else:
        lags = np.asarray(lags, dtype=float)
        i0 = np.argmin(np.abs(lags))
        lag0 = lags[i0]
        lag_step = np.median(np.diff(lags))

    if i0 == 0 or i0 == len(ccf) - 1:
        return np.nan, np.nan, np.nan, False

    y = np.array([ccf[i0 - 1], ccf[i0], ccf[i0 + 1]], dtype=float)

    def delta_from_y(y):
        ym, y0, yp = y
        denom = ym - 2*y0 + yp
        if denom >= 0 or not np.isfinite(denom):
            return np.nan
        return 0.5 * (ym - yp) / denom

    delta_pix = delta_from_y(y)

    if not np.isfinite(delta_pix) or abs(delta_pix) > max_abs_shift:
        return 0.0, ccf[i0], np.nan, False

    ym, y0, yp = y
    peak_value = y0 - 0.25 * (ym - yp) * delta_pix
    shift = lag0 + delta_pix * lag_step

    # Error propagation
    if ccferr is None:
        shift_err = np.nan
    else:
        ccferr = np.asarray(ccferr, dtype=float)
        yerr = np.array([ccferr[i0 - 1], ccferr[i0], ccferr[i0 + 1]])

        deriv = np.zeros(3)
        eps_base = 1e-6

        for j in range(3):
            eps = eps_base * max(abs(y[j]), 1.0)
            y_hi = y.copy()
            y_lo = y.copy()
            y_hi[j] += eps
            y_lo[j] -= eps

            d_hi = delta_from_y(y_hi)
            d_lo = delta_from_y(y_lo)

            if np.isfinite(d_hi) and np.isfinite(d_lo):
                deriv[j] = (d_hi - d_lo) / (2 * eps)
            else:
                deriv[j] = np.nan

        if np.all(np.isfinite(deriv)) and np.all(np.isfinite(yerr)):
            delta_err_pix = np.sqrt(np.sum((deriv * yerr)**2))
            shift_err = delta_err_pix * abs(lag_step)
        else:
            shift_err = np.nan

    return shift, peak_value, shift_err, True

def measdither(data,verbose=False):
    """ Measure dither in exposure stack """
    # first exposures is always the reference
    # measure each chip and fiber separately since there is some rotation
    if verbose:
        print('Measuring dither shift')
    nframes = data['nframes']
    shift = np.zeros((nframes-1,3,300),float)
    shifterr = np.zeros((nframes-1,3,300),float)
    num1 = data['frames'][0]
    frame1 = data['spec'][num1]
    fibers = np.arange(300)
    coefarr = np.zeros((nframes-1,3,2),float)
    for i in range(1,nframes):
        num2 = data['frames'][i]
        frame2 = data['spec'][num2]
        for j,c in enumerate(chips):
            for k in range(300):
                flux1 = frame1[c]['flux'][k,:]
                error1 = frame1[c]['error'][k,:]
                mask1 = frame1[c]['mask'][k,:]
                flux2 = frame2[c]['flux'][k,:]
                error2 = frame2[c]['error'][k,:]
                mask2 = frame2[c]['mask'][k,:]
                ccf,ccferr = xcorr(flux1,error1,mask1,flux2,error2,mask2)
                # Fit the peak
                # should be within 1 pixel of 0
                shft, peak, shft_err, ok = ccf_peak_near_zero_with_error(ccf, ccferr)
                shift[i-1,j,k] = shft
                shifterr[i-1,j,k] = shft_err

            good1, = np.where(np.isfinite(shifterr[i-1,j,:]))
            coef1 = np.polyfit(fibers[good1],shift[i-1,j,good1],1)
            #coef2 = np.polyfit(fibers[good],shift[i-1,j,good],2)
            # outlier rejection
            diff = shift[i-1,j,:]-np.polyval(coef1,fibers)
            sig1 = dln.mad(diff[good1])
            med1 = np.nanmedian(diff[good1])
            good2, = np.where(np.isfinite(shifterr[i-1,j,:]) & (np.abs(diff-med1) < 3*sig1))
            coef2 = np.polyfit(fibers[good2],shift[i-1,j,good2],1)
            coefarr[i-1,j,:] = coef2
            print(i,c,coef2)
            
        #import pdb; pdb.set_trace()

    #import pdb; pdb.set_trace()

    return shift,shifterr,coefarr

def quick_linear_scales(flux, err=None, mask=None, ref=None,
                        nsample=200, qlo=10, qhi=90):
    """
    Fast linear throughput correction:
        scale_e(x) = a_e + b_e * xn

    Returns
    -------
    coef : ndarray
        Shape (nexp, 2), where scale = coef[e,0] + coef[e,1]*xn
    ref : ndarray
        Reference spectrum.
    xn : ndarray
        Normalized pixel coordinate.
    """

    flux = np.asarray(flux, float)
    nexp, npix = flux.shape

    x = np.arange(npix, dtype=float)
    xn = (x - 0.5*(npix-1)) / (0.5*(npix-1))

    if ref is None:
        ref = np.nanmedian(flux, axis=0)

    # sample pixels instead of using all pixels
    idx = np.linspace(0, npix-1, min(nsample, npix)).astype(int)

    coef = np.zeros((nexp, 2), float)

    for e in range(nexp):
        good = np.isfinite(flux[e, idx]) & np.isfinite(ref[idx]) & (ref[idx] != 0)

        if err is not None:
            good &= np.isfinite(err[e, idx]) & (err[e, idx] > 0)

        if mask is not None:
            good &= ~mask[e, idx]

        if np.sum(good) < 10:
            coef[e] = [1.0, 0.0]
            continue

        ratio = flux[e, idx][good] / ref[idx][good]
        xx = xn[idx][good]

        #lo, hi = np.nanpercentile(ratio, [qlo, qhi])
        #use = np.isfinite(ratio) & (ratio > lo) & (ratio < hi)
        
        med = np.nanmedian(ratio)
        mad = np.nanmedian(np.abs(ratio - med))
        sig = 1.4826 * mad

        if np.isfinite(sig) and sig > 0:
            use = np.isfinite(ratio) & (np.abs(ratio - med) < 3*sig)
        else:
            use = np.isfinite(ratio)
        
        if np.sum(use) < 10:
            med = np.nanmedian(ratio)
            coef[e] = [med if np.isfinite(med) else 1.0, 0.0]
            continue

        # Linear least squares: ratio = a + b*xn
        #A = np.vstack([np.ones(np.sum(use)), xx[use]]).T
        #c, _, _, _ = np.linalg.lstsq(A, ratio[use], rcond=None)

        xu = xx[use]
        yu = ratio[use]

        xm = np.mean(xu)
        ym = np.mean(yu)

        dx = xu - xm
        dy = yu - ym

        den = np.sum(dx * dx)
        
        if den > 0:
            b = np.sum(dx * dy) / den
            a = ym - b * xm
            c = [a, b]
        else:
            c = [ym, 0.0]

        if not np.all(np.isfinite(c)) or c[0] <= 0:
            c = [1.0, 0.0]

        coef[e] = c

    return coef, ref, xn

def stack_bin_with_shifts(flux, err, shifts, wave=None,
                          bin_width_pix=4,
                          mask=None,
                          mincount=1,
                          shift_sign=-1,
                          scale_coef=None,
                          nscale_sample=51,
                          restore_scale="median"):

    flux = np.asarray(flux, float)
    err = np.asarray(err, float)
    shifts = np.asarray(shifts, float)
    if wave is not None:
        wave = np.asarray(wave, float)

    nexp, npix = flux.shape
    x = np.arange(npix, dtype=float)
    xn = (x - 0.5*(npix-1)) / (0.5*(npix-1))

    if scale_coef is None:
        scale_coef, ref, xn = quick_linear_scales(
            flux, err=err, mask=mask, ref=flux[0], nsample=nscale_sample
        )
    else:
        scale_coef = np.asarray(scale_coef, float)

    nbin = npix // bin_width_pix

    sumw = np.zeros(nbin)
    sumwf = np.zeros(nbin)
    if wave is not None:
        sumwlam = np.zeros(nbin)
    count = np.zeros(nbin, dtype=int)
    
    for e in range(nexp):

        a, bcoef = scale_coef[e]
        s = a + bcoef * xn

        good = (
            np.isfinite(flux[e]) &
            np.isfinite(err[e]) &
            (err[e] > 0) &
            np.isfinite(s) &
            (s > 0)
        )
        
        if mask is not None:
            good &= ~mask[e]

        if bin_width_pix == 1:
            bbin = np.arange(npix)
        else:
            xs = x + shift_sign * shifts[e]
            bbin = np.floor((xs + 0.5) / bin_width_pix).astype(int)

        keep = good & (bbin >= 0) & (bbin < nbin)

        bb = bbin[keep]
        ivar = 1.0 / err[e, keep]**2
        sk = s[keep]

        sumw += np.bincount(bb, weights=sk*sk*ivar, minlength=nbin)
        sumwf += np.bincount(bb, weights=sk*flux[e, keep]*ivar, minlength=nbin)

        if wave is not None:
            if wave.ndim == 1:
                lam = wave[keep]
            else:
                lam = wave[e, keep]
            sumwlam += np.bincount(bb, weights=sk*sk*ivar*lam, minlength=nbin)
            
        count += np.bincount(bb, minlength=nbin)    

    if wave is not None:
        outwave = np.full(nbin, np.nan)
    outflux = np.full(nbin, np.nan)
    outerr = np.full(nbin, np.inf)

    goodbin = sumw > 0

    if wave is not None:
        outwave[goodbin] = sumwlam[goodbin] / sumw[goodbin]
    outflux[goodbin] = sumwf[goodbin] / sumw[goodbin]
    outerr[goodbin] = np.sqrt(1.0 / sumw[goodbin])

    # Restore average throughput level
    if restore_scale == "median":
        avg_coef = np.nanmedian(scale_coef, axis=0)
        avg_scale_pix = avg_coef[0] + avg_coef[1] * xn
    elif restore_scale == "mean":
        avg_coef = np.nanmean(scale_coef, axis=0)
        avg_scale_pix = avg_coef[0] + avg_coef[1] * xn
    elif restore_scale is None:
        avg_scale_pix = np.ones(npix)
    else:
        avg_scale_pix = np.asarray(restore_scale, float)

    pixbin = np.floor((x + 0.5) / bin_width_pix).astype(int)
    #pixbin = np.floor(x / bin_width_pix).astype(int)
    keep = (pixbin >= 0) & (pixbin < nbin) & np.isfinite(avg_scale_pix)

    sumscale = np.bincount(pixbin[keep],
                           weights=avg_scale_pix[keep],
                           minlength=nbin)
    nscale = np.bincount(pixbin[keep], minlength=nbin)

    avg_scale_bin = np.ones(nbin)
    goodscale = nscale > 0
    avg_scale_bin[goodscale] = sumscale[goodscale] / nscale[goodscale]

    outflux[goodbin] *= avg_scale_bin[goodbin]
    outerr[goodbin] *= avg_scale_bin[goodbin]

    outmask = (~goodbin) | (count < mincount)

    if wave is not None:
        return outwave, outflux, outerr, outmask, count, sumw, scale_coef
    else:
        return outflux, outerr, outmask, count, sumw, scale_coef
        

def stackexposures(data, bins=4, verbose=False):

    if verbose:
        print('Stacking frames')

    nframes = data['nframes']
    fibers = np.arange(300)
    nbinpix = 2048 // bins

    outdata = {'bins': bins}

    for j, c in enumerate(chips):

        edata = {
            'flux':  np.zeros((300, nbinpix), float),
            'error': np.zeros((300, nbinpix), float),
            'mask':  np.zeros((300, nbinpix), bool),
            'wave':  np.zeros((300, nbinpix), float),
            'count': np.zeros((300, nbinpix), int),
            'scale': np.zeros((300, nframes, 2), float)
        }

        # Precompute frame objects once
        frames = [data['spec'][data['frames'][f]][c] for f in range(nframes)]

        # Precompute shifts for all fibers
        shifts_all = np.zeros((nframes, 300), float)
        for f in range(1, nframes):
            shcoef = data['dithcoefarr'][f-1, j, :]
            shifts_all[f, :] = np.polyval(shcoef, fibers)

        # Allocate once, reuse for every fiber
        flux = np.empty((nframes, 2048), float)
        err  = np.empty((nframes, 2048), float)
        mask = np.empty((nframes, 2048), bool)

        for k in range(300):

            for f, frame in enumerate(frames):
                flux[f] = frame['flux'][k]
                err[f]  = frame['error'][k]
                mask[f] = frame['mask'][k] > 0

            # only copy wave for the reference frame, not all frames
            wave0 = frames[0]['wave'][k]
            edata['wave'][k] = np.nanmean(wave0[:nbinpix*bins].reshape(nbinpix, bins), axis=1)

            out = stack_bin_with_shifts(
                flux, err, shifts_all[:, k],
                bin_width_pix=bins,
                mask=mask
            )
            outflux, outerr, outmask, count, sumw, scale_coef = out

            edata['flux'][k]  = outflux
            edata['error'][k] = outerr
            edata['mask'][k]  = outmask
            edata['count'][k] = count
            edata['scale'][k] = scale_coef

        outdata[c] = edata

    return outdata

class Telluric():

    def __init__(self):
        self.species = ['CH4','CO2','H2O']
        self.nspecies = len(self.species)
        self.models = 3*[None]
        self.loaddata()
        self.wave = self.models[0].dispersion.copy()
        self.wrange = [np.min(self.wave),np.max(self.wave)]
        self.npix = len(self.wave)
        self.range = np.array([np.min(self.models[0].training_set_labels,axis=0),
                               np.max(self.models[0].training_set_labels,axis=0)])
        
    def loaddata(self):
        datadir = '/Users/nidever/projects/apogee_drp/data/telluric/'
        for i,c in enumerate(self.species):
            filename = os.path.join(datadir,'telluric_cannon_'+c+'.pkl')
            if os.path.exists(filename)==False:
                print(filename,'not found')
                continue
            self.models[i] = CannonModel.read(filename)
            
    def __call__(self,pars):
        """ scaling parameters. """
        # [airmass, scale] for each of the 3 species
        if len(pars)!=6:
            raise Exception("pars must have 6 elements")
        spec = np.ones(self.npix,float)
        for i in range(3):
            pars1 = pars[2*i:2*i+2]
            if pars1[1] != 0.0:
                if (pars1[0] < self.range[0,0] or pars1[0] > self.range[1,0] or
                    pars1[1] < self.range[0,1] or pars1[1] > self.range[1,1]):
                    raise Exception('label outside range')
                tspec = self.models[i](pars1)
                spec *= tspec
        return spec
            
    def __str__(self):
        """ String representation of the Telluric."""
        return self.__class__.__name__+'({:.2f}<lambda<{:.2f}, Npix={:d}, [{:s}, {:s}, {:s}])'.format(
                                        self.wrange[0],self.wrange[1],self.npix,*self.species)

    def __repr__(self):
        """ String representation of the Telluric."""
        return self.__class__.__name__+'({:.2f}<lambda<{:.2f}, Npix={:d}, [{:s}, {:s}, {:s}])'.format(
                                        self.wrange[0],self.wrange[1],self.npix,*self.species)
            


class Airglow():

    def __init__(self):
        self.fiducial = None
        self.loaddata()
        self.fiducial = self()  # fiducial model
        
    def line(self,wave,pars,doublet=False,dbl_wsep=0.0):
        if doublet==False:
            g = pars[0]*np.exp(-0.5*(wave-pars[1])**2/pars[2]**2)
        else:
            g = pars[0]*np.exp(-0.5*(wave-pars[1]-dbl_wsep*0.5)**2/pars[2]**2)
            g += pars[0]*np.exp(-0.5*(wave-pars[1]+dbl_wsep*0.5)**2/pars[2]**2)
        return g
            
    def __call__(self,scale=1.0,wave=None):
        sigma = 1.0  # lsf sigma in pixels
        if wave is None:
            wave = self.wave.copy()

        
    def loaddata(self):
        datadir = '/Users/nidever/projects/apogee_drp/data/skylines/'
        linelist = ascii.read(datadir+'/airglow.txt')
        for c in linelist.colnames:linelist[c].name=c.lower()
        self.data = linelist
        self.nlines = len(linelist)
        self.wrange = [np.min(self.data['wave']),np.max(self.data['wave'])]
        logw0 = 4.179
        dlogw = 6.e-6
        nw_apStar = 8575
        self.wave = 10.**(logw0+np.arange(nw_apStar)*dlogw)
        # get wavelength mask for each line
        self.mask = np.zeros((len(self.data),2),int)
        sigma = 1.0
        for i in range(len(self.data)):
            ind, = np.where(np.abs(self.wave-self.data['wave'][i]) < 100)
            pars1 = [self.data['emission'][i],self.data['wave'][i],sigma]
            g = self.line(self.wave[ind],pars1,doublet=self.data['doublet'][i],
                          dbl_wsep=self.data['dbl_wsep'][i])
            ind2, = np.where(g>1)
            if len(ind2)==0:
                self.mask[i,:] = -1
                continue
            final_ind = ind[ind2]
            self.mask[i,:] = [final_ind[0],final_ind[-1]]
        
    def __call__(self,scale=1.0,wave=None,sigma=1.0):
        if wave is None and sigma==1.0 and self.fiducial is not None:
            return scale*self.fiducial.copy()
        # if sigma is the same and wave input, then we could just interpolate
        # the fiducial spectrum. might be faster
        
        #sigma = 1.0  # lsf sigma in pixels
        if wave is None:
            wave = self.wave.copy()
        spec = np.zeros(len(wave),float)
        for i in range(len(self.data)):
            lo = self.mask[i,0]
            hi = self.mask[i,1]+1
            pars1 = [self.data['emission'][i],self.data['wave'][i],sigma]
            g = self.line(self.wave[lo:hi],pars1,doublet=self.data['doublet'][i],
                          dbl_wsep=self.data['dbl_wsep'][i])
            spec[lo:hi] += g*scale
        return spec

    def __str__(self):
        """ String representation of the Airglow."""
        return self.__class__.__name__+'({:.2f}<lambda<{:.2f}, Npix={:d})'.format(
                                         self.wrange[0],self.wrange[1],self.nlines)

    def __repr__(self):
        """ String representation of the Airglow."""
        return self.__class__.__name__+'({:.2f}<lambda<{:.2f}, Npix={:d})'.format(
                                         self.wrange[0],self.wrange[1],self.nlines)
    
        
def run(planfile,verbose=True):
    """

    """

    # Load data
    data = loaddata(planfile,verbose=verbose)

    # Measure dither shift
    dithshift,dithshifterr,dithcoefarr = measdither(data,verbose=verbose)
    data['dithshift'] = dithshift
    data['dithshifterr'] = dithshifterr
    data['dithcoefarr'] = dithcoefarr
    
    # Stack/bin
    stackdata = stackexposures(data,verbose=verbose)

    # Initial sky/telluric correction

    # fit stellar model + RV to each fiber

    
    import pdb; pdb.set_trace()

    return data,stackdata
