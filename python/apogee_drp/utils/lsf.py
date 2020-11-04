# APOGEE LSF-relevant utilities

import numpy as np
import doppler
from doppler import lsf as doplsf
from scipy.optimize import curve_fit
#from .norm import apStarWave

logw0=4.179
dlogw=6.e-6
nw_apStar=8575
def apStarWave() :
    """ Returns apStar wavelengths
    """
    return 10.**(logw0+np.arange(nw_apStar)*dlogw)
    

class LsfGHFitter:
    def __init__(self,wave,lsfpars):
        # Make sure lsfpars are at least 2D
        self.lsfpars = lsfpars
        self.lsf = doplsf.GaussHermiteLsf(wave=wave,pars=lsfpars,lsftype='Gauss-Hermite',xtype='Pixels')

        # Useful information for parsing the array of coefficients
        params = doplsf.unpack_ghlsf_params(lsfpars)
        nGHcoefs = np.sum(params['Porder']+1)
        self.params = params
        self.nghcoefs = nGHcoefs
        self.ghcoef0 = params['Horder']+4
        self.ghcoef1 = params['Horder']+4+nGHcoefs-1
        # GHpar = lsfarr[out['Horder']+4:out['Horder']+4+nGHcoefs] #all coeffs
        # wingarr = lsfarr[3+out['Horder']+1+nGHcoefs:]
        # Wpar = wingarr[out['nWpar']+2:]
        self.wcoef0 = 3+params['Horder']+1+nGHcoefs + params['nWpar']+2
        self.wcoef1 = len(lsfpars)-1
        self.nwcoefs = self.wcoef1-self.wcoef0+1
        # Get the initial coefficients
        ghcoefs = lsfpars[self.ghcoef0:self.ghcoef1+1]
        wcoefs = lsfpars[self.wcoef0:self.wcoef1+1]
        coeffs = np.hstack((ghcoefs,wcoefs))
        self.coeffs = coeffs

    def loadcoefs(self,coefs):
        """ Load coefficients into the lsf parameters array."""

        lsfpars = self.lsf.pars.copy()
        # Stuff in GH coefficients
        ghcoefs = coefs[0:self.nghcoefs]
        lsfpars[self.ghcoef0:self.ghcoef1+1,0] = ghcoefs
        # Stuff in wing coefficients
        wcoefs = coefs[self.nghcoefs:]
        lsfpars[self.wcoef0:self.wcoef1+1,0] = wcoefs
        return lsfpars
        
    def model(self,x,*args,flatten=True):
        """ Create the LSF model."""
        coef = np.array(args)
        lsfpars = self.loadcoefs(coef)
        self.lsf.pars = lsfpars
        
        # Generate the LSF model
        lsfmodel = self.lsf.anyarray(x,xtype='pixel',order=0,nlsf=15,original=True)

        if flatten==True:
            return lsfmodel.flatten()   # must be 1D for 
        else:
            return lsfmodel
        

def fitghpars(w,comblsf,initparams):
    """ Get LSF GH coefficients for a 2D LSF array."""

    # Initialize the fitter
    fitter = LsfGHFitter(w,initparams)

    # Downweight pixels where there is no LSF
    lsfsum = np.sum(comblsf,axis=1)
    bd, = np.where(lsfsum < 0.1)
    nbd = len(bd)
    sigma = comblsf*0+1.0
    sigma[bd] = 1e10
    
    # bounds=bounds
    # Data must be flattened to 1D for curve_fit
    pinit = fitter.coeffs.copy()
    npix,nlsf = comblsf.shape
    #w2 = np.tile(w,(nlsf,1)).T
    #pars, cov = curve_fit(fitter.model, w2.flatten(), comblsf.flatten(), sigma=sigma.flatten(), p0=pinit)
    x = np.arange(npix)
    pars, cov = curve_fit(fitter.model, x, comblsf.flatten(), sigma=sigma.flatten(), p0=pinit)
    perr = np.sqrt(np.diag(cov))

    # Final LSF model and full parameter array
    lsfmodel = fitter.model(x,*pars,flatten=False)
    lsfpars = fitter.loadcoefs(pars)

    return pars, perr, lsfpars, lsfmodel


def lsfvisitcomb(visitfiles):
    """
    Get LSF of combined spectrum from multiple apVisit files.
    """

    nspec = len(visitfiles)
    
    # Load spectra into list
    specarr = []
    for vf in visitfiles:
        specarr.append(doppler.read(vf))
        
    # Create LSF array
    wstar = apStarWave() 
    npix = len(wstar)
    nlsf = 15
    lsfarr = np.zeros((npix,nlsf,nspec),np.float64)
    # Loop over spectra
    for s in range(nspec):
        print(str(s+1)+' '+visitfiles[s])
        # Loop over orders
        lsf1 = np.zeros((npix,nlsf),np.float64)
        for o in range(3):
            # Get the wstar pixels covered by this chip/order
            g, = np.where((wstar >= specarr[s].wave[:,o].min()) & (wstar <= specarr[s].wave[:,o].max()))
            print('  '+str(o)+' '+str(len(g)))
            lsf1[g,:] += specarr[s].lsf.anyarray(wstar[g],xtype='wave',order=o,nlsf=15,original=False)
        lsfarr[:,:,s] = lsf1
            
    # Get S/N per star
    snarr = np.zeros(nspec,float)
    for i in range(nspec):
        snarr[i] = specarr[i].snr
    snarr = np.maximum(0.0,snarr)  # must be >=0
        
    # Get S/N weighted LSF
    comblsf = np.zeros((npix,nlsf),np.float64)
    for i in range(nspec):
        comblsf[:,:] += snarr[i]*lsfarr[:,:,i]
    totsnr = np.sum(snarr)
    comblsf /= totsnr  # normalize
    
    # Make sure each pixel is normalized
    comblsf[comblsf<0.] = 0.
    totcomblsf = np.sum(comblsf,axis=1)
    totcomblsf[totcomblsf<0.01] = 1.0           # deal with "missing" pixels
    comblsf /= np.tile(totcomblsf,(nlsf,1)).T
    
    # Initialize GH parameters
    #params = doplsf.unpack_ghlsf_params(specarr[0].lsf.pars[:,1])
    #params['Xoffset'] = npix//2
    #coeffs = np.hstack((params['GHcoefs'].flatten(),params['Wcoefs'].flatten()))
    lsfpars = specarr[0].lsf.pars[:,1]
    lsfpars[1] = npix//2
    
    # Now fit LSF Gauss-Hermite parameters to the combined LSF
    coefs,coeferr,finalpars,lsfmodel = fitghpars(wstar,comblsf,lsfpars)
    
    return comblsf,coefs,coeferr,finalpars,lsfmodel

