# APOGEE LSF-relevant utilities

import numpy as np
import doppler
from doppler import lsf as doplsf
from .norm import apStarWave


class lsfghfitter:
    def __init__(self,lsfpars):
        self.lsfpars = lsfpars
        self.lsf = doplsf.GaussHermiteLsf(lsfpars=lsfpars,lsftype='Gauss-Hermite',lsfxtype='Pixels')

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
        self.wcoef0 = params['Horder']+4+nGHcoefs + 3+param['Horder']+1+nGHcoefs
        self.wcoef1 = len(lsfpars)-1
        self.nwcoefs = self.wcoef1-self.wcoef0+1        
        
    def model(self,x,*args):
        """ Create the LSF model."""
        lsfpars = self.lsf.pars.copy()
        coef = np.array(args)
        
        # Stuff in GH coefficients
        ghcoefs = coef[0:self.nghcoefs]
        lsfpars[self.ghcoef0:self.ghcoef1+1] = ghcoefs
        # Stuff in wing coefficients
        wcoefs = coef[self.nghcoefs:]
        lsfpars[self.wcoef0:self.wcoef1+1] = wcoefs
        self.lsf.pars = lsfpars
        # Generate the LSF model
        lsfmodel = self.lsf.anyarray(x,original=False)
        
        return lsf
        

def fitghpars(w,comblsf,initparams):
    """ Get LSF GH coefficients for a 2D LSF array."""
    
    fitter = lsfghfitter(initparams)

    # Downweight pixels where there is no LSF
    lsfsum = np.sum(comblsf,axis=1)
    bd, = np.where(lsfsum < 0.1)
    nbd = len(bd)
    sigma = comblsf*0+1.0
    sigma[bd] = 1e10

    # bounds=bounds
    pars, cov = curve_fit(fitter.model, w, comblsf, sigma=sigma, p0=pinit)
    perr = np.sqrt(np.diag(cov))
    return coef, perr


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
        # Loop over orders
        lsf1 = np.zeros((npix,lsf),np.float64)
        for o in range(3):
            g, = np.where((wstar >= specarr[s].wave[:,o].min()) & (wstar <= specarr[s].wave[:,o].max()))
            lsf1[g,:] += specarr[o].lsf.anyarray(wstar[g],xtype='wave',order=o,nlsf=15,original=False)

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
    comblsf /= np.tile(np.sum(comblsf,axis=1),(nlsf,1)).T

    # Initialize GH parameters
    #params = doplsf.unpack_ghlsf_params(specarr[0].lsf.pars[:,1])
    #params['Xoffset'] = npix//2
    #coeffs = np.hstack((params['GHcoefs'].flatten(),params['Wcoefs'].flatten()))
    lsfpars = specarr[0].lsf.pars[:,1]
    lspars[1] = npix//2
    
    # Now fit LSF Gauss-Hermite parameters to the combined LSF
    pars,cov = fitghpars(wstar,comblsf,lsfpars)

    

