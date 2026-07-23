# APOGEE LSF-relevant utilities

import numpy as np
import doppler
from doppler import lsf as doplsf
from scipy.optimize import curve_fit
#from .norm import apStarWave
import math

logw0=4.179
dlogw=6.e-6
nw_apStar=8575
def apStarWave() :
    """ Returns apStar wavelengths
    """
    return 10.**(logw0+np.arange(nw_apStar)*dlogw)
    
#def wave2pix(self,w,extrapolate=True,order=0):
#    if self.wave is None:
#        raise Exception("No wavelength information")
#    if self.ndim==2:
#        # Order is always the second dimension
#        return utils.w2p(self.wave[:,order],w,extrapolate=extrapolate)
#    else:
#        return utils.w2p(self.wave,w,extrapolate=extrapolate)

    


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
        if lsfpars.ndim==1:
            lsfpars = np.atleast_2d(lsfpars).T
        # Stuff in GH coefficients
        ghcoefs = coefs[0:self.nghcoefs]
        lsfpars[self.ghcoef0:self.ghcoef1+1,0] = ghcoefs
        # Stuff in wing coefficients
        wcoefs = coefs[self.nghcoefs:]
        lsfpars[self.wcoef0:self.wcoef1+1,0] = wcoefs
        return lsfpars

        
    def model(self,x,*args,ravel=True):
        """ Create the LSF model."""
        coef = np.array(args)
        lsfpars = self.loadcoefs(coef)
        self.lsf.pars = lsfpars
        print(lsfpars.ravel())        
        # Generate the LSF model
        lsfmodel = self.lsf.anyarray(x,xtype='pixel',order=0,nlsf=15,original=True)
        
        if ravel==True:
            return lsfmodel.ravel()   # must be 1D for 
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

    # fit every 20th column to speed up    
    xx = x[::20]
    yy = comblsf[::20,:].ravel()
    err = sigma[::20,:].ravel()
    
    pars, cov = curve_fit(fitter.model, xx, yy, sigma=err, p0=pinit, maxfev=50000)
    pars2, cov2 = curve_fit(fitter.model, x, comblsf.ravel(), sigma=sigma.ravel(), p0=pinit, maxfev=50000)
    perr = np.sqrt(np.diag(cov))

    # Final LSF model and full parameter array
    lsfmodel = fitter.model(x,*pars,ravel=False)
    lsfpars = fitter.loadcoefs(pars)

    lsfmodel2 = fitter.model(x,*pars2,ravel=False)
    lsfpars2 = fitter.loadcoefs(pars2)

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Qt5Agg')
    plt.ion()

    params=doplsf.unpack_ghlsf_params(fitter.loadcoefs(pars))
    params2=doplsf.unpack_ghlsf_params(fitter.loadcoefs(pars2))

    print('rms=',np.sqrt(np.nanmean((comblsf-lsfmodel)**2)))
    print('rms=',np.sqrt(np.nanmean((comblsf-lsfmodel2)**2)))
    
    import pdb; pdb.set_trace()
    
    return pars, perr, lsfpars, lsfmodel

def getcoefficients(x,xcenter,params):
    nowings = False
    
    # Parse x
    if len(x.shape) == 1:
        x = np.tile(x,(len(xcenter),1))
    # Unpack the LSF parameters
    if type(params) is not dict:
        params = doplsf.unpack_ghlsf_params(params)
    # Get the wing parameters at each x
    wingparams = np.empty((params['nWpar'],len(xcenter)))
    for ii in range(params['nWpar']):
        poly = np.polynomial.Polynomial(params['Wcoefs'][ii])
        wingparams[ii] = poly(xcenter+params['Xoffset'])
    # Get the GH parameters at each x
    ghparams = np.empty((params['Horder']+1,len(xcenter)))

    # note that this is modified/corrected a bit from Bovy's routines based on comparison
    # with LSF from IDL routines, noticeable when wings are non-negligible
    for ii in range(params['Horder']+1):
        poly = np.polynomial.Polynomial(params['GHcoefs'][ii])
        ghparams[ii] = poly(xcenter+params['Xoffset'])
   # for ii in range(params['Horder']+2):
   #     if ii == 1:
   #         ghparams[ii] = 1.
   #     else:
   #         poly = np.polynomial.Polynomial(params['GHcoefs'][ii-(ii > 1)])
   #         ghparams[ii] = poly(xcenter+params['Xoffset'])
   #     # normalization
   #     if ii > 0:
   #         ghparams[ii] /= np.sqrt(2.*np.pi*math.factorial(ii-1))
   #         if not nowings: ghparams[ii] *= (1.-wingparams[0])
            
    return ghparams,wingparams
    

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


    # get the GH coefficients
    lsfpars = specarr[0].lsf.pars[:,1].copy()
    vwave = specarr[0].lsf.wave[:,1].copy()
    xlsf = np.arange(nlsf)-nlsf//2
    xcenter = np.arange(4096)
    ghparams,wingparams = getcoefficients(xlsf,xcenter,lsfpars)
    # (6, 4096)
    
    # Get x-values on the final scale
    wstar = apStarWave()
    xstar = np.arange(8575)
    xint = np.interp(vwave,wstar,xstar)
    # fit new GH coefficients on the new x-array
    params = doplsf.unpack_ghlsf_params(lsfpars)
    porder = params['Porder']
    xoffset = -npix//2
    coefarr = np.zeros((6,3),float)
    for i in range(6):
        coef = np.polyfit(xint+xoffset,ghparams[i,:],porder[i])
        coefarr[i,:porder[i]+1] = coef

    # Make new params
    newparams = params.copy()
    newparams['binsize'] = 1
    newparams['Xoffset'] = xoffset
    newparams['Porder'][0] = 2
    newparams['GHcoefs'] = coefarr
    newlsfpars = doplsf.repack_ghlsf_params(newparams)
    
    
    # Get S/N per visit
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
    #lsfpars = specarr[0].lsf.pars[:,1].copy()
    #lsfpars[1] = npix//2

    # an estimate from a dr12 apStar LSF
    #lsfpars = np.array([ 1.00000000e+00,  4.28700000e+03,  5.00000000e+00,  2.00000000e+00,
    #                     1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
    #                     0.00000000e+00,  2.21925064e+00, -1.22070496e-04,  6.05504495e-09,
    #                     -2.10190968e-02,  1.39464958e-06, -1.70763672e-01, -2.25294252e-07,
    #                     8.01283666e-03, -5.56369138e-07,  3.21678185e-01, -1.30227820e-05,
    #                     -1.99480173e-02,  1.00000000e+00,  2.00000000e+00,  0.00000000e+00,
    #                     0.00000000e+00,  9.99999978e-03,  2.84032046e-02])

    # try with fewer GH components
    # one
    #lsfpars = np.array([ 1.00000000e+00,  4.28700000e+03,  0.00000000e+00,  2.00000000e+00,
    #                     2.21925064e+00, -1.22070496e-04,  6.05504495e-09,
    #                     1.00000000e+00,  2.00000000e+00,  0.00000000e+00,
    #                     0.00000000e+00,  9.99999978e-03,  2.84032046e-02])
    # two (order 2 0)
    #lsfpars = np.array([ 1.00000000e+00,  4.28700000e+03,  1.00000000e+00,  2.00000000e+00,
    #                     0.0,
    #                     2.21925064e+00, -1.22070496e-04,  6.05504495e-09,
    #                     -2.10190968e-02,
    #                     1.00000000e+00,  2.00000000e+00,  0.00000000e+00,
    #                     0.00000000e+00,  9.99999978e-03,  2.84032046e-02])

    # two (order 2 1)
    #lsfpars = np.array([ 1.00000000e+00,  4.28700000e+03,  1.00000000e+00,  2.00000000e+00,
    #                     1.0,
    #                     2.21925064e+00, -1.22070496e-04,  6.05504495e-09,
    #                     -2.10190968e-02,  1.39464958e-06,
    #                     1.00000000e+00,  2.00000000e+00,  0.00000000e+00,
    #                     0.00000000e+00,  9.99999978e-03,  2.84032046e-02])

    # three (order 2 1 1)
    #lsfpars = np.array([ 1.00000000e+00,  4.28700000e+03,  2.00000000e+00,  2.00000000e+00,
    #                     1.0, 1.0,
    #                     2.21925064e+00, -1.22070496e-04,  6.05504495e-09,
    #                     -2.10190968e-02,  1.39464958e-06,
    #                     -1.70763672e-01, -2.25294252e-07,                         
    #                     1.00000000e+00,  2.00000000e+00,  0.00000000e+00,
    #                     0.00000000e+00,  9.99999978e-03,  2.84032046e-02])

    
    
    # binsize=1
    
    #import pdb; pdb.set_trace()
    
    # Now fit LSF Gauss-Hermite parameters to the combined LSF
    coefs,coeferr,finalpars,lsfmodel = fitghpars(wstar,comblsf,newlsfpars)
    
    return comblsf,coefs,coeferr,finalpars,lsfmodel

