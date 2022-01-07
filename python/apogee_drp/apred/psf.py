#!/usr/bin/env python

import copy
import numpy as np
import os
import shutil
import time
from dlnpyutils import utils as dln, bindata
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
import statsmodels.api as sm
from apogee_drp.utils import peakfit, mmm
from numba import njit

WARNMASK = -16640
BADMASK = 16639
BADERR = 1.00000e+10
maskval = {'NOT_ENOUGH_PSF': 16384}


#####  EMPIRICAL PSF MODEL CLASS #######

def leaky_relu(z):
    """ This is the activation function used by default in all our neural networks. """
    return z*(z > 0) + 0.01*z*(z < 0)

class PSF(object):

    def __init__(self,coeffs,nxgrid=20,nygrid=50):
        # coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max, y)
        self._coeffs = coeffs
        self.xmin = coeffs['xmin']
        self.xmax = coeffs['xmax']
        self.y = coeffs['y']
        self.npix = len(self.y)
        self._xgrid = None
        self._nxgrid = nxgrid
        self._ygrid = None        
        self._nygrid = nygrid
        self._grid = None

    def __str__(self):
        """ String representation of the PSF."""
        return self.__class__.__name__+'(%.1f<X<%.1f, %.1f<X<%.1f, Npix=%d)' % \
                                        (self.xmin[0],self.xmax[0],self.xmin[1],self.xmax[1],self.npix)

    def __repr__(self):
        """ String representation of the PSF."""
        return self.__class__.__name__+'(%.1f<X<%.1f, %.1f<X<%.1f, Npix=%d)' % \
                                        (self.xmin[0],self.xmax[0],self.xmin[1],self.xmax[1],self.npix)
    
    def __call__(self,labels,y=None,ycen=None):
        """  Make the PSF. """

        if labels[0]<0 or labels[0]>2047 or labels[1]<0 or labels[1]>2047:
            raise ValueError('X/Y must be between 0 and 2047')
            
        # Interpolate in the grid
        profile = self.gridinterp(labels)

        # Pixel values input, shift and interpolate
        if y is not None:
            if ycen is None:
                ycen = labels[1]
            yfine = np.arange(self.npix)
            fullprofile = profile
            profile = interp1d(self.y,fullprofile,kind='quadratic',bounds_error=False,fill_value=np.nan)(y-ycen)
            # Set values beyond the range to 0.0
            if np.sum(~np.isfinite(profile))>0:
                profile[~np.isfinite(profile)] = 0.0
                
        return profile
        
    def scaled_labels(self,labels):
        """ Scale the labels."""
        if self.xmin is None or self.xmax is None:
            raise ValueError('No label scaling informationl')
        slabels = (labels-self.xmin)/(self.xmax-self.xmin) - 0.5   # scale the labels
        return slabels
        
    def model(self,inlabels):
        """ Make a brand-new full profile model with input labels."""
        if inlabels[0]<0 or inlabels[0]>2047 or inlabels[1]<0 or inlabels[1]>2047:
            raise ValueError('X/Y must be between 0 and 2047')
        labels = self.scaled_labels(inlabels) # scale the labels
        # We input the scaled stellar labels (not in the original unit).
        # Each label ranges from -0.5 to 0.5
        w_array_0 = self._coeffs['weight0']
        b_array_0 = self._coeffs['bias0']
        w_array_1 = self._coeffs['weight2']
        b_array_1 = self._coeffs['bias2']
        w_array_2 = self._coeffs['weight4']
        b_array_2 = self._coeffs['bias4']                
        inside = np.einsum('ij,j->i', w_array_0, labels) + b_array_0
        outside = np.einsum('ij,j->i', w_array_1, leaky_relu(inside)) + b_array_1
        m = np.einsum('ij,j->i', w_array_2, leaky_relu(outside)) + b_array_2
        # This is the log of the model
        m = 10**m
        return m

    def gridinterp(self,labels):
        """ Interpolate model in the grid."""

        if labels[0]<0 or labels[0]>2047 or labels[1]<0 or labels[1]>2047:
            raise ValueError('X/Y must be between 0 and 2047')
        
        if self._grid is None:
            self.mkgrid()

        xind = np.searchsorted(self._xgrid,labels[0])
        yind = np.searchsorted(self._ygrid,labels[1])        
        
        # Find the closest points on the grid
        #------------------------------------
        # -- At corners, use corner values --
        # bottom left
        if labels[0] < self.xmin[0] and labels[1] < self.xmin[1]:
            return self._grid[0,0,:]
        # top left
        if labels[0] < self.xmin[0] and labels[1] > self.xmax[1]:
            return self._grid[0,-1,:]
        # bottom right
        if labels[0] > self.xmax[0] and labels[1] < self.xmin[1]:
            return self._grid[-1,0,:]
        # top right
        if labels[0] > self.xmax[0] and labels[1] > self.xmax[1]:
            return self._grid[-1,-1,:]

        # -- Edges, use two points --
        # linearly interpolate so it's smooth        
        # left
        if labels[0] < self.xmin[0]:
            yind1 = yind-1
            yind2 = yind
            wt = (labels[1]-self._ygrid[yind1])/(self._ygrid[yind2]-self._ygrid[yind1])
            profile = (1-wt)*self._grid[0,yind1,:] + wt*self._grid[0,yind2,:]
            return profile
        # right
        if labels[0] > self.xmax[0]:
            yind1 = yind-1
            yind2 = yind
            wt = (labels[1]-self._ygrid[yind1])/(self._ygrid[yind2]-self._ygrid[yind1])
            profile = (1-wt)*self._grid[-1,yind1,:] + wt*self._grid[-1,yind2,:]
            return profile
        # bottom
        if labels[1] < self.xmin[1]:
            xind1 = xind-1
            xind2 = xind
            wt = (labels[0]-self._xgrid[xind1])/(self._xgrid[xind2]-self._xgrid[xind1])
            profile = (1-wt)*self._grid[xind1,0,:] + wt*self._grid[xind2,0,:]
            return profile
        # top
        if labels[1] > self.xmax[1]:
            xind1 = xind-1
            xind2 = xind
            wt = (labels[0]-self._xgrid[xind1])/(self._xgrid[xind2]-self._xgrid[xind1])
            profile = (1-wt)*self._grid[xind1,-1,:] + wt*self._grid[xind2,-1,:]
            return profile
            
        # -- In the middle --
        # linearly interpolate so it's smooth
        xind1 = xind-1
        xind2 = xind
        yind1 = yind-1
        yind2 = yind
        wtdx = (self._xgrid[xind2]-self._xgrid[xind1])
        wtdy = (self._ygrid[yind2]-self._ygrid[yind1])        
        profile = np.zeros(self.npix,float)
        totwt = 0.0
        for xind in [xind1,xind2]:
            for yind in [yind1,yind2]:
                wt = (labels[0]-self._xgrid[xind])*(labels[1]-self._ygrid[yind])/(wtdx*wtdy)
                totwt += wt
                profile += wt*self._grid[xind,yind]
        profile /= totwt
                
        return profile
        
    def mkgrid(self,nx=None,ny=None):
        """ Make a grid of models to be used later."""

        # Default values
        if nx is None and self._nxgrid is not None:
            nx = self._nxgrid
        if ny is None and self._nygrid is not None:
            ny = self._nygrid 
        if nx is None:
            nx = 20
        if ny is None:
            ny = 50

        # Limits and steps
        npix = 2048
        dx = (self.xmax[0]-self.xmin[0])/nx
        dy = (self.xmax[1]-self.xmin[1])/ny        
        x0 = self.xmin[0]
        y0 = self.xmin[1]
        
        # Loop over X and Y points and fill in the 3D grid
        xgrid = np.linspace(self.xmin[0],self.xmax[0],nx)
        ygrid = np.linspace(self.xmin[1],self.xmax[1],ny)
        grid = np.zeros((nx,ny,self.npix),float)
        for i,x1 in enumerate(xgrid):
            for j,y1 in enumerate(ygrid):
                m1 = self.model([x1,y1])
                grid[i,j,:] = m1

        # Save the information
        self._xgrid = xgrid
        self._nxgrid = nx
        self._ygrid = ygrid
        self._nygrid = ny 
        self._grid = grid
    
    @classmethod
    def read(cls,infile):
        # Load the file and return a PSF object
        coeffs = {}
        hdu = fits.open(infile)
        for i in range(9):
            coeffs[hdu[i].header['type']] = hdu[i].data
        # coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max, y)
        return PSF(coeffs)

    
#####  GET EMPIRICAL PSF #######
    

def getprofdata(fibs,cols,hdulist,fiber2hdu):
    """ Get the apEPSF profile data for a range of fibers and columns."""

    # Fiber range
    if len(fibs)==2:
        nfibers = fibs[1]-fibs[0]+1
        fibers = np.arange(nfibers)+fibs[0]
    else:
        fibers = fibs
    nfibers = len(fibers)
    ncols = cols[1]-cols[0]
    mncol = (cols[0]+cols[1])*0.5
        
    # Initialize final arrays
    # dy, flux, X and Y
    data = np.zeros((nfibers*ncols*30,4),float)
    cnt = 0
        
    # Fiber loop    
    for i,f in enumerate(fibers):
        hind = fiber2hdu.get(f)
        if hind is not None:
            psfcat = hdulist[hind].data
            psfim = psfcat['IMG'][0,:,:]
            mny = np.mean(psfcat['CENT'])
            subim = psfim[:,cols[0]:cols[1]]
            ny,nx = subim.shape
            y1 = np.arange(ny)
            ymn = np.sum(y1.reshape(-1,1)*subim,axis=0)/np.sum(subim,axis=0)
            dy = y1.reshape(-1,1)-ymn.reshape(1,-1)
            col2d = (np.arange(ncols)+cols[0]).reshape(-1,1) + np.zeros(ny).reshape(1,-1)
            y2d = psfcat['CENT'][0,cols[0]:cols[1]].reshape(-1,1) + np.zeros(ny).reshape(1,-1)
            data[cnt:cnt+ncols*ny,0] = dy.ravel()      # dy        
            data[cnt:cnt+ncols*ny,1] = subim.ravel()   # flux
            data[cnt:cnt+ncols*ny,2] = col2d.ravel()   # X
            data[cnt:cnt+ncols*ny,3] = y2d.ravel()     # Y
            cnt += ncols*ny
    # Trim data
    data = data[0:cnt,:]
    # Trim out zero flux values
    gd, = np.where(data[:,1]>0)
    data = data[gd,:]

    return data


def avgprofile(fibs,cols,hdulist,fiber2hdu):
    """ Get the average profile for a range of fibers and columns."""
    # composite profile
    
    # Get profile data
    data = getprofdata(fibs,cols,hdulist,fiber2hdu)

    # Do binning first
    xr = [-7.0,7.0]
    binsize = 0.1
    nbins = int(np.ceil((xr[1]-xr[0])/binsize)+1)
    bins = np.linspace(xr[0],xr[1],nbins)
    ybin, bin_edges, binnumber = bindata.binned_statistic(data[:,0],data[:,1],statistic='percentile',
                                                          percentile=50,bins=bins)
    xbin = bin_edges[0:-1]+0.5*binsize

    # Use Gaussian smoothing
    gd, = np.where(np.isfinite(ybin) & (ybin>0))
    temp = ybin.copy()
    temp[~np.isfinite(ybin) | (ybin<=0)] = np.nan
    ybinsm = dln.gsmooth(temp,5)
    bad = ~np.isfinite(ybinsm)
    if np.sum(bad)>0:
        ybinsm[bad] = interp1d(xbin[~bad],ybinsm[~bad])(bad)

    # Make sure it's normalized
    ybinsm /= np.sum(ybinsm)*binsize
        
    # Use LOWESS to generate empirical template
    # it will use closest frac*N data points to a given point to estimate the smooth version
    # want at least 5 points
    #gd, = np.where(np.isfinite(ybin) & (ybin>0))
    ##lowess = sm.nonparametric.lowess(ybin[gd],xbin[gd], frac=0.05)    
    ## interpolate onto fine grid, leave some overhang
    #gdl, = np.where(np.isfinite(lowess[:,1]) & (lowess[:,1]>0))
    #lowint = interp1d(lowess[gdl,0],lowess[gdl,1],kind='quadratic',bounds_error=None,
    #                  fill_value="extrapolate")(xbin)
    #return data, xbin, ybin, lowess, lowint
    
    return data, xbin, ybin, ybinsm


def getprofilegrid(psffile,sparsefile,nfbin=5,ncbin=200):
    """ Get all of the profiles across the detector."""

    allim,head = fits.getdata(sparsefile,0,header=True)
    sim = allim[1,:,:]
    psfhdu = fits.open(psffile)

    # Get fiber numbers for each hdu
    fiber2hdu = {}
    fibernum = np.zeros(len(psfhdu)-1,int)
    for i in range(len(psfhdu)-1):
        fibernum[i] = psfhdu[i+1].data['FIBER']
        fiber2hdu[psfhdu[i+1].data[0]['FIBER']] = i+1
    
    fibers = np.arange(0,300,nfbin)
    columns = np.arange(10,2000,ncbin)

    # Get sparse data
    
    #data = np.zeros((len(fibers),len(columns),700),float)
    data = []
    mnx = np.zeros((len(columns),len(fibers)),float)
    mny = np.zeros((len(columns),len(fibers)),float)
    profiles = np.zeros((len(columns),len(fibers),300),float)
    binsize = 0.1
    xx = np.arange(300)*binsize-14.95
    
    # Column loop
    for i,c in enumerate(columns):

        # Get sparse profile
        sflux = np.zeros(2048,float)
        sflux[4:2044] = np.nanmedian(sim[4:2044,c:c+ncbin],axis=1)
        # Find the peaks
        maxind, = argrelextrema(sflux, np.greater)
        gd, = np.where(sflux[maxind] > 0.1*np.max(sflux))
        peaks = maxind[gd]
        linestr0 = peakfit.peakfit(sflux,pix0=peaks)
        # Distances to neighbors
        ldiff = linestr0['pars'][:,1]-np.hstack((0,linestr0['pars'][0:-1,1]))
        rdiff = np.hstack((linestr0['pars'][1:,1],2048))-linestr0['pars'][:,1]
        gd, = np.where((ldiff >= 22) & (rdiff >= 22))
        ngd = len(gd)
        # 15
        linestr = linestr0[gd]
        
        # Fiber loop
        for j,f in enumerate(fibers):

            print(f,c)
            #data1, xbin,ybin,lowess,ylowess = avgprofile([f,f+nfbin],[c,c+ncbin],psfhdu,fiber2hdu)
            data1, xbin,ybin,ybinsm = avgprofile([f,f+nfbin],[c,c+ncbin],psfhdu,fiber2hdu)            
            
            # Get closest sparse fiber
            ytracearr = []
            for k in np.arange(f,f+nfbin):
                if fiber2hdu.get(k) is not None:
                    psfcat = psfhdu[fiber2hdu[k]].data
                    ytracearr.append(np.median(psfcat['CENT']))
            ytrace = np.median(np.array(ytracearr))
            diff = linestr['pars'][:,1]-ytrace
            bestind = np.argmin(np.abs(diff))
            linestr1 = linestr[bestind]
            ycensparse = linestr1['pars'][1]
            dysparse = np.arange(31).astype(float)-15
            fluxsparse = sflux[int(round(ycensparse))-15:int(round(ycensparse))+16]
            fluxsparse /= np.sum(fluxsparse)   # normalize
            # replace very low values with point on opposite side
            bad, = np.where(fluxsparse<1e-5)
            if len(bad)>0:
                good = len(fluxsparse)-bad-1
                fluxsparse[bad] = fluxsparse[good]
                print('fixing fluxsparse edge bad value')
                #import matplotlib.pyplot as plt
                #import pdb; pdb.set_trace()
            fluxsparse /= np.sum(fluxsparse)   # normalize again          
            ymnsparse = np.sum(dysparse*fluxsparse)/np.sum(fluxsparse)
            dysparse -= ymnsparse


            #import matplotlib.pyplot as plt
            #plt.clf()
            #plt.scatter(dysparse,fluxsparse,c='blue',s=100,marker='+')
            #plt.plot(dysparse,fluxsparse,c='blue')
            #plt.yscale('log')
            #plt.plot(xbin,ybinsm,c='r')
            #plt.show()

            # Use points +/-3 for scaling
            gdpt, = np.where((np.abs(dysparse) <= 3) & (fluxsparse > 0.4*np.max(fluxsparse)))
            ybinsm2 = interp1d(xbin,ybinsm)(dysparse[gdpt])
            ratio = np.median(ybinsm2/fluxsparse[gdpt])
            ybinsm /= ratio   # scale thin curve to sparse one
            
            # Interpolate sparse onto finer scale
            nxfine = 30/0.1
            xfine = np.arange(nxfine)*0.1-14.95
            fluxsparsefine = 10**interp1d(dysparse,np.log10(fluxsparse),kind='quadratic',bounds_error=False,fill_value=np.nan)(xfine)
            #fluxsparsefine = 10**interp1d(dysparse,np.log10(fluxsparse),kind='quadratic',bounds_error=False,fill_value=np.nan)(xbin)            

            # switch to the sparse courve around x~3, around 3sigma
            sigma = np.sqrt(np.sum(ybinsm*xbin**2)/np.sum(ybinsm))
            # use logistic curve
            wt = 1/(1+np.exp(-2*(np.abs(xbin)-2.5*sigma)))
            gdsparse, = np.where((xfine>=np.min(xbin)-0.001) & (xfine<=np.max(xbin)+0.001))            
            combflux = fluxsparsefine[gdsparse]*wt + (1-wt)*ybinsm
            #plt.plot(xbin,combflux,c='orange')

            # Stuff the central combined portion into the final profile
            yprofile = fluxsparsefine.copy()
            yprofile[gdsparse] = combflux

            # Fix any NaNs
            ind = np.arange(len(yprofile))
            good1 = np.where(np.isfinite(yprofile))[0][0]
            bad1, = np.where(~np.isfinite(yprofile) & (ind<10))
            if len(bad1)>0:
                yprofile[bad1] = yprofile[good1]
            good2 = np.where(np.isfinite(yprofile))[0][-1]
            bad2, = np.where(~np.isfinite(yprofile) & (ind>len(yprofile)-10))
            if len(bad2)>0:
                yprofile[bad2] = yprofile[good2]
            bad = ~np.isfinite(yprofile)
            if np.sum(bad)>0:
                print('some nans')
                import pdb; pdb.set_trace()
            
            # Normalize
            yprofile /= np.sum(yprofile)*binsize

            if np.min(yprofile)<1e-5:
                print('problem')
                import matplotlib.pyplot as plt
                import pdb; pdb.set_trace()
            
            #plt.clf()
            #plt.scatter(data[:,0],data[:,1],s=5)
            #plt.plot(xbin,ybin,c='r')
            ##plt.plot(lowess[:,0],lowess[:,1],c='g')
            #plt.plot(xbin,ybinsm,c='b')  
            #plt.yscale('log')
            #plt.xlim(-8,8)
            #plt.ylim(1e-5,1)
            #plt.title('fiber='+str(f)+' column='+str(c))

            data.append( [xbin,ybin,ybinsm,f,c] )
            mnx[i,j] = np.median(data1[:,2])
            mny[i,j] = np.median(data1[:,3])
            profiles[i,j,:] = yprofile 
            
            #import pdb; pdb.set_trace()


            
    import pdb; pdb.set_trace()

    return data,mnx,mny,profiles


#####  EXTRACTION #######

def loadframe(infile):
    """ Load a 2D APOGEE image."""
    head = fits.getheader(infile,0)
    flux = fits.getdata(infile,1)
    err = fits.getdata(infile,2)
    mask = fits.getdata(infile,3)    
    frame = {'flux':flux, 'err':err, 'mask':mask, 'header':head}
    return frame

def loadepsf(infile):
    # Load Empirical PSF data
    # this takes a while
    phead = fits.getheader(infile,0)
    ntrace = phead['ntrace']
    epsf = []
    hdu = fits.open(infile)
    for itrace in range(ntrace):
        ptmp = hdu[itrace+1].data
        data = {'fiber': ptmp['FIBER'][0], 'lo': ptmp['LO'][0], 'hi': ptmp['HI'][0], 'img': ptmp['IMG'][0]}
        epsf.append(data)
    hdu.close()
    return epsf

def scat_remove(a,scat=None,mask=None):
    """
    remove scattered light
    """

    if scat==1:
        # simple stupid single level removal!
        if mask is not None:
            flux = np.copy(a)
            bad = (mask & BADMASK) > 0
            flux[bad] = np.nan
        else:
            flux = np.copy(a)
        bot = np.nanmedian(flux[100:1948,5:11])
        top = np.nanmedian(flux[100:1948,2038:2043])
        scatlevel = (bot+top)/2.
        print('scatlevel: ',scatlevel)
        flux -= scatlevel

    else:
        # variable scattered light, but only works for sparse exposures
        sz = a.ndim
        t = np.copy(a)
        bad = (~np.isfinite(t) | (t < -10))
        t[bad] = 1e10
        nbox = 51
        grid = np.zeros((41,41),float)
        ii = 0
        for i in range(4,2045,nbox):
            print(i)
            jj = 0
            for j in range(4,2045,nbox):
                i1 = i-nbox//2
                i2 = i+nbox//2
                j1 = j-nbox//2
                j2 = j+nbox//2
                i1 = np.max([4,i1])
                i2 = np.min([2044,i2])
                j1 = np.max([4,j1])
                j2 = np.min([2044,j2])
                sky = t[i1:i2+1,j1:j2+1]
                val,sig,skew = mmm.mmm(sky.ravel(),highbad=1e5)
                if sig > 0: grid[ii,jj]=val
                jj += 1
            ii += 1
  
        vec1 = np.arange(nbox).astype(int)
        vec2 = np.ones(nbox,float)
        xramp = vec1.reshape(-1,1)*vec2.reshape(1,-1)
        yramp = vec1.reshape(1,-1)*vec2.reshape(-1,1)
        
        w1 = (nbox-xramp)/nbox*(nbox-yramp)/nbox
        w2 = xramp/nbox*(nbox-yramp)/nbox
        w3 = (nbox-xramp)/nbox*yramp/nbox
        w4 = xramp/nbox*yramp/nbox
        
        out = np.zeros((2048,2048),float)
        ii = 0

        for i in range(4+nbox//2,2045-nbox//2,nbox):
            jj = 0
            for j in range(4+nbox//2,2045-nbox//2,nbox):            
                v1 = grid[ii,jj]
                v2 = grid[ii+1,jj]
                v3 = grid[ii,jj+1]
                v4 = grid[ii+1,jj+1]
                if v1 > 1e9: v1=v2
                if v2 > 1e9: v2=v1
                out[i-nbox//2:i+nbox//2+1,j-nbox//2:j+nbox//2+1] = v1*w1+v2*w2+v3*w3+v4*w4
                jj += 1
            ii += 1

        flux = np.copy(a)
        flux -= out
        
    return flux


def extract_pmul(p1lo,p1hi,img,p2):
    """ Helper function for extract()."""
    
    lo = np.max([p1lo,p2['lo']])
    k1 = lo-p1lo
    l1 = lo-p2['lo']
    hi = np.min([p1hi,p2['hi']])
    k2 = hi-p1lo
    l2 = hi-p2['lo']
    if lo>hi:
        out = np.zeros(2048,float)
    img2 = p2['img'].T  # transpose
    if lo==hi:
        out = img[:,k1:k2+1]*img2[:,l1:l2+1]
    else:
        out = np.nansum(img[:,k1:k2+1]*img2[:,l1:l2+1],axis=1)
    if out.ndim==2:
        out = out.flatten()   # make sure it's 1D
    return out

@njit
def solvefibers(x,xvar,ngood,v,b,c,vvar):
    for j in np.flip(np.arange(0,ngood-1)):
        x[j] = (v[j]-c[j]*x[j+1])/b[j]
        xvar[j] = (vvar[j]+c[j]**2*xvar[j+1])/b[j]**2            
    return x,xvar

def epsfmodel(epsf,spec,skip=False,subonly=False,fibers=None,yrange=[0,2048]):
    """ Create model image using EPSF and best-fit values."""
    # spec [2048,300], best-fit flux values
    
    ntrace = len(epsf)
    if fibers is None:
        fibers = np.arange(ntrace)
    
    # Create the Model 2D image
    if yrange is not None:
        model = np.zeros((2048,yrange[1]-yrange[0]),float)
        ylo = yrange[0]
    else:
        ylo = 0
        model = np.zeros((2048,2048),float)
    t = np.copy(spec)
    bad = (t<=0)
    if np.sum(bad)>0:
        t[bad] = 0
    for k in fibers:
        nf = 1
        ns = 0
        if subonly:
            junk, = np.where(subonly==k)
            nf = len(junk)
        if skip:
            junk, = np.where(skip==k)
            ns = len(junk)
        if nf > 0 and ns==0:
            p1 = epsf[k]
            lo = epsf[k]['lo']
            hi = epsf[k]['hi']
            img = p1['img'].T
            rows = np.ones(hi-lo+1,int)
            fiber = epsf[k]['fiber']
            model[:,lo-ylo:hi+1-ylo] += img[:,:]*(rows.reshape(-1,1)*t[:,fiber]).T                                    
    model = model.T

    return model


def extract(frame,epsf,doback=False,skip=False,scat=None,subonly=False,guess=None):
    """
    This extracts spectra using an empirical PSF.

    Extract spectrum under the assumption that a given pixel only contributes
    to two neighboring traces, leading to a tridiagonal matrix inversion.

    Parameters
    ----------
    frame : dict
       The 2D input structure with FLUX, VAR and MASK.
    epsf : list
       A list with the empirical PSF.
    doback : boolean, optional
       Subtract the background.  False by default.
    guess : dict
       Initial guess of the fluxes.  This is used to subtract out the contribution
         of fibers farther away.

    Returns
    -------
    outstr : dict
        The 1D output structure with FLUX, VAR and MASK.
    back : numpy array
        The background
    model : numpy array
        The model 2D image

    Example
    -------
    outstr,back,model = extract(frame,epsf)

    By J. Holtzman  2011
      Incorporated into ap2dproc.pro  D.Nidever May 2011  

    """
    
    nframe = len(frame)
    ntrace = len(epsf)

    fibers = np.array([e['fiber'] for e in epsf])
    flux = np.copy(frame['flux'].T)
    red = np.copy(frame['flux'].T)    
    var = np.copy(frame['err'].T**2)
    inmask = np.copy(frame['mask'].T)
    # use the transposes
    
    if scat:
        red = scat_remove(red,scat=scat,mask=inmask)

    # Guess input
    if guess is not None:
        gmodel = epsfmodel(epsf,guess)
        # subtract the initial best-fit model from the data
        red -= gmodel.T
        
    # Initialize output arrays
    spec = np.zeros((2048,300),float)
    err = np.zeros((2048,300),float)+999999.09 #+baderr()
    outmask = np.ones((2048,300),int)

    # calculate extraction matrix
    if doback:
        nback = 1 
    else:
        nback = 0
    back = np.zeros(2048,float)        
    beta = np.zeros((ntrace+nback,2048),float)
    betavar = np.zeros((ntrace+nback,2048),float)
    psftot = np.zeros((ntrace+nback,2048),float)
    tridiag = np.zeros((3,ntrace+nback,2048),float)
    warnmasked = np.zeros((ntrace+nback,2048),int)
    badmasked = np.zeros((ntrace+nback,2048),int)
    inmask_warn = (inmask & WARNMASK) > 0
    inmask_bad = (inmask & BADMASK) > 0

    for k in np.arange(0,ntrace+nback):        
        # Background
        if k > ntrace-1:
            beta[k,:] = np.nansum(red[:,lo:hi+1],axis=1)
            betavar[k,:] = np.nansum(var[:,lo:hi+1],axis=1)
            psftot[k,:] = 1.

        # Fibers
        else:
            # Initial guess, add flux back in for this fiber and neighbors
            if guess is not None:
                if k==0:
                    fibs = [k,k+1]
                elif k==ntrace-1:
                    fibs = [k-1,k]
                else:
                    fibs = [k-1,k,k+1]
                ylo = 2048
                yhi = 0
                for j in fibs:
                    ylo = np.minimum(epsf[j]['lo'],ylo)
                    yhi = np.maximum(epsf[j]['hi'],yhi)
                yhi += 1
                gmodel1 = epsfmodel(epsf,guess,fibers=fibs,yrange=[ylo,yhi])
                gmodel1 = gmodel1.T
                red[:,ylo:yhi] += gmodel1
                    
            # get EPSF and set bad pixels to NaN
            p1 = epsf[k]
            lo = epsf[k]['lo']
            hi = epsf[k]['hi']
            bad = (~np.isfinite(flux[:,lo:hi+1]) | (flux[:,lo:hi+1] == 0) |
                   ((inmask[:,lo:hi+1] & BADMASK) > 0) )
            nbad = np.sum(bad)
            img = np.copy(p1['img'].T)   # transpose
            if nbad > 0:
                img[bad] = np.nan
                
            # are there any warning flags for this trace? If so, flag the output
            warnmasked[k,:] = np.sum(inmask_warn[:,lo:hi+1],axis=1)
            warntot = np.maximum(warnmasked[k,:],1)
            warnmasked[k,:] = warnmasked[k,:] / warntot
            badmasked[k,:] = np.sum(inmask_bad[:,lo:hi+1],axis=1)
            badtot = np.maximum(badmasked[k,:],1)
            badmasked[k,:] = badmasked[k,:] / badtot
            
            psftot[k,:] = np.nansum(img,axis=1)
            beta[k,:] = np.nansum(red[:,lo:hi+1]*img,axis=1)
            betavar[k,:] = np.nansum(var[:,lo:hi+1]*img**2,axis=1)
            
            # Initial guess, subtract model back out
            if guess is not None:
                red[:,ylo:yhi] -= gmodel1                
                
        # First fiber (on the bottom edge)
        if k==0:
            ll = 1
            for l in np.arange(k,k+2):
                tridiag[ll,k,:] = extract_pmul(p1['lo'],p1['hi'],img,epsf[l])
                ll += 1

        # Last fiber (on top edge)
        elif k == ntrace-1:
            ll = 0
            for l in np.arange(k-1,k+1):
                tridiag[ll,k,:] = extract_pmul(p1['lo'],p1['hi'],img,epsf[l])
                ll += 1

        # Background terms
        elif k > ntrace-1:
            tridiag[1,k,:] = hi-lo+1

        # Middle fibers (not first or last)
        else:
            ll = 0
            for l in np.arange(k-1,k+2):
                tridiag[ll,k,:] = extract_pmul(p1['lo'],p1['hi'],img,epsf[l])
                ll += 1

    for i in np.arange(4,2044):
        # Good fibers
        good, = np.where(psftot[:,i] > 0.5)
        ngood = len(good)
        bad, = np.where(psftot[:,i] <= 0.5)
        nbad = len(bad)
        if nbad > 0:
            bad0, = np.where(bad>0)
            nbad0 = len(bad0)
            if nbad0 > 0:
                tridiag[2,bad[bad0]-1,i]=0 
            bad1, = np.where(bad < ntrace-1)
            nbad1 = len(bad1)
            if nbad1 > 0:
                tridiag[0,bad[bad1]+1,i] = 0 
        if ngood>0:
            a = tridiag[0,good,i]
            b = tridiag[1,good,i]
            c = tridiag[2,good,i]
            v = beta[good,i]
            vvar = betavar[good,i]
            m = a[1:ngood]/b[0:ngood-1]
            b[1:] = b[1:]-m*c[0:ngood-1]
            v[1:] = v[1:]-m*v[0:ngood-1]
            vvar[1:] = vvar[1:]+m**2*vvar[0:ngood-1]
            x = np.zeros(ngood,float)
            xvar = np.zeros(ngood,float)
            x[ngood-1] = v[ngood-1]/b[ngood-1]
            xvar[ngood-1] = vvar[ngood-1]/b[ngood-1]**2
            # Use numba to speed up this slow lopp
            #for j in np.flip(np.arange(0,ngood-1)):
            #    x[j] = (v[j]-c[j]*x[j+1])/b[j]
            #    xvar[j] = (vvar[j]+c[j]**2*xvar[j+1])/b[j]**2
            x,xvar = solvefibers(x,xvar,ngood,v,b,c,vvar)
            spec[i,fibers[good]] = x
            err[i,fibers[good]] = np.sqrt(xvar)
            # mask the bad pixels
            outmask[i,fibers[good]] = 0
            if nbad > 0:
                outmask[i,fibers[bad]] = maskval['NOT_ENOUGH_PSF'] | badmasked[bad,i]
            # put the warning bits into the mask
            outmask[i,fibers] = outmask[i,fibers] | warnmasked[:,i]
            
        # No good fibers for this column
        else:
            spec[i,:] = 0
            err[i,:] = BADERR
            outmask[i,fibers] = maskval['NOT_ENOUGH_PSF'] | badmasked[:,i]

        if doback:
            back[i] = x[ngood-1]


    # Catch any NaNs (shouldn't be there, but ....)
    bad = ~np.isfinite(spec)
    nbad = np.sum(bad)
    if nbad > 0:
        spec[bad] = 0.
        err[bad] = BADERR
        outmask[bad] = 1


    # Put together the output dictionary
    outstr = {'flux':spec, 'err':err, 'mask':outmask}

    # Create the Model 2D image
    model = epsfmodel(epsf,spec,subonly=subonly,skip=skip)

    
    return outstr,back,model

def func_poly2d(inp,*args):
    """ 2D polynomial surface"""
    x = inp[0]
    y = inp[1]
    p = args
    np = len(p)
    if np==0:
        a = p[0]
    elif np==3:
        a = p[0] + p[1]*x + p[2]*y
    elif np==4:
        a = p[0] + p[1]*x + p[2]*x*y + p[3]*y
    else:
        raise Exception('Only 0, 3, and 4 parameters supported')
    return a
    
def getoffset(frame,traceim):
    """
    Measure the offset of an object exposure and the PSF model/traces.
    """
    
    # Find bright fibers and measure the centroid
    nfibers = traceim.shape[0]
    flux = frame['flux']
    
    # Loop over X locations
    xx = [512, 1024, 1536]
    coef = np.zeros((len(xx),2),float)
    for i,x in enumerate(xx):
        # Get median flux/centers
        medflux = np.median(flux[:,x-100:x+100],axis=1)
        medcent = np.median(traceim[:,x-100:x+100],axis=1)
        # Measure rough flux in each fiber
        boxflux = np.zeros(nfibers,float)
        fiberycen = np.zeros(nfibers,float)
        for j in range(nfibers):
            fiberycen[j] = medcent[j]
            boxflux[j] = np.sum(medflux[int(np.round(medcent[j]))-1:int(np.round(medcent[j]))+1])
        # Find bright fibers
        bright, = np.where(boxflux > 1000)
        nbright = len(bright)
        
        # Loop over bright fibers
        y = np.arange(2048)
        gcent = np.zeros(nbright,float)
        offset = np.zeros(nbright,float)
        ycen = fiberycen[bright]
        for j in range(nbright):
            ind = bright[j]
            # Fit Gaussian
            lo = int(np.floor(medcent[ind]-3))
            hi = int(np.ceil(medcent[ind]+3))
            yy = np.arange(hi-lo+1)+lo
            ff = medflux[lo:hi+1]
            initpar = [ff[3],medcent[j],1.0,0.0]
            try:
                pars,perror = dln.gaussfit(yy,ff,initpar=initpar)
                gcent[j] = pars[1]
                offset[j] = pars[1]-medcent[j]                
            except:
                gcent[j] = np.nan
                offset[j] = np.nan


        # Fit line to it
        medoff = np.nanmedian(offset)
        sigoff = dln.mad(offset[np.isfinite(offset)])
        gd, = np.where(np.isfinite(offset) & (np.abs(offset-medoff) < 3*sigoff))
        coef1 = np.polyfit(ycen[gd],offset[gd],1)
        coef[i,:] = coef1

    # Fit 2D linear model
    xvals = np.zeros((len(xx),2048),float)
    yvals = np.zeros((len(xx),2048),float)
    zvals = np.zeros((len(xx),2048),float)    
    for i,x in enumerate(xx):
        xvals[i,:] = x
        yvals[i,:] = np.arange(2048)
        zvals[i,:] = np.polyval(coef[i,:],np.arange(2048))
        
    initpar = np.zeros(4)
    coef2,cov2 = curve_fit(func_poly2d,[xvals.ravel(),yvals.ravel()],zvals.ravel(),p0=initpar)
    coeferr2 = np.sqrt(np.diag(cov2))

    return coef2


def fullepsfgrid(psf,traceim,offcoef):
    """ Generate a full EPSF grid for all fibers and columns and dealing with offsets."""

    
    nfibers = traceim.shape[0]
    
    ## Loop over X locations
    #xx = [512, 1024, 1536]
    #coef = np.zeros((len(xx),2),float)
    #for i,x in enumerate(xx):
    #    # Get median flux/centers
    #    medflux = np.median(flux[:,x-100:x+100],axis=1)
    #    medcent = np.median(traceim[:,x-100:x+100],axis=1)

    # It should be possible to do linear interpolation of all 2048 profiles
    # at once without using a loop
    
    epsf = []
    # Fiber loop
    for i in range(nfibers):
        print('fiber = ',i)
        off = func_poly2d([np.arange(2048),traceim[i,:]],*offcoef)
        ycen = traceim[i,:]+off
        ylo = int(np.min(np.round(ycen)))-14
        yhi = int(np.max(np.round(ycen)))+14
        ny = yhi-ylo+1
        y = np.arange(ny)+ylo        
        img = np.zeros((ny,2048),float)
        #ylo = np.zeros(2048,int)
        #yhi = np.zeros(2048,int)
        # Column loop
        for j in range(2048):
            m1 = psf([j,ycen[j]],y=y,ycen=ycen[j])
            m1 /= np.sum(m1)
            img[:,j] = m1
            #ylo[j] = y[0]
            #yhi[j] = y[-1]
                        
        data = {'fiber':i, 'lo':ylo, 'hi':yhi, 'img':img, 'ycen':ycen}
        #data = {'fiber':i, 'lo':ylo, 'hi':yhi, 'img':img, 'ycen':ycen}        
        #data = {'fiber':i, 'lo':np.min(ycen)-14, 'hi':np.max(ycen)+14, 'img':img, 'ycen':ycen}        
        #p = {'fiber': ptmp['FIBER'], 'lo': ptmp['LO'], 'hi': ptmp['HI'], img: ptmp['IMG']
        epsf.append(data)
        
    return epsf
        

def extractwing(frame,psf,tracefile):
    """ Extract taking wings into account."""

    # ideas for extraction with wings if I can't fit fiber and 4 neighbors simultaneously:
    # 1) do usual fiber + 2 neighbor extraction using narrower profile
    # 2) create model using the broad profile and find the residual of data-model.
    # 3) loop through each fiber and add its broad profile back in (this is the same as
    #  subtracting all other fibers only)
    # use the narrow profile to find improved flux using weighted mean of best scaled profile
    # -can iterate if wanted
    # -could do this just around bright stars?

    # Load PSF
    if type(psf) is not PSF:
        psffile = psf
        psf = PSF.read(psffile)

    # Load the data
    if type(frame) is str:
        framefile = frame
        frame = loadframe(framefile)
    # Load the trace imformation
    traceim = fits.getdata(tracefile,0)  # [Nfibers,2048]
        
    # Step 1) Measure the offset
    #  returns 2D linear of the offset
    #  c0 + c1*x + c2*x*y + c3*y
    offcoef =  getoffset(frame,traceim)

    # Step 2) Generate full PSFs for this image
    # Generate the input that extract() expects
    # this currently takes about 176 sec. to run
    #epsf = fullepsgrid(psf,traceim,offcoef)
    #np.savez('fullepsfgrid.npz',epsf=epsf)
    epsf = np.load('fullepsfgrid.npz',allow_pickle=True)['epsf']
    
    # Step 3) Regular fiber+2 neighbor extraction
    out1,back1,model1 = extract(frame,epsf)
    
    # Step 4) Subtract all profiles except the fibers+2 neighbors and refit
    out,back,model = extract(frame,epsf,guess=out1['flux'])
    
    import pdb; pdb.set_trace()
    
    return out,back,model

      
if __name__ == '__main__' :

    psfdir = '/Users/nidever/sdss5/mwm/apogee/spectro/redux/daily/cal/apogee-n/psf/'
    psffile = 'apEPSF-b-39880014.fits'
    sparsefile = 'apSparse-39870034.fits'
    allim,head = fits.getdata(psfdir+sparsefile,0,header=True)
    im = allim[1,:,:]
    psfhdu = fits.open(psfdir+psffile)

    data = getprofilegrid(psfdir+psffile,psfdir+sparsefile)


    import pdb; pdb.set_trace()

    # Check domeflat vs. quartzflat profile to see if the profile changes
    # if the light goes through the FPS octagonal fibers or not

    # To combine the 300-fiber quartzflat ("narrow") and sparse data
    # for each "narrow" profile, find the closest good sparse profile
    # then scale the narrow profile to the sparse profile for the inner
    # points (r<4 or so).  Then splice them together, or maybe use
    # an average in the crossover region

    # Interpolating the profiles
    # 1) grid interpolation
    #  Use the whole 3D grid (Nprofile,Nx,Ny) to do interpolation
    #  For each profile point, we can use RectBivariateSpline
    #  to get the value for any detector X/Y position.

    # For each fiber, you could interpolate the profile for all columns
    # at once.  Then use

    # When we do extraction, we do it one column at a time.
    # We want profiles for all 300 fibers and a single column
    # take the two closest Y positions in the 3D profile grid
    # then linearly interpolate (Nprofile,Ny) for those two X points,
    # Then we have (Nprofile,Ny) for the column of interest.
    # Use RectBivariateSpline on the (Nprofile,Ny) data for this column
    # and the trace positions for the 300 fibers for this column to
    # get full profiles for each fiber (Nprofile,300).
    # Then get the actual profiles for each fiber using the trace
    # positions and pixel values (could do the last two steps together).
    
    # 2) ANN
    # train an ANN on the profiles as an emulator
    # you give it the detector X/Y position and it returns the profile
    # for that position
  

    # When we do extraction, we want to solve the three neighbors simultanously
    # but the wings affect +/-15 pixels or two neighbors on each side.
    # The pixels nearest the peak have the most weight for solving the fiber's
    # flux, but wings will affect the solution (especially for bright fibers).
    
