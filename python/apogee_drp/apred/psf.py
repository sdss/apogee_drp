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
from ..utils import peakfit, mmm, apload, utils
from numba import njit
import copy


WARNMASK = -16640
BADMASK = 16639
BADERR = 1.00000e+10
maskval = {'NOT_ENOUGH_PSF': 16384}
chips = ['a','b','c']

#####  EMPIRICAL PSF MODEL CLASS #######

def leaky_relu(z):
    """ This is the activation function used by default in all our neural networks. """
    return z*(z > 0) + 0.01*z*(z < 0)

class PSFProfile(object):
    """ This holds an oversampled PSF profile and interpolation coefficients
         for fast interplation."""

    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.n = len(x)
        self._xrange = [np.min(x),np.max(x)]
        self._dx = self.x[1]-self.x[0]  # assuming constant steps
        self._xcoefind = None
        self._xcoefsteps = 2
        self._coef = None

        # Make the coefficients
        self.makecoefs(steps=self._xcoefsteps)
        
    def __call__(self,x):
        """ Interpolate onto x"""
        return self.interpolate(x)


    def __str__(self):
        """ String representation of the PSFProfile."""
        return self.__class__.__name__+'(%.2f<X<%.2f, Npix=%d)' % \
                                        (self._xrange[0],self._xrange[1],self.n)

    def __repr__(self):
        """ String representation of the PSFProfile."""
        return self.__class__.__name__+'(%.2f<X<%.2f, Npix=%d)' % \
                                        (self._xrange[0],self._xrange[1],self.n)
    
    def interpolate(self,x):
        """ Fast interpolation."""        
        newy = np.zeros(len(x),float)
        xind = np.floor((x-self._xrange[0])/(self._dx*self._xcoefsteps)).astype(int)
        #good, = np.where((xind >= 0) & (xind <= self.n))
        good, = np.where((x >= self._xrange[0]) & (x <= self._xrange[1]))
        ngood = len(good)
        if ngood>0:
            newy[good] = self._coef[xind[good],0]*x[good]**2 + self._coef[xind[good],1]*x[good] + self._coef[xind[good],2]
        # points outside of the range are zero by default
        return newy


    def makecoefs(self,kind=2,steps=2):
        """ Make the polynomial coefficients."""
        ncoef = self.n//steps
        coef = np.zeros((ncoef,3),float)        
        xcoefind = np.arange(1,self.n,steps)
        for i,ind in enumerate(xcoefind):
            lo = ind-1
            hi = ind+2
            if hi>self.n:
                hi = self.n
                lo = hi-3
            # a*x^2+b*x+c
            coef[i,:] = dln.quadratic_coefficients(self.x[lo:hi],self.y[lo:hi])  # a,b,c
        self._xcoefind = xcoefind
        self._coef = coef

    def copy(self):
        """ Make a copy of self."""
        return copy.deepcopy(self)
        
    def __add__(self, other):
        # Add number to profile
        if isinstance(other,int) or isinstance(other,float):
            new = self.copy()
            new.y += other
            new._coef += other
            return new
        # Add two profiles
        if isinstance(other,PSFProfile) is False:
            raise Exception('Other object must also be a PSFProfile')
        if self.n != other.n:
            raise Exception('Array lengths must be the same')
        if self.x[0] != other.x[0] or self.x[-1] != other.x[-1]:
            raise Exception('X arrays must be the same')
        new = self.copy()
        new.y = self.y + other.y
        new._coef = self._coef + other._coef
        return new

    def __sub__(self, other):
        # Subtract number to profile
        if isinstance(other,int) or isinstance(other,float):
            new = self.copy()
            new.y -= other
            new._coef -= other
            return new        
        if isinstance(other,PSFProfile) is False:
            raise Exception('Other object must also be a PSFProfile')
        if self.n != other.n:
            raise Exception('Array lengths must be the same')
        if self.x[0] != other.x[0] or self.x[-1] != other.x[-1]:
            raise Exception('X arrays must be the same')
        new = self.copy()
        new.y = self.y - other.y
        new._coef = self._coef - other._coef
        return new        

    def __mul__(self, other):
        # Multiply profile by number
        if isinstance(other,int) or isinstance(other,float):
            new = self.copy()
            new.y *= other
            new._coef *= other
            return new        
        if isinstance(other,PSFProfile) is False:
            raise Exception('Other object must also be a PSFProfile')
        if self.n != other.n:
            raise Exception('Array lengths must be the same')
        if self.x[0] != other.x[0] or self.x[-1] != other.x[-1]:
            raise Exception('X arrays must be the same')
        new = self.copy()
        new.y = self.y * other.y
        new._coef = self._coef * other._coef
        return new

    def __truediv__(self, other):
        # Divide profile by number
        if isinstance(other,int) or isinstance(other,float):
            new = self.copy()
            new.y /= other
            new._coef /= other
            return new        
        if isinstance(other,PSFProfile) is False:
            raise Exception('Other object must also be a PSFProfile')
        if self.n != other.n:
            raise Exception('Array lengths must be the same')
        if self.x[0] != other.x[0] or self.x[-1] != other.x[-1]:
            raise Exception('X arrays must be the same')
        new = self.copy()
        new.y = self.y / other.y
        new._coef = self._coef / other._coef
        return new
    
class PSF(object):

    def __init__(self,data,nxgrid=20,nygrid=50,kind='ann',log=True):
        # kind can be 'ann' or 'grid'
        if kind=='ann':
            # coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max, y)
            self.kind = kind
            coefs = data
            self._log = log
            self._coeffs = coeffs
            self.xmin = coeffs['xmin']
            self.xmax = coeffs['xmax']
            self.y = coeffs['y']
            self._grid = None
            self._xgrid = None
            self._ygrid = None        
        elif kind=='grid':
            # data should be (grid,labels,y)
            # grid should be [Ncols,Nrows,Npix]
            # labels should be [Ncols,Nrows,2]
            # y should be [Npix]
            self.kind = kind
            grid,labels,y = data
            self._grid = grid
            self._log = log        
            self._labels = labels
            self._xgrid = labels[0]
            self._ygrid = labels[1]
            self.y = y
            self.xmin = [np.min(labels[0]),np.min(labels[1])]
            self.xmax = [np.max(labels[1]),np.max(labels[1])]
            nxgrid,nygrid,npix = grid.shape
        else:
            raise ValueError('Only "ann" and "grid" supported at this time')
        self.npix = len(self.y)
        self._nxgrid = nxgrid
        self._nygrid = nygrid

    def __str__(self):
        """ String representation of the PSF."""
        return self.__class__.__name__+'(%.1f<X<%.1f, %.1f<X<%.1f, %s, Npix=%d)' % \
                                        (self.xmin[0],self.xmax[0],self.xmin[1],self.xmax[1],self,kind,self.npix)

    def __repr__(self):
        """ String representation of the PSF."""
        return self.__class__.__name__+'(%.1f<X<%.1f, %.1f<X<%.1f, %s, Npix=%d)' % \
                                        (self.xmin[0],self.xmax[0],self.xmin[1],self.xmax[1],self.kind,self.npix)
    
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
            profile = np.interp(y-ycen,self.y,fullprofile,left=fullprofile[0],right=fullprofile[-1])

        # Take to the power of
        if self._log:
            profile = 10**profile
            
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
        if self.kind=='ann':
            return self.ann_model(inlabels)
        else:
            return self.gridinterp(inlabels)
        
    def ann_model(self,inlabels):
        """ Make a brand-new full profile model with input labels and ANN model."""
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
        return m

    def gridinterp(self,labels):
        """ Interpolate model in the grid."""

        if labels[0]<0 or labels[0]>2047 or labels[1]<0 or labels[1]>2047:
            raise ValueError('X/Y must be between 0 and 2047')
        
        if self._grid is None:
            self.mkgrid()

        if self.kind=='grid':
            # xgrid/ygrid are 2D [Nx,Ny] and not quite a regular rectangular grid
            # Find closest X position
            xind = np.searchsorted(self._xgrid[:,self._nygrid//2],labels[0])
            yind = np.searchsorted(self._ygrid[np.minimum(xind,self._nxgrid-1),:],labels[1])
            xind = np.searchsorted(self._xgrid[:,np.minimum(yind,self._nygrid-1)],labels[0])
            yind = np.searchsorted(self._ygrid[np.minimum(xind,self._nxgrid-1),:],labels[1])            
        else:
            xind = np.searchsorted(self._xgrid,labels[0])
            yind = np.searchsorted(self._ygrid,labels[1])        
        
        # Find the closest points on the grid
        #------------------------------------
        # -- At corners, use corner values --
        # bottom left
        if xind==0 and yind==0:
            return self._grid[0,0,:]
        # top left
        if xind==0 and yind==self._nygrid:
            return self._grid[0,-1,:]
        # bottom right
        if xind==self._nxgrid and yind==0:        
            return self._grid[-1,0,:]
        # top right
        if xind==self._nxgrid and yind==self._nygrid:
            return self._grid[-1,-1,:]
        
        # -- Edges, use two points --
        # linearly interpolate so it's smooth        
        # Left
        #   use left-most X and interpolate only in Y
        if xind==0:
            yind1 = yind-1
            yind2 = yind
            if self.kind=='grid':
                wt = (labels[1]-self._ygrid[xind,yind1])/(self._ygrid[xind,yind2]-self._ygrid[xind,yind1])
            else:
                wt = (labels[1]-self._ygrid[yind1])/(self._ygrid[yind2]-self._ygrid[yind1])                
            profile = (1-wt)*self._grid[0,yind1,:] + wt*self._grid[0,yind2,:]
            return profile
        # Right
        #  use right-most X and interpolate only in Y
        if xind==self._nxgrid:
            yind1 = yind-1
            yind2 = yind
            if self.kind=='grid':
                wt = (labels[1]-self._ygrid[xind-1,yind1])/(self._ygrid[xind-1,yind2]-self._ygrid[xind-1,yind1])
            else:
                wt = (labels[1]-self._ygrid[yind1])/(self._ygrid[yind2]-self._ygrid[yind1])
            profile = (1-wt)*self._grid[-1,yind1,:] + wt*self._grid[-1,yind2,:]
            return profile
        # Bottom
        #  use Bottom-most Y and interpolate only in X
        if yind==0:
            xind1 = xind-1
            xind2 = xind
            if self.kind=='grid':
                wt = (labels[0]-self._xgrid[xind1,yind])/(self._xgrid[xind2,yind]-self._xgrid[xind1,yind])
            else:
                wt = (labels[0]-self._xgrid[xind1])/(self._xgrid[xind2]-self._xgrid[xind1])
            profile = (1-wt)*self._grid[xind1,0,:] + wt*self._grid[xind2,0,:]
            return profile
        # Top
        #  use top-most Y and interpolate only in X
        if yind==self._nygrid:
            xind1 = xind-1
            xind2 = xind
            if self.kind=='grid':
                wt = (labels[0]-self._xgrid[xind1,yind-1])/(self._xgrid[xind2,yind-1]-self._xgrid[xind1,yind-1])
            else:
                wt = (labels[0]-self._xgrid[xind1])/(self._xgrid[xind2]-self._xgrid[xind1])            
            profile = (1-wt)*self._grid[xind1,-1,:] + wt*self._grid[xind2,-1,:]
            return profile
            
        # -- In the middle --
        # linearly interpolate so it's smooth
        xind1 = xind-1
        xind2 = xind
        yind1 = yind-1
        yind2 = yind
        if self.kind=='grid':
            wtdx = (self._xgrid[xind2,yind1]-self._xgrid[xind1,yind1])
            wtdy = (self._ygrid[xind1,yind2]-self._ygrid[xind1,yind1])
        else:
            wtdx = (self._xgrid[xind2]-self._xgrid[xind1])
            wtdy = (self._ygrid[yind2]-self._ygrid[yind1])                
        profile = np.zeros(self.npix,float)
        totwt = 0.0
        for xind in [xind1,xind2]:
            for yind in [yind1,yind2]:
                if self.kind=='grid':
                    wt = np.abs(labels[0]-self._xgrid[xind,yind])*np.abs(labels[1]-self._ygrid[xind,yind])/(wtdx*wtdy)
                else:
                    wt = np.abs(labels[0]-self._xgrid[xind])*np.abs(labels[1]-self._ygrid[yind])/(wtdx*wtdy)
                totwt += wt
                profile += wt*self._grid[xind,yind]
        profile /= totwt
        
        return profile

    # Make a new method that does the interpolation for an entire fiber all at once (all 2048 pixels)
    # might allow for some speedups.  Would need to have y values (trace) input.
    def fiber(self,y):
        """ Construct profiles for all columns of a fiber."""
        # all y-values must be given

        # Just pick ONE y-value
        x = np.arange(2048)
        ymn = np.mean(y)
        profile = np.zeros((2048,self.npix),float)

        if self.kind=='grid':
            # xgrid/ygrid are 2D [Nx,Ny] and not quite a regular rectangular grid
            yind = np.searchsorted(self._ygrid[self._nxgrid//2,:],ymn)
            xarr = self._xgrid[:,yind]
        else:
            yind = np.searchsorted(self._ygrid,ymn)
            xarr = self._xgrid

        yfine = np.arange(self.npix)            
        for i in range(2048):
            if self.kind=='grid':
                xind = np.searchsorted(self._xgrid[:,yind],i)
            else:
                xind = np.searchsorted(self._xgrid,i)
            xind2 = None
            if xind==0:
                xind1 = xind
            elif xind==self._nxgrid-1:
                xind1 = xind
            else:
                xind1 = xind-1
                xind2 = xind
            prof1 = self._grid[xind1,yind,:]
            # Shift and interpolate
            ycen = y[i]
            prof = np.interp(y-ycen,y,prof1,left=0.0,right=0.0)
            if xind2 is not None:
                prof2 = self._grid[xind2,yind,:]
                # Shift and interpolate
                iprof2 = np.interp(y-ycen,y,prof2,left=0.0,right=0.0)
                wt = (self._xgrid[xind1,yind]-i)/(self._xgrid[xind2,yind]-self._xgrid[xind1,yind])
                prof = profile*wt + (1-wt)*iprof2
            profile[:,i] = prof
            import pdb; pdb.set_trace()
            
        import pdb; pdb.set_trace()
            
        
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


    def write(self,outfile):
        # Write to a file
        hdu = fits.HDUList()
        hdu.append(fits.ImageHDU(self._grid))
        hdu[0].header['TYPE'] = self.kind
        hdu[0].header['LOG'] = self._log
        hdu[0].header['COMMENT'] = 'Data (log)'
        hdu.append(fits.ImageHDU(self._labels))
        hdu[1].header['COMMENT'] = 'Labels'
        hdu.append(fits.ImageHDU(self.y))
        hdu[2].header['COMMENT'] = 'x'
        hdu.writeto(outfile,overwrite=True)
    
    @classmethod
    def read(cls,infile):
        # Load the file and return a PSF object
        if infile[-4:]=='fits':
            hdu = fits.open(infile)
        else:
            raise ValueError('Only fits files allowed')

        kind = hdu[0].header['type']
        log = hdu[0].header['log']
        if log is None: log=True  # True by default
        if kind=='grid':
            grid = hdu[0].data
            labels = hdu[1].data
            y = hdu[2].data
            return PSF((grid,labels,y),kind='grid',log=log)
        elif kind=='ann':
            coeffs = {}
            hdu = fits.open(infile)
            for i in range(9):
                coeffs[hdu[i].header['type']] = hdu[i].data
            # coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max, y)
            return PSF(coeffs,kind='ann',log=log)
        else:
            raise ValueError('Only grid or ann types allowed')
        
    
#####  GET EMPIRICAL PSF #######


def mkfiber2hdu(hdulist):
    # Get fiber numbers for each hdu of an apEPSF file
    fiber2hdu = {}
    fibernum = np.zeros(len(hdulist)-1,int)
    for i in range(len(hdulist)-1):
        fibernum[i] = hdulist[i+1].data['FIBER']
        fiber2hdu[hdulist[i+1].data[0]['FIBER']] = i+1
    return fiber2hdu
    

def getprofdata(fibs,cols,hdulist,fiber2hdu):
    """
    Load the apEPSF profile data for a range of fibers and columns from the HDUList.

    Parameters
    ----------
    fibs : list
      List or fiber numbers or two-element list of upper/lower range to use.
    cols : list
      Two-element list of upper/lower range of columns to use.
    hdulist : HDUList
      HDUList containing the data.
    fiber2hdu : dict
      Fiber to HDU conversion dictionary.

    Returns
    -------
    data : numpy array
      The profile data for the range of fibers and columns [Nfibers*Ncols*30,4].
      The values in the second dimension are: dy, flux, X, Y.

    Example
    -------

    data = getprofdata(fibs,cols,hdulist,fiber2hdu)

    """

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
    """
    Calculate the average profile for a range of fibers and columns.

    Parameters
    ----------
    fibers : list
      List or 2-element upper/lower range of fiber numbers to average.
    cols : list
      List or 2-element upper/lower range of column numbers to average.
    hdulist : HDUList
      HDUList containing the data.
    fiber2hdu : dict
      Fiber to HDU conversion dictionary.

    Returns
    -------
    data : numpy array
      The profile data for the range of fibers and columns [Nfibers*Ncols*30,4].
      The values in the second dimension are: dy, flux, X, Y.

    xbin : numpy array
      Binned X-values.
    ybin : numpy array
      Binned Y-values.
    profile : numpy array
      The binned and normalized profiles for each grid point.

    Example
    -------

    data, xbin, ybin, profile = avgprofile(fibs,cols,hdulist,fiber2hdu):

    """
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
        bd, = np.where(bad)
        gd, = np.where(~bad)
        fill_value = (ybinsm[gd[0]],ybinsm[gd[1]])
        ybinsm[bd] = interp1d(xbin[~bad],ybinsm[~bad],bounds_error=False,fill_value=fill_value)(bd)

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


def makeprofilegrid(psffile,sparsefile,nfbin=5,ncbin=200,verbose=False):
    """
    Construct a grid in X and Y across the detector of average
    PSF profiles.

    Parameters
    ----------
    psffile : str
      Filename of apEPSF file with empirical PSF profiles.
    sparsefile : str
      Filename of apSparse file with APOGEE sparse PSF profile data.
    nfbin : int
      Number of fibers to bin/average.  Default is 5.
    ncbin : int
      Number of column to bin/average.  Default is 200.
    verbose : boolean, optional
      Verbose output to the screen.

    Returns
    -------
    data : numpy array
      List of all profile data for the grid.  There is an element
        for each grid point that contains:
         [xbin,ybin,profile,fiber,column]
    mnx : numpy array
      Mean X values for each average/grid profile point [Ncols,Nfibers].
    mny : numpy array
      Mean Y values for each average/grid profile point [Ncols,Nfibers].
    profiles : numpy array
      Averaged profile data [Ncols, Nfibers, 300].
    xx : numpy array
      The profile X values [300].

    Example
    -------

    data,mnx,mny,profiles,xx = makeprofilegrid(psffile,sparsefile,nfbin=5,ncbin=200)

    """

    if verbose:
        print('Making Model PSF grid')
        print('EPSF file: '+psffile)
        print('Sparse file: '+sparsefile)
        print('Fiber binning: '+str(nfbin))
        print('Column binning: '+str(ncbin))

    allim,head = fits.getdata(sparsefile,0,header=True)
    sim = allim[1,:,:]
    psfhdu = fits.open(psffile)

    # Get fiber numbers for each hdu
    fiber2hdu = mkfiber2hdu(psfhdu)
    
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

            if verbose:
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
                if verbose:
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

    return data,mnx,mny,profiles,xx


def mkmodelpsf(name,psfid,sparseid,apred,telescope,nfbin=5,ncbin=200,verbose=False):
    """
    Makes the Model PSF calibration file.

    Parameters
    ----------
    name : int
      Name of the output model PSF file (apPSFModel).
    psfid : int
      ID of apEPSF exposure empirical PSF profiles.
    sparseid : int
      ID of apSparse file with APOGEE sparse PSF profile data.
    apred : str
      APOGEE Reduction version.
    telescope : str
      Telescope name: apo25m or lco25m.
    nfbin : int
      Number of fibers to bin/average.  Default is 5.
    ncbin : int
      Number of column to bin/average.  Default is 200.
    verbose : boolean, optional
      Verbose output to the screen.

    Returns
    -------

    Example
    -------

    mkmodelpsf(psfid,sparseid)

    """

    print('Making Model PSF calibration file')
    print('EPSF ID: '+str(psfid))
    print('Sparse ID: '+str(+sparseid))
    print('Fiber binning: '+str(nfbin))
    print('Column binning: '+str(ncbin))

    load = apload.ApLoad(apred=apred,telescope=telescope)
    sparsefile = load.filename('Sparse',num=sparseid,chips=True)
    psffile = load.filename('EPSF',num=psfid,chips=True)
    for ch in chips:
        psffile1 = psffile.replace('EPSF-','EPSF-'+ch+'-')
        data,mnx,mny,profiles,y = makeprofilegrid(psffile1,sparsefile,verbose=verbose)
        labels = [mnx,mny]
        p = PSF((profiles,labels,y),kind='grid',log=False)
        outfile = load.filename('PSFModel',num=name,chips=True).replace('PSFModel-','PSFModel-'+ch+'-')
        print('Writing to '+outfile)
        p.write(outfile)


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
    """
    Load Empirical PSF data
    this takes a while

    Parameters
    ----------
    infile : str
       Filename of apEPSF file.

    Returns
    -------
    epsf : list
       List of dictionaries with information on each trace.

    Example
    -------

    epsf = loadepsf(infile)
 
    """
    phead = fits.getheader(infile,0)
    ntrace = phead.get('ntrace')
    if ntrace is None:
        print('No NTRACE in header')
        return []
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
        print('scatlevel: %.5f ' % scatlevel)
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
       The 2D input structure with flux, err, mask and header.
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
    inmask_warn = (inmask & WARNMASK)
    inmask_bad = (inmask & BADMASK)

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
            warnmasked[k,:] = np.bitwise_or.reduce(inmask_warn[:,lo:hi+1],axis=1)
            badmasked[k,:] = np.bitwise_or.reduce(inmask_bad[:,lo:hi+1],axis=1)
            
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
    outstr = {'flux':spec, 'err':err, 'mask':outmask, 'header':frame['header'].copy()}

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

def measuretrace(frame,traceim,xcen,nbin,avgtype='median',nrepeat=12,fibers=None,fitmethod='gaussian'):
    """
    Measure trace position given the data and method.

    Parameters
    ----------
    frame : dict
       Dictionary with information a single detector 2D APOGEE image.  Must contain 'flux', 'err'
       and 'mask'.
    traceim : numpy array
       The 2D image containing the Y trace values from a reference image (with shape [300,2048]).
    xcen : int
       Central column to find trace positions for.
    nbin : int
       Number of columns to average (using avgtype method) +/-nbin from xcen.  Therefore,
       2*xbin+1 columns will be combined/averaged.
    avgtype : str, optional
       Column averaging method to use.  Default is "median".
    nrepeat : int, optional
       Number of repeats to use for avgtype='rollmedian'.  Only used for FPI images.  Default is 12.
    fibers : list, optional
       List of fibers to extract.  Default is to extract all "bright" ones.
    fitmethod : str, optional
       Method to determine the central Y value.  Options are 'gaussian' or 'empirical'.
       Default is 'gaussian'.

    Returns
    -------
    tab : table
       Table of values for all the traces.

    Example
    -------

    tab = measuretrace(frame,traceim,1024,100,avgtype='median')

    """

    flux = frame['flux']
    fluxerr = frame['err']
    ntraces = traceim.shape[0]
    
    tab = np.zeros(ntraces,dtype=np.dtype([('fiber',int),('x',float),('ytemp',float),('flux',float),('snr',float),
                                           ('ycent',float),('ycenterr',float),('yoffset',float),('bright',bool)]))
    tab['ycent'] = np.nan
    tab['yoffset'] = np.nan    
    tab['x'] = xcen
    tab['fiber'] = np.arange(300)
    
    # Get the average/median/sum profile flux
    xlo = np.maximum(xcen-nbin,0)
    xhi = np.minimum(xcen+nbin,2048)
    nxpix = xhi-xlo
    if avgtype == 'median':
        profileflux = np.nanmedian(flux[:,xlo:xhi],axis=1)
        # use standard error, more robust against large uncertainties in some pixels
        profilefluxerr = np.nanmedian(fluxerr[:,xlo:xhi],axis=1)/np.sqrt(nxpix)
    if avgtype == 'mean':
        profileflux = np.nanmean(flux[:,xlo:xhi],axis=1)
        profilefluxerr = np.nanmedian(fluxerr[:,xlo:xhi],axis=1)/np.sqrt(nxpix)
    elif avgtype == 'sum':
        profileflux = np.nansum(flux[:,xlo:xhi],axis=1)
        profilefluxerr = np.nanmedian(fluxerr[:,xlo:xhi],axis=1)*np.sqrt(nxpix)
    elif avgtype == 'summedian':
        # First sum, then median
        binflux = dln.rebin(flux[:,xlo:xhi],binsize=(1,18),tot=True)
        nbinflux = binflux.shape[1]        
        binfluxerr = dln.rebin(flux[:,xlo:xhi],binsize=(1,18),med=True)*np.sqrt(18)
        profileflux = np.nanmedian(binflux,axis=1)
        profilefluxerr = np.nanmedian(binfluxerr[:,xlo:xhi],axis=1)/np.sqrt(nbinflux)
    elif avgtype == 'smoothmedian':
        # First "smooth" in X, then take the median
        smflux = utils.smooth(flux,[1,2*nrepeat+1])
        smfluxerr = np.sqrt(utils.smooth(fluxerr**2,[2*nrepeat+1]))
        profileflux = np.nanmedian(smflux[:,xlo:xhi],axis=1)
        profilefluxerr = np.nanmedian(smfluxerr[:,xlo:xhi],axis=1)/np.sqrt(nxpix)        
    elif avgtype == 'rollmedian':        
        # First "smooth" by repeating the peaks multiple times shifted
        # (basically boxcar smoothing), then taking the median
        smflux = np.zeros(flux.shape,float)
        smfluxerr = np.zeros(flux.shape,float)        
        for k in np.arange(-nrepeat//2,nrepeat//2):
            smflux += np.roll(flux,k,axis=1)
            smfluxerr += np.roll(fluxerr,k,axis=1)**2  # add in quadrature
        smfluxerr = np.sqrt(smfluxerr)
        profileflux = np.nanmedian(smflux[:,xlo:xhi],axis=1)
        profilefluxerr = np.nanmedian(smfluxerr[:,xlo:xhi],axis=1)/np.sqrt(nxpix)

    # Get template trace center
    ytempcent = np.nanmedian(traceim[:,xlo:xhi],axis=1)
    tab['ytemp'] = ytempcent
    
    # Measure rough flux in each fiber and S/N
    boxflux = np.zeros(ntraces,float)
    fiberycen = np.zeros(ntraces,float)
    for j in range(ntraces):
        ylo = np.maximum(int(np.round(ytempcent[j]))-2,0)
        yhi = np.minimum(int(np.round(ytempcent[j]))+3,2048)
        totflux = np.sum(profileflux[ylo:yhi])
        totfluxerr = np.sqrt(np.sum(profilefluxerr[ylo:yhi]**2))
        tab['flux'][j] = totflux
        tab['snr'][j] = totflux/totfluxerr

    # Find bright fibers to measure
    if fibers is None:
        fibers, = np.where((tab['flux'] > 1000) | (tab['snr'] > 100))
        if len(fibers)<5:
            fibers, = np.where((tab['flux'] > 500) | (tab['snr'] > 50))
        if len(fibers)<5:
            fibers, = np.where((tab['flux'] > 100) | (tab['snr'] > 10))
        if len(fibers)<5:
            fibers = np.argsort(tab['flux'])[0:30]  # take brightest 30 fibers
    nfibers = len(fibers)
    tab['bright'][fibers] = True
    
    # Loop over fibers to measure
    y = np.arange(2048)
    gcent = np.zeros(nfibers,float)
    offset = np.zeros(nfibers,float)
    for j in range(nfibers):
        ind = fibers[j]
        ytemp = ytempcent[ind]
        if fitmethod == 'gaussian':
            # Fit Gaussian
            lo = int(np.floor(ytemp-3))
            hi = int(np.ceil(ytemp+3))
            yy = np.arange(hi-lo+1)+lo
            ff = profileflux[lo:hi+1]
            initpar = [ff[3],ytemp,1.0,0.0]
            try:
                pars,pcov = dln.gaussfit(yy,ff,initpar=initpar,binned=True,bounds=(-np.inf,np.inf))
                perror = np.sqrt(np.diag(pcov))
                ycent = pars[1]
                ycenterr = perror[1]
            except:
                ycent = np.nan
                ycenterr = np.nan                    
        # Empirical centroids
        else:
            # apmkpsf_epsf.pro used this centroiding method
            #  to create the apEPSF reference values
            lo = np.maximum(int(np.round(ytemp)-2),0)
            hi = np.minimum(int(np.round(ytemp)+3),2048)
            yy = np.arange(hi-lo)+lo
            ff = np.maximum(profileflux[lo:hi],0)
            fferr = np.maximum(np.sqrt(ff),1)
            ycent = np.sum(yy*ff)/np.sum(ff)
            ycenterr = np.sqrt(np.sum((yy*fferr)**2))/np.sum(ff)
        tab['ycent'][ind] = ycent
        tab['ycenterr'][ind] = ycenterr        

    return tab


def getoffset(frame,traceframe,traceim):
    """
    Measure the spatial offset of an object exposure and the PSF model/traces.

    Parameters
    ----------
    frame : dict
       The 2D input dictionary with flux, err, mask and header.
    traceframe : dict
       The 2D input dictionary of the trace quartzflat image with flux, err, mask and header.
    traceim : numpy array
       APOGEE trace information (Y-position) from a trace file [Nfibers, 2048].

    Returns
    -------
    offcoef : numpy array
       Additive offset coefficients (4-elements) of the 2D linear equation:
          c0 + c1*X + c2*X*Y + c3*Y
    medoff : float
       Median offset.

    Example
    -------

    offcoef,medoff = getoffset(frame,traceframe,traceim)

    """

    fitmethod = 'centroid'
    #fitmethod = 'gaussian'
    
    # Find bright fibers and measure the centroid
    nfibers = traceim.shape[0]
    flux = frame['flux']
    header = frame['header']
    exptype = header['exptype'].lower()
    chip = header['chip'].strip().lower()

    # The IDL apmkpsf_epsf.pro that creates the apEPSF trace image
    # applies a scattered light correction  (scat_remove.pro)
    # We need to do that here as well to get consistent trace results
    #bot = np.median(flux[5:10+1,100:1947+1])
    #top = np.median(flux[2038:2042+1,100:1947+1])
    #scatlevel = (bot+top)/2.
    ##print,'scatlevel: ',scatlevel
    #flux -= scatlevel

    nrepeat = 0
    # Use different X positions for arclamps
    if exptype == 'arclamp' and header['LAMPUNE']:
        avgtype = 'sum'
        xdict = {'a':[415,607,1490,2022], 'b':[90,594,1460], 'c':[1220,1750,2020]}
        nxbin = 20
        xx = xdict[chip]
    # THARNE
    elif exptype == 'arclamp' and header['LAMPTHAR']:
        #avgtype = 'sum'
        #avgtype = 'rollmedian'
        avgtype = 'smoothmedian'
        nrepeat = 15        
        #xdict = {'a':[60,950,1840], 'b':[905,1110,1570,1870], 'c':[1240,1780,1860,2010]}
        xdict = {'a':[950], 'b':[905,1110,1570,1870], 'c':[1240,1780,1860,2010]}        
        nxbin = 15
        xx = xdict[chip]        
    # FPI
    elif exptype == 'arclamp' and header['LAMPUNE']==False and header['LAMPTHAR']==False:
        avgtype = 'rollmedian'
        nrepeat = {'a':16,'b':12,'c':10}[chip]
        nxbin = 100
        xx = [512, 1024, 1536] 
    # Object/dome/quartz exposures
    else:
        avgtype = 'median'
        nxbin = 100
        #xx = [204, 614, 1024, 1434, 1844]
        xx = [512, 1024, 1536]                

    # Loop over X column locations
    ntraces = traceim.shape[0]
    coef = np.zeros((len(xx),2),float) + np.nan
    ngood = np.zeros(len(xx),int)
    sigma = np.zeros(len(xx),float)
    alloffset = np.array([],float)
    tab = np.zeros((len(xx),ntraces),dtype=np.dtype([('fiber',int),('x',float),('ytemp',float),('flux',float),('snr',float),
                                                     ('ycent',float),('ycenterr',float),('yoffset',float),('bright',bool)]))
    tab['ycent'] = np.nan
    tab['yoffset'] = np.nan    
    tabcount = 0
    for i,x in enumerate(xx):

        # Measure trace values from this exposure
        tab1 = measuretrace(frame,traceim,x,nxbin,avgtype=avgtype,nrepeat=nrepeat,
                            fibers=None,fitmethod=fitmethod)
        gdfiber, = np.where(tab1['bright']==True)
        ngood[i] = len(gdfiber)
        fibers = tab1['fiber'][gdfiber]
        # Measure trace values from quartzflat
        reftab1 = measuretrace(traceframe,traceim,x,nxbin,avgtype=avgtype,nrepeat=nrepeat,
                               fibers=fibers,fitmethod=fitmethod)
        # Measure the offsets
        tab1['yoffset'][gdfiber] = tab1['ycent'][gdfiber] - reftab1['ycent'][gdfiber]
        tab[i,:] = tab1
        yoffset = tab1['yoffset'][gdfiber]
        ycen = tab1['ycent'][gdfiber]

        # Fit line to it
        medoff = np.nanmedian(yoffset)
        sigoff = np.maximum(dln.mad(yoffset[np.isfinite(yoffset)]),0.02)
        sigma[i] = sigoff
        gd, = np.where(np.isfinite(yoffset) & (np.abs(yoffset-medoff) < 3*sigoff))
        if len(gd) > 5:
            coef1 = np.polyfit(ycen[gd],yoffset[gd],1)
            coef[i,:] = coef1
        else:
            coef[i,:] = [0.0, medoff]
        alloffset = np.hstack((alloffset,yoffset))
        
    avgngood = np.mean(ngood)
    print('Average bright fibers = {:.1f}'.format(avgngood))
    if avgngood < 5:
        print('Not enough bright fibers to measure the offset. Assuming zero offset.')
        # c0 + c1*x + c2*x*y + c3*y 
        coef2 = np.zeros(4,float)
        medoff = 0.0
        return coef2,medoff,[]
        
    # Fit 2D linear model
    if len(xx) >= 3:
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

    # Not enough X columns to fit 2-D model, use 1-D instead        
    else:
        # c0 + c1*x + c2*x*y + c3*y 
        print('Not enough columns to fit 2-D model.  Using 1-D.')
        coef2 = np.zeros(4,float)
        coef2[0] = np.mean(coef[:,1])  # constant term
        coef2[3] = np.mean(coef[:,0])  # linear y term

    medoff = np.nanmedian(alloffset)
    sigoff = np.mean(sigma)/np.sqrt(len(alloffset))
    print('Median offset = {:.3f} +/- {:.4f} pixels'.format(medoff,sigoff))
    print('Offset coefficients = ',coef2)

    return coef2,medoff,tab


def fullepsfgrid(psf,traceim,fibers,offcoef,verbose=True):
    """
    Generate a full EPSF grid for all fibers and columns and applying spatial offsets.

    Parameters
    ----------
    psf : 
       PSF information.
    traceim : numpy array
       APOGEE trace information (Y-position) from a trace file [Nfibers, 2048].
    fibers : list or numpy array
       List of fiber numbers.
    offcoef : numpy array
       Additive offset coefficients (4-elements) of the 2D linear equation:
         c0 + c1*X + c2*X*Y + c3*Y
    verbose : boolean, optional
       Verbose output to the screen.

    Returns
    -------
    epsf : list
      Empirical PSF model for the full image.

    Example
    -------

    epsf = fullepsfgrid(psf,traceim,fibers,offcoef)

    """
    
    nfibers = traceim.shape[0]
    if nfibers != len(fibers):
        raise ValueError('traceim dimensions do NOT agree with fibers')

    epsf = []
    # Fiber loop
    for i in range(len(fibers)):
        if verbose:
            if i % 50==0: print('fiber = ',i)
        off = func_poly2d([np.arange(2048),traceim[i,:]],*offcoef)
        ycen = traceim[i,:]+off
        ylo = int(np.min(np.round(ycen)))-14
        ylo = np.maximum(ylo,0)
        yhi = int(np.max(np.round(ycen)))+14
        yhi = np.minimum(yhi,2047)
        ny = yhi-ylo+1
        y = np.arange(ny)+ylo        
        img = np.zeros((ny,2048),float)
        # Column loop
        for j in range(2048):
            try:
                m1 = psf([j,ycen[j]],y=y,ycen=ycen[j])
            except:
                print('problem')
                import pdb; pdb.set_trace()
            m1 /= np.sum(m1)
            img[:,j] = m1
                
        data = {'fiber':fibers[i], 'lo':ylo, 'hi':yhi, 'img':img, 'ycen':ycen}
        epsf.append(data)
        
    return epsf
        

def extractwing(frame,modelpsffile,epsffile,tracefile):
    """
    Extract taking wings into account.

    Parameters
    ----------
    frame : dict
       The 2D input structure with flux, err, mask and header.
    modelpsffile : str
       Model PSF filename.
    epsffile : str
       Name of the EPSF filename.
    tracefile : str
       Name of the trace filename.

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

    outstr,back,model = extractwing(frame,modelpsffile,epsffile,tracefile)

    """

    # Ideas for extraction with wings if I can't fit fiber and 4 neighbors simultaneously:
    # 1) do usual fiber + 2 neighbor extraction using narrower profile
    # 2) create model using the broad profile and find the residual of data-model.
    # 3) loop through each fiber and add its broad profile back in (this is the same as
    #  subtracting all other fibers only)
    # use the narrow profile to find improved flux using weighted mean of best scaled profile
    # -can iterate if wanted
    # -could do this just around bright stars?

    # Load PSF
    psf = PSF.read(modelpsffile)

    # Load the data
    if type(frame) is str:
        framefile = frame
        frame = loadframe(framefile)
    # Load the trace imformation
    traceim = fits.getdata(tracefile,0)  # [Nfibers,2048]
    nfibers,npix = traceim.shape
    # Load the 2D image for the trace quartzflat
    traceid = os.path.basename(tracefile)
    chip = traceid.split('-')[1]
    traceid = int(traceid.split('-')[2][0:8])  # asETrace-a-44840002.fits
    traceframefile = load.filename('2D',num=traceid,chips=True).replace('2D','2D-'+chip)
    traceframe = loadframe(traceframefile)
    
    # Load the EPSF fiber information
    # Need this to get the missing fiber numbers
    hdu = fits.open(epsffile)
    fibers = []
    for i in np.arange(1,len(hdu)):
        fibers.append(hdu[i].data['FIBER'][0])
    hdu.close()

    # Step 1) Measure the offset
    #  returns 2D linear of the offset
    #  c0 + c1*x + c2*x*y + c3*y
    offcoef,medoff,tab = getoffset(frame,traceframe,traceim)

    # Step 2) Generate full PSFs for this image
    # Generate the input that extract() expects
    # this currently takes about 176 sec. to run
    print('Generating full EPSF grid with spatial offsets')
    epsf = fullepsfgrid(psf,traceim,fibers,offcoef)
    #np.savez('fullepsfgrid.npz',epsf=epsf)
    #epsf = np.load('fullepsfgrid.npz',allow_pickle=True)['epsf']
    
    # Step 3) Regular fiber+2 neighbor extraction
    out1,back1,model1 = extract(frame,epsf)
    
    # Step 4) Subtract all profiles except the fibers+2 neighbors and refit
    out,back,model = extract(frame,epsf,guess=out1['flux'])

    # Add information to header
    out['header']['HISTORY'] = 'psf.extractwing: Extracting '+str(nfibers)+' fibers at '+time.asctime()
    out['header']['HISTORY'] = 'psf.extractwing: EPSF file: '+epsffile
    out['header']['HISTORY'] = 'psf.extractwing: Median Trace offset %.3f pixels' % medoff
    out['header']['medtroff'] = medoff
    out['header']['HISTORY'] = 'psf.extractwing: Additive trace offset coefficients:'
    out['header']['HISTORY'] = 'psf.extractwing: %.3e %.3e %.3e %.3e' % tuple(offcoef)
    out['header']['HISTORY'] = 'psf.extractwing: c0 + c1*X + c2*X*Y + c3*Y'
    out['header']['toffpar0'] = offcoef[0],'constant term'
    out['header']['toffpar1'] = offcoef[1],'X term'
    out['header']['toffpar2'] = offcoef[2],'X*Y term'
    out['header']['toffpar3'] = offcoef[3],'Y term'
    
    return out,back,model
