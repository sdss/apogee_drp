#!/usr/bin/env python

import copy
import numpy as np
import os
import shutil
import time
from pathlib import Path
from dlnpyutils import utils as dln, bindata
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from astropy.table import Table
import statsmodels.api as sm
from ..utils import peakfit, mmm, apload, utils, plan
from numba import njit
import copy
import matplotlib
import matplotlib.pyplot as plt
import subprocess

WARNMASK = -16640
BADMASK = 16639
BADERR = 1.00000e+10
maskval = {'NOT_ENOUGH_PSF': 16384}
chips = ['a','b','c']


#####  EMPIRICAL PSF MODEL CLASS #######

def leaky_relu(z):
    """ This is the activation function used by default in all our neural networks. """
    return z*(z > 0) + 0.01*z*(z < 0)

def interpolate(x):
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

@njit(cache=True)
def func_poly2d_numba(x,y,pars):
    """ 2D polynomial surface"""
    npars = len(pars)
    a = np.zeros(len(x), dtype=np.float64)
    if npars==0:
        a[:] = pars[0]
    elif npars==3:
        a[:] = pars[0] + pars[1]*x + pars[2]*y
    elif npars==4:
        a[:] = pars[0] + pars[1]*x + pars[2]*x*y + pars[3]*y
    else:
        raise Exception('Only 0, 3, and 4 parameters supported')
    return a

@njit(cache=True)
def gridinterp(labels,xgrid,ygrid,grid):
    """ Interpolate model in the grid."""

    if labels[0]<0 or labels[0]>2047 or labels[1]<0 or labels[1]>2047:
        raise ValueError('X/Y must be between 0 and 2047')

    nxgrid,nygrid,npix = grid.shape
    
    # xgrid/ygrid are 2D [Nx,Ny] and not quite a regular rectangular grid
    # Find closest X position
    xind = np.searchsorted(xgrid[:,nygrid//2],labels[0])
    yind = np.searchsorted(ygrid[np.minimum(xind,nxgrid-1),:],labels[1])
    xind = np.searchsorted(xgrid[:,np.minimum(yind,nygrid-1)],labels[0])
    yind = np.searchsorted(ygrid[np.minimum(xind,nxgrid-1),:],labels[1])            
        
    # Find the closest points on the grid
    #------------------------------------
    # -- At corners, use corner values --
    # bottom left
    if xind==0 and yind==0:
        return grid[0,0,:]
    # top left
    if xind==0 and yind==nygrid:
        return grid[0,-1,:]
    # bottom right
    if xind==nxgrid and yind==0:        
        return grid[-1,0,:]
    # top right
    if xind==nxgrid and yind==nygrid:
        return grid[-1,-1,:]
        
    # -- Edges, use two points --
    # linearly interpolate so it's smooth        
    # Left
    #   use left-most X and interpolate only in Y
    if xind==0:
        yind1 = yind-1
        yind2 = yind
        wt = (labels[1]-ygrid[xind,yind1])/(ygrid[xind,yind2]-ygrid[xind,yind1])
        profile = (1-wt)*grid[0,yind1,:] + wt*grid[0,yind2,:]
        return profile
    # Right
    #  use right-most X and interpolate only in Y
    if xind==nxgrid:
        yind1 = yind-1
        yind2 = yind
        wt = (labels[1]-ygrid[xind-1,yind1])/(ygrid[xind-1,yind2]-ygrid[xind-1,yind1])
        profile = (1-wt)*grid[-1,yind1,:] + wt*grid[-1,yind2,:]
        return profile
    # Bottom
    #  use Bottom-most Y and interpolate only in X
    if yind==0:
        xind1 = xind-1
        xind2 = xind
        wt = (labels[0]-xgrid[xind1,yind])/(xgrid[xind2,yind]-xgrid[xind1,yind])
        profile = (1-wt)*grid[xind1,0,:] + wt*grid[xind2,0,:]
        return profile
    # Top
    #  use top-most Y and interpolate only in X
    if yind==nygrid:
        xind1 = xind-1
        xind2 = xind
        wt = (labels[0]-xgrid[xind1,yind-1])/(xgrid[xind2,yind-1]-xgrid[xind1,yind-1])
        profile = (1-wt)*grid[xind1,-1,:] + wt*grid[xind2,-1,:]
        return profile
            
    # -- In the middle --
    # linearly interpolate so it's smooth
    xind1 = xind-1
    xind2 = xind
    yind1 = yind-1
    yind2 = yind
    # bilinear interpolation
    x1 = xgrid[xind1,yind1]
    x2 = xgrid[xind2,yind1]
    y1 = ygrid[xind1,yind1]
    y2 = ygrid[xind1,yind2]
        
    tx = (labels[0]-x1)/(x2-x1)
    ty = (labels[1]-y1)/(y2-y1)
    
    profile = (
        (1-tx)*(1-ty)*grid[xind1,yind1,:] +
        (1-tx)*ty    *grid[xind1,yind2,:] +
        tx    *(1-ty)*grid[xind2,yind1,:] +
        tx    *ty    *grid[xind2,yind2,:]
    )
        
    return profile

@njit(cache=True)
def build_fiber_epsf(trace_y,detector_y,profile_dy,xgrid,
                     ygrid,profile_grid,logscale):
    """
    Construct the effective PSF for every column of one fiber.

    At each detector column, the profile is interpolated from the
    irregular PSF grid at the fiber-center position. The oversampled
    profile is then resampled onto the requested detector rows and
    normalized to unit sum.

    Parameters
    ----------
    trace_y : ndarray, shape (n_columns,)
        Detector y coordinate of the fiber center at every column.

    detector_y : ndarray, shape (n_detector_rows,)
        Absolute detector-row coordinates at which the final profile
        should be evaluated. For example,
        ``[1005, 1006, ..., 1035, 1036]``.

    profile_dy : ndarray, shape (n_profile_samples,)
        Relative, possibly oversampled, y coordinates of the model
        profile with respect to its center. For example,
        ``[-14.95, -14.85, ..., 14.85, 14.95]``.

    xgrid, ygrid : ndarray, shape (nxgrid, nygrid)
        Detector coordinates of the irregular PSF grid.

    profile_grid : ndarray, shape (nxgrid, nygrid, n_profile_samples)
        Model profile at every PSF-grid position. Values may be linear
        or base-10 logarithmic, as specified by ``logscale``.

    logscale : bool
        If True, ``profile_grid`` contains base-10 logarithmic profile
        values, which are converted to linear values after interpolation.

    Returns
    -------
    epsf_image : ndarray, shape (n_detector_rows, n_columns)
        Normalized effective PSF evaluated at every requested detector
        row and detector column.

    Notes
    -----
    Values requested outside the range of ``profile_dy`` are assigned
    the corresponding endpoint value by ``np.interp``.
    """
    n_columns = trace_y.size
    n_detector_rows = detector_y.size

    epsf_image = np.empty(
        (n_detector_rows, n_columns),
        dtype=np.float64,
    )

    # Reuse this small array instead of allocating it in every iteration.
    grid_position = np.empty(2, dtype=np.float64)

    for column in range(n_columns):
        fiber_center_y = trace_y[column]

        grid_position[0] = column
        grid_position[1] = fiber_center_y

        oversampled_profile = gridinterp(
            grid_position,
            xgrid,
            ygrid,
            profile_grid,
        )

        # Convert absolute detector rows to offsets from the fiber
        # center and resample the model profile at those positions.
        detector_profile = np.interp(
            detector_y - fiber_center_y,
            profile_dy,
            oversampled_profile,
        )

        if logscale:
            detector_profile = 10.0 ** detector_profile

        profile_sum = np.sum(detector_profile)

        if profile_sum <= 0.0:
            raise ValueError("Interpolated profile has a non-positive sum")

        epsf_image[:, column] = detector_profile / profile_sum

    return epsf_image


@njit(cache=True)
def build_epsf_grid(trace_y,fiber_indices,offset_coefficients,profile_dy,
                    xgrid,ygrid,profile_grid,logscale):
    """
    Construct an effective-PSF image for multiple fiber traces.

    For each requested fiber, this function evaluates the polynomial trace
    offset, determines the detector rows covered by the profile, and
    interpolates the effective PSF at every detector column.

    Parameters
    ----------
    trace_y : ndarray, shape (n_trace_fibers, n_columns)
        Nominal detector y coordinate of every fiber trace as a function
        of detector column.

    fiber_indices : ndarray, shape (n_output_fibers,)
        Indices of the fibers in ``trace_y`` for which profiles should be
        generated.

    offset_coefficients : ndarray
        Coefficients passed to ``func_poly2d_numba`` to calculate the
        position-dependent y offset from the nominal trace.

    profile_dy : ndarray, shape (n_profile_samples,)
        Relative detector-y coordinates at which the model profiles are
        sampled.

    xgrid, ygrid : ndarray, shape (nxgrid, nygrid)
        Detector coordinates of the irregular effective-PSF grid.

    profile_grid : ndarray, shape (nxgrid, nygrid, n_profile_samples)
        Effective-PSF profiles sampled at each grid position.

    logscale : bool
        Passed to ``fiber_grid``. Indicates whether the profile-grid
        values are logarithmically scaled.

    Returns
    -------
    epsf_cube : ndarray, shape (n_output_fibers, 100, n_columns)
        Effective-PSF image for each requested fiber. Only rows
        ``0:profile_height`` contain the profile for a given fiber.

    trace_centers_y : ndarray, shape (n_output_fibers, n_columns)
        Offset-corrected y coordinate of each fiber center.

    row_start : ndarray, shape (n_output_fibers,)
        First detector row represented in each fiber's PSF image.

    row_stop : ndarray, shape (n_output_fibers,)
        Last detector row represented in each fiber's PSF image,
        inclusive.

    Notes
    -----
    The second dimension of ``epsf_cube`` is currently fixed at 100.
    Therefore, the detector-row range occupied by any fiber must not
    exceed 100 pixels.
    """
    n_trace_fibers, n_columns = trace_y.shape
    n_output_fibers = len(fiber_indices)

    max_profile_rows = 100
    detector_row_min = 0
    detector_row_max = 2047

    row_start = np.zeros(n_output_fibers, dtype=np.int64)
    row_stop = np.zeros(n_output_fibers, dtype=np.int64)

    trace_centers_y = np.zeros(
        (n_output_fibers, n_columns),
        dtype=np.float64,
    )
    epsf_cube = np.zeros(
        (n_output_fibers, max_profile_rows, n_columns),
        dtype=np.float64,
    )

    column_x = np.arange(n_columns)

    for output_index in range(n_output_fibers):
        fiber_index = fiber_indices[output_index]
        nominal_trace_y = trace_y[fiber_index, :]

        trace_offset_y = func_poly2d_numba(
            column_x,
            nominal_trace_y,
            offset_coefficients,
        )
        fiber_center_y = nominal_trace_y + trace_offset_y

        # Determine the detector rows covered by the profile. The model
        # currently extends approximately 14 pixels from its center.
        first_row = int(np.round(np.min(fiber_center_y))) - 14
        last_row = int(np.round(np.max(fiber_center_y))) + 14

        first_row = max(first_row, detector_row_min)
        last_row = min(last_row, detector_row_max)

        profile_height = last_row - first_row + 1

        if profile_height > max_profile_rows:
            raise ValueError(
                "Fiber profile requires more than 100 detector rows"
            )

        detector_y = np.arange(first_row, last_row + 1)

        fiber_profile = build_fiber_epsf(
            fiber_center_y,
            detector_y,
            profile_dy,
            xgrid,
            ygrid,
            profile_grid,
            logscale,
        )

        epsf_cube[
            output_index,
            :profile_height,
            :,
        ] = fiber_profile

        trace_centers_y[output_index, :] = fiber_center_y
        row_start[output_index] = first_row
        row_stop[output_index] = last_row

    return epsf_cube, trace_centers_y, row_start, row_stop



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
            coef[i,:] = np.polyfit(self.x[lo:hi],self.y[lo:hi],2)
            #coef[i,:] = dln.quadratic_coefficients(self.x[lo:hi],self.y[lo:hi])  # a,b,c
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
            # Adding a constant changes only the constant polynomial term.
            new._coef[:, 2] += other
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
            # Subtracting a constant changes only the constant term.
            new._coef[:, 2] -= other
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
            # Make sure the arrays are native-endian, numba needs this
            for c in ['_labels','_xgrid','_ygrid','_grid','y']:
                data = getattr(self,c)
                data = np.asarray(data, dtype=data.dtype.newbyteorder("="))
                setattr(self,c,data)
            nxgrid,nygrid,npix = grid.shape
        else:
            raise ValueError('Only "ann" and "grid" supported at this time')
        self.npix = len(self.y)
        self._nxgrid = nxgrid
        self._nygrid = nygrid

    def __str__(self):
        """ String representation of the PSF."""
        return self.__class__.__name__+'(%.1f<X<%.1f, %.1f<Y<%.1f, %s, Npix=%d)' % \
                                        (self.xmin[0],self.xmax[0],self.xmin[1],self.xmax[1],self,kind,self.npix)

    def __repr__(self):
        """ String representation of the PSF."""
        return self.__class__.__name__+'(%.1f<X<%.1f, %.1f<Y<%.1f, %s, Npix=%d)' % \
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

        labels = np.asarray(labels, dtype=np.float64)
        
        if self._grid is None:
            self.mkgrid()

        if self.kind=='grid':
            return gridinterp(labels,self._xgrid,self._ygrid,self._grid)
        else:
            raise Exception('not implemented yet')
        
    # Make a new method that does the interpolation for an entire fiber all at once (all 2048 pixels)
    # might allow for some speedups.  Would need to have y values (trace) input.
    def fiber(self,trace_y):
        """ Construct profiles for all columns of a fiber."""
        # all y-values must be given
        trace_y = np.asarray(trace_y, dtype=trace_y.dtype.newbyteorder("="))
        epsfimg = fiber_grid(trace_y, self.y, self._xgrid, self._ygrid, self._grid, self._log)
        return epsfimg


    def buildepsf(self,traceim_y,fibers=np.arange(300),offcoef=np.zeros(4)):
        """Construct profiles for the requested fibers and columns."""

        # Numba requires native-endian arrays.
        traceim_y = np.asarray(traceim_y,dtype=traceim_y.dtype.newbyteorder("="))
        ntrace, ncolumns = traceim_y.shape

        if fibers is None:
            fibers = np.arange(ntrace)

        fibers = np.asarray(fibers,dtype=np.int64)
        nfibers = len(fibers)

        if offcoef is None:
            offcoef = np.zeros(4)
        offcoef = np.asarray(offcoef,dtype=np.float64)

        epsfimg, ycen, ylo, yhi = build_epsf_grid(traceim_y,fibers,offcoef,self.y,
                                                  self._xgrid,self._ygrid,self._grid,self._log)
        
        epsf = []
        for i in range(nfibers):
            ny = yhi[i] - ylo[i] + 1
            data = {"fiber": fibers[i],"lo": ylo[i],"hi": yhi[i],
                    "img": epsfimg[i, :ny, :],"ycen": ycen[i, :]}
            epsf.append(data)

        return epsf

    
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
        hdu[0].header['V_APRED'] = (plan.getgitvers(), 'APOGEE software version')
        hdu[0].header['EXTNAME'] = 'DATA'
        hdu.append(fits.ImageHDU(self._labels))
        hdu[1].header['COMMENT'] = 'Labels'
        hdu[1].header['EXTNAME'] = 'LABELS'
        hdu.append(fits.ImageHDU(self.y))
        hdu[2].header['COMMENT'] = 'x'
        hdu[2].header['EXTNAME'] = 'X'
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

    ndata = len(data)
    xdata = data[:,0]
    ydata = np.log10(data[:,1])

    # Bspline with one 10 sigma outlier rejection round
    spl = dln.bspline(xdata,ydata)
    diff = ydata-spl(xdata)
    sig = dln.mad(diff)
    good, = np.where(np.abs(diff) < 10*sig)
    spl = dln.bspline(xdata[good],ydata[good])
    
    #plt.clf()
    #plt.scatter(xdata,ydata,s=10)
    #plt.scatter(xdata[good],ydata[good],s=10)
    #xx = np.arange(-6.5,6.5,0.1)
    #plt.scatter(xx,spl(xx),c='green')
    ##plt.scatter(xbin1,ybin1,s=100,c='green')
    ##plt.plot(xbin1,ybin1,c='green')
        
    # Binning
    xr = [-7.0,7.0]
    binsize = 0.10
    nbins = int(np.ceil((xr[1]-xr[0])/binsize)+1)
    bins = np.linspace(xr[0],xr[1],nbins)
    ybin, bin_edges, binnumber = bindata.binned_statistic(xdata[good],ydata[good],statistic='percentile',
                                                          percentile=50,bins=bins)
    xbin = bin_edges[0:-1]+0.5*binsize
    ybin = 10**ybin  # back to linear
    
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
        ybinsm[bd] = interp1d(xbin[~bad],ybinsm[~bad],bounds_error=False,fill_value=fill_value)(xbin[bd])

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

    # Get chip from psf filename
    psfbase = os.path.basename(psffile)
    chiptag = psfbase.split('-')[1]
    chip = {'a':0,'b':1,'c':2}[chiptag]
    # Load the sparse image
    allim,head = fits.getdata(sparsefile,0,header=True)
    sim = allim[chip,:,:]
    # Subtract scatter light from the Sparse image
    #  this takes the bottom 10 rows and takes the median in eight 256x10 chunks
    #  same for the top 10 rows
    medlobin = np.nanmedian(sim[5:15,:].T.reshape(8,10*256),axis=1)
    medhibin = np.nanmedian(sim[2030:2040,:].T.reshape(8,10*256),axis=1)
    xbin = np.arange(8)*256+128
    locoef = np.polyfit(xbin,medlobin,2)
    hicoef = np.polyfit(xbin,medhibin,2)
    x = np.arange(2048)
    medlo = np.polyval(locoef,x)
    medhi = np.polyval(hicoef,x)
    # Create ramp across the detector
    slp = (medhi-medlo)/(2035-10)
    off = medlo-slp*10
    xx,yy = np.meshgrid(np.arange(2048),np.arange(2048))
    scatim = yy*slp.reshape(1,-1) + medlo.reshape(1,-1)
    # Now subtract the scattered light from the sparse image
    print('Mean scattered light level = {:.1f} counts'.format(np.nanmedian(scatim)))
    sim -= scatim

    
    # Get fiber numbers for each PSF hdu
    psfhdu = fits.open(psffile)    
    fiber2hdu = mkfiber2hdu(psfhdu)
    
    fibers = np.arange(0,300,nfbin)
    columns = np.arange(10,2000,ncbin)

    # Get sparse data
    
    #data = np.zeros((len(fibers),len(columns),700),float)
    data = []
    mnx = np.zeros((len(columns),len(fibers)),float)
    mny = np.zeros((len(columns),len(fibers)),float)
    profiles = np.zeros((len(columns),len(fibers),300),float)
    fsparse = np.zeros((len(columns),len(fibers),31),float)    
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
        # we don't want fibers with close contaminating neighbors 
        gd, = np.where((ldiff >= 22) & (rdiff >= 22))
        ngd = len(gd)
        # 15
        linestr = linestr0[gd]
        
        # Fiber loop
        for j,f in enumerate(fibers):
            
            if verbose:
                print(f,c)
            data1, xbin,ybin,ybinsm = avgprofile([f,f+nfbin],[c,c+ncbin],psfhdu,fiber2hdu)            

            # Get average sparse flux profile
            # get median ytrace of the five fibers we are using in our "block"
            ytracearr = []
            for k in np.arange(f,f+nfbin):
                if fiber2hdu.get(k) is not None:
                    psfcat = psfhdu[fiber2hdu[k]].data
                    ytracearr.append(np.median(psfcat['CENT']))
            ytrace = np.median(np.array(ytracearr))  # average trace of our 5 fibers
            diff = linestr['pars'][:,1]-ytrace
            si = np.argsort(np.abs(diff))
            useind = si[0:5]
            fluxsparsearr = np.zeros([5,31],float)
            for k in range(5):
                bestind = useind[k]
                linestr1 = linestr[bestind]
                ycensparse = linestr1['pars'][1]
                dysparse = np.arange(31).astype(float)-15
                fluxsparse1 = sflux[int(round(ycensparse))-15:int(round(ycensparse))+16].copy()
                fluxsparse1 /= np.sum(fluxsparse1)   # normalize
                fluxsparsearr[k,:] = fluxsparse1
            # Now average with outlier rejection
            fluxsparsearr = np.log10(np.maximum(fluxsparsearr,1e-6))
            medfluxsparse = np.median(fluxsparsearr,axis=0)
            fluxdiff = fluxsparsearr-medfluxsparse.reshape(1,-1)
            sigfluxsparse = dln.mad(fluxdiff)
            # Mask outlier pixels
            goodmask = (np.abs(fluxdiff) < 5*sigfluxsparse)
            tempfluxsparse = fluxsparsearr.copy()
            tempfluxsparse[~goodmask] = np.nan
            fluxsparse = np.nanmedian(tempfluxsparse,axis=0)
            # Fix NaNs with median value
            bdnan, = np.where(~np.isfinite(fluxsparse))
            if len(bdnan)>0:
                fluxsparse[bdnan] = medfluxsparse[bdnan]
            # Back to linear
            fluxsparse = 10**fluxsparse
            
            # Replace very low values with point on opposite side
            mededge = np.median(np.concatenate((fluxsparse[:4],fluxsparse[-4:])))
            bad, = np.where( (fluxsparse < 1e-5) |
                             (fluxsparse < mededge/3))
            if len(bad)>0:
                good = len(fluxsparse)-bad-1
                fluxsparse[bad] = fluxsparse[good]
                if verbose:
                    print('fixing fluxsparse edge bad value')
                #import matplotlib.pyplot as plt
                #import pdb; pdb.set_trace()
            # If still low, use second point
            if fluxsparse[0]<1e-5:
                fluxsparse[0] = fluxsparse[1]
            if fluxsparse[-1]<1e-5:
                fluxsparse[-1] = fluxsparse[-2] 
            fluxsparse /= np.sum(fluxsparse)   # normalize again          
            ymnsparse = np.sum(dysparse*fluxsparse)/np.sum(fluxsparse)
            dysparse -= ymnsparse


            #import matplotlib.pyplot as plt
            #import matplotlib
            #matplotlib.use('Qt5Agg')
            #plt.clf()
            #plt.scatter(dysparse,fluxsparse,c='blue',s=100,marker='+')
            #plt.plot(dysparse,fluxsparse,c='blue')
            #plt.yscale('log')
            #plt.plot(xbin,ybinsm,c='r')
            #plt.show()
            #import pdb; pdb.set_trace()
            
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

            # switch to the sparse curve around x~3, around 3sigma
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

            if np.min(yprofile)<1e-6:
                print('some low fluxes <1e-6')
                #import matplotlib.pyplot as plt
                #import pdb; pdb.set_trace()

            if 0:
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')
                plt.figure()
                plt.scatter(data1[:,0],data1[:,1],s=5)
                plt.plot(xbin,ybin,c='r',label='binned')
                #plt.plot(lowess[:,0],lowess[:,1],c='g')
                plt.plot(xbin,ybinsm,c='b',label='smoothed binned')  
                plt.yscale('log')
                plt.xlim(-8,8)
                plt.ylim(1e-5,1)
                plt.xlabel('Pixel offset')
                plt.ylabel('Profile flux')
                plt.title('fiber='+str(f)+' column='+str(c))
                plt.legend()
                plt.savefig('gridprofile_fiber'+str(f)+'_column'+str(c)+'.png',bbox_inches='tight')
                plt.close()
                #plt.show()
                #import pdb; pdb.set_trace()

            #if i==4 and j==53:
            #    print('problem profile')
            #    import pdb; pdb.set_trace()
            
            data.append( [xbin,ybin,ybinsm,f,c] )
            mnx[i,j] = np.median(data1[:,2])
            mny[i,j] = np.median(data1[:,3])
            profiles[i,j,:] = yprofile
            fsparse[i,j,:] = fluxsparse

            #import pdb; pdb.set_trace()
            
    return data,mnx,mny,profiles,xx,fsparse


def mkmodelpsf(name,psfid,sparseid,apred,telescope,nfbin=5,ncbin=200,verbose=False):
    """
    Makes the Model PSF calibration file.

    Parameters
    ----------
    name : str
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
        data,mnx,mny,profiles,y,fsparse = makeprofilegrid(psffile1,sparsefile,verbose=verbose)
        labels = [mnx,mny]
        p = PSF((profiles,labels,y),kind='grid',log=False)
        outfile = load.filename('PSFModel',num=name,chips=True).replace('PSFModel-','PSFModel-'+ch+'-')
        print('Writing to '+outfile)
        p.write(outfile)

        # Save a diagnostic plot
        matplotlib.use('Agg')
        nx,ny,_ = profiles.shape
        for i in range(nx):
            for j in range(ny):
                plt.plot(y,profiles[i,j,:])
        plt.xlabel('Pixel Offset')
        plt.ylabel('Normalized Flux')
        plt.title(os.path.basename(outfile))
        plt.yscale('log')
        pltdir = os.path.dirname(outfile)+'/plots/'
        if os.path.exists(pltdir)==False:
            os.makedirs(pltdir,exist_ok=True)
        figfile = pltdir+os.path.basename(outfile).replace('.fits','.png')
        plt.savefig(figfile,bbox_inches='tight')

        
#####  EXTRACTION #######

def loadframe(infile):
    """ Load a 2D APOGEE image."""
    head = fits.getheader(infile,0)
    flux = fits.getdata(infile,1)
    err = fits.getdata(infile,2)
    mask = fits.getdata(infile,3)    
    frame = {'flux':flux, 'err':err, 'mask':mask, 'header':head}
    return frame

def saveepsf(filename, epsf, *, header=None, compress=True):
    """Write empirical PSF profiles to a FITS file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Output FITS filename.
    epsf : sequence of dict
        Empirical profiles. Each profile must contain ``fiber``, ``lo``,
        ``hi``, and ``img``. The optional ``cent`` entry contains the
        trace center in each detector column.
    header : astropy.io.fits.Header, optional
        Header cards to copy into the primary HDU.
    compress : bool, optional
        Compress the completed file using ``fpack``. Default is True for
        backward compatibility.
    """
    filename = str(filename)
    output_header = fits.Header() if header is None else header.copy()
    output_header["NTRACE"] = len(epsf)

    hdus = [fits.PrimaryHDU(header=output_header)]

    for profile in epsf:
        image = np.asarray(profile["img"])
        fields = [
            ("fiber", np.int32),
        ]

        if "cent" in profile:
            center = np.asarray(profile["cent"], dtype=np.float64)
            fields.append(("cent", np.float64, center.shape))

        fields.extend([
            ("lo", np.int32),
            ("hi", np.int32),
            ("img", np.float64, image.shape),
        ])

        row = np.zeros(1, dtype=np.dtype(fields))
        row["fiber"] = int(profile["fiber"])
        row["lo"] = int(profile["lo"])
        row["hi"] = int(profile["hi"])
        row["img"] = image

        if "cent" in profile:
            row["cent"] = center

        hdu = fits.table_to_hdu(Table(row))
        hdu.header["EXTNAME"] = f"EPSF{int(profile['fiber'])}"
        hdus.append(hdu)

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(filename, overwrite=True)

    if compress:
        compressed = filename + ".fz"
        if os.path.exists(compressed):
            os.remove(compressed)

        result = subprocess.run(
            ["fpack", "-D", "-Y", filename],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"fpack failed for {filename}")

#def saveepsf(filename,epsf,compress=True):
#    """
#    Save Empirical PSF data
#
#    Parameters
#    ----------
#    filename : str
#       Filename to save the EPSF information to.
#    epsf : list
#       Empirical PSF information.
#    compress : bool, optional
#       Fpack compress the EPSF file.  Default is True.
#
#    Results
#    -------
#    The empirical PSF information is saved to disk.
#    Nothing is returned.
#
#    Example
#    -------
#
#    saveepsf('apEPSFmodel-30330011.fits',epsf))
#
#    """
#
#    hdu = fits.HDUList()
#    hdu.append(fits.PrimaryHDU())
#    hdu[0].header['ntrace'] = len(epsf)
#    for i in range(len(epsf)):
#        if 'cent' in epsf[i].keys():
#            dt = [('fiber',int),('cent',float,len(epsf[i]['cent'])),('lo',int),('hi',int),('img',float,epsf[i]['img'].shape)]
#        else:
#            dt = [('fiber',int),('lo',int),('hi',int),('img',float,epsf[i]['img'].shape)]            
#        data = np.zeros(1,dtype=np.dtype(dt))
#        data['fiber'] = epsf[i]['fiber']
#        if 'cent' in epsf[i].keys():
#            data['cent'] = epsf[i]['cent']
#        data['lo'] = epsf[i]['lo']
#        data['hi'] = epsf[i]['hi']
#        data['img'] = epsf[i]['img']
#        hdu.append(fits.table_to_hdu(Table(data)))
#        hdu[i+1].header['EXTNAME'] = 'EPSF'+str(epsf[i]['fiber'])
#    hdu.writeto(filename,overwrite=True)
#    hdu.close()
#
#    if compress:
#        if os.path.exists(filename+'.fz'): os.remove(filename+'.fz')
#        sout = subprocess.run(['fpack','-D','-Y',filename],shell=False)
        
    # from apmkpsf_epsf.pro
    # file = apogee_filename('EPSF',chip=chip[ichip],num=im)
    # sxdelpar,head,'NAXIS1'
    # sxdelpar,head,'NAXIS2'
    # MWRFITS,0,file,head,/create
    #
    # # Put the PSFs in the output structure
    # for k=0,ntrace-1 do begin
    #   m = TOTAL(bpsf[*,*,k],1,/nan)
    #   ind = where(finite(m) and m ne 0)
    #   i1 = MIN(ind)
    #   i2 = MAX(ind)
    #   if i1 ge 0 then begin
    #     outpsf = {fiber: fiber[k], cent: trace[*,k], lo: i1, hi: i2, img: bpsf[*,i1:i2,k]}
    #     MWRFITS,outpsf,file,/silent
    #   endif else print,'not halted, but bad PSF at: ',k
    # endfor

def loadepsf(filename):
    """Load empirical PSF profiles from a FITS file.

    Parameters
    ----------
    filename : str or pathlib.Path
        EPSF FITS filename.

    Returns
    -------
    epsf : list of dict
        Empirical PSF profiles. Each dictionary contains ``fiber``,
        ``lo``, ``hi``, and ``img`` and includes ``cent`` when present.
    """
    filename = str(filename)

    with fits.open(filename, memmap=False) as hdus:
        ntrace = int(hdus[0].header.get("NTRACE", len(hdus) - 1))

        if ntrace > len(hdus) - 1:
            raise ValueError(
                f"NTRACE={ntrace} but {filename} contains only "
                f"{len(hdus) - 1} profile extensions"
            )

        epsf = []
        for hdu in hdus[1:ntrace + 1]:
            if hdu.data is None or len(hdu.data) == 0:
                raise ValueError(f"Empty EPSF extension in {filename}")
            names = {name.upper(): name for name in hdu.data.names}
            required = {"FIBER", "LO", "HI", "IMG"}
            missing = required - set(names)
            if missing:
                raise ValueError(
                    f"EPSF extension in {filename} is missing columns: "
                    f"{', '.join(sorted(missing))}"
                )

            row = hdu.data[0]
            profile = {
                "fiber": int(row[names["FIBER"]]),
                "lo": int(row[names["LO"]]),
                "hi": int(row[names["HI"]]),
                "img": np.asarray(row[names["IMG"]]).copy(),
            }

            if "CENT" in names:
                profile["cent"] = np.asarray(row[names["CENT"]]).copy()

            epsf.append(profile)

    return epsf

    
#def loadepsf(infile):
#    """
#    Load Empirical PSF data
#    this takes a while
#
#    Parameters
#    ----------
#    infile : str
#       Filename of apEPSF file.
#
#    Returns
#    -------
#    epsf : list
#       List of dictionaries with information on each trace.
#
#    Example
#    -------
#
#    epsf = loadepsf(infile)
# 
#    """
#    phead = fits.getheader(infile,0)
#    ntrace = phead.get('ntrace')
#    if ntrace is None:
#        print('No NTRACE in header')
#        return []
#    epsf = []
#    hdu = fits.open(infile)
#    for itrace in range(ntrace):
#        ptmp = hdu[itrace+1].data
#        data = {'fiber': ptmp['FIBER'][0], 'lo': ptmp['LO'][0], 'hi': ptmp['HI'][0], 'img': ptmp['IMG'][0]}
#        epsf.append(data)
#    hdu.close()
#    return epsf

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
    # No overlap
    if l1<0 or l2<0 or k1<0 or k2<0:
        out = np.zeros(2048,float)
        return out
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
    #for k in fibers:
    #    nf = 1
    #    ns = 0
    #    if subonly:
    #        junk, = np.where(subonly==k)
    #        nf = len(junk)
    #    if skip:
    #        junk, = np.where(skip==k)
    #        ns = len(junk)
    #    if nf > 0 and ns==0:
    #        p1 = epsf[k]
    #        lo = epsf[k]['lo']
    #        hi = epsf[k]['hi']
    #        img = p1['img'].T
    #        rows = np.ones(hi-lo+1,int)
    #        fiber = epsf[k]['fiber']
    #        model[:,lo-ylo:hi+1-ylo] += img[:,:]*(rows.reshape(-1,1)*t[:,fiber]).T                                    

    for k in fibers:
        include = True
        if subonly is not False and subonly is not None:
            subonly_array = np.asarray(subonly,dtype=int)
            include = np.any(subonly_array == k)
        if skip is not False and skip is not None:
            skip_array = np.asarray(skip,dtype=int)
            if np.any(skip_array == k):
                include = False
        if include:
            p1 = epsf[k]
            lo = p1["lo"]
            hi = p1["hi"]
            img = p1["img"].T
            fiber = p1["fiber"]
            model[:, lo-ylo:hi+1-ylo] += img * t[:, fiber].reshape(-1, 1)

    model = model.T
        
    return model


@njit(cache=True)
def solve_all_columns(
        tridiag,
        beta,
        betavar,
        psftot,
        fibers,
        warnmasked,
        badmasked,
        doback=False,
        nout=300,
        min_psf=0.5,
        baderr=1.0e10,
        initial_baderr=999999.09,
        not_enough_psf=16384,
):
    """
    Solve the tridiagonal extraction system for all detector columns.

    Parameters
    ----------
    tridiag : ndarray, shape (3, nsystem, ncol)
        Lower diagonal, main diagonal, and upper diagonal of the
        extraction matrix.

    beta, betavar : ndarray, shape (nsystem, ncol)
        Right-hand side and its variance.

    psftot : ndarray, shape (nsystem, ncol)
        Total valid PSF contribution for each spectrum and column.

    fibers : ndarray, shape (ntrace,)
        Output fiber index associated with each trace. The optional
        background element is not included in this array.

    warnmasked, badmasked : ndarray, shape (nsystem, ncol)
        Combined warning and bad-pixel mask values.

    doback : bool, optional
        If True, the final system element is the background.

    nout : int, optional
        Number of spectra in the output arrays.

    Returns
    -------
    spec : ndarray, shape (ncol, nout)
        Extracted spectra.

    err : ndarray, shape (ncol, nout)
        Extracted uncertainties.

    outmask : ndarray, shape (ncol, nout)
        Output pixel masks.

    back : ndarray, shape (ncol,)
        Extracted background. Zero when ``doback=False``.
    """
    nsystem = tridiag.shape[1]
    ncol = tridiag.shape[2]
    ntrace = fibers.size

    spec = np.zeros((ncol, nout), dtype=np.float64)
    err = np.full((ncol, nout), initial_baderr, dtype=np.float64)
    outmask = np.ones((ncol, nout), dtype=np.int64)
    back = np.zeros(ncol, dtype=np.float64)

    # Each detector column is independent.
    for i in range(4, ncol - 4):

        # Store the system indices having sufficient valid PSF coverage.
        good = np.empty(nsystem, dtype=np.int64)
        ngood = 0

        for k in range(nsystem):
            if psftot[k, i] > min_psf:
                good[ngood] = k
                ngood += 1

        if ngood == 0:
            for fiber in range(nout):
                spec[i, fiber] = 0.0
                err[i, fiber] = baderr

            for k in range(ntrace):
                fiber = fibers[k]
                outmask[i, fiber] = (
                    not_enough_psf | badmasked[k, i]
                )

            continue

        # Local work arrays. These replace the small temporary arrays
        # created by advanced NumPy indexing for every detector column.
        a = np.empty(ngood, dtype=np.float64)
        b = np.empty(ngood, dtype=np.float64)
        c = np.empty(ngood, dtype=np.float64)
        v = np.empty(ngood, dtype=np.float64)
        vvar = np.empty(ngood, dtype=np.float64)

        for j in range(ngood):
            k = good[j]

            a[j] = tridiag[0, k, i]
            b[j] = tridiag[1, k, i]
            c[j] = tridiag[2, k, i]
            v[j] = beta[k, i]
            vvar[j] = betavar[k, i]

            # Disconnect this system element from a rejected neighbor.
            # This is equivalent to modifying the corresponding entries
            # of tridiag in the original Python loop.
            if k > 0 and psftot[k - 1, i] <= min_psf:
                a[j] = 0.0

            if k < nsystem - 1 and psftot[k + 1, i] <= min_psf:
                c[j] = 0.0

        # Forward elimination.
        for j in range(1, ngood):
            m = a[j] / b[j - 1]
            b[j] -= m * c[j - 1]
            v[j] -= m * v[j - 1]
            vvar[j] += m * m * vvar[j - 1]

        # Back substitution.
        x = np.empty(ngood, dtype=np.float64)
        xvar = np.empty(ngood, dtype=np.float64)

        j = ngood - 1
        x[j] = v[j] / b[j]
        xvar[j] = vvar[j] / (b[j] * b[j])

        for j in range(ngood - 2, -1, -1):
            x[j] = (v[j] - c[j] * x[j + 1]) / b[j]
            xvar[j] = (
                vvar[j] + c[j] * c[j] * xvar[j + 1]
            ) / (b[j] * b[j])

        # Mark rejected traces first.
        for k in range(ntrace):
            if psftot[k, i] <= min_psf:
                fiber = fibers[k]
                outmask[i, fiber] = (
                    not_enough_psf | badmasked[k, i]
                )

        # Copy fitted trace values into their output fiber positions.
        # The optional background element has k == ntrace and is handled
        # separately below.
        for j in range(ngood):
            k = good[j]

            if k < ntrace:
                fiber = fibers[k]
                spec[i, fiber] = x[j]
                err[i, fiber] = np.sqrt(xvar[j])
                outmask[i, fiber] = 0

            elif doback and k == ntrace:
                back[i] = x[j]

        # Propagate warning bits to all trace outputs.
        for k in range(ntrace):
            fiber = fibers[k]
            outmask[i, fiber] |= warnmasked[k, i]

    return spec, err, outmask, back


def extract2(frame, epsf, doback=False, skip=False, scat=None,
            subonly=False, guess=None):
    """
    Extract spectra using an empirical PSF.

    The extraction assumes that a given detector pixel receives significant
    contributions from at most two neighboring traces. This produces a
    tridiagonal system that is solved independently for every detector
    column.

    Parameters
    ----------
    frame : dict
        Input 2D frame containing ``flux``, ``err``, ``mask``, and
        ``header``.

    epsf : list
        Empirical PSF information for each trace.

    doback : bool, optional
        Fit a background term. Default is False.

    skip : array-like or bool, optional
        Fibers to omit from the final model.

    scat : optional
        Scattered-light removal option.

    subonly : array-like or bool, optional
        If supplied, only these fibers are included in the final model.

    guess : ndarray, shape (2048, 300), optional
        Initial estimates of the extracted spectra. The complete guess
        model is subtracted before extraction. For each trace, the model
        contributions from that trace and its immediate neighbors are
        temporarily added back.

    Returns
    -------
    outstr : dict
        Extracted spectra containing ``flux``, ``err``, ``mask``, and
        ``header``.

    back : ndarray, shape (2048,)
        Extracted background.

    model : ndarray, shape (2048, 2048)
        Final two-dimensional model image.
    """
    ntrace = len(epsf)

    fibers = np.asarray(
        [e['fiber'] for e in epsf],
        dtype=np.int64,
    )

    # Internally use transposed detector images with shape
    # (spectral column, spatial row).
    flux = np.copy(frame['flux'].T)
    red = np.copy(frame['flux'].T)
    var = np.copy(frame['err'].T**2)
    inmask = np.copy(frame['mask'].T)

    if scat:
        red = scat_remove(red, scat=scat, mask=inmask)

    # Subtract the complete initial model once. epsfmodel() previously
    # clipped non-positive fluxes internally on every call. Do that once
    # here so that the same sanitized guess can be reused below.
    if guess is not None:
        guess1 = np.copy(guess)
        guess1[guess1 <= 0] = 0.0

        gmodel = epsfmodel(epsf, guess1)
        red -= gmodel.T
    else:
        guess1 = None

    # The optional background is an additional system component but is
    # not an output fiber.
    if doback:
        nback = 1
    else:
        nback = 0

    nsystem = ntrace + nback
    ncol = flux.shape[0]
    nout = 300

    # Arrays describing the extraction equations.
    beta = np.zeros((nsystem, ncol), dtype=float)
    betavar = np.zeros((nsystem, ncol), dtype=float)
    psftot = np.zeros((nsystem, ncol), dtype=float)
    tridiag = np.zeros((3, nsystem, ncol), dtype=float)

    warnmasked = np.zeros((nsystem, ncol), dtype=np.int64)
    badmasked = np.zeros((nsystem, ncol), dtype=np.int64)

    inmask_warn = inmask & WARNMASK
    inmask_bad = inmask & BADMASK

    # These retain the bounds of the last trace, matching the original
    # behavior of the optional background calculation.
    lo = 0
    hi = inmask.shape[1] - 1

    # Construct the extraction equations.
    for k in range(nsystem):

        # --------------------------------------------------------------
        # Background system element
        # --------------------------------------------------------------
        if k >= ntrace:
            beta[k, :] = np.nansum(
                red[:, lo:hi+1],
                axis=1,
            )

            betavar[k, :] = np.nansum(
                var[:, lo:hi+1],
                axis=1,
            )

            psftot[k, :] = 1.0
            tridiag[1, k, :] = hi - lo + 1
            continue

        # --------------------------------------------------------------
        # Trace system element
        # --------------------------------------------------------------
        if guess1 is not None:
            if k == 0:
                fibs = (k, k + 1)
            elif k == ntrace - 1:
                fibs = (k - 1, k)
            else:
                fibs = (k - 1, k, k + 1)

            # The complete guess model was subtracted above. Temporarily
            # add back this trace and its immediate neighbors.
            for j in fibs:
                pj = epsf[j]
                jlo = pj['lo']
                jhi = pj['hi']
                jfiber = pj['fiber']
                jimg = pj['img'].T

                red[:, jlo:jhi+1] += (
                    jimg * guess1[:, jfiber].reshape(-1, 1)
                )

        p1 = epsf[k]
        lo = p1['lo']
        hi = p1['hi']

        # Identify unusable detector pixels within this trace.
        trace_flux = flux[:, lo:hi+1]
        trace_mask = inmask[:, lo:hi+1]

        bad = (
            ~np.isfinite(trace_flux)
            | (trace_flux == 0)
            | ((trace_mask & BADMASK) > 0)
        )

        img = np.copy(p1['img'].T)

        if np.any(bad):
            img[bad] = np.nan

        # Combine warning and bad mask values along the spatial
        # footprint of the trace.
        warnmasked[k, :] = np.bitwise_or.reduce(
            inmask_warn[:, lo:hi+1],
            axis=1,
        )

        badmasked[k, :] = np.bitwise_or.reduce(
            inmask_bad[:, lo:hi+1],
            axis=1,
        )

        # Construct the right-hand side and its variance.
        psftot[k, :] = np.nansum(
            img,
            axis=1,
        )

        beta[k, :] = np.nansum(
            red[:, lo:hi+1] * img,
            axis=1,
        )

        betavar[k, :] = np.nansum(
            var[:, lo:hi+1] * img**2,
            axis=1,
        )

        if guess1 is not None:
            # Return red to the image with the complete initial model
            # subtracted.
            for j in fibs:
                pj = epsf[j]
                jlo = pj['lo']
                jhi = pj['hi']
                jfiber = pj['fiber']
                jimg = pj['img'].T

                red[:, jlo:jhi+1] -= (
                    jimg * guess1[:, jfiber].reshape(-1, 1)
                )

        # --------------------------------------------------------------
        # Construct the tridiagonal matrix
        # --------------------------------------------------------------
        if k == 0:
            # First trace: main diagonal and upper diagonal.
            tridiag[1, k, :] = extract_pmul(
                p1['lo'],
                p1['hi'],
                img,
                epsf[k],
            )

            tridiag[2, k, :] = extract_pmul(
                p1['lo'],
                p1['hi'],
                img,
                epsf[k + 1],
            )

        elif k == ntrace - 1:
            # Last trace: lower diagonal and main diagonal.
            tridiag[0, k, :] = extract_pmul(
                p1['lo'],
                p1['hi'],
                img,
                epsf[k - 1],
            )

            tridiag[1, k, :] = extract_pmul(
                p1['lo'],
                p1['hi'],
                img,
                epsf[k],
            )

        else:
            # Middle traces: lower, main, and upper diagonals.
            tridiag[0, k, :] = extract_pmul(
                p1['lo'],
                p1['hi'],
                img,
                epsf[k - 1],
            )

            tridiag[1, k, :] = extract_pmul(
                p1['lo'],
                p1['hi'],
                img,
                epsf[k],
            )

            tridiag[2, k, :] = extract_pmul(
                p1['lo'],
                p1['hi'],
                img,
                epsf[k + 1],
            )

    # Solve every detector column inside a single Numba call.
    spec, err, outmask, back = solve_all_columns(
        tridiag,
        beta,
        betavar,
        psftot,
        fibers,
        warnmasked,
        badmasked,
        doback=doback,
        nout=nout,
        min_psf=0.5,
        baderr=BADERR,
        initial_baderr=999999.09,
        not_enough_psf=maskval['NOT_ENOUGH_PSF'],
    )

    # Catch unexpected non-finite extracted values.
    bad = ~np.isfinite(spec)

    if np.any(bad):
        spec[bad] = 0.0
        err[bad] = BADERR
        outmask[bad] = 1

    outstr = {
        'flux': spec,
        'err': err,
        'mask': outmask,
        'header': frame['header'].copy(),
    }

    # Construct the final 2D model from the extracted spectra.
    model = epsfmodel(
        epsf,
        spec,
        subonly=subonly,
        skip=skip,
    )

    return outstr, back, model




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
            # Use numba to speed up this slow loop
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
    npp = len(p)
    if npp==0:
        a = p[0]
    elif npp==3:
        a = p[0] + p[1]*x + p[2]*y
    elif npp==4:
        a = p[0] + p[1]*x + p[2]*x*y + p[3]*y
    else:
        raise Exception('Only 0, 3, and 4 parameters supported')
    return a

def measuretrace(frame,traceim,xcen,nbin,avgtype='median',nrepeat=12,fibers=None,fitmethod='gaussian'):
    """
    Measure trace position given the data and method.  Called by getoffset().

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
    tab['fiber'] = np.arange(ntraces)
    
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
        smflux = utils.smooth(flux,[1,nrepeat])        
        smfluxerr = np.sqrt(utils.smooth(fluxerr**2,[1,2*nrepeat+1]))
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
            fibers, = np.where((tab['flux'] > 100) | (tab['snr'] > 25))
        if len(fibers)<5:
            fibers, = np.where((tab['flux'] > 50) | (tab['snr'] > 10))
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
    # empirical centroid is faster and gives similar precision
    
    # Find bright fibers and measure the centroid
    nfibers = traceim.shape[0]
    flux = frame['flux']
    header = frame['header']
    exptype = header['exptype'].lower()
    chip = header['chip'].strip().lower()

    nrepeat = 0
    # Use different X positions for arclamps
    if exptype == 'arclamp' and header['LAMPUNE']:
        avgtype = 'smoothmedian'
        nrepeat = 15
        #xdict = {'a':[416,613,1486,2022], 'b':[86,592,1457], 'c':[1221,1745,2008]}
        xdict = {'a':[416,613,1486], 'b':[86,592,1457], 'c':[1221,2008]}
        nxbin = 15
        xx = xdict[chip]
    # THARNE
    elif exptype == 'arclamp' and header['LAMPTHAR']:
        avgtype = 'smoothmedian'
        nrepeat = 15        
        #xdict = {'a':[56,946,1728,1843], 'b':[910,1112,1570,1872], 'c':[1240,1780,1861,2008]}
        xdict = {'a':[946,1728], 'b':[910,1112,1872], 'c':[1234,1745]}
        nxbin = 15
        xx = xdict[chip]        
    # FPI
    elif exptype == 'arclamp' and header['LAMPUNE']==False and header['LAMPTHAR']==False:
        avgtype = 'smoothmedian'
        nrepeat = {'a':16,'b':12,'c':10}[chip]
        nxbin = 25
        xx = [512, 1024, 1536] 
    # Object/dome/quartz exposures
    elif exptype == 'object' or exptype == 'skyflat':
        # Get flux values
        tab0 = measuretrace(frame,traceim,1024,50,avgtype='median',
                            fibers=np.arange(nfibers),fitmethod=fitmethod)
        # Check the fluxes
        gd, = np.where(tab0['snr'] > 50)
        # Essentially no continuume flux, use sky lines
        if len(gd) < 20:
            avgtype = 'smoothmedian'
            nrepeat = 15        
            #xdict = {'a':[249,531,1096,1727,1928], 'b':[166,431,728,874,1107,1445,1642],
            #         'c':[321,471,736,1159,1270,1457,1590,1728,1885]}
            xdict = {'a':[531,1096,1727,1928], 'b':[728,874,1107,1445,1642],
                     'c':[471,736,1159,1270,1457,1590,1728]}
            nxbin = 15
            xx = xdict[chip]        
        # Regular object exposure with lots of continuum flux in fibers
        else:
            avgtype = 'median'
            nxbin = 100
            xx = [512, 1024, 1536]
    # Dome/quartz exposures        
    else:
        avgtype = 'median'
        nxbin = 100
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

    # DEPRICATED!!  Use PSF.buildepsf() method now
    
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
        

def extractwing(frame,modelpsffile,epsffile,tracefile,trace2dfile):
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
    trace2dfile : str
       Name of the ap2D file used to generate the traces.

    Returns
    -------
    outstr : dict
       The 1D output structure with FLUX, VAR and MASK.
    back : numpy array
       The background
    model : numpy array
       The model 2D image
    epsf : epsf object
       The empirical EPSF generated for this exposure from the PSF model.

    Example
    -------

    outstr,back,model = extractwing(frame,modelpsffile,epsffile,tracefile,trace2dfile)

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
    traceframe = loadframe(trace2dfile)
    
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
    #epsf = fullepsfgrid(psf,traceim,fibers,offcoef)
    epsf = psf.buildepsf(traceim,fibers,offcoef)
    
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
    
    return out,back,model,epsf
