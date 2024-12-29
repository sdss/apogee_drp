import os
import numpy as np
from scipy.interpolate import interp1d,RegularGridInterpolator

""" Work with APOGEE Telluric models."""

class Telluric(object):

    def __init__(self,filename):
        hdu = fits.open(filename)
        self.wave = hdu[0].data[0,:]
        self.airmass = np.arange(7)*0.25+1.0
        self.scale = np.arange(4)*0.5+0.5
        points = (self.airmass,self.scale,np.arange(300),self.scale)
        data = np.zeros((3,7,4,300,4176),float)
        for s in range(3):
            for i in range(1,8):
                data[s,i-1,:,:,:] = hdu[i].data[:,s,:,:]
            self._data = data
            #self._interpolators[s] = RegularGridInterpolator(data)
        hdu.close()

    def interpolate(self,airmass,scale,species=0,fiber=0,wave=None,method='linear'):
        """ interpolate in airmass and scale """
        points = (self.airmass,self.scale,tel.wave)
        vals = tel._data[species,:,:,fiber,:]
        interpolator = RegularGridInterpolator(points,vals,method=method)
        coords = np.zeros((len(self.wave),3),float)
        coords[:,0] = airmass
        coords[:,1] = scale
        coords[:,2] = self.wave
        sp = interpolator(coords)
        # Interpolation in wavelength, use higher order interpolation here
        if wave is not None:
            sp1 = sp
            sp = interp1d(self.wave[::-1],sp1[::-1],kind='quadratic',assume_sorted=True)(wave)
        return sp

    def __getitem__(self,vals):
        """ get values or a spectrum."""
        # species, airmass, scale, fiber, wavelength
        # initialize with slice tuple that will return all elements
        slc = [5*[slice(None)]]
        vals = np.atleast_1d(vals)
        for i in range(len(vals)):
            slc[i] = vals[i]
        return self._data[slc].squeeze()  # remove any dangling dimensions

    def __call__(self,vals):
        pass

