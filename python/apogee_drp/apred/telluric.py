import os
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d,RegularGridInterpolator

""" Work with APOGEE Telluric models."""

chips = ['a','b','c']

def rangeoverlap(a,b):
    """ Does the range (start1, end1) overlap with (start2, end2) """
    return max(a) >= min(b) and min(a) <= max(b)

class TelluricChip(object):

    def __init__(self,filename,trim=True):
        hdu = fits.open(filename)
        self.wave = hdu[0].data[0,:]
        self.airmass = np.arange(7)*0.25+1.0
        self.scale = np.arange(4)*0.5+0.5
        points = (self.airmass,self.scale,np.arange(300),self.scale)
        data = np.zeros((3,7,4,300,4176),float)
        for s in range(3):
            for i in range(1,8):
                data[s,i-1,:,:,:] = hdu[i].data[:,s,:,:]
            self.data = data
            #self._interpolators[s] = RegularGridInterpolator(data)
        hdu.close()
        if trim:
            self.data = self.data[:,:,:,:,30:-30]
            self.wave = self.wave[30:-30]
        self.species = ['CH4','CO2','H2O']
        self.wrange = [np.min(self.wave),np.max(self.wave)]

    def __len__(self):
        """ Return the number of models."""
        return np.prod(self.shape[:-1])
            
    @property
    def shape(self):
        """ Return the shape. """
        return self.data.shape
        
    def interpolate(self,airmass,scale,species=0,fiber=0,wave=None,method='linear'):
        """ interpolate in airmass and scale """
        points = (self.airmass,self.scale,self.wave)
        vals = self.data[species,:,:,fiber,:]
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
        return self.data[slc].squeeze()  # remove any dangling dimensions

    def __call__(self,vals):
        pass


class Telluric(object):

    def __init__(self,filename):
        # Get all three chip filenames
        filenames = [filename.replace('Telluric-','Telluric-'+ch+'-') for ch in chips]
        self.data = 3*[None]
        self.wrange = np.zeros((3,2),float)
        for i in range(3):
            tchip = TelluricChip(filenames[i])
            self.data[i] = tchip
            self.wrange[i,:] = tchip.wrange

    def __len__(self):
        """ Return the number of models."""
        return np.prod(self.shape[:-1])
            
    @property
    def shape(self):
        """ Return the shape. """
        shape = tuple([3]+list(self.data[0].data.shape))
        return shape

    @property
    def wave(self):
        """ Return the wavelengths. """
        return np.array([d.wave for d in self.data])

    @property
    def species(self):
        """ Return the species names. """
        return self.data[0].species
    
    @property
    def airmass(self):
        """ Return the airmass array. """
        return self.data[0].airmass

    @property
    def scale(self):
        """ Return the scale array. """
        return self.data[0].scale

    def interpolate(self,airmass,scale,species=0,fiber=0,wave=None,method='linear'):
        """ interpolate in airmass and scale """
        if wave is not None:
            wr = [np.min(wave),np.max(wave)]
        else:
            wr = None
        out,owave,owr = [],[],[]
        for i in range(3):
            if wave is None or rangeoverlap(wr,self.data[i].wrange):
                out1 = self.data[i].interpolate(airmass,scale,species,fiber,method=method)
                out.append(out1)
                owave.append(self.data[i].wave)
                owr.append(self.data[i].wrange)
        # Interpolation in wavelength, use higher order interpolation here
        if wave is not None:
            wave1d = wave.ravel()
            ointerp = np.zeros(wave1d.shape,float)
            for i in range(len(out)):
                ind, = np.where((wave1d >= owr[i][0]) & (wave1d <= owr[i][1]))
                o = interp1d(owave[i][::-1],out[i][::-1],kind='quadratic',assume_sorted=True)(wave1d[ind])
                ointerp[ind] = o
            out = ointerp.reshape(wave.shape)
        else:
            out = np.array(out)
        return out

    def __getitem__(self,vals):
        """ get values or a spectrum."""
        # chip, species, airmass, scale, fiber, wavelength
        # initialize with slice tuple that will return all elements
        chip = vals[0]
        slc = [5*[slice(None)]]
        vals = np.atleast_1d(vals)
        for i in range(1,len(vals)):
            slc[i-1] = vals[i]
        return self.data[chip][slc]

    def __repr__(self):
        """ Print out the string representation of the Telluric object."""
        prefix = self.__class__.__name__ + '('
        # chips, species, airmass, scale, fiber, wavelength
        shape = self.shape
        nmodels = np.prod(shape[:-1])
        body = '{:d} models [{:d} chips, {:d} species, {:d} airmass, {:d} scale, {:d} fibers, {:d} waves]'.format(nmodels,*shape)
        out = ''.join([prefix, body, ')']) + '\n'
        return out
