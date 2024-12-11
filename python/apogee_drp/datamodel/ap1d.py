import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from ..utils import apload

class Frame(object):
    """
    Data model for Frame/ap2D files
    """

    def __init__(self,flux,header=None,err=None,mask=None,filename=''):
        # Initialize the object
        self._flux = flux
        if header is None:
            self.header = fits.PrimaryHDU().header
        else:
            self.header = header
        self._err = err
        self._mask = mask
        self.filename = filename
        if flux.ndim==1:
            npix = len(flux)
            norder = 1
        else:
            norder,npix = flux.shape
        self.ndim = flux.ndim
        self.npix = npix
        self.datatype = 'Frame'
        self.instrument = 'APOGEE'
        
        if filename is not None and filename != '':
            if os.path.basename(filename)[:2]=='ap':
                self.observatory = 'apo'
            else:
                self.observatory = 'lco'
        else:
            self.observatory = None
            
    @property
    def flux(self):
        """ Return the flux array."""
        if hasattr(self,'_flux')==False or self._flux is None:
            return None
        return self._flux

    @property
    def err(self):
        """ Return the error array."""
        if hasattr(self,'_err')==False or self._err is None:
            return None
        return self._err

    @property
    def mask(self):
        """ Return the bitmask array."""
        if hasattr(self,'_mask')==False or self._mask is None:
            return None
        return self._mask

    def __getitem__(self,index):
        # return one of the spectra
        if isinstance(index,int)==False:
            raise ValueError('index must be an integer')
        if index>self.norder-1:
            raise IndexError('index '+str(index)+' is out of bounds for axis 0 with size '+str(self.norder))
        if self.norder > 1:
            # Get the individual spectra
            kw = {'header':self.header,'filename':self.filename}
            for c in ['err','mask']:
                if getattr(self,c) is not None and getattr(self,c).ndim>1:
                    kw[c] = getattr(self,c)[index,:]
            # Initialize the object
            sp = Frame(self.flux[index,:],**kw)
        else:
            sp = self
        return sp
        
    def __repr__(self):
        """ Print out the string representation of the apBPM object."""
        s = repr(self.__class__)+"\n"
        if self.instrument is not None:
            s += self.instrument+"\n"
        if self.filename is not None:
            s += "File = "+self.filename+"\n"
        if self.norder > 1:
            s += 'Dimensions: ['+str(self.npix)+','+str(self.norder)+']\n'
        else:
            s += 'Dimensions: ['+str(self.npix)+']\n'
        s += "Flux = "+str(self.flux)+"\n"
        if self.err is not None:
            s += "Err = "+str(self.err)+"\n"
        return s
    
    @classmethod
    def read(cls,fname=None,**kwargs):
        """ Read from file """
        if fname is None and len(kwargs)==0:
            raise ValueError("Must input filename or sdss_access/tree keyword parameters")
        # You can input a filename OR input the sdss_access/tree information
        #   obj, apred, telescope, [mjd]
        if fname is None and len(kwargs)>0:
            for c in ['num','apred','telescope']:
                if c not in kwargs.keys():
                    raise ValueError(c+' parameter must be input')
            load = apload.ApLoad(apred=kwargs['apred'],telescope=kwargs['telescope'])
            filename = load.filename('2D',num=kwargs['num'])
        else:
            filename = fname

        # APOGEE ap2D, bad pixel mask
        # HISTORY AP2D: Output File:
        # HISTORY AP2D:  HDU0 - Header only
        # HISTORY AP2D:  HDU1 - image (ADU)                                               
        # HISTORY AP2D:  HDU2 - error (ADU)                                               
        # HISTORY AP2D:  HDU3 - flag mask

        if os.path.exists(filename)==False:
            raise FileNotFoundError(filename)
        hdu = fits.open(filename)

        # Initialize the object
        data = Frame(hdu[1].data,header=hdu[0].header,err=hdu[2].data,
                     mask=hdu[3].data,filename=filename)
        hdu.close()
        
        return data

    def write(self,filename,overwrite=True):
        """ Write data to a file """

        chips = ['a','b','c']
        for i in range(len(chips)):
            ichip = chips[i]   # chip index, 0-first chip
            # HDU0 - header only
            hdu = fits.HDUList()
            hdu.append(fits.PrimaryHDU(header=self.header)) 
            # HDU1 - Flux
            flux = self.flux[i,:]  #frame_wave[i]['flux']
            if outlong:
                flux = np.round(flux).astype(int)
            else:
                flux = flux.astype(np.float32)
            hdu.append(fits.ImageHDU(flux.T))
            hdu[1].header['CTYPE1'] = 'Pixel'
            hdu[1].header['CTYPE2'] = 'Fiber'
            hdu[1].header['BUNIT'] = 'Flux (ADU)'
            hdu[1].header['EXTNAME'] = 'FLUX'
            # HDU2 - error
            err = errout(self.err[i,:])  #frame_wave[i]['err']) 
            if outlong:
                err = np.round(err).astype(np.int32) 
            else:
                err = err.astype(np.float32)
            hdu.append(fits.ImageHDU(err.T))
            hdu[2].header['CTYPE1'] = 'Pixel'
            hdu[2].header['CTYPE2'] = 'Fiber'
            hdu[2].header['BUNIT'] = 'Error (ADU)'
            hdu[2].header['EXTNAME'] = 'ERROR'
            # HDU3 - mask
            #mask = frame_wave[i]['mask']
            mask = self.mask[i,:].astype(np.int16)
            hdu.append(fits.ImageHDU(mask.T))
            hdu[3].header['CTYPE1'] = 'Pixel'
            hdu[3].header['CTYPE2'] = 'Fiber'
            hdu[3].header['EXTNAME'] = 'MASK'
            if self.wave is not None:
                # HDU4 - Wavelengths
                wave = self.wave[i,:] #frame_wave[i]['wavelength']
                hdu.append(fits.ImageHDU(wave.T))
                hdu[4].header['CTYPE1'] = 'Pixel'
                hdu[4].header['CTYPE2'] = 'Fiber'
                hdu[4].header['BUNIT'] = 'Wavelength (Angstroms)' 
                hdu[4].header['EXTNAME'] = 'WAVELENGTH'
                # HDU5 - Wavelength solution coefficients [DOUBLE]
                #-------------------------------------------------
                #wcoef = frame_wave[i]['wcoef'].astype(float)
                hdu.append(fits.ImageHDU(self.wcoef[i,:].astype(float)))
                hdu[5].header['CTYPE1'] = 'Pixel'
                hdu[5].header['CTYPE2'] = 'Parameters'
                hdu[5].header['BUNIT'] = 'Wavelength Coefficients'
                hdu[5].header['HISTORY'] = 'Wavelength Coefficients to be used with PIX2WAVE.PRO:'
                hdu[5].header['HISTORY'] = ' 1 Global additive pixel offset'
                hdu[5].header['HISTORY'] = ' 4 Sine Parameters'
                hdu[5].header['HISTORY'] = ' 7 Polynomial parameters (first is a zero-point offset'
                hdu[5].header['HISTORY'] = '                     in addition to the pixel offset)'
                hdu[5].header['EXTNAME'] = 'WAVECOEF'

            # Write the data to disk
            outdir = os.path.dirname(filename)
            outbase = os.path.basename(filename)
            outfile = os.path.join(outdir,outbase.replace('1D-','1D-'+chips[i]))
            if os.path.exists(outfile) and overwrite==True:
                os.remove(outfile)  # make sure there's nothing there
            hdu.writeto(outfile,overwrite=overwrite)
            hdu.close()
