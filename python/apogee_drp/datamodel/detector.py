import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from ..apred import wav,sincint
from ..utils import apload

class Detector(object):
    """
    Data model for apDetector files
    """

    def __init__(self,flux,header=None,err=None,wave=None,mask=None,filename=''):
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
        self.datatype = 'BPM'
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
            kw = {'header':self.header,'filename':self.filename,
                  'lsfcoef':self.lsfcoef,'rvtab':self.rvtab}
            for c in ['err','mask','sky','skyerr','telluric','telerr']:
                if getattr(self,c) is not None and getattr(self,c).ndim>1:
                    kw[c] = getattr(self,c)[index,:]
            # Initialize the object
            sp = BPM(self.flux[index,:],**kw)
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
        if self.snr is not None:
            s += ("S/N = %7.2f" % self.snr)+"\n"
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
            filename = load.filename('BPM',num=kwargs['num'])
        else:
            filename = fname
        
        # APOGEE apBPM, bad pixel mask
        # HISTORY APSTAR:  HDU0 = Header only
        # HISTORY APSTAR:  HDU1 - Flux (10^-17 ergs/s/cm^2/Ang)
        # HISTORY APSTAR:  HDU2 - Error (10^-17 ergs/s/cm^2/Ang)
        # HISTORY APSTAR:  HDU3 - Flag mask:
        # HISTORY APSTAR:    row 1: bitwise OR of all visits
        # HISTORY APSTAR:    row 2: bitwise AND of all visits
        # HISTORY APSTAR:    row 3-nvisits+2: individual visit masks

        if os.path.exists(filename)==False:
            raise FileNotFoundError(filename)
        hdu = fits.open(filename)

        # Initialize the object
        data = BPM(hdu[1].data,header=hdu[0].header,err=hdu[2].data,mask=hdu[3].data,
                   filename=filename)
        hdu.close()
        
        return data

    def write(self,filename,overwrite=True):
        """ Write data to a file """
        hdulist = fits.HDUList()
        hdu = fits.PrimaryHDU()
        hdu.header = self.header
        hdu.header['HISTORY'] = 'APOGEE Reduction Pipeline Version: {:s}'.format(os.environ['APOGEE_DRP_VER'])
        hdu.header['HISTORY'] = 'HDU0 : header'
        hdu.header['HISTORY'] = 'HDU1 : flux'
        hdu.header['HISTORY'] = 'HDU2 : flux uncertainty'
        hdu.header['HISTORY'] = 'HDU3 : pixel bitmask'
        hdulist.append(hdu)
        header = fits.Header()
        header['CRVAL1'] = hdu.header['CRVAL1']
        header['CDELT1'] = hdu.header['CDELT1']
        header['CRPIX1'] = hdu.header['CRPIX1']
        header['CTYPE1'] = hdu.header['CTYPE1']
        header['BUNIT'] = 'Flux (10^-17 erg/s/cm^2/Ang)'
        header['EXTNAME'] = 'FLUX'
        hdulist.append(fits.ImageHDU(self.flux,header=header))
        header['BUNIT'] = 'Err (10^-17 erg/s/cm^2/Ang)'
        header['EXTNAME'] = 'ERROR'
        hdulist.append(fits.ImageHDU(self.err,header=header))
        header['BUNIT'] = 'Pixel bitmask'
        header['EXTNAME'] = 'MASK'
        hdulist.append(fits.ImageHDU(self.mask,header=header))
        hdulist.writeto(filename,overwrite=overwrite)

