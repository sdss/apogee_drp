import os
import numpy as np
from astropy.io import fits
from ..utils import apload
from .base import APOGEEBase

chips = ['a','b','c']

def getfilename(fname=None,**kwargs):
    """ Get the filenames """
    if fname is None and len(kwargs)==0:
        raise ValueError("Must input filename or sdss_access/tree keyword parameters")
    # You can input a filename OR input the sdss_access/tree information
    #   num, apred, telescope
    if fname is None and len(kwargs)>0:
        if 'num' not in kwargs.keys():
            raise ValueError('num parameter must be input')
        # Local filename
        if 'apred' not in kwargs.keys() and 'telescope' not in kwargs.keys():
            load = apload.ApLoad(apred='daily',telescope='apo25m')
            filename = os.path.basename(load.filename('2D',num=kwargs['num'],chips=True))
        # Full filename
        elif 'apred' in kwargs.keys() and 'telescope' in kwargs.keys():
            load = apload.ApLoad(apred=kwargs['apred'],telescope=kwargs['telescope'])
            filename = load.filename('2D',num=kwargs['num'],chips=True)
        else:
            raise ValueError('apred and telescope must be input')
    else:
        filename = fname
    # Loop over the three chips
    fdir = os.path.dirname(filename)
    base = os.path.basename(filename)
    files = 3*[None]
    for i,ch in enumerate(chips):
        files[i] = os.path.join(fdir,base.replace('2D-','2D-'+ch+'-'))
    return files

class Frame(APOGEEBase):
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
        self.chipfiles = filename
        if filename is not None:
            if isinstance(filename,list):
                self.filename = filename[0].replace('-a-','-')
            else:
                self.filename = filename
        else:
            self.filename = ''
        if flux.ndim==2:
            ny,nx = flux.shape
            nchips = 1
            self.chips = ['a']
        else:
            nchips,ny,nx = flux.shape
            self.chips = chips
        self.ndim = flux.ndim
        self.nx = nx
        self.ny = ny
        self.nchips = nchips
        self.datatype = 'Frame'
        self.instrument = 'APOGEE'
        
        if filename is not None and filename[0] != '':
            if os.path.basename(filename[0])[:2]=='ap':
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
        if index>self.nchips-1:
            raise IndexError('index '+str(index)+' is out of bounds for axis 0 with size '+str(self.nchips))
        if self.nchips > 1:
            # Get the individual spectra
            kw = {'header':self.header,'filename':self.chipfiles[index]}
            for c in ['err','mask']:
                if getattr(self,c) is not None and getattr(self,c).ndim>1:
                    kw[c] = getattr(self,c)[index,:,:]
            # Initialize the object
            fr = Frame(self.flux[index,:,:],**kw)
            fr.chips = [chips[index]]
        else:
            fr = self
        return fr
        
    def __repr__(self):
        """ Print out the string representation of the apBPM object."""
        s = repr(self.__class__)+"\n"
        if self.instrument is not None:
            s += self.instrument+"\n"
        if self.filename is not None:
            s += "File = "+self.filename+"\n"
        if self.nchips > 1:
            s += 'Dimensions: ['+str(self.nchips)+','+str(self.ny)+','+str(self.nx)+']\n'
        else:
            s += 'Dimensions: ['+str(self.ny)+','+str(self.nx)+']\n'
        s += "Flux = "+str(self.flux)+"\n"
        if self.err is not None:
            s += "Err = "+str(self.err)+"\n"
        return s

    @classmethod
    def exists(cls,fname=None,**kwargs):
        """ Check if the ap2D files exist """
        chipfiles = getfilename(fname,**kwargs)
        exts = [os.path.exists(f) for f in chipfiles]
        if np.sum(exts)==3:
            return True
        else:
            return False

    @classmethod
    def read(cls,fname=None,**kwargs):
        """ Read from file """
        chipfiles = getfilename(fname,**kwargs)
        if Frame.exists(fname=None,**kwargs)==False:
            raise FileNotFoundError(chipfiles[0].replace('-a-','-abc')+' not found')

        #fr = []
        #fr = [Frame.read(f) for f in chipfiles]
        #fr1 = Frame()
        # Load individual files and then combine into one
        
        hdu1 = fits.open(chipfiles[0])
        hdu2 = fits.open(chipfiles[1])
        hdu3 = fits.open(chipfiles[2])
        header = hdu1[0].header
        flux = np.stack((hdu1[1].data,hdu2[1].data,hdu3[1].data))
        err = np.stack((hdu1[2].data,hdu2[2].data,hdu3[2].data))
        mask = np.stack((hdu1[3].data,hdu2[3].data,hdu3[3].data))
        hdu1.close()
        hdu2.close()
        hdu3.close()
        # Initialize the object
        data = Frame(flux,header=header,err=err,mask=mask,filename=chipfiles)
        return data

    def write(self,filename,overwrite=True):
        """ Write data to a file """
        # for f in self:
        #     f.write(filename)
        
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

