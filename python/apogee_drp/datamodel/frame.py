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
        # Loop over the three chips
        fdir = os.path.dirname(filename)
        base = os.path.basename(filename)
        files = 3*[None]
        for i,ch in enumerate(chips):
            files[i] = os.path.join(fdir,base.replace('2D-','2D-'+ch+'-'))
    else:
        files = [fname]
    return files

def combinechips(frames):
    """ Combine frames for multiple chips."""
    shape = [len(frames)]+list(frames[0].shape)
    dtype = frames[0].flux.dtype
    zr = np.zeros(shape,dtype)
    fr = Frame(zr,header=frames[0].header,err=zr,
               mask=zr.astype(frames[0].mask.dtype),filename=frames[0].filename)
    for i in range(len(frames)):
        if frames[i].flux is not None:
            fr.flux[i,:,:] = frames[i].flux
        if frames[i].err is not None:
            fr.err[i,:,:] = frames[i].err
        if frames[i].mask is not None:
            fr.mask[i,:,:] = frames[i].mask
    # observatory, chips, nchips, files
    fr.observatory = frames[0].observatory
    fr.nchips = len(frames)
    fr.chips = [f.chips[0] for f in frames]
    fr.chipfiles = [f.filename for f in frames]
    return fr

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
            if isinstance(filename,list) and len(filename)>1:
                self.filename = filename[0].replace('-a-','-')
            elif isinstance(filename,list) and len(filename)==1:
                self.filename = filename[0]
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

    def __len__(self):
        """ Return len.  Number of chips """
        return self.nchips

    @property
    def shape(self):
        """ Return shape of data """
        return self.flux.shape

    @property
    def size(self):
        """ Return size of data """
        return self.flux.size

    def __array__(self):
        """ Return the main data array """
        return self.flux

    def __iter__(self):
        self._count = 0
        return self
        
    def __next__(self):
        if self._count < len(self):
            self._count += 1            
            return self[self._count-1]
        else:
            raise StopIteration
    
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

    def copy(self):
        """ Copy this object """
        return copy.deepcopy(self)

    @classmethod
    def exists(cls,fname=None,**kwargs):
        """ Check if the ap2D files exist """
        files = getfilename(fname,**kwargs)
        exts = [os.path.exists(f) for f in files]
        if np.sum(exts)==len(files):
            return True
        else:
            return False

    @classmethod
    def read(cls,fname=None,**kwargs):
        """ Read from file """
        files = getfilename(fname,**kwargs)
        if Frame.exists(fname,**kwargs)==False:
            raise FileNotFoundError(files[0].replace('-a-','-[abc]-')+' not found')
        if len(files)>1:
            fr = [Frame.read(f) for f in files]
            data = combinechips(fr)
        else:
            hdu = fits.open(files[0])
            data = Frame(hdu[1].data,header=hdu[0].header,err=hdu[2].data,mask=hdu[3].data,filename=files)
            hdu.close()
        return data

    def write(self,fname=None,overwrite=True,**kwargs):
        """ Write data to a file """
        outfiles = getfilename(fname,**kwargs)
        if len(self)==1:
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
            hdulist.writeto(outfiles[0],overwrite=overwrite)
        elif len(self)>1:
            for i,f in enumerate(self):
                if os.path.exists(outfiles[i]) and overwrite==False:
                    raise FileExistsError(outfiles[i])
                f.write(outfiles[i],overwrite=overwrite)
        else:
            raise Exception('nchips is zero')
            
