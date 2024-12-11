import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from ..apred import sincint,wave as wav
from ..utils import apload

class Star(object):
    """
    Data model for apStar files
    """

    def __init__(self,flux,header=None,err=None,wave=None,mask=None,
                 sky=None,skyerr=None,telluric=None,telerr=None,filename='',
                 lsfcoef=None,rvtab=None):
        # Initialize the object
        self._flux = flux
        if header is None:
            self.header = fits.PrimaryHDU().header
        else:
            self.header = header
        self._err = err
        self._mask = mask
        self._wavevac = True
        self._wave = wave
        self._sky = sky
        self._skyerr = skyerr
        self._telluric = telluric
        self._telerr = telerr
        if rvtab is not None:
            self.rvtab = Table(rvtab)
        else:
            self.rvtab = None
        if lsfcoef is not None and len(lsfcoef)!=0:
            self._lsfcoef = lsfcoef
        else:
            self._lsfcoef = None
        self.filename = filename
        if flux.ndim==1:
            npix = len(flux)
            norder = 1
        else:
            norder,npix = flux.shape
        self.ndim = flux.ndim
        self.npix = npix
        self.norder = norder
        self.datatype = 'Star'
        self.waveregime = 'NIR'
        self.instrument = 'APOGEE'
        
        # Contruct the wavelength array
        if self.header is not None:
            w0 = np.float64(self.header["CRVAL1"])
            dw = np.float64(self.header["CDELT1"])
            nw = self.npix
            wave = 10**(np.arange(nw)*dw+w0)
            self._wave = wave
        else:
            # use the "standard" wavelength solution
            w0 = 4.179
            dw = 6e-06
            nw = 8575
            wave = 10**(np.arange(nw)*dw+w0)
            self._wave = wave

        # Create the bad pixel mask
        # "bad" pixels:
        #   flag = ['BADPIX','CRPIX','SATPIX','UNFIXABLE','BADDARK','BADFLAT','BADERR','NOSKY',
        #           'LITTROW_GHOST','PERSIST_HIGH','PERSIST_MED','PERSIST_LOW','SIG_SKYLINE','SIG_TELLURIC','NOT_ENOUGH_PSF','']
        #   badflag = [1,1,1,1,1,1,1,1,
        #              0,0,0,0,0,0,1,0]
        if self._mask is not None and self._flux is not None:
            badmask = (np.bitwise_and(self.mask,16639)!=0) | (np.isfinite(self.flux)==False)
            self.badmask = badmask
        else:
            self.badmask = np.zeros(self.flux.shape,bool)

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

    @property
    def wave(self):
        """ Return the wave array."""
        if hasattr(self,'_wave')==False or self._wave is None:
            return None
        return self._wave

    @property
    def sky(self):
        """ Return the sky array."""
        if hasattr(self,'_sky')==False or self._sky is None:
            return None
        return self._sky

    @property
    def skyerr(self):
        """ Return the sky error array."""
        if hasattr(self,'_skyerr')==False or self._skyerr is None:
            return None
        return self._skyerr
    
    @property
    def telluric(self):
        """ Return the telluric array."""
        if hasattr(self,'_telluric')==False or self._telluric is None:
            return None
        return self._telluric

    @property
    def telerr(self):
        """ Return the telluric error array."""
        if hasattr(self,'_telerr')==False or self._telerr is None:
            return None
        return self._telerr

    @property
    def lsfcoef(self):
        """ Return the lsf coefficient array."""
        if hasattr(self,'_lsfcoef')==False or self._lsfcoef is None:
            return None
        return self._lsfcoef

    @property
    def snr(self):
        """ Return the median S/N per pixel."""
        if self.flux is not None and self.err is not None:
            if self.badmask is not None:
                return np.nanmedian(self.flux[~self.badmask]/self.err[~self.badmask])
            else:
                return np.nanmedian(self.flux/self.err)                
        else:
            return None

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
            sp = Star(self.flux[index,:],**kw)
        else:
            sp = self
        return sp
        
    def __repr__(self):
        """ Print out the string representation of the apStar object."""
        s = repr(self.__class__)+"\n"
        if self.instrument is not None:
            s += self.instrument+" spectrum\n"
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
        if self.wave is not None:
            s += "Wave = "+str(self.wave)
        return s
    
    @classmethod
    def read(cls,fname=None,**kwargs):
        """ Read from file """
        if fname is None and len(kwargs)==0:
            raise ValueError("Must input filename or sdss_access/tree keyword parameters")
        # You can input a filename OR input the sdss_access/tree information
        #   obj, apred, telescope, [mjd]
        if fname is None and len(kwargs)>0:
            for c in ['obj','apred','telescope']:
                if c not in kwargs.keys():
                    raise ValueError(c+' parameter must be input')
            if 'mjd' in kwargs.keys():
                mjd = kwargs['mjd']
            else:
                mjd = None
            load = apload.ApLoad(apred=kwargs['apred'],telescope=kwargs['telescope'])
            filename = load.filename('Star',obj=kwargs['obj'],mjd=mjd)
        else:
            filename = fname
        
        # APOGEE apStar, combined spectrum
        # HISTORY APSTAR:  HDU0 = Header only
        # HISTORY APSTAR:  All image extensions have:
        # HISTORY APSTAR:    row 1: combined spectrum with individual pixel weighting
        # HISTORY APSTAR:    row 2: combined spectrum with global weighting
        # HISTORY APSTAR:    row 3-nvisits+2: individual resampled visit spectra
        # HISTORY APSTAR:   unless nvisits=1, which only have a single row
        # HISTORY APSTAR:  All spectra shifted to rest (vacuum) wavelength scale
        # HISTORY APSTAR:  HDU1 - Flux (10^-17 ergs/s/cm^2/Ang)
        # HISTORY APSTAR:  HDU2 - Error (10^-17 ergs/s/cm^2/Ang)
        # HISTORY APSTAR:  HDU3 - Flag mask:
        # HISTORY APSTAR:    row 1: bitwise OR of all visits
        # HISTORY APSTAR:    row 2: bitwise AND of all visits
        # HISTORY APSTAR:    row 3-nvisits+2: individual visit masks
        # HISTORY APSTAR:  HDU4 - Sky (10^-17 ergs/s/cm^2/Ang)
        # HISTORY APSTAR:  HDU5 - Sky Error (10^-17 ergs/s/cm^2/Ang)
        # HISTORY APSTAR:  HDU6 - Telluric
        # HISTORY APSTAR:  HDU7 - Telluric Error
        # HISTORY APSTAR:  HDU8 - LSF coefficients
        # HISTORY APSTAR:  HDU9 - RV and CCF structure

        if os.path.exists(filename)==False:
            raise FileNotFoundError(filename)
        hdu = fits.open(filename)

	# Spectrum, error, sky, skyerr are in units of 1e-17
        #  these are 2D arrays with [Nvisit+2,Npix]
        #  the first two are combined and the rest are the individual spectra

        if len(hdu)>=9:
            rvtab = hdu[9].data
        else:
            rvtab = None

        # Initialize the object
        sp = Star(hdu[1].data,header=hdu[0].header,err=hdu[2].data,mask=hdu[3].data,
                  sky=hdu[4].data,skyerr=hdu[5].data,telluric=hdu[6].data,
                  telerr=hdu[7].data,lsfcoef=hdu[8].data,rvtab=rvtab,filename=filename)
        hdu.close()
        
        return sp

    def setmask(self,bdval) :
        """ Make boolean mask from bitmask with input pixelmask for bad values """
        self.mask = (np.bitwise_and(self.bitmask,bdval)!=0) | (np.isfinite(self.flux)==False)

    def interp(self,newwave,nres) :
        """ Interpolate to new wavelengths """
        pix = wav.wave2pix(newwave,self.wave)
        gd = np.where(np.isfinite(pix))[0]
        raw = [[self.flux,self.err]]
        out = sincint.sincint(pix[gd],nres,raw)
        newflux,newerr = out[0][0],out[0][1]
        return newflux,newerr

    def write(self,filename,overwrite=True):
        """ Write data to a file """
        hdulist = fits.HDUList()
        hdu = fits.PrimaryHDU()
        hdu.header = self.header
        leadstr = 'Star: '
        hdu.header['HISTORY'] = leadstr+time.asctime()
        hdu.header['HISTORY'] = leadstr+getpass.getuser()+' on '+socket.gethostname()
        pyvers = sys.version.split()[0]
        hdu.header['HISTORY'] = leadstr+'Python '+pyvers+' '+platform.system()+' '+platform.release()+' '+platform.architecture()[0]
        hdu.header['HISTORY'] = 'APOGEE software git hash:' +str(plan.getgitvers())
        hdu.header['HISTORY'] = leadstr+' APOGEE Reduction Pipeline Version: {:s}'.format(os.environ['APOGEE_DRP_VER'])
        hdu.header['HISTORY'] = 'HDU0 : header'
        hdu.header['HISTORY'] = 'HDU1 : flux'
        hdu.header['HISTORY'] = 'HDU2 : flux uncertainty'
        hdu.header['HISTORY'] = 'HDU3 : pixel bitmask'
        hdu.header['HISTORY'] = 'HDU4 : sky'
        hdu.header['HISTORY'] = 'HDU5 : sky uncertainty'
        hdu.header['HISTORY'] = 'HDU6 : telluric'
        hdu.header['HISTORY'] = 'HDU7 : telluric uncertainty'
        hdu.header['HISTORY'] = 'HDU8 : LSF table'
        hdu.header['HISTORY'] = 'HDU9 : RV table'
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
        header['BUNIT'] = 'Sky (10^-17 erg/s/cm^2/Ang)'
        header['EXTNAME'] = 'SKY FLUX'
        hdulist.append(fits.ImageHDU(self.sky,header=header))
        header['BUNIT'] = 'Sky error (10^-17 erg/s/cm^2/Ang)'
        header['EXTNAME'] = 'SKY ERROR'
        hdulist.append(fits.ImageHDU(self.skyerr,header=header))
        header['BUNIT'] = 'Telluric'
        header['EXTNAME'] = 'TELLURIC'
        hdulist.append(fits.ImageHDU(self.telluric,header=header))
        header['BUNIT'] = 'Telluric error'
        header['EXTNAME'] = 'TELLURIC ERROR'
        hdulist.append(fits.ImageHDU(self.telerr,header=header))
        if self.lsfcoef is not None:
            hdulist.append(fits.ImageHDU(self.lsfcoef))
        else:
            hdulist.append(fits.ImageHDU())
        hdulist[-1].header['EXTNAME'] = 'LSF TABLE'
        if self.rvtab is not None:
            hdulist.append(fits.table_to_hdu(self.rvtab))
        else:
            hdulist.append(fits.ImageHDU())
        hdulist[-1].header['EXTNAME'] = 'RV TABLE'       
        hdulist.writeto(filename,overwrite=overwrite)

