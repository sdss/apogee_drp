# encoding: utf-8
#
# @Author: Jon Holtzman
# @Date: March 2018
# @Filename: synth.py
# @License: BSD 3-Clause
# @Copyright: Jon Holtzman

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from astropy.io import fits
from astropy.table import Table
import os
try :
    from sdss_access.path import path
    from sdss_access.sync.http import HttpAccess
except :
    print('sdss_access or dependencies not available!')
import sys
import numpy as np
import healpy as hp

from ..apred import wave,sincint
from . import spectra,yanny

class ApSpec() :
    """ a simple class to hold APOGEE spectra
    """
    def __init__(self,flux,header=None,err=None,wave=None,mask=None,bitmask=None,
                 sky=None,skyerr=None,telluric=None,telerr=None,cont=None,template=None,filename='',
                 lsftab=Table(),rvtab=Table(),sptype='apStar',waveregime='NIR',instrument='APOGEE',snr=100):
        # Initialize the object
        self.flux = flux
        if header is None : self.header = fits.PrimaryHDU().header
        else : self.header = header
        self.err = err
        self.bitmask = bitmask
        self.wavevac = True
        self.wave = wave
        self.sky = sky
        self.skyerr = skyerr
        self.telluric = telluric
        self.telerr = telerr
        self.cont = cont
        self.template = template
        self.filename = filename
        self.rvtab = rvtab
        self.lsftab = lsftab
        self.sptype = sptype
        self.waveregime = waveregime
        self.instrument = instrument
        self.snr = snr
        if flux.ndim==1:
            npix = len(flux)
            norder = 1
        else:
            norder,npix = flux.shape
        self.ndim = flux.ndim
        self.npix = npix
        self.norder = norder

        return

    def setmask(self,bdval) :
        """ Make boolean mask from bitmask with input pixelmask for bad values
        """
        self.mask = (np.bitwise_and(self.bitmask,bdval)!=0) | (np.isfinite(self.flux)==False)

    def interp(self,new,nres) :
        """ Interpolate to new wavelengths
        """
        pix = wave.wave2pix(new,self.wave)
        gd = np.where(np.isfinite(pix))[0]
        raw = [[self.flux,self.err]]
        out = sincint.sincint(pix[gd],nres,raw)
        self.wave = new
        self.flux = out[0][0]
        self.err = out[0][1]

    def write(self,filename,overwrite=True) :
        hdulist = fits.HDUList()
        hdu = fits.PrimaryHDU()
        hdu.header = self.header
        hdu.header['HISTORY'] = 'APOGEE Reduction Pipeline Version: {:s}'.format(os.environ['APOGEE_DRP_VER'])
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
        hdulist.append(fits.ImageHDU(self.flux,header=header))
        header['BUNIT'] = 'Err (10^-17 erg/s/cm^2/Ang)'
        hdulist.append(fits.ImageHDU(self.err,header=header))
        header['BUNIT'] = 'Pixel bitmask'
        hdulist.append(fits.ImageHDU(self.bitmask,header=header))
        header['BUNIT'] = 'Sky (10^-17 erg/s/cm^2/Ang)'
        hdulist.append(fits.ImageHDU(self.sky,header=header))
        header['BUNIT'] = 'Sky error (10^-17 erg/s/cm^2/Ang)'
        hdulist.append(fits.ImageHDU(self.skyerr,header=header))
        header['BUNIT'] = 'Telluric'
        hdulist.append(fits.ImageHDU(self.telluric,header=header))
        header['BUNIT'] = 'Telluric error'
        hdulist.append(fits.ImageHDU(self.telerr,header=header))
        hdulist.append(fits.table_to_hdu(self.lsftab))
        hdulist.append(fits.table_to_hdu(self.rvtab))
        hdulist.writeto(filename,overwrite=overwrite)

