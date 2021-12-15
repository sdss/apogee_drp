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
import pdb
import sys
import numpy as np
import healpy as hp

from ..apred import wave,sincint
from . import spectra,yanny
from .apspec import ApSpec

class ApData(object):
    def __init__(self,filename=None,datatype=None,header=None,chip=None,flux=None,error=None,mask=None):
        self.filename = filename
        self.datatype = datatype
        if header is not None:
            self.header = header
        if chip is not None:
            self.chip = chip
        if flux is not None:
            self.flux = flux
        if error is not None:
            self.error = error
        if mask is not None:
            self.mask = mask

    def __len__(self):
        if hasattr(self,'flux') is False:
            return None
        return self.flux.shape

    def __repr__(self):
        """ Print out the string representation of the Spec1D object."""
        s = repr(self.__class__)+"\n"
        if self.filename is not None:
            s += "File = "+self.filename+"\n"
        if self.datatype is not None:
            s += "Type = "+self.datatype+"\n"
        if hasattr(self,'chip') is not False:
            s += "Chip = "+self.chip+"\n"
        if hasattr(self,'flux') is not False:
            nx,ny = self.flux.shape
            s += 'Dimensions: ['+str(nx)+','+str(ny)+']\n'
            s += "Flux = "+str(self.flux)+"\n"
        if hasattr(self,'error') is not False:
            s += "Error = "+str(self.error)+"\n"
        if hasattr(self,'mask') is not False:
            s += "Mask = "+str(self.mask)+"\n"
        if hasattr(self,'wave') is not False:
            s += "Wave = "+str(self.wave)+"\n"
        if hasattr(self,'sky') is not False:
            s += "Sky = "+str(self.sky)+"\n"
        if hasattr(self,'skyerr') is not False:
            s += "Skyerr = "+str(self.skyerr)+"\n"
        if hasattr(self,'telluric') is not False:
            s += "Telluric = "+str(self.telluric)+"\n"
        if hasattr(self,'telerr') is not False:
            s += "Telerr = "+str(self.telerr)+"\n"
        if hasattr(self,'wcoef') is not False:
            s += "Wcoef = "+str(self.wcoef)+"\n"
        if hasattr(self,'lsf') is not False:
            s += "LSF = "+str(self.lsf)+"\n"
        if hasattr(self,'wcoef') is not False:
            s += "Wcoef = "+str(self.wcoef)+"\n"
        if hasattr(self,'plugmap') is not False:
            s += "Plugmap = "+str(self.plugmap[[0,-1]])+"\n"
        if hasattr(self,'plhead') is not False:
            s += "Plhead = "+str(self.plhead)+"\n"
        if hasattr(self,'telstr') is not False:
            s += "Telstr = "+str(len(self.telstr))+" elements\n"
        if hasattr(self,'shiftstr') is not False:
            s += "Shiftstr = "+str(len(self.shiftstr))+" elements\n"
        if hasattr(self,'pairstr') is not False:
            s += "Pairstr = "+str(self.pairstr)+"\n"
        if hasattr(self,'fluxfactor') is not False:
            s += "Fluxfactor = "+str(self.fluxfactor[[0,1,2]])+"\n"

        for k in self.__dict__.keys():
            if k not in ['filename','datatype','chip','header','flux','error','mask','wave','sky','skyerr','telluric',
                         'telerr','wcoef','lsf','wcoef','plugmap','plhead','telstr','shiftstr','pairstr','fluxfactor']:
                s += k+" = "+str(getattr(self,k))
        return s

class ApDataArr(object):
    def __init__(self,datatype=None,header=None):
        self.datatype = datatype
        self.header = header
        self.ndata = 0
        self._data = []

    def __repr__(self):
        """ Print out the string representation of the Spec1D object."""
        s = repr(self.__class__)+"\n"
        if self.datatype is not None:
            s += "Type = "+self.datatype+"\n"
        s += "Ndata = "+str(self.ndata)
        return s    

    def __setitem__(self,index,data):
        if index>self.ndata:
            raise ValueError('index must be '+str(self.ndata)+' or '+str(self.ndata+1))
        if index<=(self.ndata-1):
            self._data[index] = data
        else:
            self._data.append(data)
            self.ndata = len(self._data)

    def __getitem__(self,index):
        if self.ndata==0:
            raise ValueError('no data to get')
        if index>(self.ndata-1):
            raise ValueError('index must be <='+str(self.ndata-1))
        return self._data[index]

    def __iter__(self):
        self._count = 0
        return self

    def __next__(self):
        if self._count < self.ndata:
            self._count += 1
            return self._data[self._count-1]
        else:
            raise StopIteration




class ApLoad:

    def __init__(self,dr=None,apred='r8',apstar='stars',aspcap='l31c',results='l31c.2',
                 telescope='apo25m',instrument=None,verbose=False,pathfile=None) :
        self.apred=apred
        self.apstar=apstar
        self.aspcap=aspcap
        self.results=results
        self.settelescope(telescope)
        if instrument is not None : self.instrument=instrument
        self.verbose=verbose
        if dr == 'dr10' : self.dr10()
        elif dr == 'dr12' : self.dr12()
        elif dr == 'dr13' : self.dr13()
        elif dr == 'dr14' : self.dr14()
        elif dr == 'dr16' : self.dr16()
        # set up 
        self.sdss_path=path.Path()
        self.http_access=HttpAccess(verbose=verbose)
        self.http_access.remote()
   
    def settelescope(self,telescope) :
        self.telescope = telescope
        if 'apo' in telescope:
            self.instrument = 'apogee-n'
            self.observatory = 'apo'
        if 'lco' in telescope:
            self.instrument = 'apogee-s'
            self.observatory = 'lco'
 
    def setinst(self,instrument) :
        self.instrument=instrument
 
    def dr10(self) :
        self.apred='r3'
        self.apstar='s3'
        self.aspcap='v304'
        self.results='v304'

    def dr12(self) :
        self.apred='r5'
        self.aspcap='l25_6d'
        self.results='v603'

    def dr13(self) :
        self.apred='r6'
        self.aspcap='l30e'
        self.results='l30e.2'

    def dr14(self) :
        self.apred='r8'
        self.aspcap='l31c'
        self.results='l31c.2'

    def dr16(self) :
        self.apred='r12'
        self.aspcap='l33'

    def printerror(self) :
        print('cannot find file: do you have correct version? permission? wget authentication?')

    def allStar(self,hdu=None) :
        ''' Read allStar file (downloading if necesssary)'''
        file = self.allfile('allStar')
        try :
            file = self.allfile('allStar')
            return self._readhdu(file,hdu=hdu)
        except :
            self.printerror()

    def allVisit(self,hdu=None) :
        ''' Read allVisit file (downloading if necesssary)'''
        try :
            file = self.allfile('allVisit')
            return self._readhdu(file,hdu=hdu)
        except :
            self.printerror()
    
    def allPlates(self,hdu=None) :
        ''' Read allPlates file (downloading if necesssary)'''
        try :
            file = self.allfile('allPlates')
            return self._readhdu(file,hdu=hdu)
        except :
            self.printerror()
    
    def allExp(self,hdu=None) :
        ''' Read allExp file (downloading if necesssary)'''
        try :
            file = self.allfile('allExp')
            return self._readhdu(file,hdu=hdu)
        except :
            self.printerror()
    
    def allSci(self,hdu=None) :
        ''' Read allSci file (downloading if necesssary)'''
        try :
            file = self.allfile('allSci')
            return self._readhdu(file,hdu=hdu)
        except :
            self.printerror()
    
    def allCal(self,hdu=None) :
        ''' Read allCal file (downloading if necesssary)'''
        try :
            file = self.allfile('allCal')
            return self._readhdu(file,hdu=hdu)
        except :
            self.printerror()
    
    def apR(self,*args,**kwargs) :
        """
        NAME: apload.apR
        PURPOSE:  read apR file (downloading if necessary)
        USAGE:  ret = apload.apR(imagenumber[,hdu=N,tuple=True])
        RETURNS: if hdu==None : dictionary of ImageHDUs (all extensions) 
                                for chips 'a', 'b', 'c'
                 if hdu=N : returns dictionaries (data, header) for specified HDU
                 if tuple=True : returns tuples rather than dictionaries
        """
        if len(args) != 1 :
            print('Usage: apR(imagenumber)')
        else :
            try :
                file = self.allfile(
                   'R',num=args[0],mjd=self.cmjd(args[0]),chips=True)
                return self._readchip(file,'R',**kwargs)
            except :
                self.printerror()
    
    def apFlat(self,*args,**kwargs) :
        """
        NAME: apload.apFlat
        PURPOSE:  read apFlat file (downloading if necessary)
        USAGE:  ret = apload.apFlat(imagenumber[,hdu=N,tuple=True])
        RETURNS: if hdu==None : dictionary of ImageHDUs (all extensions) 
                                for chips 'a', 'b', 'c'
                 if hdu=N : returns dictionaries (data, header) for specified HDU
                 if tuple=True : returns tuples rather than dictionaries
        """
        if len(args) != 1 :
            print('Usage: apFlat(imagenumber)')
        else :
            try :
                file = self.allfile(
                   'Flat',num=args[0],mjd=self.cmjd(args[0]),chips=True)
                return self._readchip(file,'Flat',**kwargs)
            except :
                self.printerror()
    
    def apFlux(self,*args,**kwargs) :
        """
        NAME: apload.apFlux
        PURPOSE:  read apFlux file (downloading if necessary)
        USAGE:  ret = apload.apFlux(imagenumber[,hdu=N,tuple=True])
        RETURNS: if hdu==None : dictionary of ImageHDUs (all extensions) 
                                for chips 'a', 'b', 'c'
                 if hdu=N : returns dictionaries (data, header) for specified HDU
                 if tuple=True : returns tuples rather than dictionaries
        """
        if len(args) != 1 :
            print('Usage: apFlux(imagenumber)')
        else :
            try :
                file = self.allfile(
                   'Flux',num=args[0],mjd=self.cmjd(args[0]),chips=True)
                return self._readchip(file,'Flux',**kwargs)
            except :
                self.printerror()
    
    def apWave(self,*args,**kwargs) :
        """
        NAME: apload.apWave
        PURPOSE:  read apWave file (downloading if necessary)
        USAGE:  ret = apload.apWave(imagenumber[,hdu=N,tuple=True])
        RETURNS: if hdu==None : dictionary of ImageHDUs (all extensions) 
                                for chips 'a', 'b', 'c'
                 if hdu=N : returns dictionaries (data, header) for specified HDU
                 if tuple=True : returns tuples rather than dictionaries
        """
        if len(args) != 1 :
            print('Usage: apWave(imagenumber)')
        else :
            try :
                file = self.allfile(
                   'Wave',num=args[0],mjd=self.cmjd(args[0]),chips=True)
                return self._readchip(file,'Wave',**kwargs)
            except :
                self.printerror()
    
    def apLSF(self,*args,**kwargs) :
        """
        NAME: apload.apLSF
        PURPOSE:  read apLSF file (downloading if necessary)
        USAGE:  ret = apload.apLSF(imagenumber[,hdu=N,tuple=True])
        RETURNS: if hdu==None : dictionary of ImageHDUs (all extensions) 
                                for chips 'a', 'b', 'c'
                 if hdu=N : returns dictionaries (data, header) for specified HDU
                 if tuple=True : returns tuples rather than dictionaries
        """
        if len(args) != 1 :
            print('Usage: apLSF(imagenumber)')
        else :
            try :
                file = self.allfile(
                   'LSF',num=args[0],mjd=self.cmjd(args[0]),chips=True)
                return self._readchip(file,'LSF',**kwargs)
            except :
                self.printerror()
    
    def apPSF(self,*args,**kwargs) :
        """
        NAME: apload.apPSF
        PURPOSE:  read apPSF file (downloading if necessary)
        USAGE:  ret = apload.apPSF(imagenumber[,hdu=N,tuple=True])
        RETURNS: if hdu==None : dictionary of ImageHDUs (all extensions) 
                                for chips 'a', 'b', 'c'
                 if hdu=N : returns dictionaries (data, header) for specified HDU
                 if tuple=True : returns tuples rather than dictionaries
        """
        if len(args) != 1 :
            print('Usage: apPSF(imagenumber)')
        else :
            try :
                file = self.allfile(
                   'PSF',num=args[0],mjd=self.cmjd(args[0]),chips=True)
                return self._readchip(file,'PSF',**kwargs)
            except :
                self.printerror()
    
    def apEPSF(self,*args,**kwargs) :
        """
        NAME: apload.apEPSF
        PURPOSE:  read apEPSF file (downloading if necessary)
        USAGE:  ret = apload.apEPSF(imagenumber[,hdu=N,tuple=True])
        RETURNS: if hdu==None : dictionary of ImageHDUs (all extensions) 
                                for chips 'a', 'b', 'c'
                 if hdu=N : returns dictionaries (data, header) for specified HDU
                 if tuple=True : returns tuples rather than dictionaries
        """
        if len(args) != 1 :
            print('Usage: apEPSF(imagenumber)')
        else :
            try :
                file = self.allfile(
                   'EPSF',num=args[0],mjd=self.cmjd(args[0]),chips=True)
                return self._readchip(file,'EPSF',**kwargs)
            except :
                self.printerror()
    
    def ap1D(self,*args,**kwargs) :
        """
        NAME: apload.ap1D
        PURPOSE:  read ap1D file (downloading if necessary)
        USAGE:  ret = apload.ap1D(imagenumber[,hdu=N,tuple=True])
        RETURNS: if hdu==None : dictionary of ImageHDUs (all extensions) 
                                for chips 'a', 'b', 'c'
                 if hdu=N : returns dictionaries (data, header) for specified HDU
                 if tuple=True : returns tuples rather than dictionaries
        """
        if len(args) != 1 :
            print('Usage: ap1D(imagenumber)')
        else :
            try :
                file = self.allfile(
                   '1D',num=args[0],mjd=self.cmjd(args[0]),chips=True)
                return self._readchip(file,'1D',**kwargs)
            except :
                self.printerror()
    
    
    def ap2D(self,*args,**kwargs) :
        """
        NAME: apload.ap2D
        PURPOSE:  read ap2D file (downloading if necessary)
        USAGE:  ret = apload.ap2D(imagenumber)
        RETURNS: if hdu==None : dictionary of ImageHDUs (all extensions) 
                                for chips 'a', 'b', 'c'
                 if hdu=N : returns dictionaries (data, header) for specified HDU
                 if tuple=True : returns tuples rather than dictionaries
        """
        if len(args) != 1 :
            print('Usage: ap2D(imagenumber)')
        else :
            try :
                file = self.allfile(
                   '2D',num=args[0],mjd=self.cmjd(args[0]),chips=True,**kwargs)
                print('file: ', file)
                return self._readchip(file,'2D',**kwargs)
            except :
                self.printerror()
    
    def ap2Dmodel(self,*args,**kwargs) :
        """
        NAME: apload.ap2Dmodel
        PURPOSE:  read ap2Dmodel file (downloading if necessary)
        USAGE:  ret = apload.ap2Dmodel(imagenumber)
        RETURNS: if hdu==None : dictionary of ImageHDUs (all extensions) 
                                for chips 'a', 'b', 'c'
                 if hdu=N : returns dictionaries (data, header) for specified HDU
                 if tuple=True : returns tuples rather than dictionaries
        """
        if len(args) != 1 :
            print('Usage: ap2Dmodel(imagenumber)')
        else :
            try :
                file = self.allfile(
                   '2Dmodel',num=args[0],mjd=self.cmjd(args[0]),chips=True,**kwargs)
                return self._readchip(file,'2Dmodel',**kwargs)
            except :
                self.printerror()
    
    def apCframe(self,*args, **kwargs) :
        """
        NAME: apload.apCframe
        PURPOSE:  read apCframe file (downloading if necessary)
        USAGE:  ret = apload.apCframe(plate,mjd,imagenumber[,hdu=N,tuple=True])
        RETURNS: if hdu==None : dictionary of ImageHDUs (all extensions) 
                                for chips 'a', 'b', 'c'
                 if hdu=N : returns dictionaries (data, header) for specified HDU
                 if tuple=True : returns tuples rather than dictionaries
        """
        if len(args) != 4 :
            print('Usage: apCframe(field,plate,mjd,imagenumber)')
        else :
            try :
                file = self.allfile(
                   'Cframe',field=args[0],plate=args[1],mjd=args[2],num=args[3],chips=True)
                return self._readchip(file,'Cframe',**kwargs)
            except :
                self.printerror()
    
    def apPlate(self,*args, **kwargs) :
        """
        NAME: apload.apPlate
        PURPOSE:  read apPlate file (downloading if necessary)
        USAGE:  ret = apload.ap2D(plate,mjd[,hdu=N,tuple=True])
        RETURNS: if hdu==None : dictionary of ImageHDUs (all extensions) 
                                for chips 'a', 'b', 'c'
                 if hdu=N : returns dictionaries (data, header) for specified HDU
                 if tuple=True : returns tuples rather than dictionaries
        """
        if len(args) != 2 :
            print('Usage: apPlate(plate,mjd)')
        else :
            try :
                file = self.allfile(
                   'Plate',plate=args[0],mjd=args[1],chips=True)
                return self._readchip(file,'Plate',**kwargs)
            except :
                self.printerror()
    
    def apVisit(self,*args, load=False, **kwargs) :
        """
        NAME: apload.apVisit
        PURPOSE:  read apVisit file (downloading if necessary)
        USAGE:  ret = apload.apVisit(plate,mjd,fiber,[hdu=N])
        RETURNS: if hdu==None : ImageHDUs (all extensions)
                 if hdu=N : returns (data, header) for specified HDU
        """
        if len(args) != 3 :
            print('Usage: apVisit(plate,mjd,fiber)')
        else :
            try :
                if kwargs.get('field') is None:
                    filePath = self.allfile('Visit',plate=args[0],mjd=args[1],fiber=args[2])
                else:
                    filePath = self.allfile('Visit',plate=args[0],mjd=args[1],fiber=args[2],field=kwargs['field'])
                if load : 
                    hdulist=self._readhdu(filePath)
                    spec=ApSpec(hdulist[1].data,header=hdulist[0].header,
                                err=hdulist[2].data,bitmask=hdulist[3].data,wave=hdulist[4].data,
                                sky=hdulist[5].data,skyerr=hdulist[5].data,
                                telluric=hdulist[7].data,telerr=hdulist[8].data)
                    return spec
                return self._readhdu(filePath,**kwargs)
            except :
                self.printerror()
    
    def apVisit1m(self,*args, load=False, **kwargs) :
        """
        NAME: apload.apVisit
        PURPOSE:  read apVisit file (downloading if necessary)
        USAGE:  ret = apload.apVisit(program,mjd,object,[hdu=N])
        RETURNS: if hdu==None : ImageHDUs (all extensions)
                 if hdu=N : returns (data, header) for specified HDU
        """
        if len(args) != 3 :
            print('Usage: apVisit1m(program,mjd,object)')
        else :
            try :
                file = self.allfile(
                   'Visit',plate=args[0],mjd=args[1],reduction=args[2])
                if load : 
                    hdulist=self._readhdu(file)
                    spec=ApSpec(hdulist[1].data,header=hdulist[0].header,
                                err=hdulist[2].data,bitmask=hdulist[3].data,wave=hdulist[4].data,
                                sky=hdulist[5].data,skyerr=hdulist[6].data,
                                telluric=hdulist[7].data,telerr=hdulist[8].data)
                    return spec
                return self._readhdu(file,**kwargs)
            except :
                self.printerror()
    
    def apVisitSum(self,*args, **kwargs) :
        """
        NAME: apload.apVisitSum
        PURPOSE:  read apVisitSum file (downloading if necessary)
        USAGE:  ret = apload.apVisitSum(plate,mjd)
        RETURNS: if hdu==None : ImageHDUs (all extensions)
                 if hdu=N : returns (data, header) for specified HDU
        """
        if len(args) != 2 :
            print('Usage: apVisitSum(plate,mjd)')
        else :
            try :
                file = self.allfile(
                   'VisitSum',plate=args[0],mjd=args[1])
                return self._readhdu(file,**kwargs)
            except :
                self.printerror()
    
    def apStar(self,*args, load=False, **kwargs) :
        """
        NAME: apload.apStar
        PURPOSE:  read apStar file (downloading if necessary)
        USAGE:  ret = apload.apStar(field,object)
        RETURNS: if hdu==None : ImageHDUs (all extensions)
                 if hdu=N : returns (data, header) for specified HDU
        """

        if len(args)!=2 and len(kwargs)<2 :
            print('Usage: apStar(field,object)')
        else :
            if len(args)>0:
                field = args[0]
                obj = args[1]
            else:
                field = kwargs.get('field')
                obj = kwargs.get('obj')
                healpix = kwargs.get('healpix')
            try :
                filePath = self.allfile('Star',field=field,obj=obj)
                if load : 
                    hdulist=self._readhdu(filePath)
                    wave=spectra.fits2vector(hdulist[1].header,1)
                    spec=ApSpec(hdulist[1].data,header=hdulist[0].header,
                                err=hdulist[2].data,bitmask=hdulist[3].data,wave=wave,
                                sky=hdulist[4].data,skyerr=hdulist[5].data,
                                telluric=hdulist[6].data,telerr=hdulist[7].data)
                    return spec
                return self._readhdu(filePath,**kwargs)
            except :
                self.printerror()
    
    def apStar1m(self,*args, **kwargs) :
        """
        NAME: apload.apStar1m
        PURPOSE:  read apStar file (downloading if necessary)
        USAGE:  ret = apload.apStar1m(location,object)
        RETURNS: if hdu==None : ImageHDUs (all extensions)
                 if hdu=N : returns (data, header) for specified HDU
        """
        if len(args) != 2 :
            print('Usage: apStar(location,object)')
        else :
            try :
                file = self.allfile(
                   'Star',location=args[0],obj=args[1])
                return self._readhdu(file,**kwargs)
            except :
                self.printerror()
    
    def aspcapStar(self,*args, **kwargs) :
        """
        NAME: apload.aspcapStar
        PURPOSE:  read aspcapStar file (downloading if necessary)
        USAGE:  ret = apload.aspcapStar(location,object)
        RETURNS: if hdu==None : ImageHDUs (all extensions)
                 if hdu=N : returns (data, header) for specified HDU
        """
        if len(args) != 2 :
            print('Usage: aspcapStar(location,object)')
        else :
            try :
                file = self.allfile(
                   'aspcapStar',field=args[0],obj=args[1])
                return self._readhdu(file,**kwargs)
            except :
                self.printerror()
    
    def apField(self,*args, **kwargs) :
        """
        NAME: apload.apField
        PURPOSE:  read apField file (downloading if necessary)
        USAGE:  ret = apload.apField(field)
        RETURNS: if hdu==None : ImageHDUs (all extensions)
                 if hdu=N : returns (data, header) for specified HDU
        """
        if len(args) != 1 :
            print('Usage: apField(field)')
        else :
            try :
                file = self.allfile('Field',field=args[0])
                return self._readhdu(file,**kwargs)
            except :
                self.printerror()
    
    def apFieldVisits(self,*args, **kwargs) :
        """
        NAME: apload.apFieldVisits
        PURPOSE:  read apFieldVisits file (downloading if necessary)
        USAGE:  ret = apload.apFieldVisits(field)
        RETURNS: if hdu==None : ImageHDUs (all extensions)
                 if hdu=N : returns (data, header) for specified HDU
        """
        if len(args) != 1 :
            print('Usage: apFieldVisits(field)')
        else :
            try :
                file = self.allfile('FieldVisits',field=args[0])
                return self._readhdu(file,**kwargs)
            except :
                self.printerror()
    
    def aspcapField(self,*args, **kwargs) :
        """
        NAME: apload.aspcapField
        PURPOSE:  read aspcapField file (downloading if necessary)
        USAGE:  ret = apload.aspcapField(field)
        RETURNS: if hdu==None : ImageHDUs (all extensions)
                 if hdu=N : returns (data, header) for specified HDU
        """
        if len(args) != 1 :
            print('Usage: aspcapField(field)')
        else :
            try :
                file = self.allfile( 'aspcapField',field=args[0])
                return self._readhdu(file,**kwargs)
            except :
                self.printerror()
    
    def cmjd(self,frame) :
        """ Get chracter MJD from frame number """
        num = (frame - frame%10000 ) / 10000
        return('{:05d}'.format(int(num)+55562) )
    
    def _readchip(self,file,root,hdu=None,tuple=None,fz=None) :
        """ low level routine to read set of 3 chip files and return data as requested"""
        if self.verbose : print('Reading from file: ', file)
        try:
            if self.verbose : print (file.replace(root,root+'-a'))
            a=fits.open(file.replace(root,root+'-a'))
            if self.verbose : print (file.replace(root,root+'-b'))
            b=fits.open(file.replace(root,root+'-b'))
            if self.verbose : print (file.replace(root,root+'-c'))
            c=fits.open(file.replace(root,root+'-c'))
        except:
            print("Can't open file: ", file)
            return(0)
    
        if hdu is None :
            if tuple :
               if self.verbose : print('file: ', file,' read into tuple')
               return a,b,c 
            else :
               if self.verbose : print('file: ', file,' read into dictionary with entries a, b, c')
               return {'a' : a, 'b' : b, 'c' : c, 'filename' : file}
        else :
            a[hdu].header.set('filename',os.path.basename(file.replace(root,root+'-a')))
            b[hdu].header.set('filename',os.path.basename(file.replace(root,root+'-b')))
            c[hdu].header.set('filename',os.path.basename(file.replace(root,root+'-c')))
            if tuple :
                data =( a[hdu].data, b[hdu].data, c[hdu].data)
                header =( a[hdu].header, b[hdu].header, c[hdu].header)
            else :
                data ={'a' : a[hdu].data, 'b' : b[hdu].data, 'c': c[hdu].data}
                header = {'a' : a[hdu].header, 'b' : b[hdu].header, 'c': c[hdu].header}
            a.close()
            b.close()
            c.close()
            return data, header
    
    def _readhdu(self,file,hdu=None) :
        '''
        internal routine for reading all HDU or specified HDU and returning data and header
        '''
        if self.verbose :
            print('Reading from file: ', file)
        if hdu is None :
            fits.open(file)
            return fits.open(file)
        else :
            hd = fits.open(file)
            data = hd[hdu].data 
            header = hd[hdu].header
            hd.close()
            return data, header
    
    def filename(self,root,
                 location=None,obj=None,plate=None,mjd=None,num=None,fiber=None,chips=False,
                 field=None,configid=None,fps=None) :

        return self.allfile(root,
                            location=location,obj=obj,plate=plate,mjd=mjd,num=num,fiber=fiber,chips=chips,field=field,
                            configid=configid,fps=fps,download=False)

    def allfile(self,root,
                location=None,obj=None,reduction=None,plate=None,mjd=None,num=None,fiber=None,chips=False,field=None,
                healpix=None,configid=None,fps=None,download=True,fz=False) :
        '''
        Uses sdss_access to create filenames and download files if necessary
        '''

        if self.verbose: 
            print('allfile... chips=',chips)
            pdb.set_trace()
        if self.instrument == 'apogee-n' : prefix='ap'
        else : prefix='as'
        if fz : suffix = '.fz'
        else : suffix = ''

        # get the sdss_access root file name appropriate for telescope and file 
        # usually just 'ap'+root, but not for "all" files, raw files, and 1m files, since
        # those require different directory paths
        if 'all' in root or 'aspcap' in root or 'cannon' in root :
            sdssroot = root 
        elif root == 'R' :
            if mjd is None:
                mjd = self.cmjd(num)
            if 'lco' in self.telescope: sdssroot = 'asR'
            elif 'apo1m' in self.telescope: sdssroot = 'apR-1m'
            else : sdssroot = 'apR'
        elif (self.telescope == 'apo1m' and 
           (root == 'Plan' or root == 'PlateSum' or root == 'Visit' or root == 'VisitSum' or root == 'Tellstar' or 
            root == 'Cframe' or root == 'Plate') ) :
            sdssroot = 'ap'+root+'-1m'
        elif root=='confSummary':
            sdssroot = root
        else :
            sdssroot = 'ap'+root

        if (plate is not None) and (field is None):
            field = apfield(plate,telescope=self.telescope,fps=fps)[0]
 
        if chips == False :
            # First make sure the file doesn't exist locally
            #print(sdssroot,apred,apstar,aspcap,results,location,obj,self.telescope,field,prefix)

            # apStar, calculate HEALPix
            if root=='Star':
                healpix = obj2healpix(obj)
            else:
                healpix = None

            filePath = self.sdss_path.full(sdssroot,
                                           apred=self.apred,apstar=self.apstar,aspcap=self.aspcap,results=self.results,
                                           field=field,location=location,obj=obj,reduction=reduction,plate=plate,mjd=mjd,num=num,
                                           telescope=self.telescope,fiber=fiber,prefix=prefix,instrument=self.instrument,
                                           healpix=healpix,configid=configid,obs=self.observatory)
            if self.verbose: print('filePath',filePath)
            if os.path.exists(filePath) is False and download: 
                downloadPath = self.sdss_path.url(sdssroot,
                                      apred=self.apred,apstar=self.apstar,aspcap=self.aspcap,results=self.results,
                                      field=field,location=location,obj=obj,reduction=reduction,plate=plate,mjd=mjd,num=num,
                                      telescope=self.telescope,fiber=fiber,prefix=prefix,instrument=self.instrument,
                                                  healpix=healpix,configid=configid,obs=self.observatory)
                if self.verbose: print('downloadPath',downloadPath)
                self.http_access.get(sdssroot,
                                apred=self.apred,apstar=self.apstar,aspcap=self.aspcap,results=self.results,
                                field=field,location=location,obj=obj,reduction=reduction,plate=plate,mjd=mjd,num=num,
                                telescope=self.telescope,fiber=fiber,prefix=prefix,instrument=self.instrument,
                                     healpix=healpix,configid=configid,obs=self.observatory)
            return filePath
        else :
            for chip in ['a','b','c'] :
                #print(chip,root,num,mjd,prefix)
                filePath = self.sdss_path.full(sdssroot,
                                apred=self.apred,apstar=self.apstar,aspcap=self.aspcap,results=self.results,
                                field=field, location=location,obj=obj,reduction=reduction,plate=plate,mjd=mjd,num=num,
                                telescope=self.telescope,fiber=fiber,
                                chip=chip,prefix=prefix,instrument=self.instrument)+suffix
                if self.verbose : print('filePath: ', filePath, os.path.exists(filePath))
                if os.path.exists(filePath) is False and download : 
                  try:
                    self.http_access.get(sdssroot,
                                apred=self.apred,apstar=self.apstar,aspcap=self.aspcap,results=self.results,
                                field=field, location=location,obj=obj,reduction=reduction,plate=plate,mjd=mjd,num=num,
                                telescope=self.telescope,fiber=fiber,
                                chip=chip,prefix=prefix,instrument=self.instrument)
                  except: pdb.set_trace()
            return filePath.replace('-c','')

    def apread(self,root,
                location=None,obj=None,reduction=None,plate=None,mjd=None,num=None,fiber=None,chips=False,field=None,
                healpix=None,download=True):
        '''
        Similar to allfile but returns data in a more useful format (similar to apread.pro).
        '''

        fz = False
        if root=='2Dmodel':
            fz = True
        if self.instrument == 'apogee-n' : prefix='ap'
        else : prefix='as'
        if fz : suffix = '.fz'
        else : suffix = ''

        # get the sdss_access root file name appropriate for telescope and file 
        # usually just 'ap'+root, but not for "all" files, raw files, and 1m files, since
        # those require different directory paths
        if 'all' in root or 'aspcap' in root or 'cannon' in root:
            sdssroot = root 
        elif root == 'R':
            if 'lco' in self.telescope: sdssroot = 'asR'
            elif 'apo1m' in self.telescope: sdssroot = 'apR-1m'
            else : sdssroot = 'apR'
        elif (self.telescope == 'apo1m' and 
           (root == 'Plan' or root == 'PlateSum' or root == 'Visit' or root == 'VisitSum' or root == 'Tellstar' or 
            root == 'Cframe' or root == 'Plate') ) :
            sdssroot = 'ap'+root+'-1m'
        else :
            sdssroot = 'ap'+root

        if (plate is not None) and (field is None):
            field = apfield(plate,telescope=self.telescope)[0]

        if root=='1D' or root=='2D' or root=='2Dmodel' or root=='Cframe' or root=='R':
            mjd = self.cmjd(num)

        # Load the data
        if root=='Raw':
            pass
        elif root=='Dark':
            pass
        elif root=='1D' or root=='2D' or root=='2Dmodel' or root=='Cframe' or root=='Plate':
            # 1D or 2D: flux, error, mask
            # 2Dmodel: flux
            # Cframe: flux, error, mask
            # Plate: flux, error, mask
            chips = ['a','b','c']
            out = ApDataArr(datatype=root)
            # Chip loop
            for i in range(3):
                filePath = self.sdss_path.full(sdssroot,apred=self.apred,apstar=self.apstar,aspcap=self.aspcap,results=self.results,
                                           field=field,location=location,obj=obj,reduction=reduction,plate=plate,mjd=mjd,num=num,
                                           telescope=self.telescope,fiber=fiber,prefix=prefix,instrument=self.instrument,
                                           healpix=healpix,chip=chips[i])+suffix
                head = fits.getheader(filePath,0)
                ch = ApData(filename=filePath,datatype=root,header=head,chip=chips[i])
                flux,fhead = fits.getdata(filePath,1,header=True)
                ch.flux = flux.T
                if root!='2Dmodel':
                    err,ehead = fits.getdata(filePath,2,header=True)
                    ch.error = err.T
                    mask,mhead = fits.getdata(filePath,3,header=True)
                    ch.mask = mask.T
                if root=='Cframe' or root=='Plate':
                    wave,whead = fits.getdata(filePath,4,header=True)
                    ch.wave = wave.T
                    sky,shead = fits.getdata(filePath,5,header=True)
                    ch.sky = sky.T
                    skyerr,sehead = fits.getdata(filePath,6,header=True)
                    ch.skyerr = skyerr.T
                    telluric,thead = fits.getdata(filePath,7,header=True)
                    ch.telluric = telluric.T
                    telerr,tehead = fits.getdata(filePath,8,header=True)
                    ch.telerr = telerr.T
                    wcoef,whead = fits.getdata(filePath,9,header=True)
                    ch.wcoef = wcoef.T
                    lsf,lhead = fits.getdata(filePath,10,header=True)
                    ch.lsf = lsf.T
                    plugmap,plhead = fits.getdata(filePath,11,header=True)
                    ch.plugmap = plugmap
                    plhead = fits.getdata(filePath,12)
                    ch.plhead = plhead
                if root=='Cframe':
                    telstr = fits.getdata(filePath,13)
                    ch.telstr = telstr
                    shiftstr = fits.getdata(filePath,14)
                    ch.shiftstr = shiftstr
                if root=='Plate':
                    shiftstr = fits.getdata(filePath,13)
                    ch.shiftstr = shiftstr
                    phead = fits.getheader(filePath,14)
                    if phead['naxis']>0:
                        pairstr = fits.getdata(filePath,14)
                        ch.pairstr = pairstr
                    fluxfactor = fits.getdata(filePath,15)
                    ch.fluxfactor = fluxfactor

                # Add to the ApDataArr object
                out[i] = ch

        elif root=='Star':
            # apStar, calculate HEALPix
            healpix = obj2healpix(obj)
            out = self.apStar(field=field,obj=obj,healpix=healpix,load=True)
        elif root=='Visit':
            out = self.apVisit(plate,mjd,fiber,load=True)
        else:
            pass

        return out
 

plans=None

def apfield(plateid,loc=0,addloc=False,telescope='apo25m',fps=False):
    """ Get field name given plateid and plateplans
    """
    global plans

    if plateid==0:
        return 0, None, None

    if telescope == 'apo1m' :
        # for apo1m, plateid is the field and programname
        survey='apo1m'
        return plateid, survey, plateid

    nj = 0
    if plans == None and fps==False: 
        print('reading platePlans')
        plans = yanny.yanny(os.environ['PLATELIST_DIR']+'/platePlans.par')['PLATEPLANS']
        j, = np.where(np.array(plans['plateid']) == int(plateid))
        nj = len(j)
        if nj>0:
            j = j[0]

    # FPS
    if nj==0 or fps:
        # Pull this from the confSummary file
        observatory = {'apo25m':'apo','apo1m':'apo','lco25m':'lco'}[telescope]
        configgrp = '{:0>4d}XX'.format(int(plateid) // 100)
        configfile = os.environ['SDSSCORE_DIR']+'/'+observatory+'/summary_files/'+configgrp+'/confSummary-'+str(plateid)+'.par'
        planstr = yanny.yanny(configfile)
        field = planstr.get('field_id')
        return field, 'SDSS-V', None

    # None found
    if nj==0:
        return None, None, None

    survey = plans['survey'][j]
    programname = plans['programname'][j]
    if survey == 'manga-apogee2' : field = plans['comments'][j]
    else : field = plans['name'][j]

    field = field.split()[0]
    field = field.replace('APGS_','')
    field = field.replace('APG_','')
    field = field.replace('MC-','MC')

    if survey == 'manga-apogee2' and addloc : field = '{:s}_loc{:04d}'.format(field,loc)

    return field, survey, programname

def obj2coords(tmassname):
    """ Get RA/DEC coordinates (in deg) from 2MASS-style name."""

    # Check length of string
    if len(tmassname)!=18:
        print(tmassname+' is not in the correct format')
        return None,None

    # apogeetarget/pro/make_2mass_style_id.pro makes these
    # APG-Jhhmmss[.]ss±ddmmss[.]s
    # http://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec1_8a.html

    # Parse 2MASS-style name
    #  2M00034301-7717269
    name = tmassname[-16:]  # remove any prefix by counting from the end  
    # RA: 00034301 = 00h 03m 43.02s
    ra = np.float64(name[0:2]) + np.float64(name[2:4])/60. + np.float64(name[4:8])/100./3600.
    ra *= 15     # convert to degrees
    # DEC: -7717269 = -71d 17m 26.9s
    dec = np.float64(name[9:11]) + np.float64(name[11:13])/60. + np.float64(name[13:])/10./3600.
    dec *= np.float(name[8]+'1')  # dec sign

    return ra,dec


def obj2healpix(tmassname,nside=128):
    """ Calculate healpix number for a star given it's 2MASS-style name."""
    
    # Check length of string
    if len(tmassname)!=18:
        print(tmassname+' is not in the correct format')
        return None

    # Get coordinates from the 2MASS-style name
    ra,dec = obj2coords(tmassname)

    # Calculate HEALPix number
    pix = hp.ang2pix(nside,ra,dec,lonlat=True)
    return pix
