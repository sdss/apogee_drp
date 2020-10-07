#!/usr/bin/env python

"""CAL.PY - APOGEE Calibration Software

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20200918'  # yyyymmdd                                                                                                                           

import os
import numpy as np
#import warnings
#from astropy.io import fits
from astropy.table import Table
from dlnpyutils import utils as dln
from collections import OrderedDict


def loadcaltype(lines,caltype,dt):
   """ A small helper function for readcal(). """
   # Add a space at the end to make sure we are getting the right calibration type
   # e.g., "persist" and not "persistmodel"
   gd,ngd = dln.where(lines.find(caltype+' ') == 0)
   cat = None
   if ngd>0:
      cat = np.zeros(ngd,dtype=dt)
      for i in range(ngd):
         dum = lines[gd[i]].split()
         for j,n in enumerate(cat.dtype.names):
            cat[n][i] = dum[j+1]
   return cat

def readcal(calfile):
    """
    This reads all of the information from a master calibration index and returns
    it in a dictionary where each calibration type has a structured arrays that
    can be accessed by the calibration name (e.g. 'dark').
    """

    if os.path.exists(calfile) == False:
        raise ValueError(calfile+' NOT FOUND')
    lines = dln.readlines(calfile)
    lines = np.char.array(lines)
    # Get rid of comment and blank lines
    gd,ngd = dln.where(( lines.find('#') != 0) & (lines=='')==False )
    if ngd==0:
        raise ValueError('No good calibration lines')
    lines = lines[gd]

    # Initialize calibration dictionary
    caldict = OrderedDict()
    dtdict = OrderedDict()

    # -- Darks --
    #  mjd1, mjd2, name, frames
    #  dark       55600 56860 12910009 12910009-12910037
    #  dark       56861 99999 15640003 15640003-15640021
    dtdict['dark'] = np.dtype([('mjd1',int),('mjd2',int),('name',np.str,50),('frames',np.str,100)])
    # -- Flats --
    #  mjd1, mjd2, name, frames, nrep, dithered
    #  flat       99999 55761 01380106 1380106-1380134 1 1
    #  flat       99999 99999 02410013 2410013-2410022 1 0
    dtdict['flat'] = np.dtype([('mjd1',int),('mjd2',int),('name',np.str,50),('frames',np.str,100),
                               ('nrep',int),('dithered',int)])
    # -- Sparse --
    #  mjd1, mjd2, name, frames, darkframes, dmax, maxread
    #  sparse     55600 55761 01590015 1590015-1590024  0                   21  30,30,20
    #  sparse     55797 99999 02410059 2410059-2410068  2410058,2410069     21  30,30,20
    dtdict['sparse'] = np.dtype([('mjd1',int),('mjd2',int),('name',np.str,50),('frames',np.str,100),
                                 ('darkframes',np.str,100),('dmax',int),('maxread',np.str,100)])
    # -- Fiber --
    #  mjd1, mjd2, name
    #  fiber      55600 55761 01970078
    #  fiber      55797 56860 02410024
    dtdict['fiber'] = np.dtype([('mjd1',int),('mjd2',int),('name',np.str,50)])
    # -- Badfiber --
    #  mjd1, mjd2, frames
    #  badfiber   55600 57008   0
    #  badfiber   57009 57177   195
    dtdict['badfiber'] = np.dtype([('mjd1',int),('mjd2',int),('frames',np.str,100)])
    # -- Fixfiber --
    #  mjd1, mjd2, name
    #  fixfiber   56764 56773   1
    #  fixfiber   58038 58046   2
    dtdict['fixfiber'] = np.dtype([('mjd1',int),('mjd2',int),('name',np.str,50)])
    # -- Wave --
    #  mjd1, mjd2, name, frames, psfid
    #  wave       55699 55699 01370096 1370096,1370099  1370098
    #  wave       55700 55700 01380079 1380079          1380081
    dtdict['wave'] = np.dtype([('mjd1',int),('mjd2',int),('name',np.str,50),('frames',np.str,100),
                               ('psfid',int)])
    # -- Multiwave --
    #  mjd1, mjd2, name, frames
    #  multiwave 55800 56130 2380000  02390007,02390008,02500007
    #  multiwave 56130 56512 5680000  05870007,05870008,05870018,05870019
    dtdict['multiwave'] = np.dtype([('mjd1',int),('mjd2',int),('name',np.str,50),('frames',np.str,500)])
    # -- LSF --
    #  mjd1, mjd2, name, frames, psfid
    #  lsf 55800 56130 03430016 03430016 03430020
    #  lsf 56130 56512 07510018 07510018 07510022
    dtdict['lsf'] = np.dtype([('mjd1',int),('mjd2',int),('name',np.str,50),('frames',np.str,100),
                               ('psfid',int)])
    # -- Det --
    #  mjd1, mjd2, name, linid
    #  det        99999 99999 55640     0
    #  det        55600 56860 11870003 11870003
    dtdict['det'] = np.dtype([('mjd1',int),('mjd2',int),('name',np.str,50),('linid',int)])
    # -- BPM --
    #  mjd1, mjd2, name, darkid, flatid
    #  bpm        99999 99999 05560001 5560001  4750009
    #  bpm        55600 56860 12910009 12910009 4750009
    dtdict['bpm'] = np.dtype([('mjd1',int),('mjd2',int),('name',np.str,50),('darkid',int),
                              ('flatid',int)])
    # -- Littrow --
    #  mjd1, mjd2, name, psfid
    #  littrow    55600 56860 06670109 6670109
    #  littrow    56861 99999 13400052 13400052
    dtdict['littrow'] = np.dtype([('mjd1',int),('mjd2',int),('name',np.str,50),('psfid',int)])
    # -- Persist --
    #  mjd1, mjd2, name, darkid, flatid, thresh
    #  persist    55600 56860 04680019 4680019 4680018 0.03
    #  persist    56861 99999 13400061 13400061 13400060 0.03
    dtdict['persist'] = np.dtype([('mjd1',int),('mjd2',int),('name',np.str,50),('darkid',int),
                                  ('flatid',int),('thresh',float)])
    # -- Persistmodel --
    #  mjd1, mjd2, name
    #  persistmodel    55600 56860 57184
    #  persistmodel    56861 99999 0
    dtdict['persistmodel'] = np.dtype([('mjd1',int),('mjd2',int),('name',np.str,50)])
    # -- Response --
    #  mjd1, mjd2, name, fluxid, psfid, temp
    #  response   55600 99999  0  0   0 0
    dtdict['response'] = np.dtype([('mjd1',int),('mjd2',int),('name',np.str,50),('fluxid',int),
                                   ('psfid',int),('temp',float)])
    # Readnoise
    #  frame1, frame2
    #  rn 1380094 1380095
    #  rn 1380102 1380103
    #dtdict['rn'] = np.dtype([('frame1',int),('frame2',int)])
    # Gain
    #  frame1, frame2
    #dtdict['gain'] = np.dtype([('frame1',int),('frame2',int)])
    # READNOISE and GAIN lines are NOT used

    # Load the data
    for caltype in dtdict.keys():
       cat = loadcaltype(lines,caltype,dtdict[caltype])
       caldict[caltype.strip()] = cat

    return caldict


def parsecaldict(caldict,mjd):
    """ Small helper function for getcal() to select the entry in a calibration
    dictionary that is valid for a MJD."""
    gd,ngd = dln.where( (mjd >= caldict['mjd1']) & (mjd <= caldict['mjd2']) )
    if ngd>0:
       if ngd>1:
          gd = gd[-1]
          print('Multiple cal products found for mjd '+str(mjd)+' will use last: '+caldict['name'][gd])
       return caldict['name'][gd]
    else:
       return None


def getcal(calfile,mjd):
    """ Return the needed calibration products for a given night."""

    #caldir = os.environ['APOGEE_DRP_DIR']+'/data/cal/'
    #calfile = caldir+'apogee-n.par'

    # Read in the master calibration index
    allcaldict = readcal(calfile)

    # Loop over the calibration types and get the ones we need
    caldict = OrderedDict()
    for caltype in allcaldict.keys():
       val = parsecaldict(allcaldict[caltype],mjd)
       if type(val) is np.ndarray:
          if len(val)==1: val=val[0]
       caldict[caltype] = val

    return caldict


def mkcal():
    """ This makes a particular calibration file."""

def mkdet():
   #ret = subprocess.call(["idl","-e","mkdet,"])
   pass

def mkbpm():
   pass

def mkdark():
   pass

def mkflat():
   pass

def mkwave():
   pass

def mkmultiwave():
   pass

def mklittrow():
   pass

def mklinearity():
   pass

def mklsf():
   pass

def mkpersist():
   pass

def mkflux():
   pass
