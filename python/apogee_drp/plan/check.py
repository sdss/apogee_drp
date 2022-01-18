import copy
import numpy as np
import os
import shutil
from glob import glob
import pdb
import subprocess
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from dlnpyutils import utils as dln
from apogee_drp.utils import spectra,yanny,apload,platedata,utils
from sdss_access.path import path
from astropy.io import fits
from collections import OrderedDict
from scipy.signal import medfilt2d
from ..utils import info

# This checks APOGEE exposures to make sure that they are okay

# bitmask
# 0 - 3D does not exist
# 1 - less than 3 reads
# 2 - wrong gangstate
# 3 - wrong shutterstate
# 4 - Wrong flux

def bitmask(mask):
    """ Return values set in mask, both integers and string values"""

    # bitmask values
    # 0 - 3D does not exist
    # 1 - less than 3 reads
    # 2 - wrong gangstate
    # 3 - wrong shutterstate
    # 4 - Wrong flux
    bitnames = ['apR files do not exist','Too few reads','Wrong gangstate','Wrong shutter state','Wrong flux']
    bits = []
    bset = []
    for i in range(5):
        if mask & 2**i > 0:
            bits.append(i)
            bset.append(bitnames[i])
    return bits,bset


def getinfo(num,apred,telescope):
    """ Get the info needed to check various things."""

    observatory = telescope[0:3]
    tab = info.expinfo(observatory=observatory,expnum=num)
    # convert to dictionary
    tab = dict((key, tab[key][0]) for key in tab.dtype.names) # all CAPS
    # add some more fields
    tab.update({'filename3d':'', 'exists3d':False,'filename2d':'', 'exists2d':False})

    load = apload.ApLoad(apred=apred,telescope=telescope)

    # Does 3D file exist
    filename3d = load.filename('R',num=num,chips=True).replace('R-','R-b-')
    tab['filename3d'] = filename3d
    if os.path.exists(filename3d)==False:
        tab['exists3d'] = False
        return tab
    else:
        tab['exists3d'] = True
    # Does 2D file exist
    mjd = int(load.cmjd(num))
    filename2d = load.filename('2D',num=num,mjd=mjd,chips=True).replace('2D-','2D-b-')
    tab['filename2d'] = filename2d
    if os.path.exists(filename2d)==False:
        tab['exists2d'] = False
    else:
        tab['exists2d'] = True

    return tab


def check_dark(num,apred,telescope):
    """ check DARK exposure."""

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstat, shutter, exists2d
    tab = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if tab['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if tab['nread']<3 or tab['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    #   gang state doesn't matter for dark
    # 3 - Wrong shutter state
    if tab['shutter'] is not None:
        # shutter must be closed for dark
        if tab['shutter']=='Open':
            mask |= 2**3
    # 4 - Wrong flux
    if tab['exists2d']==True:
        im = fits.getdata(tab['filename2d'],1)
        med = np.median(im[:,900:1100])
        # Check the flux
        if med/tab['nread']>100:
            mask |= 2**4

    return mask


def check_object(num,apred,telescope):
    """ check OBJECT exposure."""

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstat, shutter, exists2d
    tab = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if tab['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if tab['nread']<3 or tab['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    if tab['gangstate'] is not None:
        if tab['gangstate']=='Podium':
            mask |= 2**2
    # 3 - Wrong shutter state
    if tab['shutter'] is not None:
        # shutter must be open for object exposures
        if tab['shutter']=='Closed':
            mask |= 2**3
    # 4 - Wrong flux
    if tab['exists2d']==True:
        im = fits.getdata(tab['filename2d'],1)
        # There's a bright sky line around X=1117
        sub = im[:,1117-100:1117+100]
        smsub = medfilt2d(sub,(1,7))  # smooth in spectral axis
        resmsub = dln.rebin(smsub,(2048//8,200),tot=True) # rebin in spatial axis
        peakflux = np.nanmax(resmsub,axis=1)  # peak flux feature in spectral dim.
        avgpeakflux = np.nanmean(peakflux)
        # Check skyline flux
        if avgpeakflux/tab['nread']<200:
            mask |= 2**4
        #print('object',med/tab['nread'])

    return mask


def check_domeflat(num,apred,telescope):
    """ check DOMEFLAT exposure."""

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstate, shutter, exists2d
    tab = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if tab['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if tab['nread']<3 or tab['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    if tab['gangstate'] is not None:
        if tab['gangstate']=='Podium':
            mask |= 2**2
    # 3 - Wrong shutter state
    if tab['shutter'] is not None:
        # shutter must be open for domeflat exposures
        if tab['shutter']=='Closed':
            mask |= 2**3
    # 4 - Wrong flux
    if tab['exists2d']==True:
        im = fits.getdata(tab['filename2d'],1)
        medim = np.nanmedian(im[:,900:1100],axis=1)
        remedim = dln.rebin(medim,2048//8,tot=True) # rebin in spatial axis
        avgpeakflux = np.nanmean(remedim)
        # Check the flux
        if avgpeakflux/tab['nread']<500:
            mask |= 2**4
        #print('domeflat',med/tab['nread'])

    return mask


def check_quartzflat(num,apred,telescope):
    """ check QUARTZFLAT exposure."""

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstate, shutter, exists2d
    tab = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if tab['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if tab['nread']<3 or tab['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    if tab['gangstate'] is not None:
        if tab['gangstate']!='Podium':
            mask |= 2**2
    # 3 - Wrong APOGEE shutter state
    if tab['shutter'] is not None:
        # shutter must be open for quartzflat exposures
        if tab['shutter']=='Closed':
            mask |= 2**3
    # cal shutter state
    if tab['calshutter'] is not None:
        # shutter must be open for quartzflat exposures
        if tab['calshutter']==False:
            mask |= 2**3            
    # 4 - Wrong flux
    if tab['exists2d']==True:
        im = fits.getdata(tab['filename2d'],1)
        medim = np.nanmedian(im[:,900:1100],axis=1)
        remedim = dln.rebin(medim,2048//8,tot=True) # rebin in spatial axis
        avgpeakflux = np.nanmean(remedim)
        # Check the flux
        if avgpeakflux/tab['nread']<500:
            mask |= 2**4
        #print('quartzflat',med/tab['nread'])

    return mask


def check_arclamp(num,apred,telescope):
    """ check ARCLAMP exposure."""

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstate, shutter, exists2d
    tab = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if tab['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if tab['nread']<3 or tab['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    if tab['gangstate'] is not None:
        if tab['gangstate']!='Podium':
            mask |= 2**2
    # 3 - Wrong shutter state
    if tab['shutter'] is not None:
        # shutter must be open for arclamp exposures
        if tab['shutter']=='Closed':
            mask |= 2**3
    # cal shutter state
    if tab['calshutter'] is not None:
        # shutter must be open for arclamp exposures
        if tab['calshutter']==False:
            mask |= 2**3            
    # 4 - Wrong flux
    if tab['exists2d']==True:
        im = fits.getdata(tab['filename2d'],1)
        head = fits.getheader(tab['filename2d'])
        # UNE
        #  bright line at X=1452
        if head.get('LAMPUNE')==True:
            sub = im[:,1452-100:1452+100]
            thresh = 40
        # THARNE
        #  bright line at X=1566
        elif head.get('LAMPTHAR')==True:
            sub = im[:,1566-100:1566+100]
            thresh = 1000
        else:
            sub = im[:,900:1100]
            thresh = 10
        smsub = medfilt2d(sub,(1,7))  # smooth in spectral axis
        resmsub = dln.rebin(smsub,(2048//8,200),tot=True) # rebin in spatial axis
        peakflux = np.nanmax(resmsub,axis=1)  # peak flux feature in spectral dim.
        avgpeakflux = np.nanmean(peakflux)
        # Check the line flux
        if avgpeakflux/tab['nread']<thresh:
            mask |= 2**4
        #print('arclamp',med/tab['nread'])

    return mask


def check_fpi(num,apred,telescope):
    """ check FPI exposure."""

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstate, shutter, exists2d
    tab = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if tab['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if tab['nread']<3 or tab['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    if tab['gangstate'] is not None:
        if tab['gangstate']!='Podium':
            mask |= 2**2
    # 3 - Wrong shutter state
    if tab['shutter'] is not None:
        # shutter must be open for FPI exposures
        if tab['shutter']=='Closed':
            mask |= 2**3
    # cal shutter state
    if tab['calshutter'] is not None:
        # shutter must be open for fpi exposures
        if tab['calshutter']==False:
            mask |= 2**3            
    # 4 - Wrong flux
    if tab['exists2d']==True:
        im = fits.getdata(tab['filename2d'],1)
        sub = im[:,900:1100]
        smsub = medfilt2d(sub,(1,7))  # smooth in spectral axis
        resmsub = dln.rebin(smsub,(2048//8,200),tot=True) # rebin in spatial axis
        peakflux = np.nanmax(resmsub,axis=1)  # peak flux feature in spectral dim.
        avgpeakflux = np.nanmean(peakflux)
        # Check the flux
        if avgpeakflux/tab['nread']<70:
            mask |= 2**4

    return mask


def check_internalflat(num,apred,telescope):
    """ check INTERNALFLAT exposure."""

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstate, shutter, exists2d
    tab = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if tab['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if tab['nread']<3 or tab['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    if tab['gangstate'] is not None:
        if tab['gangstate']!='Podium':
            mask |= 2**2
    # 3 - Wrong shutter state
    if tab['shutter'] is not None:
        # shutter must be open good internalflat exposures
        if tab['shutter']=='Closed':
            mask |= 2**3
    # 4 - Wrong flux
    if tab['exists2d']==True:
        im = fits.getdata(tab['filename2d'],1)
        med = np.nanmedian(im)
        # Check the flux
        if med/tab['nread']<300:
            mask |= 2**4
        #print('internalflat',med/tab['nread'])

    return mask


def check_skyflat(num,apred,telescope):
    """ check SKYFLAT exposure."""

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstate, shutter, exists2d
    tab = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if tab['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if tab['nread']<3 or tab['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    if tab['gangstate'] is not None:
        if tab['gangstate']=='Podium':
            mask |= 2**2
    # 3 - Wrong shutter state
    if tab['shutter'] is not None:
        # shutter must be open for skyflat exposures
        if tab['shutter']=='Closed':
            mask |= 2**3
    # 5 - Wrong flux
    if tab['exists2d']==True:
        im = fits.getdata(tab['filename2d'],1)
        # There's a bright sky line around X=1117
        sub = im[:,1117-100:1117+100]
        smsub = medfilt2d(sub,(1,7))  # smooth in spectral axis
        resmsub = dln.rebin(smsub,(2048//8,200),tot=True) # rebin in spatial axis
        peakflux = np.nanmax(resmsub,axis=1)  # peak flux feature in spectral dim.
        avgpeakflux = np.nanmean(peakflux)
        # Check skyline flux
        if avgpeakflux/tab['nread']<200:
            mask |= 2**4

    return mask


def check(nums,apred,telescope,verbose=True):
    """ Check a list of exposures."""

    if type(nums) is np.ndarray:
        nums = list(nums)
    if type(nums) is not list:
        nums = [nums]
    nexp = len(nums)

    dt = np.dtype([('num',int),('exptype',(np.str,50)),('mask',int),('okay',bool)])
    out = np.zeros(nexp,dtype=dt)
    out['mask'] = -1
    out['okay'] = False
    if verbose:
        print('NUM           EXPTYPE      MASK   OKAY')
    for i in range(nexp):
        num = nums[i]
        tab = getinfo(num,apred,telescope) 
        #print(num,tab['exists3d'],tab['gangstate'],tab['shutter'])
        out['num'][i] = num
        out['exptype'][i] = tab['exptype']
        mask = None
        # Exposure types
        if tab['exptype'].lower()=='dark':
            mask = check_dark(num,apred,telescope)
        elif tab['exptype'].lower()=='object':
            mask = check_object(num,apred,telescope)
        elif tab['exptype'].lower()=='domeflat':
            mask = check_domeflat(num,apred,telescope)
        elif tab['exptype'].lower()=='quartzflat':
            mask = check_quartzflat(num,apred,telescope)
        elif tab['exptype'].lower()=='arclamp':
            mask = check_arclamp(num,apred,telescope)
        elif tab['exptype'].lower()=='fpi':
            mask = check_fpi(num,apred,telescope)
        elif tab['exptype'].lower()=='internalflat':
            mask = check_internalflat(num,apred,telescope)
        elif tab['exptype'].lower()=='skyflat':
            mask = check_skyflat(num,apred,telescope)
        else:
            print('exptype: ',tab['exptype'],' not known')
        if mask is not None:
            out['mask'][i] = mask
            if mask == 0:
                out['okay'][i] = True
        if verbose:
            bits,bset = bitmask(mask)
            sbset = ', '.join(bset)
            
            print('%8d  %13s  %5d  %6s  %-100s' % (out['num'][i],out['exptype'][i],out['mask'][i],out['okay'][i],sbset))

    return out
