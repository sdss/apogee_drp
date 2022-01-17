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

# This checks APOGEE exposures to make sure that they are okay

# bitmask
# 1 - 3D does not exist
# 2 - less than 3 reads
# 3 - wrong gangstate
# 4 - wrong shutterstate
# 5 - Wrong flux

def getinfo(num,apred,telescope):
    """ Get the info needed to check various things."""

    info = {'num':int(num), 'mjd':0, 'filename3d':'', 'exists3d':False,
            'exptype':'', 'nread':-1, 'gangstat':None, 'shutter':None,
            'calshutter':'', 'filename2d':'', 'exists2d':False}

    load = apload.ApLoad(apred=apred,telescope=telescope)
    mjd = int(load.cmjd(num))
    info['mjd'] = mjd

    # Does 3D file exist
    filename3d = load.filename('R',num=num,chips=True).replace('R-','R-b-')
    info['filename3d'] = filename3d
    if os.path.exists(filename3d)==False:
        info['exists3d'] = False
        return info
    else:
        info['exists3d'] = True
    # Get header
    head = fits.getheader(filename3d,1)
    exptype = head.get('exptype')
    info['exptype'] = exptype
    # Nread
    nread = head.get('nread')
    info['nread'] = nread
    # Gang state
    info['gangstat'] = head.get('gangstat')
    # APOGEE Shutter state
    info['shutter'] = head.get('shutter')
    # CalBox shutter status
    info['calshutter'] = head.get('lampshtr')
    # Does 2D file exist
    mjd = int(load.cmjd(num))
    filename2d = load.filename('2D',num=num,mjd=mjd,chips=True).replace('2D-','2D-b-')
    info['filename2d'] = filename2d
    if os.path.exists(filename2d)==False:
        info['exists2d'] = False
    else:
        info['exists2d'] = True

    return info


def check_dark(num,apred,telescope):
    """ check DARK exposure."""

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstat, shutter, exists2d
    info = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if info['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if info['nread']<3 or info['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    #   gang state doesn't matter for dark
    # 3 - Wrong shutter state
    if info['shutter'] is not None:
        # shutter must be closed for dark
        if info['shutter']=='Open':
            mask |= 2**3
    # 4 - Wrong flux
    if info['exists2d']==True:
        im = fits.getdata(info['filename2d'],1)
        med = np.median(im[:,900:1100])
        # Check the flux
        if med/info['nread']>100:
            mask |= 2**4

    return mask


def check_object(num,apred,telescope):
    """ check OBJECT exposure."""

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstat, shutter, exists2d
    info = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if info['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if info['nread']<3 or info['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    if info['gangstat'] is not None:
        if info['gangstat']=='Podium':
            mask |= 2**2
    # 3 - Wrong shutter state
    if info['shutter'] is not None:
        # shutter must be open for object exposures
        if info['shutter']=='Closed':
            mask |= 2**3
    # 4 - Wrong flux
    if info['exists2d']==True:
        im = fits.getdata(info['filename2d'],1)
        # There's a bright sky line around X=1117
        sub = im[:,1117-100:1117+100]
        smsub = medfilt2d(sub,(1,7))  # smooth in spectral axis
        resmsub = dln.rebin(smsub,(2048//8,200),tot=True) # rebin in spatial axis
        peakflux = np.nanmax(resmsub,axis=1)  # peak flux feature in spectral dim.
        avgpeakflux = np.nanmean(peakflux)
        # Check skyline flux
        if avgpeakflux/info['nread']<200:
            mask |= 2**4
        #print('object',med/info['nread'])

    return mask


def check_domeflat(num,apred,telescope):
    """ check DOMEFLAT exposure."""

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstat, shutter, exists2d
    info = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if info['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if info['nread']<3 or info['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    if info['gangstat'] is not None:
        if info['gangstat']=='Podium':
            mask |= 2**2
    # 3 - Wrong shutter state
    if info['shutter'] is not None:
        # shutter must be open for domeflat exposures
        if info['shutter']=='Closed':
            mask |= 2**3
    # 4 - Wrong flux
    if info['exists2d']==True:
        im = fits.getdata(info['filename2d'],1)
        medim = np.nanmedian(im[:,900:1100],axis=1)
        remedim = dln.rebin(medim,2048//8,tot=True) # rebin in spatial axis
        avgpeakflux = np.nanmean(remedim)
        # Check the flux
        if avgpeakflux/info['nread']<500:
            mask |= 2**4
        #print('domeflat',med/info['nread'])

    return mask


def check_quartzflat(num,apred,telescope):
    """ check QUARTZFLAT exposure."""

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstat, shutter, exists2d
    info = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if info['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if info['nread']<3 or info['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    if info['gangstat'] is not None:
        if info['gangstat']!='Podium':
            mask |= 2**2
    # 3 - Wrong APOGEE shutter state
    if info['shutter'] is not None:
        # shutter must be open for quartzflat exposures
        if info['shutter']=='Closed':
            mask |= 2**3
    # cal shutter state
    if info['calshutter'] is not None:
        # shutter must be open for quartzflat exposures
        if info['calshutter']==False:
            mask |= 2**3            
    # 4 - Wrong flux
    if info['exists2d']==True:
        im = fits.getdata(info['filename2d'],1)
        medim = np.nanmedian(im[:,900:1100],axis=1)
        remedim = dln.rebin(medim,2048//8,tot=True) # rebin in spatial axis
        avgpeakflux = np.nanmean(remedim)
        # Check the flux
        if avgpeakflux/info['nread']<500:
            mask |= 2**4
        #print('quartzflat',med/info['nread'])

    return mask


def check_arclamp(num,apred,telescope):
    """ check ARCLAMP exposure."""

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstat, shutter, exists2d
    info = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if info['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if info['nread']<3 or info['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    if info['gangstat'] is not None:
        if info['gangstat']!='Podium':
            mask |= 2**2
    # 3 - Wrong shutter state
    if info['shutter'] is not None:
        # shutter must be open for arclamp exposures
        if info['shutter']=='Closed':
            mask |= 2**3
    # cal shutter state
    if info['calshutter'] is not None:
        # shutter must be open for arclamp exposures
        if info['calshutter']==False:
            mask |= 2**3            
    # 4 - Wrong flux
    if info['exists2d']==True:
        im = fits.getdata(info['filename2d'],1)
        head = fits.getheader(info['filename2d'])
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
        if avgpeakflux/info['nread']<thresh:
            mask |= 2**4
        #print('arclamp',med/info['nread'])

    return mask


def check_fpi(num,apred,telescope):
    """ check FPI exposure."""

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstat, shutter, exists2d
    info = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if info['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if info['nread']<3 or info['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    if info['gangstat'] is not None:
        if info['gangstat']!='Podium':
            mask |= 2**2
    # 3 - Wrong shutter state
    if info['shutter'] is not None:
        # shutter must be open for FPI exposures
        if info['shutter']=='Closed':
            mask |= 2**3
    # cal shutter state
    if info['calshutter'] is not None:
        # shutter must be open for fpi exposures
        if info['calshutter']==False:
            mask |= 2**3            
    # 4 - Wrong flux
    if info['exists2d']==True:
        im = fits.getdata(info['filename2d'],1)
        sub = im[:,900:1100]
        smsub = medfilt2d(sub,(1,7))  # smooth in spectral axis
        resmsub = dln.rebin(smsub,(2048//8,200),tot=True) # rebin in spatial axis
        peakflux = np.nanmax(resmsub,axis=1)  # peak flux feature in spectral dim.
        avgpeakflux = np.nanmean(peakflux)
        # Check the flux
        if avgpeakflux/info['nread']<70:
            mask |= 2**4

    return mask


def check_internalflat(num,apred,telescope):
    """ check INTERNALFLAT exposure."""

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstat, shutter, exists2d
    info = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if info['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if info['nread']<3 or info['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    if info['gangstat'] is not None:
        if info['gangstat']!='Podium':
            mask |= 2**2
    # 3 - Wrong shutter state
    if info['shutter'] is not None:
        # shutter must be open good internalflat exposures
        if info['shutter']=='Closed':
            mask |= 2**3
    # 4 - Wrong flux
    if info['exists2d']==True:
        im = fits.getdata(info['filename2d'],1)
        med = np.nanmedian(im)
        # Check the flux
        if med/info['nread']<300:
            mask |= 2**4
        #print('internalflat',med/info['nread'])

    return mask


def check_skyflat(num,apred,telescope):
    """ check SKYFLAT exposure."""

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstat, shutter, exists2d
    info = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if info['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if info['nread']<3 or info['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    if info['gangstat'] is not None:
        if info['gangstat']=='Podium':
            mask |= 2**2
    # 3 - Wrong shutter state
    if info['shutter'] is not None:
        # shutter must be open for skyflat exposures
        if info['shutter']=='Closed':
            mask |= 2**3
    # 5 - Wrong flux
    if info['exists2d']==True:
        im = fits.getdata(info['filename2d'],1)
        # There's a bright sky line around X=1117
        sub = im[:,1117-100:1117+100]
        smsub = medfilt2d(sub,(1,7))  # smooth in spectral axis
        resmsub = dln.rebin(smsub,(2048//8,200),tot=True) # rebin in spatial axis
        peakflux = np.nanmax(resmsub,axis=1)  # peak flux feature in spectral dim.
        avgpeakflux = np.nanmean(peakflux)
        # Check skyline flux
        if avgpeakflux/info['nread']<200:
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
        info = getinfo(num,apred,telescope) 
        #print(num,info['exists3d'],info['gangstat'],info['shutter'])
        out['num'][i] = num
        out['exptype'][i] = info['exptype']
        mask = None
        # Exposure types
        if info['exptype'].lower()=='dark':
            mask = check_dark(num,apred,telescope)
        elif info['exptype'].lower()=='object':
            mask = check_object(num,apred,telescope)
        elif info['exptype'].lower()=='domeflat':
            mask = check_domeflat(num,apred,telescope)
        elif info['exptype'].lower()=='quartzflat':
            mask = check_quartzflat(num,apred,telescope)
        elif info['exptype'].lower()=='arclamp':
            mask = check_arclamp(num,apred,telescope)
        elif info['exptype'].lower()=='fpi':
            mask = check_fpi(num,apred,telescope)
        elif info['exptype'].lower()=='internalflat':
            mask = check_internalflat(num,apred,telescope)
        elif info['exptype'].lower()=='skyflat':
            mask = check_skyflat(num,apred,telescope)
        else:
            print('exptype: ',info['exptype'],' not known')
        if mask is not None:
            out['mask'][i] = mask
            if mask == 0:
                out['okay'][i] = True
        if verbose:
            print('%8d  %13s  %5d  %6s' % (out['num'][i],out['exptype'][i],out['mask'][i],out['okay'][i]))

    return out
