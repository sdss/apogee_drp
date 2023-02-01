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
    """
    Get the info needed to check various things.

    Parameters
    ----------
    num : int
       APOGEE 8-digit exposure number.
    apred : str
       APOGEE reduction version, e.g. 'daily'.
    telescope : str
       Telescope name: 'apo25m' or 'lco25m'.

    Returns
    -------
    expinfo : dict
       Dictionary with information about an exposure.

    Usage
    -----
    
    expinfo = getinfo(40300010,'daily','apo25m')

    """

    observatory = telescope[0:3]
    expinfo = info.expinfo(observatory=observatory,expnum=num)
    # convert to dictionary
    expinfo = dict((key, expinfo[key][0]) for key in expinfo.dtype.names) # all CAPS
    # add some more fields
    expinfo.update({'filename3d':'', 'exists3d':False,'filename2d':'', 'exists2d':False})

    load = apload.ApLoad(apred=apred,telescope=telescope)

    # Does 3D file exist
    filename3d = load.filename('R',num=num,chips=True).replace('R-','R-b-')
    expinfo['filename3d'] = filename3d
    if os.path.exists(filename3d)==False:
        expinfo['exists3d'] = False
        return expinfo
    else:
        expinfo['exists3d'] = True
    # Does 2D file exist
    mjd = int(load.cmjd(num))
    filename2d = load.filename('2D',num=num,mjd=mjd,chips=True).replace('2D-','2D-b-')
    expinfo['filename2d'] = filename2d
    if os.path.exists(filename2d)==False:
        expinfo['exists2d'] = False
    else:
        expinfo['exists2d'] = True

    return expinfo


def check_dark(num,apred,telescope):
    """
    Perform quality checks on a DARK exposure.

    Parameters
    ----------
    num : int
       APOGEE 8-digit exposure number.
    apred : str
       APOGEE reduction version, e.g. 'daily'.
    telescope : str
       Telescope name: 'apo25m' or 'lco25m'.

    Returns
    -------
    mask : int
       Bitmask of "bad" QA values (see bitmask()) for what
        the values mean.  A "good" exposure will have mask=0.

    Usage
    -----
    
    mask = check_dark(40300010,'daily','apo25m')

    """

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstat, shutter, exists2d
    expinfo = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if expinfo['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if expinfo['nread']<3 or expinfo['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    #   gang state doesn't matter for dark
    # 3 - Wrong shutter state
    if expinfo['shutter'] is not None:
        # shutter must be closed for dark
        if expinfo['shutter']=='Open':
            mask |= 2**3
    # 4 - Wrong flux
    if expinfo['exists2d']==True:
        im = fits.getdata(expinfo['filename2d'],1)
        med = np.median(im[:,900:1100])
        # Check the flux
        if med/expinfo['nread']>100:
            mask |= 2**4

    return mask


def check_object(num,apred,telescope):
    """
    Perform quality checks on a OBJECT exposure.

    Parameters
    ----------
    num : int
       APOGEE 8-digit exposure number.
    apred : str
       APOGEE reduction version, e.g. 'daily'.
    telescope : str
       Telescope name: 'apo25m' or 'lco25m'.

    Returns
    -------
    mask : int
       Bitmask of "bad" QA values (see bitmask()) for what
        the values mean.  A "good" exposure will have mask=0.

    Usage
    -----
    
    mask = check_object(40300010,'daily','apo25m')

    """

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstat, shutter, exists2d
    expinfo = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if expinfo['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if expinfo['nread']<3 or expinfo['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    # gang state not working for LCO FPS commissioning
    if expinfo['gangstate'] != '' and telescope != 'lco25m':
        if expinfo['gangstate']=='Podium':
            mask |= 2**2
    # 3 - Wrong shutter state
    if expinfo['shutter'] != '':
        # shutter must be open for object exposures
        if expinfo['shutter']=='Closed':
            mask |= 2**3
    # 4 - Wrong flux
    if expinfo['exists2d']==True:
        im = fits.getdata(expinfo['filename2d'],1)
        # There's a bright sky line around X=1117
        sub = im[:,1117-100:1117+100]
        smsub = medfilt2d(sub,(1,7))  # smooth in spectral axis
        resmsub = dln.rebin(smsub,(2048//8,200),tot=True) # rebin in spatial axis
        peakflux = np.nanmax(resmsub,axis=1)  # peak flux feature in spectral dim.
        avgpeakflux = np.nanmean(peakflux)
        # Check skyline flux
        if avgpeakflux/expinfo['nread']<100:  # DLN 08/22/2022, changed from 200->100
            mask |= 2**4
        #print('object',med/expinfo['nread'])

    return mask


def check_domeflat(num,apred,telescope):
    """
    Perform quality checks on a DOMEFLAT exposure.

    Parameters
    ----------
    num : int
       APOGEE 8-digit exposure number.
    apred : str
       APOGEE reduction version, e.g. 'daily'.
    telescope : str
       Telescope name: 'apo25m' or 'lco25m'.

    Returns
    -------
    mask : int
       Bitmask of "bad" QA values (see bitmask()) for what
        the values mean.  A "good" exposure will have mask=0.

    Usage
    -----
    
    mask = check_domeflat(40300010,'daily','apo25m')

    """

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstate, shutter, exists2d
    expinfo = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if expinfo['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if expinfo['nread']<3 or expinfo['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    # gang state not working for LCO FPS commissioning
    if expinfo['gangstate'] != '' and telescope != 'lco25m':    
        if expinfo['gangstate']=='Podium':
            mask |= 2**2
    ## 3 - Wrong shutter state
    #if expinfo['shutter'] != '':
    #    # shutter must be open for domeflat exposures
    #    if expinfo['shutter']=='Closed':
    #        mask |= 2**3
    # 4 - Wrong flux
    if expinfo['exists2d']==True:
        im = fits.getdata(expinfo['filename2d'],1)
        medim = np.nanmedian(im[:,900:1100],axis=1)
        remedim = dln.rebin(medim,2048//8,tot=True) # rebin in spatial axis
        avgpeakflux = np.nanmean(remedim)
        # Check the flux
        if avgpeakflux/expinfo['nread']<500:
            mask |= 2**4
        #print('domeflat',med/expinfo['nread'])

    return mask


def check_quartzflat(num,apred,telescope):
    """
    Perform quality checks on a QUARTZFLAT exposure.

    Parameters
    ----------
    num : int
       APOGEE 8-digit exposure number.
    apred : str
       APOGEE reduction version, e.g. 'daily'.
    telescope : str
       Telescope name: 'apo25m' or 'lco25m'.

    Returns
    -------
    mask : int
       Bitmask of "bad" QA values (see bitmask()) for what
        the values mean.  A "good" exposure will have mask=0.

    Usage
    -----
    
    mask = check_quartzflat(40300010,'daily','apo25m')

    """

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstate, shutter, exists2d
    expinfo = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if expinfo['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if expinfo['nread']<3 or expinfo['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    if expinfo['gangstate'] != '':
        if expinfo['gangstate']!='Podium':
            mask |= 2**2
    # 3 - Wrong APOGEE shutter state
    if expinfo['shutter'] != '':
        # shutter must be open for quartzflat exposures
        if expinfo['shutter']=='Closed':
            mask |= 2**3
    # cal shutter state
    if expinfo['calshutter'] != '':
        # shutter must be open for quartzflat exposures
        if expinfo['calshutter']==False:
            mask |= 2**3            
    # 4 - Wrong flux
    if expinfo['exists2d']==True:
        im = fits.getdata(expinfo['filename2d'],1)
        medim = np.nanmedian(im[:,900:1100],axis=1)
        remedim = dln.rebin(medim,2048//8,tot=True) # rebin in spatial axis
        avgpeakflux = np.nanmean(remedim)
        # Check the flux
        if avgpeakflux/expinfo['nread']<500:
            mask |= 2**4
        #print('quartzflat',med/expinfo['nread'])

    return mask


def check_arclamp(num,apred,telescope):
    """
    Perform quality checks on a ARCLAMP exposure.

    Parameters
    ----------
    num : int
       APOGEE 8-digit exposure number.
    apred : str
       APOGEE reduction version, e.g. 'daily'.
    telescope : str
       Telescope name: 'apo25m' or 'lco25m'.

    Returns
    -------
    mask : int
       Bitmask of "bad" QA values (see bitmask()) for what
        the values mean.  A "good" exposure will have mask=0.

    Usage
    -----
    
    mask = check_arclamp(40300010,'daily','apo25m')

    """

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstate, shutter, exists2d
    expinfo = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if expinfo['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if expinfo['nread']<3 or expinfo['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    if expinfo['gangstate'] != '':
        if expinfo['gangstate']!='Podium':
            mask |= 2**2
    # 3 - Wrong shutter state
    if expinfo['shutter'] != '':
        # shutter must be open for arclamp exposures
        if expinfo['shutter']=='Closed':
            mask |= 2**3
    # cal shutter state
    if expinfo['calshutter'] != '':
        # shutter must be open for arclamp exposures
        if expinfo['calshutter']==False:
            mask |= 2**3            
    # 4 - Wrong flux
    if expinfo['exists2d']==True:
        im = fits.getdata(expinfo['filename2d'],1)
        head = fits.getheader(expinfo['filename2d'])
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
        if avgpeakflux/expinfo['nread']<thresh:
            mask |= 2**4
        #print('arclamp',med/expinfo['nread'])
        
    return mask


def check_fpi(num,apred,telescope):
    """
    Perform quality checks on a FPI exposure.

    Parameters
    ----------
    num : int
       APOGEE 8-digit exposure number.
    apred : str
       APOGEE reduction version, e.g. 'daily'.
    telescope : str
       Telescope name: 'apo25m' or 'lco25m'.

    Returns
    -------
    mask : int
       Bitmask of "bad" QA values (see bitmask()) for what
        the values mean.  A "good" exposure will have mask=0.

    Usage
    -----
    
    mask = check_fpi(40300010,'daily','apo25m')

    """

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstate, shutter, exists2d
    expinfo = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if expinfo['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if expinfo['nread']<3 or expinfo['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    if expinfo['gangstate'] != '':
        if expinfo['gangstate']!='Podium':
            mask |= 2**2
    # 3 - Wrong shutter state
    if expinfo['shutter'] != '':
        # shutter must be open for FPI exposures
        if expinfo['shutter']=='Closed':
            mask |= 2**3
    # cal shutter state
    if expinfo['calshutter'] != '':
        # shutter must be open for fpi exposures
        if expinfo['calshutter']==False:
            mask |= 2**3            
    # 4 - Wrong flux
    if expinfo['exists2d']==True:
        im = fits.getdata(expinfo['filename2d'],1)
        sub = im[:,900:1100]
        smsub = medfilt2d(sub,(1,7))  # smooth in spectral axis
        resmsub = dln.rebin(smsub,(2048//8,200),tot=True) # rebin in spatial axis
        peakflux = np.nanmax(resmsub,axis=1)  # peak flux feature in spectral dim.
        avgpeakflux = np.nanmean(peakflux)
        # Check the flux
        if avgpeakflux/expinfo['nread']<70:
            mask |= 2**4

    return mask


def check_internalflat(num,apred,telescope):
    """
    Perform quality checks on a INTERNALFLAT exposure.

    Parameters
    ----------
    num : int
       APOGEE 8-digit exposure number.
    apred : str
       APOGEE reduction version, e.g. 'daily'.
    telescope : str
       Telescope name: 'apo25m' or 'lco25m'.

    Returns
    -------
    mask : int
       Bitmask of "bad" QA values (see bitmask()) for what
        the values mean.  A "good" exposure will have mask=0.

    Usage
    -----
    
    mask = check_internalflat(40300010,'daily','apo25m')

    """

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstate, shutter, exists2d
    expinfo = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if expinfo['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if expinfo['nread']<3 or expinfo['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    if expinfo['gangstate'] != '':
        if expinfo['gangstate']!='Podium':
            mask |= 2**2
    # 3 - Wrong shutter state
    if expinfo['shutter'] != '':
        # shutter must be open good internalflat exposures
        if expinfo['shutter']=='Closed':
            mask |= 2**3
    # 4 - Wrong flux
    if expinfo['exists2d']==True:
        im = fits.getdata(expinfo['filename2d'],1)
        med = np.nanmedian(im)
        # Check the flux
        if med/expinfo['nread']<300:
            mask |= 2**4
        #print('internalflat',med/expinfo['nread'])

    return mask


def check_skyflat(num,apred,telescope):
    """
    Perform quality checks on a SKYFLAT exposure.

    Parameters
    ----------
    num : int
       APOGEE 8-digit exposure number.
    apred : str
       APOGEE reduction version, e.g. 'daily'.
    telescope : str
       Telescope name: 'apo25m' or 'lco25m'.

    Returns
    -------
    mask : int
       Bitmask of "bad" QA values (see bitmask()) for what
        the values mean.  A "good" exposure will have mask=0.

    Usage
    -----
    
    mask = check_skyflat(40300010,'daily','apo25m')

    """

    mask = 0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    # Get information
    # exists3d, nread, gangstate, shutter, exists2d
    expinfo = getinfo(num,apred,telescope)

    # Go over the cases
    #------------------

    # 0 - 3D file does not exist 
    if expinfo['exists3d']==False:
        mask |= 2**0
        return mask
    # 1 - Less than 3 reads
    if expinfo['nread']<3 or expinfo['nread'] is None:
        mask |= 2**1
        return mask
    # 2 - Wrong gang state
    if expinfo['gangstate'] != '':
        if expinfo['gangstate']=='Podium':
            mask |= 2**2
    # 3 - Wrong shutter state
    if expinfo['shutter'] != '':
        # shutter must be open for skyflat exposures
        if expinfo['shutter']=='Closed':
            mask |= 2**3
    # 5 - Wrong flux
    if expinfo['exists2d']==True:
        im = fits.getdata(expinfo['filename2d'],1)
        # There's a bright sky line around X=1117
        sub = im[:,1117-100:1117+100]
        smsub = medfilt2d(sub,(1,7))  # smooth in spectral axis
        resmsub = dln.rebin(smsub,(2048//8,200),tot=True) # rebin in spatial axis
        peakflux = np.nanmax(resmsub,axis=1)  # peak flux feature in spectral dim.
        avgpeakflux = np.nanmean(peakflux)
        # Check skyline flux
        if avgpeakflux/expinfo['nread']<200:
            mask |= 2**4

    return mask


def check(nums,apred,telescope,verbose=True,logger=None):
    """
    Perform quality checks on a list of exposures.

    Parameters
    ----------
    nums : list
       List of APOGEE 8-digit exposure numbers.
    apred : str
       APOGEE reduction version, e.g. 'daily'.
    telescope : str
       Telescope name: 'apo25m' or 'lco25m'.
    verbose : bool, optional
       Verbose output to the screen.  Default is True.
    logger : logger, optional
       Logging object.

    Returns
    -------
    mask : int
       Bitmask of "bad" QA values (see bitmask()) for what
        the values mean.  A "good" exposure will have mask=0.

    Usage
    -----
    
    mask = check([40300010,40300011,40300012]'daily','apo25m')

    """


    """ Check a list of exposures."""

    # Convert to list
    if hasattr(nums,'__iter__')==False or type(nums)==str:
        nums = [nums]
    if hasattr(nums,'__iter__')==True and type(nums) != str:
        nums = list(nums)
    nexp = len(nums)

    if logger is None:
        logger = dln.basiclogger()

    dt = np.dtype([('num',int),('exptype',(np.str,50)),('mask',int),('okay',bool)])
    out = np.zeros(nexp,dtype=dt)
    out['mask'] = -1
    out['okay'] = False
    if verbose:
        logger.info('NUM           EXPTYPE      MASK   OKAY')
    for i in range(nexp):
        num = nums[i]
        expinfo = getinfo(num,apred,telescope) 
        #print(num,expinfo['exists3d'],expinfo['gangstate'],expinfo['shutter'])
        out['num'][i] = num
        out['exptype'][i] = expinfo['exptype']
        mask = None
        # Exposure types
        if expinfo['exptype'].lower()=='dark':
            mask = check_dark(num,apred,telescope)
        elif expinfo['exptype'].lower()=='object':
            mask = check_object(num,apred,telescope)
        elif expinfo['exptype'].lower()=='domeflat':
            mask = check_domeflat(num,apred,telescope)
        elif expinfo['exptype'].lower()=='quartzflat':
            mask = check_quartzflat(num,apred,telescope)
        elif expinfo['exptype'].lower()=='arclamp':
            mask = check_arclamp(num,apred,telescope)
        elif expinfo['exptype'].lower()=='fpi':
            mask = check_fpi(num,apred,telescope)
        elif expinfo['exptype'].lower()=='internalflat':
            mask = check_internalflat(num,apred,telescope)
        elif expinfo['exptype'].lower()=='skyflat':
            mask = check_skyflat(num,apred,telescope)
        else:
            mask = None
            logger.info('exptype: '+str(expinfo['exptype'])+' not known')
        if mask is not None:
            out['mask'][i] = mask
            if mask == 0:
                out['okay'][i] = True
        else:
            out['mask'][i] = 1
            out['okay'][i] = False
        if verbose:
            if mask is not None:
                bits,bset = bitmask(mask)
                sbset = ', '.join(bset)
                lensbset = len(sbset)
            else:
                sbset = ''
                lensbset = 0
            fmt = '%8d  %13s  %5d  %6s  %-'+str(lensbset+2)+'s'
            logger.info(fmt % (out['num'][i],out['exptype'][i],out['mask'][i],out['okay'][i],sbset))

    return out
