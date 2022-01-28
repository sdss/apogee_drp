import copy
import numpy as np
import os
import shutil
from glob import glob
import pdb

from dlnpyutils import utils as dln
from . import apload
from astropy.io import fits


def expinfo(observatory=None,mjd5=None,files=None,expnum=None):
    """
    Get header information about raw APOGEE files.
    This program can be run with observatory+mjd5 or
    by giving a list of files.

    Parameters
    ----------
    observatory : str, optional
        APOGEE observatory (apo or lco).
    mjd5 : int, optional
        The MJD5 night to get exposure information for.
    files : list of str, optional
        List of APOGEE apz filenames.
    expnum : list, optional
        List of exposure numbers.

    Returns
    -------
    cat : numpy structured array
        Table with information for each file grabbed from the header.

    Examples
    --------
    info = expinfo(files)

    By D.Nidever,  Oct 2020
    """

    # Types of inputs:
    #  files, observatory+mjd5, observatory+expnum
    if (files is None and observatory is None) or (files is None and mjd5 is None and expnum is None):
        raise ValueError('Either files or observatory+mjd5 or observatory+expnum must be input')
    if (mjd5 is not None and expnum is not None):
        raise ValueError('Input either observatory+mjd5 or observatory+expnum')

    load = apload.ApLoad(apred='daily',telescope='apo25m')

    # Get the exposures info for this MJD5        
    if files is None and expnum is None:
        if observatory not in ['apo','lco']:
            raise ValueError('observatory must be apo or lco')
        datadir = {'apo':os.environ['APOGEE_DATA_N'],'lco':os.environ['APOGEE_DATA_S']}[observatory]
        files = glob(datadir+'/'+str(mjd5)+'/a?R-c*.apz')
        files = np.array(files)
        nfiles = len(files)
        if nfiles==0:
            return []
        files = files[np.argsort(files)]  # sort        
    # Exposure numbers input
    if files is None and expnum is not None:
        if observatory not in ['apo','lco']:
            raise ValueError('observatory must be apo or lco')
        datadir = {'apo':os.environ['APOGEE_DATA_N'],'lco':os.environ['APOGEE_DATA_S']}[observatory]
        if type(expnum) is not list and type(expnum) is not np.ndarray:
            expnum = [expnum]
        nfiles = len(expnum)
        files = []
        for i in range(nfiles):
            file1 = load.filename('R',num=expnum[i],chips=True).replace('R-','R-c-')
            files.append(file1)
        files = np.array(files)
        files = files[np.argsort(files)]  # sort        

    nfiles = len(files)
    dtype = np.dtype([('num',int),('nread',int),('exptype',np.str,20),('arctype',np.str,20),('plateid',np.str,20),
                      ('configid',np.str,50),('designid',np.str,50),('fieldid',np.str,50),('exptime',float),
                      ('dateobs',np.str,50),('gangstate',np.str,20),('shutter',np.str,20),('calshutter',np.str,20),
                      ('mjd',int),('observatory',(np.str,10)),('dithpix',float)])
    tab = np.zeros(nfiles,dtype=dtype)
    # Loop over the files
    for i in range(nfiles):
        if os.path.exists(files[i]):
            head = fits.getheader(files[i],1)
            base,ext = os.path.splitext(os.path.basename(files[i]))
            # apR-c-12345678.apz
            num = base.split('-')[2]
            tab['num'][i] = num
            tab['nread'][i] = head['nread']
            tab['exptype'][i] = head['exptype']
            tab['plateid'][i] = head['plateid']
            tab['configid'][i] = head.get('configid')
            tab['designid'][i] = head.get('designid')
            tab['fieldid'][i] = head.get('fieldid')
            tab['exptime'][i] = head['exptime']
            tab['dateobs'][i] = head['date-obs']
            mjd = int(load.cmjd(int(num)))
            tab['mjd'] = mjd
            #    tab['mjd'] = utils.getmjd5(head['date-obs'])
            if observatory is not None:
                tab['observatory'] = observatory
            else:
                tab['observatory'] = {'p':'apo','s':'lco'}[base[1]]
            # arc types
            if tab['exptype'][i]=='ARCLAMP':
                if head['lampune']==1:
                    tab['arctype'][i] = 'UNE'
                elif head['lampthar']==1:
                    tab['arctype'][i] = 'THAR'
                else:
                    tab['arctype'][i] = 'None'
            # FPI
            if tab['exptype'][i]=='ARCLAMP' and tab['arctype'][i]=='None' and head.get('OBSCMNT')=='FPI':
                tab['exptype'][i] = 'FPI'

            # Sky flat
            if tab['exptype'][i]=='OBJECT' and tab['nread'][i]>10 and tab['nread'][i]<13 and head.get('OBSCMNT').lower().replace(' ','')[0:3]=='sky':
                tab['exptype'][i] = 'SKYFLAT'

            # Dither position
            tab['dithpix'][i] = head['dithpix']
            # Gang state
            #  gangstat wasn't working properly until MJD=59592
            if mjd>=59592:
                tab['gangstate'][i] = head.get('gangstat')
            # APOGEE Shutter state
            #  shutter wasn't working perly until MJD=59592
            if mjd>=59592:
                tab['shutter'][i] = head.get('shutter')
            # CalBox shutter status
            lampshtr = head.get('lampshtr')
            if lampshtr is not None:
                if lampshtr:
                    tab['calshutter'][i] = 'Open'
                else:
                    tab['calshutter'][i] = 'Closed'

    return tab


