import numpy as np
import os
import glob
import yaml
from dlnpyutils import utils as dln,coords
from apogee_drp.utils import spectra,yanny,apload
from apogee_drp.plan import mkslurm
from apogee_drp.apred import mkcal
from sdss_access.path import path
from astropy.io import fits

def load(plugfile,verbose=False,fixfiberid=None):
    """
    This program loads an APOGEE plugmap file.

    Parameters
    ----------
    plugfile : str
           The absolute path of the plugmap file

    Returns
    -------
    plugmap : dict
           The plugmap structure with all the relevant information

    Example
    -------
    pmap = plugmap.load(plugfile)

    By D.Nidever  May 2010
    converted to python, Oct 2020
    """

    # Check that the plug file exists
    if os.path.exists(plugfile)==False:
        raise ValueError(plugfile+' NOT FOUND')

    # Load the plugmap yanny file
    plugmap = yanny.yanny(plugfile,np=-True)

    # Add ETA/ZETA to plugmap structure
    fiberdata = plugmap['PLUGMAPOBJ']
    del plugmap['PLUGMAPOBJ']   # gets replaced with fiberdata below
    fiberdata = dln.addcatcols(fiberdata,np.dtype([('zeta',np.float64),('eta',np.float64)]))
    zeta,eta = coords.rotsphcen(fiberdata['ra'],fiberdata['dec'],np.float64(plugmap['raCen']),np.float64(plugmap['decCen']),gnomic=True)
    fiberdata['zeta'] = zeta
    fiberdata['eta'] = eta

    # Fix bit 6 erroneously set in plugmap files for tellurics 
    ind, = np.where((fiberdata['spectrographId']==2) &
                    (fiberdata['holeType'].astype(str)=='OBJECT') &
                    (fiberdata['objType'].astype(str)=='HOT_STD'))
    if len(ind)>0:
        fiberdata['secTarget'][ind] = np.int32(fiberdata['secTarget'][ind] & 0xFFFFFFDF)
        #fiberdata['secTarget'][ind] = np.uint64(fiberdata['secTarget'][ind] & 0xFFFFFFDF)

    # Custom errors in mapping?
    if fixfiberid is not None:
        if fixfiberid==1:
            starind = np.where(fiberdata['spectrographId']==2)
            for istar in range(len(star)):
                fiberid = fiberdata['fiberId'][starind[istar]]
                if fiberid>=0:
                    subid = (fiberid - 1) % 30
                    bundleid = (fiberid-subid)//30
                    fiberdata['fiberId'][star[istar]] = (9-bundleid)*30 + subid +1
                print(star,fiberid,subid,bundleid,fiberdata['fiberId'][star[istar]])
        if fixfiberid==2:
            # MTP#2 rotated
            starind, = np.where((fiberdata['spectrographId']==2) &
                               (fiberdata['holeType']=='OBJECT'))
            fiberid = fiberdata['fiberId'][starind]
            j, = np.where((fiberid==31) & (fiberid<=36))
            fiberdata['fiberId'][starind[j]] = fiberid[j] + 23
            j, = np.where((fiberid==37) & (fiberid<=44))
            fiberdata['fiberId'][starind[j]] = fiberid[j] + 8
            j, = np.where((fiberid==45) & (fiberid<=52))
            fiberdata['fiberId'][starind[j]] = fiberid[j] - 8
            j, = np.where((fiberid==54) & (fiberid<=59))
            fiberdata['fiberId'][starind[j]] = fiberid[j] - 23
            # Missing fibers from unpopulated 2 of MTP
            j, = np.where((fiberdata['fiberId'][star]==53) | (fiberdata['fiberId'][star]==60))
            fiberdata['fiberId'][starind[j]] = -1
        
    # Plug fixed fiberdata back in
    plugmap['fiberdata'] = fiberdata

    return plugmap
