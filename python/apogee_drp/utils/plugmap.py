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
from . import bitmask as bmask

def plugmapfilename(plate,mjd,instrument,plugid=None,mapa=False,mapper_data=None,
                    logger=None,verbose=True):
    """
    Construct the plugmap filename using plate and mjd.
    Works for both plate and FPS data.
    Should this be in apload???

    Parameters
    ----------
    plate : int
       ID for the desired plate.  For the FPS era, this is the configid.
    mjd : int
       MJD for the plugmap information.
    instrument : str
       APOGEE instrument name: 'apogee-n' or 'apogee-s'.
    plugid : str, optional
       Name of plugmap file.
    mapa : bool, optional
       Use "plPlugMapA" file, otherwise "plPlugMapM".
    mapper_data : str, optional
       Directory for mapper information (optional).  If not input, this will
         be obtained from the environmental variables MAPPER_DATA_N/S.
    logger : logging object, optional
       Logging object used for logging output.
    verbose : bool, optional
       Verbose output to the screen.  Default is True.

    Returns
    -------
    plugfile : str
       Full path to plugmap filename.

    Examples
    --------

    plugfile = plugmapfilename(15073,59190,'apogee-n')

    """

    if logger is None:
        logger = dln.basiclogger()
    
    # Plates or FPS
    fps = False  # default
    if mjd>=59556:
        fps = True

    if instrument=='apogee-n':
        telescope = 'apo25m'
        datadir = os.environ['APOGEE_DATA_N']
        if mapper_data is None:
            mapper_data = os.environ['MAPPER_DATA_N']
    elif instrument=='apogee-s':
        telescope = 'lco25m'
        datadir = os.environ['APOGEE_DATA_S']
        if mapper_data is None:
            mapper_data = os.environ['MAPPER_DATA_S']
    else:
        raise ValueError('no instrument '+str(instrument))
            
    # SDSS-V FPS configuration files
    #-------------------------------
    if fps:
        aload = apload.ApLoad(apred='daily',telescope=telescope)
        plugmapfile = aload.filename('confSummary',configid=plate)
        plugdir = os.path.dirname(plugmapfile)+'/'
        plugfile = os.path.basename(plugmapfile)

    # SDSS-III/IV/V Plate plugmap files
    #----------------------------------
    else:
        # Do we want to use a plPlugMapA file with the matching already done?
        havematch = 0
        if mapa==True:
            root = 'plPlugMapA'
        else:
            root = 'plPlugMapM'
        if plugid is not None:
            base,ext = os.path.splitext(os.path.basename(plugid))
            if base.find('plPlug')>-1:
                tplugid = base[11:]
            else:
                tplugid = base
            plugfile = root+'-'+tplugid+'.par'
        else:
            #tplugid = root+'-'+str(plate)
            #plugfile = tplugid+'.par'            
            tplugid = str(plate)+'-'+str(mjd)+'-01'
            plugfile = root+'-'+tplugid+'.par'
        if mapa==True:
            plugdir = os.path.join(datadir,str(mjd))
        else:
            plugmjd = tplugid.split('-')[1]
            plugdir = os.path.join(mapper_data,plugmjd)

        # Rare cases where the plugmap exists in the data directory but NOT in the mapper directory
        if os.path.exists(os.path.join(plugdir,plugfile))==False:
            if os.path.exists(os.path.join(datadir,str(mjd),plugfile.replace('MapM','MapA')))==True:
                if verbose:
                    logger.info('Cannot find plPlugMapM file in mapper directory.  Using plPlugMapA file in data directory instead.')
                plugdir = os.path.join(datadir,str(mjd))
                root = 'plPlugMapA'
                plugfile = plugfile.replace('MapM','MapA')
            else:
                logger.info('Cannot find plugmap file for '+str(mjd)+' '+str(plugfile))
                return ''

    return os.path.join(plugdir,plugfile)

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
    if 'PLUGMAPOBJ' in plugmap.keys():
        fiberdata = plugmap['PLUGMAPOBJ']  # plate plugmap
        del plugmap['PLUGMAPOBJ']   # gets replaced with fiberdata below
        FPS = False
    else:
        fiberdata = plugmap['FIBERMAP']   # FPS configuration file
        del plugmap['FIBERMAP']   # gets replaced with fiberdata below
        FPS = True
    fiberdata = dln.addcatcols(fiberdata,np.dtype([('zeta',np.float64),('eta',np.float64)]))
    zeta,eta = coords.rotsphcen(fiberdata['ra'],fiberdata['dec'],np.float64(plugmap['raCen']),np.float64(plugmap['decCen']),gnomic=True)
    fiberdata['zeta'] = zeta
    fiberdata['eta'] = eta

    # SDSS-V FPS configuration data
    if FPS:
        # Fix some early duplicates
        # P0650 = 275 (was duplicate 175)
        # P0880 = 276 (was duplicate 176)
        # P0177 = 286 (was duplicate 186)
        bd, = np.where((fiberdata['positionerId']==650) & (fiberdata['spectrographId']==2) & (fiberdata['fiberId']==175))
        if len(bd)>0: fiberdata['fiberId'][bd]=275
        bd, = np.where((fiberdata['positionerId']==880) & (fiberdata['spectrographId']==2) & (fiberdata['fiberId']==176))
        if len(bd)>0: fiberdata['fiberId'][bd]=276
        bd, = np.where((fiberdata['positionerId']==177) & (fiberdata['spectrographId']==2) & (fiberdata['fiberId']==186))
        if len(bd)>0: fiberdata['fiberId'][bd]=286        

        # Add objType
        fiberdata = dln.addcatcols(fiberdata,np.dtype([('objType',np.str,20)]))
        fiberdata['objType'] = 'OBJECT'   # default
        category = np.char.array(fiberdata['category'].astype(str)).upper()
        skyind, = np.where( (fiberdata['fiberId']>=0) & (fiberdata['spectrographId']==2) &
                            ((category=='SKY') | (category=='SKY_APOGEE') | (category=='SKY_BOSS')))
                            #(bmask.is_bit_set(fiberdata['sdssv_apogee_target0'],0)==1))    # SKY
        if len(skyind)>0:
            fiberdata['objType'][skyind] = 'SKY'
        tellind, = np.where( (fiberdata['fiberId']>=0) & (fiberdata['spectrographId']==2) &
                             ((category=='STANDARD_APOGEE') | (category=='HOT_STD')))
                             #(bmask.is_bit_set(fiberdata['sdssv_apogee_target0'],1)==1))    # HOT_STD/telluric
        if len(tellind)>0:
            fiberdata['objType'][tellind] = 'HOT_STD'
        # Plug fixed fiberdata back in
        plugmap['fiberdata'] = fiberdata

    # SDSS-III/IV/V Plate plugmap data
    else:
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
