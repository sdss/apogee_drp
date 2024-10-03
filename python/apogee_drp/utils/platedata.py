import copy
import numpy as np
import os
import glob
import pdb
import subprocess
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from dlnpyutils import utils as dln, coords
from apogee_drp.utils import spectra,yanny,apload
from apogee_drp.utils import plugmap as plmap,bitmask as bmask
from apogee_drp.plan import mkslurm
from apogee_drp.apred import mkcal
from apogee_drp.database import catalogdb,apogeedb
from sdss_access.path import path
from astropy.io import fits
from astropy.table import Table,vstack,hstack
from astropy.coordinates import Angle
from astropy import units as u
import numpy.lib.recfunctions as rfn

# filter the warnings

def coords2tmass(ra,dec):
    """ Construct 2MASS-style name (without prefix) from RA/DEC coordinates (in deg)."""

    # apogeetarget/pro/make_2mass_style_id.pro makes these
    # APG-Jhhmmss[.]ss±ddmmss[.]s
    # http://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec1_8a.html

    # 2M00034301-7717269
    # RA: 00034301 = 00h 03m 43.01s
    # DEC: -7717269 = -71d 17m 26.9s
    # 8 digits for RA and 7 digits for DEC
    name = Angle(ra,unit=u.degree).to_string(unit=u.hour,sep='',pad=True,precision=2).replace('.','')[0:9]
    name += Angle(dec,unit=u.degree).to_string(unit=u.degree,sep='',pad=True,alwayssign=True,precision=1).replace('.','')[0:8]

    return name

def save(filename,plug):
    """
    Save plugmap data to a PlateData file.

    Parameters
    ----------
    filename : str
      Filename to save the plugmap data to.
    plug : dict
      Plugmap data as returned from getdata().
    
    Returns
    -------
    The plugmap data is saved to disk.  Nothing is returned.

    Example
    -------

    save('apPlateData-3745-59636.fits',plug)

    """

    # Save the data to file
    hdu = fits.HDUList()
    hdu.append(fits.PrimaryHDU())
    hdu[0].header['plate'] = plug.get('plate')
    hdu[0].header['mjd'] = plug.get('mjd')
    hdu[0].header['location'] = plug.get('locationid')
    hdu[0].header['field'] = plug.get('field')
    hdu[0].header['program'] = plug.get('programname')
    if plug.get('ha') is not None:
        hdu[0].header['ha1'] = plug['ha'][0]
        hdu[0].header['ha2'] = plug['ha'][1]
        hdu[0].header['ha3'] = plug['ha'][2]
    fiber = plug.get('fiberdata')
    if fiber is not None:
        hdu.append(fits.table_to_hdu(Table(fiber)))
    else:
        hdu.append(fits.ImageHDU())
    hdu[1].header['EXTNAME'] = 'FIBERDATA'
    guidedata = plug.get('guidedata')
    if guidedata is not None:
        hdu.append(fits.table_to_hdu(Table(guidedata)))
    else:
        hdu.append(fits.ImageHDU())
    hdu[2].header['EXTNAME'] = 'GUIDEDATA'
    hdu.writeto(filename,overwrite=True)
    hdu.close()

    
def read(filename):
    """
    Read in plugmap data from apPlateData file.

    Parameters
    ----------
    filename : str
       PlateData filename to load.

    Returns
    -------
    plug : dict
       Plugmap data.

    Example
    -------

    plug = read('apPlateData-3745-59636.fits')

    """

    if os.path.exists(filename)==False:
        raise FileNotFoundError(filename)
    
    hdu = fits.open(filename)
    head = hdu[0].header
    if len(hdu)>1:
        fiberdata = hdu[1].data
    else:
        fiberdata = None
    if len(hdu)>2:
        guidedata = hdu[2].data
    else:
        guidedata = None
    plug = {}
    plug['plate'] = head.get('plate')
    plug['mjd'] = head.get('mjd')
    plug['locationid'] = head.get('location')
    plug['field'] = head.get('field')
    plug['programname'] = head.get('program')
    plug['ha'] = [head.get('ha1'),head.get('ha2'),head.get('ha3')]
    plug['fiberdata'] = fiberdata
    plug['guidedata'] = guidedata
    
    return plug


def getdata(plate,mjd,apred,telescope,plugid=None,asdaf=None,mapa=False,obj1m=None,
            fixfiberid=False,noobject=False,skip=False,twilight=False,
            badfiberid=None,mapper_data=None,starfiber=None,clobber=False,
            logger=None):
    """
    Getdata loads up a structure with plate information and information about the 300 APOGEE fibers
    This is obtained from a plPlugMapA file or from a 
    plPlugMapM+plateHolesSorted combination
    returned structure includes:
    fiberid, ra, dec, eta, zeta, hmag, objtype, obj (name)
    for each of 300 APOGEE (spectrographid=2) files

    Parameters
    ----------
    plate : int
       ID for the desired plate.
    mjd : int
       MJD for the plugmap information.
    apred : str
       Reduction version.
    telescope : str
       Telescope name.
    plugid : str, optional
       Name of plugmap file.
    asdaf : list, optional
       Array of fiberIDs for stars in a ASDAF
        (Any-Star-Down-Any Fiber) observation.
    mapa : bool, optional
       Use "plPlugMapA" file, otherwise "plPlugMapM".
    obj1m : str, optional
       Object name for APO-1m observation.
    fixfiberid : bool, optional
       Fix issues with fibers.  Default is False.
    noobject : bool, optional
       Don't load the apogeeObject targeting information.  Default is False.
    twilight : bool, optional
       This is a twilight observation, no stars.  Default is False.
    badiberid : list, optional
       Array of fiberIDs of bad fibers.
    mapper_data : str, optional
       Directory for mapper information (optional).
    starfiber : int, optional
       FiberID of the star for APO-1m observations.
    skip : bool, optional
       Don't load the plugmap information.  Default is False.
    clobber : bool, optional
       Overwrite any existing apPlateData file.  Default is False.
    logger : logger, optional
       Logging object.  If not is input, then a default one will be created.   

    Returns
    -------
    plandata : dict
       Targeting information for an APOGEE plate.

    Example
    -------

    plugmap = getdata(plate,mjd,plugid=plugmap.plugid)

    By J. Holtzman, 2011?
    Doc updates by D. Nidever, Sep 2020
    Many updates and fixed by D. Nidever, 2020-2024
    """

    if logger is None:
        logger = dln.basiclogger()
    
    load = apload.ApLoad(apred=apred,telescope=telescope)
    if mapper_data is None:
        if load.instrument=='apogee-n':
            mapper_data = os.environ['MAPPER_DATA_N']
        else:
            mapper_data = os.environ['MAPPER_DATA_S']
    # Data directory
    datadir = {'apo25m':os.environ['APOGEE_DATA_N'],
               'apo1m':os.environ['APOGEE_DATA_N'],
               'lco25m':os.environ['APOGEE_DATA_S']}[telescope]
    observatory = {'apo25m':'apo','apo1m':'apo','lco25m':'lco'}[telescope]

    # Check if the apPlateData file already exists
    #  if yes, load it unless clobber is set
    # Use planfile name to construct PlateData filename
    if obj1m is not None:
        planfile = load.filename('Plan',plate=plate,reduction=objid,mjd=mjd)
    else:
        if int(mjd)>=59556:  # FPS
            # get fieldid from confSummary file
            configfile = load.filename('confSummary',configid=plate)
            configlines = dln.readlines(configfile)
            cline = dln.grep(configlines,'field_id')
            fieldid = cline[0].split()[1]
            planfile = load.filename('Plan',plate=plate,mjd=mjd,
                                     field=str(fieldid)) 
        else:
            planfile = load.filename('Plan',plate=plate,mjd=mjd)
    platedatafile = planfile.replace('Plan','PlateData').replace('.yaml','.fits')
    if os.path.exists(platedatafile) and clobber==False:
        logger.info('Reading platedata from '+platedatafile)
        pdata = read(platedatafile)
        fiber = Table(pdata['fiberdata'])  # make sure columns are lowercase
        for c in fiber.colnames: fiber[c].name = c.lower()
        pdata['fiberdata'] = fiber
        return pdata

    
    # Create the output fiber structure
    dtype = np.dtype([('fiberid',int),('ra',np.float64),('dec',np.float64),
                      ('raobs',float),('decobs',float),
                      ('eta',np.float64),('zeta',np.float64),('objtype',str,10),
                      ('holetype',str,10),('object',str,30),('assigned',int),
                      ('on_target',int),('valid',int),('tmass_style',str,30),
                      ('sdss_id',int),('ra_sdss_id',float),('dec_sdss_id',float),
                      ('healpix',int),('target1',int),('target2',int),('target3',int),
                      ('target4',int),('spectrographid',int),('mag',float,5),
                      ('alt_id',str,30),('twomass_designation',str,30),('jmag',float),
                      ('jerr',float),('hmag',float),('herr',float),('kmag',float),
                      ('kerr',float),('phflag',str,50),('src_h',str,50),('pmra',float),
                      ('pmdec',float),('pm_src',str,50),('sdss5_target_pks',str,1000),
                      ('sdss5_target_catalogids',str,1000),('sdss5_target_carton_pks',str,1000),
                      ('sdss5_target_cartons',str,1000),('sdss5_target_flagshex',str,150),
                      ('brightneicount',np.int16),('brightneiflag',np.int16),
                      ('brightneifluxfrac',np.float32),('catalogid',int),
                      ('version_id',int),('sdssv_apogee_target0',int),('firstcarton',str,100),
                      ('cadence',str,100),('program',str,100),('category',str,100),
                      ('gaia_release',str,10),('gaia_sourceid',str,25),
                      ('gaia_ra',float),('gaia_dec',float),('gaia_plx',float),
                      ('gaia_plx_error',float),('gaia_pmra',float),
                      ('gaia_pmra_error',float),('gaia_pmdec',float),
                      ('gaia_pmdec_error',float),('gaia_gmag',float),('gaia_gerr',float),
                      ('gaia_bpmag',float),('gaia_bperr',float),('gaia_rpmag',float),
                      ('gaia_rperr',float)])    
    guide = np.zeros(16,dtype=dtype)
    loc = 0

    # APO-1M observations
    if obj1m is not None:
        if fixfiberid==True:
            fiberid = [218,220,222,223,226,227,228,229,230,231]
            if starfiber is None:
                starfiber = 229
        else:
            fiberid = [218,219,221,223,226,228,230]
            if starfiber is None:
                starfiber = 223
        fiber = np.zeros(len(fiberid),dtype=dtype)
        fiber['objtype'] = 'none'
        fiber['holetype'] = 'OBJECT'
        fiber['spectrographid'] = 2
        platedata = {'plate':plate, 'mjd':mjd, 'plateid':str(plate), 'locationid':1, 'field':plate, 'programname':'',
                     'ha':[-99.,-99.,-99.]}
        fiber['fiberid'] = fiberid
        fiber['objtype'] = 'SKY'
        starind, = np.where(fiber['fiberid']==starfiber)
        fiber['objtype'][starind] = 'STAR'
        skyind, = np.where(fiber['objtype'] == 'SKY')
        fiber['target2'][skyind] = 2**4
        platefile = os.environ['APOGEEREDUCEPLAN_DIR']+'/data/1m/'+str(plate)+'.fits'
        if os.path.exists(platefile):
            obj = fits.getdata(platefile,1)
            ind, = np.where(obj['name']==obj1m)
            if len(ind)>0:
                ifiber, = np.where(fiber['fiberid']==starfiber)
                fiber['object'][ifiber] = obj['name'][ind]
                fiber['tmass_style'][ifiber] = obj['name'][ind]
                fiber['hmag'][ifiber] = obj['h'][ind]
                fiber['mag'][ifiber][1] = obj['h'][ind]
                fiber['ra'][ifiber] = obj['ra'][ind]
                fiber['dec'][ifiber] = obj['dec'][ind]
                fiber['target2'][ifiber] = 2**22
        else:
            raise Exception('no file found with object information!')

        platedata['fiberdata'] = fiber  
        platedata['guidedata'] = guide
        return platedata

    # Twilight observations
    if twilight==True:
        fiber = np.zeros(300,dtype=dtype)
        platedata = {'plate':plate, 'mjd':mjd, 'locationid':1, 'field':plate, 'programname':'',
                     'ha':[-99.,-99.,-99.]}
        fiber['hmag'] = 10.
        fiber['mag'] = [10.,10.,10.,10.,10.]
        fiber['objtype'] = 'STAR'
        fiber['fiberid'] = np.arange(300)+1
        platedata['fiberdata'] = fiber
        platedata['guidedata'] = guide
        return platedata

    # Plates or FPS
    fps = False  # default
    if mjd>=59556:
        fps = True

    # INITIALIZING the FIBER table
    fiber = np.zeros(300,dtype=dtype)
    # default values
    fiber['holetype'] = 'OBJECT'
    fiber['spectrographid'] = 2
    fiber['catalogid'] = -1
    fiber['version_id'] = -1
    fiber['ra_sdss_id'] = -9999.0
    fiber['dec_sdss_id'] = -9999.0
    fiber['healpix'] = -1
    for c in ['ra','dec','raobs','decobs','eta','zeta']:
        fiber[c] = 999999.9
    gaia_cols = ['gaia_ra','gaia_dec','gaia_plx',
                 'gaia_plx_error','gaia_pmra','gaia_pmra_error','gaia_pmdec',
                 'gaia_pmdec_error','gaia_gmag','gaia_gerr','gaia_bpmag',
                 'gaia_bperr','gaia_rpmag','gaia_rperr']
    for c in gaia_cols:
        fiber[c] = -9999.99
    fiber['mag'] = 99.99  # default to faint magnitudes
    fiber['jmag'] = 99.99  # default to faint magnitudes
    fiber['hmag'] = 99.99  # default to faint magnitudes
    fiber['kmag'] = 99.99  # default to faint magnitudes
    platedata = {'plate':plate, 'mjd':mjd, 'locationid':0, 'field':' ',
                 'programname':'', 'ha':[-99.,-99.,-99.], 'fiberdata':fiber,
                 'guidedata':guide}
    field, survey, programname = apload.apfield(plate,loc,telescope=load.telescope,
                                                fps=fps)
    platedata['field'] = field
    platedata['locationid'] = loc
    platedata['programname'] = programname

    # SDSS-V FPS configuration files
    #-------------------------------
    if fps:
        # ra/dec are the observed epoch coordinates
        # racat/deccat are the catalog coordinates
        plugmapfile = plmap.plugmapfilename(plate,60000,load.instrument)
        #plugmapfile = load.filename('confSummary',configid=plate)
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
            plugdir = datadir+'/'+str(mjd)+'/'
        else:
            plugmjd = tplugid.split('-')[1]
            plugdir = mapper_data+'/'+plugmjd+'/'

        # Rare cases where the plugmap exists in the data directory but NOT in the mapper directory
        if os.path.exists(plugdir+'/'+plugfile)==False:
            if os.path.exists(datadir+'/'+str(mjd)+'/'+plugfile.replace('MapM','MapA'))==True:
                logger.info('Cannot find plPlugMapM file in mapper directory.  Using plPlugMapA file in data directory instead.')
                plugdir = datadir+'/'+str(mjd)+'/'
                root = 'plPlugMapA'
                plugfile = plugfile.replace('MapM','MapA')
            
    # Does the plugfile exist? If so, load it
    if os.path.exists(plugdir+'/'+plugfile):
        plugmap = plmap.load(plugdir+'/'+plugfile,fixfiberid=fixfiberid)
    else:
        if skip==True:
            return None
        else:
            raise Exception('Cannot find plugmap/fibermap file '+plugdir+'/'+plugfile)
        
    if fps:
        plugmap['locationId'] = plugmap['field_id']
    platedata['locationid'] = plugmap['locationId']
    if fps==False:
        platedata['ha'][0] = plugmap['ha'][0]
        platedata['ha'][1] = plugmap['ha_observable_min'][0]
        platedata['ha'][2] = plugmap['ha_observable_max'][0]

    ## Need to get tmass_style IDs from plateHolesSorted
    #platedir = os.environ['PLATELIST_DIR']+'/plates/%04dXX/%06d' % (plate//100,plate)
    #holefile = 'plateHolesSorted-%06d.par' % plate
    #print('yanny_read,'+platedir+'/'+holefile)
    #pdata = yanny.yanny(platedir+'/'+holefile,np=True)
    #gd, = np.where(pdata['STRUCT1']['holetype'].astype(str)=='APOGEE')   
    #pdata['STRUCT1']['targetids'][gd].astype(str)
    
    if mapa==False and fps==False:
        # Get the plateHolesSorted file for this plate and read it
        platestr = '{:06d}'.format(plate)
        platedir = os.environ['PLATELIST_DIR']+'/plates/%04dXX/%06d' % (plate//100,plate)
        holefile = 'plateHolesSorted-'+platestr+'.par'
        logger.info('yanny_read,'+platedir+'/'+holefile)
        pdata = yanny.yanny(platedir+'/'+holefile,np=True)
        ph = Table(pdata['STRUCT1'])
        # Use locationid from plateHoles files as there are a few cases
        #  where plugmapM is wrong
        loc = pdata['locationId']
        platedata['locationid'] = loc

        # Early S-V plate data were missing some columns
        if 'tmass_id' not in ph.colnames and 'targetids' in ph.colnames:
            ph['tmass_id'] = ph['targetids']
        if 'ra' not in ph.colnames and 'target_ra' in ph.colnames:
            ph['ra'] = ph['target_ra']
        if 'dec' not in ph.colnames and 'target_dec' in ph.colnames:
            ph['dec'] = ph['target_dec']
            
        # Rename a few columns
        ph['tmass_id'].name = 'twomass'
        ph['tmass_j'].name = 'jmag'
        ph['tmass_h'].name = 'hmag'
        ph['tmass_k'].name = 'kmag'

        # Sometimes TMASS_ID in "p" is 2MASS-J23580454-0007415
        for i in range(len(ph)):
            if 'twomass' in ph.dtype.names:
                if str(ph['twomass'][i]).find('2MASS-J') > -1:
                    ph['twomass'][i] = str(ph['twomass'][i]).replace('2MASS-J','2M')
                    
        # Fix telluric catalogIDs
        # There is a problem with some of the telluric catalogIDs due to
        # overflow.  We need to add 2**32 to them.
        # Use the minimum catalogid in the v0 cross-match
        # (version_id=21). That number is 4204681993.
        bdcatid = []
        if 'catalogid' in ph.dtype.names:
            bdcatid, = np.where((np.char.array(ph['holetype']).astype(str).find('APOGEE') > -1) & 
                                (ph['catalogid']>0) & (ph['catalogid']<4204681993))
        if len(bdcatid)>0:
            logger.info('KLUDGE!!!  Fixing overflow catalogIDs for '+str(len(bdcatid))+' telluric stars')
            logger.info(ph['catalogid'][bdcatid])
            ph['catalogid'][bdcatid] += 2**32

        # Read flag correction data
        have_flag_changes = 0
        logger.info(platedir+'/flagModifications-'+platestr+'.txt')
        if os.path.exists(platedir+'/flagModifications-'+platestr+'.txt'):
            logger.info('Reading flagModifications file: flagModifications-'+platestr+'.txt')
            flag_changes = Table.read(platedir+'/flagModifications-'+platestr+'.txt',format='ascii')
            have_flag_changes = 1
            
        # Get SDSS_ID, Gaia DR3 and other catalogdb information
        if 'catalogid' in ph.colnames and np.sum(ph['catalogid']>0)>0:
            gdcat, = np.where(ph['catalogid']>0)
            if len(gdcat)>0:
                cdata = catalogdb.getdata(catid=ph['catalogid'][gdcat])
                dt = [('version_id',int),('sdss_id',int),('ra_sdss_id',float),
                      ('dec_sdss_id',float),('gaia_sourceid',int),('gaia_ra',float),
                      ('gaia_dec',float),('gaia_pmra',float),('gaia_pmra_error',float),
                      ('gaia_pmdec',float),('gaia_pmdec_error',float),('gaia_plx',float),
                      ('gaia_plx_error',float),('gaia_gmag',float),('gaia_gerr',float),
                      ('gaia_bpmag',float),('gaia_bperr',float),('gaia_rpmag',float),
                      ('gaia_rperr',float),('gaia_release',str,10),
                      ('healpix',int),('sdss5_target_pks',str,1000),
                      ('sdss5_target_catalogids',str,1000),
                      ('sdss5_target_carton_pks',str,1000),
                      ('sdss5_target_cartons',str,1000),
                      ('sdss5_target_flagshex',str,1000)]
                temp = Table(np.zeros(len(ph),dtype=np.dtype(dt)))
                temp['version_id'] = -1
                temp['sdss_id'] = -1
                temp['ra_sdss_id'] = -9999
                temp['dec_sdss_id'] = -9999
                _,ind1,ind2 = np.intersect1d(ph['catalogid'],cdata['catalogid'],
                                             return_indices=True)
                for c in temp.colnames:
                    temp[c][ind1] = cdata[c][ind2]
                ph = hstack((ph,temp))
        else:
            apgind, = np.where(np.char.array(ph['holetype'].astype(str)).find('APOGEE')>-1)
            ph = ph[apgind]
            gdph, = np.where(ph['targettype'].astype(str)!='sky')
            cdata = catalogdb.getdata(ra=ph['ra'][gdph],dec=ph['dec'][gdph])
            if len(gdph)>0:
                dt = [('catalogid',int),('version_id',int),('sdss_id',int),('ra_sdss_id',float),
                      ('dec_sdss_id',float),('gaia_sourceid',int),('gaia_ra',float),
                      ('gaia_dec',float),('gaia_pmra',float),('gaia_pmra_error',float),
                      ('gaia_pmdec',float),('gaia_pmdec_error',float),('gaia_plx',float),
                      ('gaia_plx_error',float),('gaia_gmag',float),('gaia_gerr',float),
                      ('gaia_bpmag',float),('gaia_bperr',float),('gaia_rpmag',float),
                      ('gaia_rperr',float),('gaia_release',str,10),
                      ('healpix',int),('sdss5_target_pks',str,1000),
                      ('sdss5_target_catalogids',str,1000),
                      ('sdss5_target_carton_pks',str,1000),
                      ('sdss5_target_cartons',str,1000),
                      ('sdss5_target_flagshex',str,1000)]
                temp = Table(np.zeros(len(ph),dtype=np.dtype(dt)))
                temp['catalogid'] = -1
                temp['version_id'] = -1
                temp['sdss_id'] = -1
                temp['ra_sdss_id'] = -9999
                temp['dec_sdss_id'] = -9999
                ind1,ind2,dist = coords.xmatch(ph['ra'],ph['dec'],cdata['ra_sdss_id'],
                                               cdata['dec_sdss_id'],2,unique=True)
                if len(ind1)>0:
                    for c in temp.colnames:
                        if c in cdata.colnames:
                            temp[c][ind1] = cdata[c][ind2]
                    temp['catalogid'][ind1] = cdata['catalogid31'][ind2]
                    temp['version_id'][ind1] = 31

                # Delete catalogid and version_id in ph, if they exist already
                if 'catalogid' in ph.colnames: del ph['catalogid']
                if 'version_id' in ph.colnames: del ph['version_id']

                # Add the new columns to ph
                ph = hstack((ph,temp))
                
    # Get SDSS-V FPS photometry from catalogdb/targetdb
    if fps:
        logger.info('Querying catalogdb/targetdb')
        ph = catalogdb.getdata(designid=plugmap['design_id'])
        if len(ph)>0:
            ph['target_ra'] = ph['ra']
            ph['target_dec'] = ph['dec']
        else:
            print('No APOGEE assignments for this design.')

    # I don't understand the purpose of this next section???
    # querying by design or by catalogid gives the same results!
    # Maybe that used to be the case
        
    #if fps and len(ph)>0:
    #    # The extra "if" statement is to deal with cases when there are no apogee
    #    # assigned targets for a design and therefore "ph" is empty
    #    # Add missing columns
    #    for c in ['jmag','hmag','kmag','gaia_gmag','gaia_bpmag','gaia_rpmag']:
    #        if c not in ph.colnames: ph[c]=99.99
    #    for c in ['e_jmag','e_hmag','e_kmag','gaia_gerr','gaia_bperr','gaia_rperr']:
    #        if c not in ph.colnames: ph[c]=9.99            
    #    for c in ['pmra','pmdec','gaia_ra','gaia_dec','gaia_pmra','gaia_pmra_error',
    #              'gaia_pmdec','gaia_pmdec_error','gaia_plx','gaia_plx_error']:
    #        if c not in ph.colnames: ph[c]=np.nan
    #    if 'tmass_id' not in ph.colnames: ph['tmass_id'] = 18*' '
    #    if 'gaia_sourceid' not in ph.colnames: ph['gaia_sourceid'] = 0
    #    if 'gaia_release' not in ph.colnames: ph['gaia_release'] = 0
    #    ph['target_ra'] = ph['ra']
    #    ph['target_dec'] = ph['dec']
    #    ph['tmass_j'] = ph['jmag']
    #    ph['tmass_h'] = ph['hmag']
    #    ph['tmass_k'] = ph['kmag']
    #    # get 2MASS IDs and other info from catalogdb
    #    gdid, = np.where( (plugmap['fiberdata']['catalogid']>0) &
    #                      (plugmap['fiberdata']['spectrographId']==2) &
    #                      (plugmap['fiberdata']['objType']!='SKY'))
    #    if len(gdid)>0:
    #        catinfo = catalogdb.getdata(catid=plugmap['fiberdata']['catalogid'][gdid])
    #        if len(catinfo)>0:
    #            dum,ind1,ind2 = np.intersect1d(ph['catalogid'],catinfo['catalogid'],return_indices=True)
    #        else:
    #            ind1 = []
    #        if len(ind1)>0:
    #            ph['tmass_id'] = '                      '
    #            ph['tmass_id'][ind1] = catinfo['twomass'][ind2]
    #            ph['sdss_id'] = 0
    #            ph['sdss_id'][ind1] = catinfo['sdss_id'][ind2]
    #            ph['sdss_ra'] = -9999.0
    #            ph['sdss_ra'][ind1] = catinfo['ra_sdss_id'][ind2]
    #            ph['sdss_dec'] = -9999.0
    #            ph['sdss_dec'][ind1] = catinfo['dec_sdss_id'][ind2]
    #            ph['healpix'] = -1
    #            ph['healpix'] = catinfo['healpix'][ind2]
    #            ph['e_jmag'] = 0.0
    #            ph['e_jmag'][ind1] = catinfo['e_jmag'][ind2]
    #            ph['e_hmag'] = 0.0
    #            ph['e_hmag'][ind1] = catinfo['e_hmag'][ind2]
    #            ph['e_kmag'] = 0.0
    #            ph['e_kmag'][ind1] = catinfo['e_kmag'][ind2]
    #            ph['phflag'] = '                                  '
    #            ph['phflag'][ind1] = catinfo['twomflag'][ind2]
    #            #ph['gaia_release'] = '          '
    #            #ph['gaia_release'][ind1] = catinfo['gaia_release'][ind2]
    #            #ph['gaia_sourceid'] = '                                  '
    #            ph['gaia_sourceid'] = -1                
    #            ph['gaia_sourceid'][ind1] = catinfo['gaia'][ind2]
    #            ph['gaia_ra'] = 0.0
    #            ph['gaia_ra'][ind1] = catinfo['ra'][ind2]
    #            ph['gaia_dec'] = 0.0
    #            ph['gaia_dec'][ind1] = catinfo['dec'][ind2]
    #            ph['gaia_pmra'] = 0.0
    #            ph['gaia_pmra'][ind1] = catinfo['pmra'][ind2]
    #            ph['gaia_pmdec'] = 0.0
    #            ph['gaia_pmdec'][ind1] = catinfo['pmdec'][ind2]
    #            ph['gaia_pmra_error'] = 0.0
    #            ph['gaia_pmra_error'][ind1] = catinfo['e_pmra'][ind2]
    #            ph['gaia_pmdec_error'] = 0.0
    #            ph['gaia_pmdec_error'][ind1] = catinfo['e_pmdec'][ind2]
    #            ph['gaia_plx'] = 0.0
    #            ph['gaia_plx'][ind1] = catinfo['plx'][ind2]
    #            ph['gaia_plx_error'] = 0.0
    #            ph['gaia_plx_error'][ind1] = catinfo['e_plx'][ind2]
    #            ph['gaia_gmag'] = 0.0
    #            ph['gaia_gmag'][ind1] = catinfo['gaiamag'][ind2]
    #            ph['gaia_gerr'] = 0.0
    #            ph['gaia_gerr'][ind1] = catinfo['e_gaiamag'][ind2]
    #            ph['gaia_bpmag'] = 0.0
    #            ph['gaia_bpmag'][ind1] = catinfo['gaiabp'][ind2]
    #            ph['gaia_bperr'] = 0.0
    #            ph['gaia_bperr'][ind1] = catinfo['e_gaiabp'][ind2]
    #            ph['gaia_rpmag'] = 0.0
    #            ph['gaia_rpmag'][ind1] = catinfo['gaiarp'][ind2]
    #            ph['gaia_rperr'] = 0.0
    #            ph['gaia_rperr'][ind1] = catinfo['e_gaiarp'][ind2]

    # Load guide stars
    if fps==False:
        gind, = np.where(plugmap['fiberdata']['holeType'].astype(str) == 'GUIDE')
        for i,gi in enumerate(gind):
            guide['fiberid'][i] = plugmap['fiberdata']['fiberId'][gi]
            guide['ra'][i] = plugmap['fiberdata']['ra'][gi]
            guide['dec'][i] = plugmap['fiberdata']['dec'][gi]
            guide['eta'][i] = plugmap['fiberdata']['eta'][gi]
            guide['zeta'][i] = plugmap['fiberdata']['zeta'][gi]
            guide['spectrographid'][i] = plugmap['fiberdata']['spectrographId'][gi]
        platedata['guidedata'] = guide
        
    # Find matching plugged entry for each spectrum and load up the output information from correct source(s)
    nomatch = 0
    for i in range(300):
        fiber['spectrographid'][i] = -1
        if fps:
            m, = np.where((plugmap['fiberdata']['spectrographId'] == 2) &
                          (plugmap['fiberdata']['fiberId'] == 300-i))
        else:
            m, = np.where((plugmap['fiberdata']['holeType'].astype(str) == 'OBJECT') &
                          (plugmap['fiberdata']['spectrographId'] == 2) &
                          (plugmap['fiberdata']['fiberId'] == 300-i))
        nm = len(m)
        if badfiberid is not None:
            j, = np.where(badfiberid == 300-i)
            if len(j)>0:
                logger.info('fiber index '+str(i)+' declared as bad')
                nm = 0
        
        if nm>1:
            logger.info('halt: more than one match for fiber id !! MARVELS??')
            #logger.info(plugmap['fiberdata']['fiberId'][m],plugmap['fiberdata']['primTarget'][m],
            #      plugmap['fiberdata']['secTarget'][m])
            import pdb; pdb.set_trace()
        if nm==1:
            m = m[0]
            fiber['fiberid'][i] = plugmap['fiberdata']['fiberId'][m]
            fiber['raobs'][i] = plugmap['fiberdata']['ra'][m]
            fiber['decobs'][i] = plugmap['fiberdata']['dec'][m]
            fiber['eta'][i] = plugmap['fiberdata']['eta'][m]
            fiber['zeta'][i] = plugmap['fiberdata']['zeta'][m]
            if fps:
                fiber['assigned'][i] = plugmap['fiberdata']['assigned'][m]
                fiber['on_target'][i] = plugmap['fiberdata']['on_target'][m]
                fiber['valid'][i] = plugmap['fiberdata']['valid'][m]
                fiber['sdssv_apogee_target0'][i] = plugmap['fiberdata']['sdssv_apogee_target0'][m]
                fiber['catalogid'][i] = plugmap['fiberdata']['catalogid'][m]
                fiber['category'][i] = plugmap['fiberdata']['category'][m]
            else:
                fiber['target1'][i] = plugmap['fiberdata']['primTarget'][m]
                fiber['target2'][i] = plugmap['fiberdata']['secTarget'][m]
            fiber['spectrographid'][i] = plugmap['fiberdata']['spectrographId'][m]

            # Unassigned, off target, invalid targets
            if fps:
                # ASSIGNED: assigned=1 indicates a science/sky target is
                #   assigned to that fiber in robostrategy. assigned=0 means no
                #   target was assigned for this fiber (likely used for BOSS instead).
                #   There's no targeting information for this fiber.
                # ON_TARGET: on_target=1 indicates that the fiber actually was
                #   placed on the target. In some cases you'll find that
                #   on_target=0 although assigned=1; that usually happens when
                #   there are collisions or deadlocks that we need to solve.
                # VALID: valid=0 means there was a problem converting from
                #   on-sky coordinates to alpha/beta for the robots. Position
                #   was unreachable.
                if fiber['assigned'][i]==0 or fiber['on_target'][i]==0 or fiber['valid'][i]==0:
                    cmt = 'FiberID '+str(300-i)+' is '
                    badcmt = []
                    if fiber['assigned'][i]==0: badcmt.append('UNASSIGNED')
                    if fiber['on_target'][i]==0: badcmt.append('NOT on target')
                    if fiber['valid'][i]==0: badcmt.append('INVALID')
                    cmt += ', '.join(badcmt)
                    logger.info(cmt)
                # Unassigned, generaly no information for this target
                #   but allow for use of sky fibers for boss-assigned robots
                if fiber['assigned'][i]==0 and str(fiber['category'][i]).upper().find('SKY')==-1:
                    fiber['hmag'][i] = 99.99
                    fiber['mag'][i] = 99.99
                    continue

            # Special for asdaf object plates
            if asdaf is not None:
                # ASDAF fiber
                if 300-i == asdaf:
                    fiber['objtype'][i] = 'STAR'
                    fiber['hmag'][i] = 0.
                else:
                    fiber['objtype'][i] = 'SKY'
                    fiber['hmag'][i] = -99.999

            # Normal plate/configuration
            else:
                fiber['objtype'][i] = plugmap['fiberdata']['objType'][m].astype(str)
                # Fix up objtype
                fiber['objtype'][i] = 'STAR'
                if fps==False:
                    fiber['holetype'][i] = plugmap['fiberdata']['holeType'][m].astype(str)
                if mapa==True:
                    # HMAG's are correct from plPlugMapA files
                    fiber['hmag'][i] = plugmap['fiberdata']['mag'][m][1]
                    fiber['object'][i] = plugmap['fiberdata']['tmass_style'][m]
                    fiber['tmass_style'][i] = plugmap['fiberdata']['tmass_style'][m]
                    if bmask.is_bit_set(fiber['target2'][i],9) == 1: fiber['objtype'][i]='HOT_STD'
                    if bmask.is_bit_set(fiber['target2'][i],4) == 1: fiber['objtype'][i]='SKY'
                else:
                    # FPS object type from CATEGORY
                    if fps:
                        # objtype: STAR, HOT_STD, or SKY
                        objtype = 'STAR'
                        if plugmap['fiberdata']['category'][m].astype(str).upper() == 'SCIENCE': objtype='STAR'
                        if plugmap['fiberdata']['category'][m].astype(str).upper() == 'SKY_APOGEE': objtype='SKY'
                        if plugmap['fiberdata']['category'][m].astype(str).upper() == 'SKY_BOSS': objtype='SKY'
                        if plugmap['fiberdata']['category'][m].astype(str).upper() == 'STANDARD_APOGEE': objtype='HOT_STD'
                        fiber['objtype'][i] = objtype
                        if objtype=='SKY': continue
                        
                    # Get matching stars using catalogid for FPS
                    if fps:
                        match, = np.where(fiber['catalogid'][i]==ph['catalogid'])
                    # Get matching stars from coordinate match
                    else:
                        match, = np.where((np.abs(ph['target_ra']-fiber['raobs'][i]) < 0.00002) &
                                          (np.abs(ph['target_dec']-fiber['decobs'][i]) < 0.00002))
                    if len(match)>0:
                        # APOGEE-2 plate
                        if ('apogee2_target1' in ph.dtype.names) and (plate > 7500) and (plate < 15000) and fps==False:
                            fiber['target1'][i] = ph['apogee2_target1'][match]
                            fiber['target2'][i] = ph['apogee2_target2'][match]
                            fiber['target3'][i] = ph['apogee2_target3'][match]
                            apogee2 = 1
                            if have_flag_changes==True:
                                jj, = np.where((flag_changes['PlateID'] == plate) &
                                               (flag_changes['TARGETID'] == ph['targetids'][match]))
                                if len(jj)>0:
                                    logger.info('modifying flags for '+str(ph['targetids'][match]))
                                    fiber['target1'][i] = flag_changes['at1'][jj]
                                    fiber['target2'][i] = flag_changes['at2'][jj]
                                    fiber['target3'][i] = flag_changes['at3'][jj]
                                    fiber['target4'][i] = flag_changes['at4'][jj]
                        # APOGEE-1 plate
                        if ('apogee2_target1' not in ph.dtype.names) and (plate <= 7500) and fps==False:
                            fiber['target1'][i] = ph['apogee_target1'][match]
                            fiber['target2'][i] = ph['apogee_target2'][match]
                            apogee2 = 0
                        # SDSS-V data
                        if (plate >= 15000) or fps:
                            sdss5 = True
                        if fps==False:   # plate data
                            fiber['ra'][i] = ph['target_ra'][match]
                            fiber['dec'][i] = ph['target_dec'][match]
                            fiber['catalogid'][i] = ph['catalogid'][match]
                            fiber['version_id'][i] = ph['version_id'][match]
                            fiber['sdss_id'][i] = ph['sdss_id'][match]
                            fiber['ra_sdss_id'][i] = ph['ra_sdss_id'][match]
                            fiber['dec_sdss_id'][i] = ph['dec_sdss_id'][match]
                            fiber['healpix'][i] = ph['healpix'][match]
                            fiber['pmra'][i] = ph['pmra'][match]
                            fiber['pmdec'][i] = ph['pmdec'][match]
                            # Copy S-V catalogdb information if available
                            if plate >= 15000:
                                fiber['sdssv_apogee_target0'][i] = ph['sdssv_apogee_target0'][match]
                                fiber['firstcarton'][i] = ph['firstcarton'].astype(str)[match][0]
                                if 'sdss5_target_pks' in ph.colnames:
                                    fiber['sdss5_target_pks'][i] = ph['sdss5_target_pks'][match[0]]
                                    fiber['sdss5_target_catalogids'][i] = ph['sdss5_target_catalogids'][match[0]]
                                    fiber['sdss5_target_carton_pks'][i] = ph['sdss5_target_carton_pks'][match[0]]
                                    fiber['sdss5_target_cartons'][i] = ph['sdss5_target_cartons'][match[0]]
                                    fiber['sdss5_target_flagshex'][i] = ph['sdss5_target_flagshex'][match[0]]
                            if 'gaia_sourceid' in ph.colnames:
                                fiber['gaia_sourceid'][i] = ph['gaia_sourceid'][match][0].astype(str)
                                fiber['gaia_release'][i] = ph['gaia_release'][match][0].astype(str)
                                for n in ['gaia_ra','gaia_dec','gaia_plx','gaia_plx_error',
                                          'gaia_pmra','gaia_pmra_error','gaia_pmdec','gaia_pmdec_error',
                                          'gaia_gmag','gaia_gerr','gaia_bpmag','gaia_bperr','gaia_rpmag',
                                          'gaia_rperr']:
                                    fiber[n][i] = ph[n][match]
                            # objtype: STAR, HOT_STD, or SKY
                            objtype = 'STAR'
                            if bmask.is_bit_set(fiber['sdssv_apogee_target0'][i],0)==1: objtype='SKY'
                            if bmask.is_bit_set(fiber['sdssv_apogee_target0'][i],1)==1: objtype='HOT_STD'
                        # SDSS-V FPS data
                        else:
                            fiber['ra'][i] = ph['ra'][match]
                            fiber['dec'][i] = ph['dec'][match]
                            fiber['catalogid'][i] = ph['catalogid'][match]
                            fiber['version_id'][i] = ph['version_id'][match]
                            fiber['sdss_id'][i] = ph['sdss_id'][match]
                            fiber['ra_sdss_id'][i] = ph['ra_sdss_id'][match]
                            fiber['dec_sdss_id'][i] = ph['dec_sdss_id'][match]
                            fiber['healpix'][i] = ph['healpix'][match]
                            fiber['twomass_designation'][i] = ph['twomass'][match][0].astype(str)
                            fiber['jmag'][i] = ph['jmag'][match]
                            fiber['jerr'][i] = ph['e_jmag'][match]
                            fiber['hmag'][i] = ph['hmag'][match]
                            fiber['herr'][i] = ph['e_hmag'][match]
                            fiber['kmag'][i] = ph['kmag'][match]
                            fiber['kerr'][i] = ph['e_kmag'][match]
                            fiber['sdss5_target_pks'][i] = ph['sdss5_target_pks'][match[0]]
                            fiber['sdss5_target_catalogids'][i] = ph['sdss5_target_catalogids'][match[0]]
                            fiber['sdss5_target_carton_pks'][i] = ph['sdss5_target_carton_pks'][match[0]]
                            fiber['sdss5_target_cartons'][i] = ph['sdss5_target_cartons'][match[0]]
                            fiber['sdss5_target_flagshex'][i] = ph['sdss5_target_flagshex'][match[0]]
                            fiber['brightneicount'][i] = ph['brightneicount'][match[0]]
                            fiber['brightneiflag'][i] = ph['brightneiflag'][match[0]]
                            fiber['brightneifluxfrac'][i] = ph['brightneifluxfrac'][match[0]]
                            fiber['firstcarton'][i] = plugmap['fiberdata']['firstcarton'][m]
                            fiber['cadence'][i] = plugmap['fiberdata']['cadence'][m]
                            fiber['program'][i] = plugmap['fiberdata']['program'][m]
                            fiber['category'][i] = plugmap['fiberdata']['category'][m]
                            fiber['pmra'][i] = ph['gaia_pmra'][match]
                            fiber['pmdec'][i] = ph['gaia_pmdec'][match]
                            fiber['gaia_sourceid'][i] = ph['gaia_sourceid'][match][0].astype(str)
                            fiber['gaia_release'][i] = ph['gaia_release'][match][0].astype(str)
                            for n in ['gaia_ra','gaia_dec','gaia_plx','gaia_plx_error',
                                      'gaia_pmra','gaia_pmra_error','gaia_pmdec','gaia_pmdec_error',
                                      'gaia_gmag','gaia_gerr','gaia_bpmag','gaia_bperr','gaia_rpmag',
                                      'gaia_rperr']:
                                fiber[n][i] = ph[n][match]

                        # APOGEE-1/2 target types
                        if (plate < 15000) and not fps:
                            objtype = 'STAR'
                            if bmask.is_bit_set(fiber['target2'][i],9)==1: objtype='HOT_STD'
                            if bmask.is_bit_set(fiber['target2'][i],4)==1: objtype='SKY'
                        # SKY's
                        if (objtype=='SKY'):
                            objname = 'SKY' 
                            hmag = 99.99
                            fiber['mag'][i] = [hmag,hmag,hmag,hmag,hmag]
                            fiber['objtype'][i] = 'SKY'
                        else:
                            fiber['objtype'][i] = objtype
                            if (plate<15000) and fps==False:
                                tmp = ph['targetids'].astype(str)[match][0]
                            else:
                                tmp = ph['twomass'].astype(str)[match][0]
                            objname = tmp
                            if len(tmp) != 16:
                                # use +/- sign to figure out what characters to use
                                posind = tmp.find('+')
                                negind = tmp.find('-')
                                signind = posind
                                if posind == -1: signind = negind
                                if signind != -1:
                                    objname = tmp[signind-8:signind+8]  # trim extra characters
                                else:
                                    objname = tmp[-16:]   # not sure what else to do
                            # sometimes objname can be "None", try to fix with catalogdb info below
                            # add prefix
                            if tmp.find('A')==0:
                                objname = 'AP'+objname
                            else:
                                objname = '2M'+objname
                            hmag = ph['hmag'][match]
                            fiber['mag'][i] = [ph['jmag'][match],ph['hmag'][match],
                                               ph['kmag'][match],0.,0.]
                            # Adopt PM un-adjusted  coordinates
                            #fiber['ra'][i] -= ph['pmra'][match]/1000./3600./np.cos(fiber['dec'][i]*np.pi/180.)*(ph['epoch'][match]-2000.)
                            #fiber['dec'][i] -= ph['pmdec'][match]/1000./3600.*(ph['epoch'][match]-2000.)
                        fiber['hmag'][i] = hmag
                        fiber['object'][i] = objname
                        fiber['tmass_style'][i] = objname
                    else:
                        #raise Exception('no match found in plateHoles!',fiber['ra'][i],fiber['dec'][i], i)
                        logger.info('no match found to photometry '+str(fiber['ra'][i])+' '+str(fiber['dec'][i])+' '+str(300-i))
                        nomatch += 1
        else:
            fiber['fiberid'][i] = -1
            logger.info('no match for fiber index '+str(i))
            
    # SDSS-V, get catalogdb information for stars that are missing it
    #----------------------------------------------------------------
    # early on tellurics didn't have catalogids
    if plate >= 15000:
        logger.info('Getting catalogdb information')
        objind, = np.where((fiber['objtype']=='OBJECT') | (fiber['objtype']=='STAR') |
                           (fiber['objtype']=='HOT_STD'))
        nobjind = len(objind)
        objdata = fiber[objind]
        # Loop over objects
        for i in range(nobjind):
            istar = objind[i]
            # Already have catalogdb information, skip
            if fiber['hmag'][istar] < 50:
                continue
            nmatch = 0
            
            # Try catalogid
            if fiber['catalogid'][istar] > 0:
                catdb = catalogdb.getdata(catid=fiber['catalogid'][istar])
                if len(catdb)>0 and 'sdss_id' in catdb.colnames:
                    ind1 = [0]
                    nmatch = 1
            # Try coordinates
            if nmatch==0:
                catdb = catalogdb.getdata(ra=fiber['ra'][istar],dec=fiber['dec'][istar])
                if len(catdb)>0 and 'sdss_id' in catdb.colnames:
                    ind1 = [0]
                    nmatch = 1
            # There's a match, fill in the information
            if nmatch>0:
                ind1 = np.atleast_1d(ind1)
                # Sometimes the plateHoles tmass_style are "None", try to fix with catalogdb information
                if fiber['tmass_style'][istar]=='2MNone':
                    # Use catalogdb.tic_v8 twomass name
                    if catdb['twomass'][ind1[0]] != 'None':
                        fiber['tmass_style'][istar] = '2M'+catdb['twomass'][ind1[0]]
                    # Construct 2MASS-style name from GaiaDR2 RA/DEC
                    else:
                        fiber['tmass_style'][istar] = '2M'+coords2tmass(catdb['ra'][ind1[0]],catdb['dec'][ind1[0]])
                    logger.info('Fixing tmass_style ID for '+fiber['tmass_style'][istar])
                if fiber['catalogid'][istar]<0:
                    fiber['catalogid'][istar]=catdb['catalogid'][ind1[0]]
                fiber['twomass_designation'][istar] = catdb['twomass'][ind1[0]]
                fiber['version_id'][istar] = catdb['version_id'][ind1[0]]
                fiber['sdss_id'][istar] = catdb['sdss_id'][ind1[0]]
                fiber['ra_sdss_id'][istar] = catdb['ra_sdss_id'][ind1[0]]
                fiber['dec_sdss_id'][istar] = catdb['dec_sdss_id'][ind1[0]]
                fiber['healpix'][istar] = catdb['healpix'][ind1[0]]
                fiber['jmag'][istar] = catdb['jmag'][ind1[0]]
                fiber['jerr'][istar] = catdb['e_jmag'][ind1[0]]
                fiber['hmag'][istar] = catdb['hmag'][ind1[0]]
                fiber['herr'][istar] = catdb['e_hmag'][ind1[0]]
                fiber['kmag'][istar] = catdb['kmag'][ind1[0]]
                fiber['kerr'][istar] = catdb['e_kmag'][ind1[0]]
                fiber['phflag'][istar] = catdb['twomflag'][ind1[0]]
                fiber['gaia_release'][istar] = catdb['gaia_release'][ind1[0]]
                fiber['gaia_sourceid'][istar] = catdb['gaia_sourceid'][ind1[0]]                
                fiber['gaia_ra'][istar] = catdb['gaia_ra'][ind1[0]]
                fiber['gaia_dec'][istar] = catdb['gaia_dec'][ind1[0]]
                fiber['gaia_pmra'][istar] = catdb['gaia_pmra'][ind1[0]]
                fiber['gaia_pmra_error'][istar] = catdb['gaia_pmra_error'][ind1[0]]
                fiber['gaia_pmdec'][istar] = catdb['gaia_pmdec'][ind1[0]]
                fiber['gaia_pmdec_error'][istar] = catdb['gaia_pmdec_error'][ind1[0]]
                fiber['gaia_plx'][istar] = catdb['gaia_plx'][ind1[0]]
                fiber['gaia_plx_error'][istar] = catdb['gaia_plx_error'][ind1[0]]
                fiber['gaia_gmag'][istar] = catdb['gaia_gmag'][ind1[0]]
                fiber['gaia_gerr'][istar] = catdb['gaia_gerr'][ind1[0]]
                fiber['gaia_bpmag'][istar] = catdb['gaia_bpmag'][ind1[0]]
                fiber['gaia_bperr'][istar] = catdb['gaia_bperr'][ind1[0]]
                fiber['gaia_rpmag'][istar] = catdb['gaia_rpmag'][ind1[0]]
                fiber['gaia_rperr'][istar] = catdb['gaia_rperr'][ind1[0]]
            else:
                logger.info('no match catalogdb match for '+str(fiber['object'][istar]))

    # SDSS-V FPS data
    if fps:
        platedata['field'] = plugmap['field_id']
        if len(ph)>0:
            platedata['programname'] = ph['carton'][0]
            
    ## Load apogeeObject file to get proper name and coordinates
    ## Get apogeeObject catalog info for this field
    #if apogee2 then apogeeobject='apogee2Object' else apogeeobject='apogeeObject'
    #if not keyword_set(noobject):
    #    targetdir = getenv('APOGEE_TARGET')
    #
    ## Get apogeeObject catalog info for this field
    ## Find all matching apogeeObject files and loop through them looking for matches
    #field=strtrim(apogee_field(platedata.locationid,platenum),2)
    #files=file_search(targetdir+'/apogee*Object/*'+field+'*')
    #if files[0] eq '':
    #    stop,'cant find apogeeObject file: '+field
    #    return,field
    #else:
    #    if n_elements(files) gt 1 then print,'using multiple apogeeObject files: '+ files
    #
    #    # We will only save tags we will use, to avoid conflict between apogeeObject and apogee2Object
    #    objects = []
    #    for ifile=0,n_elements(files)-1:
    #        print,files[ifile]
    #        tmpobject = mrdfits(files[ifile],1)
    #        tmp_cat = replicate(catalog_info_common(),n_elements(tmpobject))
    #        struct_assign, tmpobject, tmp_cat
    #        print,n_elements(tmpobject)
    #        objects = [objects,tmp_cat]
    #    # Fix NaNs, etc.
    #    aspcap_fixobject,objects
    #
    #    spherematch,objects.ra,objects.dec,fiber.ra,fiber.dec,10./3600.,match1,match2,dist,maxmatch=1
    #    for i=0,299:
    #        if fiber[i].objtype eq 'STAR' or fiber[i].objtype eq 'HOT_STD' then begin
    #        j, = np.where(match2 == i)
    #        if len(j)>0:
    #            if strtrim(fiber[i].object,2) ne strtrim(objects[match1[j]].apogee_id):
    #                print,'apogeeObject differs from plateHoles: '
    #                print,fiber[i].object+' '+objects[match1[j]].apogee_id
    #                print,fiber[i].ra,' ',objects[match1[j]].ra
    #                print,fiber[i].dec,' ',objects[match1[j]].dec
    #            fiber[i].tmass_style = objects[match1[j]].apogee_id
    #            fiber[i].ra = objects[match1[j]].ra
    #            fiber[i].dec = objects[match1[j]].dec
    #            if finite(objects[match1[j]].ak_targ) then fiber[i].ak_targ=objects[match1[j]].ak_targ
    #            fiber[i].ak_targ_method = objects[match1[j]].ak_targ_method
    #            if finite(objects[match1[j]].ak_wise) then fiber[i].ak_wise=objects[match1[j]].ak_wise
    #            if finite(objects[match1[j]].sfd_ebv) then fiber[i].sfd_ebv=objects[match1[j]].sfd_ebv
    #            fiber[i].j = objects[match1[j]].j
    #            fiber[i].j_err = objects[match1[j]].j_err
    #            fiber[i].h = objects[match1[j]].h
    #            fiber[i].h_err = objects[match1[j]].h_err
    #            fiber[i].k = objects[match1[j]].k
    #            fiber[i].k_err = objects[match1[j]].k_err
    #            fiber[i].alt_id = objects[match1[j]].alt_id
    #            fiber[i].src_h = objects[match1[j]].src_h
    #            fiber[i].wash_m = objects[match1[j]].wash_m
    #            fiber[i].wash_m_err = objects[match1[j]].wash_m_err
    #            fiber[i].wash_t2 = objects[match1[j]].wash_t2
    #            fiber[i].wash_t2_err = objects[match1[j]].wash_t2_err
    #            fiber[i].ddo51 = objects[match1[j]].ddo51
    #            fiber[i].ddo51_err = objects[match1[j]].ddo51_err
    #            fiber[i].irac_3_6 = objects[match1[j]].irac_3_6
    #            fiber[i].irac_3_6_err = objects[match1[j]].irac_3_6_err
    #            fiber[i].irac_4_5 = objects[match1[j]].irac_4_5
    #            fiber[i].irac_4_5_err = objects[match1[j]].irac_4_5_err
    #            fiber[i].irac_5_8 = objects[match1[j]].irac_5_8
    #            fiber[i].irac_5_8_err = objects[match1[j]].irac_5_8_err
    #            fiber[i].irac_8_0 = objects[match1[j]].irac_8_0
    #            fiber[i].irac_8_0_err = objects[match1[j]].irac_8_0_err
    #            fiber[i].wise_4_5 = objects[match1[j]].wise_4_5
    #            fiber[i].wise_4_5_err = objects[match1[j]].wise_4_5_err
    #            fiber[i].targ_4_5 = objects[match1[j]].targ_4_5
    #            fiber[i].targ_4_5_err = objects[match1[j]].targ_4_5_err
    #            fiber[i].wash_ddo51_giant_flag = objects[match1[j]].wash_ddo51_giant_flag
    #            fiber[i].wash_ddo51_star_flag = objects[match1[j]].wash_ddo51_star_flag
    #            fiber[i].pmra = objects[match1[j]].pmra
    #            fiber[i].pmdec = objects[match1[j]].pmdec
    #            fiber[i].pm_src = objects[match1[j]].pm_src
    #        else:
    #            print('not halted: no match in object')

    platedata['fiberdata'] = fiber
    
    # Save the platedata to file
    logger.info('Saving platedata to '+platedatafile)    
    save(platedatafile,platedata)
    
    return platedata


