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
from apogee_drp.database import catalogdb
from sdss_access.path import path
from astropy.io import fits
from astropy.table import Table,vstack

# filter the warnings

def getdata(plate,mjd,apred,telescope,plugid=None,asdaf=None,mapa=False,obj1m=None,
            fixfiberid=False,noobject=False,skip=False,twilight=False,
            badfiberid=None,mapper_data=None,starfiber=None):
    """
    Getdata loads up a structure with plate information and information about the 300 APOGEE fibers
    This is obtained from a plPlugMapA file or from a 
    plPlugMapM+plateHolesSorted combination
    returned structure includes:
    fiberid, ra, dec, eta, zeta, hmag, objtype, obj (name)
    for each of 300 APOGEE (spectrographid=2) files

    Parameters
    ----------
    plate          ID for the desired plate.
    mjd            MJD for the plugmap information.
    apred          Reduction version.
    telescope      Telescope name.
    =plugid        Name of plugmap file.
    =asdaf         Array of fiberIDs for stars in a ASDAF
                     (Any-Star-Down-Any Fiber) observation.
    /mapa          Use "plPlugMapA" file, otherwise "plPlugMapM".
    =obj1m         Object name for APO-1m observation.
    /fixfiberid    Fix issues with fibers.
    /noobject      Don't load the apogeeObject targeting information.
    /twilight      This is a twilight observation, no stars.
    =badiberid     Array of fiberIDs of bad fibers.
    =mapper_data   Directory for mapper information (optional).
    =starfiber     FiberID of the star for APO-1m observations.
    /skip          Don't load the plugmap information.

    Returns
    -------
    plandata       Targeting information for an APOGEE plate.

    Example
    -------
    plugmap = getdata(plate,mjd,plugid=plugmap.plugid)


    By J. Holtzman, 2011?
    Doc updates by D. Nidever, Sep 2020

    """

    load = apload.ApLoad(apred=apred,telescope=telescope)
    if mapper_data is None:
        if load.instrument=='apogee-n':
            mapper_data = os.environ['MAPPER_DATA_N']
        else:
            mapper_data = os.environ['MAPPER_DATA_S']
    # Data directory
    datadir = {'apo25m':os.environ['APOGEE_DATA_N'],'apo1m':os.environ['APOGEE_DATA_N'],
               'lco25m':os.environ['APOGEE_DATA_S']}[telescope]

    # Create the output fiber structure
    dtype = np.dtype([('fiberid',int),('ra',np.float64),('dec',np.float64),('eta',np.float64),('zeta',np.float64),
                      ('objtype',np.str,10),('holetype',np.str,10),('object',np.str,30),
                      ('tmass_style',np.str,30),('target1',int),('target2',int),('target3',int),('target4',int),
                      ('spectrographid',int),('mag',float,5),('alt_id',np.str,30),('twomass_designation',np.str,30),
                      ('jmag',float),('jerr',float),('hmag',float),('herr',float),('kmag',float),('kerr',float),
                      ('phflag',np.str,50),('src_h',np.str,50),('pmra',float),('pmdec',float),('pm_src',np.str,50),
                      ('catalogid',int),('gaia_g',float),('gaia_bp',float),('gaia_rp',float),('sdssv_apogee_target0',int),
                      ('firstcarton',np.str,100),('gaiadr2_sourceid',np.str,25),('gaiadr2_ra',float),('gaiadr2_dec',float),
                      ('gaiadr2_plx',float),('gaiadr2_plx_error',float),('gaiadr2_pmra',float),('gaiadr2_pmra_error',float),
                      ('gaiadr2_pmdec',float),('gaiadr2_pmdec_error',float),('gaiadr2_gmag',float),('gaiadr2_gerr',float),
                      ('gaiadr2_bpmag',float),('gaiadr2_bperr',float),('gaiadr2_rpmag',float),('gaiadr2_rperr',float)])
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
        platedata = {'plate':plate, 'mjd':mjd, 'locationid':1, 'field':plate, 'programname':'',
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

    fiber = np.zeros(300,dtype=dtype)
    platedata = {'plate':plate, 'mjd':mjd, 'locationid':0, 'field':' ', 'programname':'',
                 'ha':[-99.,-99.,-99.], 'fiberdata':fiber, 'guidedata':guide}
    field, survey, programname = apload.apfield(plate,loc,telescope=load.telescope)
    platedata['field'] = field
    platedata['locationid'] = loc
    platedata['programname'] = programname

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
        tplugid = root+'-'+str(plate)
        plugfile = tplugid+'.par'
    if mapa==True:
        plugdir = datadir+'/'+str(mjd)+'/'
    else:
        plugmjd = tplugid.split('-')[1]
        plugdir = mapper_data+'/'+plugmjd+'/'

    # Does the plugfile exist? If so, load it
    if os.path.exists(plugdir+'/'+plugfile):
        plugmap = plmap.load(plugdir+'/'+plugfile,fixfiberid=fixfiberid)
    else:
        if skip==True:
            return None
        else:
            raise Exception('Cannot find plugmap file '+plugdir+'/'+plugfile)

    platedata['locationid'] = plugmap['locationId']
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

    if mapa==False:
        # Get the plateHolesSorted file for this plate and read it
        platestr = '{:06d}'.format(plate)
        platedir = os.environ['PLATELIST_DIR']+'/plates/%04dXX/%06d' % (plate//100,plate)
        holefile = 'plateHolesSorted-'+platestr+'.par'
        print('yanny_read,'+platedir+'/'+holefile)
        pdata = yanny.yanny(platedir+'/'+holefile,np=True)
        ph = pdata['STRUCT1']
        # Use locationid from plateHoles files as there are a few cases
        #  where plugmapM is wrong
        loc = pdata['locationId']
        platedata['locationid'] = loc

        # Read flag correction data
        have_flag_changes = 0
        print(platedir+'/flagModifications-'+platestr+'.txt')
        if os.path.exists(platedir+'/flagModifications-'+platestr+'.txt'):
            print('Reading flagModifications file: ','flagModifications-'+platestr+'.txt')
            flag_changes = Table.read(platedir+'/flagModifications-'+platestr+'.txt',format='ascii')
            have_flag_changes = 1

    # Load guide stars
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
    for i in range(300):
        fiber['spectrographid'][i] = -1
        m, = np.where((plugmap['fiberdata']['holeType'].astype(str) == 'OBJECT') &
                      (plugmap['fiberdata']['spectrographId'] == 2) &
                      (plugmap['fiberdata']['fiberId'] == 300-i))
        nm = len(m)
        if badfiberid is not None:
            j, = np.where(badfiberid == 300-i)
            if len(j)>0:
                print('fiber index ',i,' declared as bad')
                nm = 0
        if nm>1:
            print('halt: more than one match for fiber id !! MARVELS??')
            print(plugmap['fiberdata']['fiberId'][m],plugmap['fiberdata']['primTarget'][m],
                  plugmap['fiberdata']['secTarget'][m])
            import pdb; pdb.set_trace()
        if nm==1:
            m = m[0]
            fiber['fiberid'][i] = plugmap['fiberdata']['fiberId'][m]
            fiber['ra'][i] = plugmap['fiberdata']['ra'][m]
            fiber['dec'][i] = plugmap['fiberdata']['dec'][m]
            fiber['eta'][i] = plugmap['fiberdata']['eta'][m]
            fiber['zeta'][i] = plugmap['fiberdata']['zeta'][m]
            fiber['target1'][i] = plugmap['fiberdata']['primTarget'][m]
            fiber['target2'][i] = plugmap['fiberdata']['secTarget'][m]
            fiber['spectrographid'][i] = plugmap['fiberdata']['spectrographId'][m]
            
            # Special for asdaf object plates
            if asdaf is not None:
                # ASDAF fiber
                if 300-i == asdaf:
                    fiber['objtype'][i] = 'STAR'
                    fiber['hmag'][i] = 0.
                else:
                    fiber['objtype'][i] = 'SKY'
                    fiber['hmag'][i] = -99.999

            # Normal plate
            else:
                fiber['objtype'][i] = plugmap['fiberdata']['objType'][m].astype(str)
                # Fix up objtype
                fiber['objtype'][i] = 'STAR'
                fiber['holetype'][i] = plugmap['fiberdata']['holeType'][m].astype(str)
                if mapa==True:
                    # HMAG's are correct from plPlugMapA files
                    fiber['hmag'][i] = plugmap['fiberdata']['mag'][m][1]
                    fiber['object'][i] = plugmap['fiberdata']['tmass_style'][m]
                    fiber['tmass_style'][i] = plugmap['fiberdata']['tmass_style'][m]
                    if bmask.is_bit_set(fiber['target2'][i],9) == 1: fiber['objtype'][i]='HOT_STD'
                    if bmask.is_bit_set(fiber['target2'][i],4) == 1: fiber['objtype'][i]='SKY'
                else:
                    # Get matching stars from coordinate match
                    match, = np.where((np.abs(ph['target_ra']-fiber['ra'][i]) < 0.00002) &
                                      (np.abs(ph['target_dec']-fiber['dec'][i]) < 0.00002))
                    if len(match)>0:
                        # APOGEE-2 plate
                        if ('apogee2_target1' in ph.dtype.names) and (plate > 7500) and (plate < 15000):
                            fiber['target1'][i] = ph['apogee2_target1'][match]
                            fiber['target2'][i] = ph['apogee2_target2'][match]
                            fiber['target3'][i] = ph['apogee2_target3'][match]
                            apogee2 = 1
                            if have_flag_changes==True:
                                jj, = np.where((flag_changes['PlateID'] == plate) &
                                               (flag_changes['TARGETID'] == ph['targetids'][match]))
                                if len(jj)>0:
                                    print('modifying flags for',ph['targetids'][match])
                                    fiber['target1'][i] = flag_changes['at1'][jj]
                                    fiber['target2'][i] = flag_changes['at2'][jj]
                                    fiber['target3'][i] = flag_changes['at3'][jj]
                                    fiber['target4'][i] = flag_changes['at4'][jj]
                        # APOGEE-1 plate
                        if ('apogee2_target1' not in ph.dtype.names) and (plate <= 7500):
                            fiber['target1'][i] = ph['apogee_target1'][match]
                            fiber['target2'][i] = ph['apogee_target2'][match]
                            apogee2 = 0
                        # SDSS-V plate
                        if (plate >= 15000):
                            fiber['catalogid'][i] = ph['catalogid'][match]
                            fiber['gaia_g'][i] = ph['gaia_g'][match]
                            fiber['gaia_bp'][i] = ph['gaia_bp'][match]
                            fiber['gaia_rp'][i] = ph['gaia_rp'][match]
                            fiber['sdssv_apogee_target0'][i] = ph['sdssv_apogee_target0'][match]
                            fiber['firstcarton'][i] = ph['firstcarton'][match][0].astype(str)
                            fiber['pmra'][i] = ph['pmra'][match]
                            fiber['pmdec'][i] = ph['pmdec'][match]
                            # objtype: OBJECT, HOT_STD, or SKY                                                                                                           
                            objtype = 'OBJECT'
                            if bmask.is_bit_set(fiber['sdssv_apogee_target0'][i],0)==1: objtype='SKY'
                            if bmask.is_bit_set(fiber['sdssv_apogee_target0'][i],1)==1: objtype='HOT_STD'
                            sdss5 = 1                            
                        # APOGEE-1/2 target types
                        if (plate < 15000):
                            objtype = 'OBJECT'
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
                            if (plate<15000):
                                tmp = ph['targetids'][match][0].astype(str)
                            else:
                                tmp = ph['tmass_id'][match][0].astype(str)
                            objname = tmp[-16:]
                            if tmp.find('A')==0:
                                objname = 'AP'+objname
                            else:
                                objname = '2M'+objname
                            hmag = ph['tmass_h'][match]
                            fiber['mag'][i] = [ph['tmass_j'][match],ph['tmass_h'][match],ph['tmass_k'][match],0.,0.]
                            # Adopt PM un-adjusted  coordinates
                            #fiber['ra'][i] -= ph['pmra'][match]/1000./3600./np.cos(fiber['dec'][i]*np.pi/180.)*(ph['epoch'][match]-2000.)
                            #fiber['dec'][i] -= ph['pmdec'][match]/1000./3600.*(ph['epoch'][match]-2000.)
                        fiber['hmag'][i] = hmag
                        fiber['object'][i] = objname
                        fiber['tmass_style'][i] = objname
                    else:
                        raise Exception('no match found in plateHoles!',fiber['ra'][i],fiber['dec'][i], i)
        else:
            fiber['fiberid'][i] = -1
            print('no match for fiber index',i)


    # SDSS-V, get catalogdb information
    #----------------------------------
    if plate >= 15000:
        print('Getting catalogdb information')
        objind, = np.where((fiber['objtype']=='OBJECT') | (fiber['objtype']=='HOT_STD'))
        nobjind = len(objind)
        objdata = fiber[objind]
        gdid,ngdid,bdid,nbdid = dln.where(objdata['catalogid'] > 0,comp=True)
        # Get catalogdb information using catalogID
        catdb = None
        if ngdid>0:
            print('Querying catalogdb using catalogID for '+str(ngdid)+' stars')
            catdb1 = catalogdb.getdata(catid=objdata['catalogid'][gdid])
            # Got some results
            if len(catdb1)>0:
                print('Got results for '+str(len(catdb1))+' stars')
                catdb = catdb1.copy()
            else:
                print('No results')
        # Get catalogdb information using coordinates (tellurics don't have IDs)    
        if nbdid>0:
            print('Querying catalogdb using coordinates for ',str(nbdid)+' stars')
            catdb2 = catalogdb.getdata(ra=objdata['ra'][bdid],dec=objdata['dec'][bdid])
            # this returns a q3c_dist columns that we don't want to keep
            if len(catdb2)>0:
                print('Got results for '+str(len(catdb2))+' stars')
                del catdb2['q3c_dist']
                if catdb is None:
                    catdb = catdb2.copy()
                else:
                    catdb = vstack((catdb,catdb2))
            else:
                print('No results')
        # Add catalogdb information
        for i in range(nobjind):
            istar = objind[i]
            ind1,ind2 = dln.match(catdb['catalogid'],fiber['catalogid'][istar])
            nmatch = len(ind1)
            # some stars are missing ids, use coordinate matching indeas   
            if nmatch==0:
                dist = coords.sphdist(catdb['ra'],catdb['dec'],fiber['ra'][istar],fiber['dec'][istar])*3600
                ind1, = np.where(dist < 0.5)
                nmatch = len(ind1)
                if nmatch > 1:
                    ind1 = np.argmin(dist)
            if nmatch>0:
                ind1 = np.atleast_1d(ind1)
                if fiber['catalogid'][istar]<0:
                    fiber['catalogid'][istar]=catdb['catalogid'][ind1[0]]
                fiber['twomass_designation'][istar] = catdb['twomass'][ind1[0]]
                fiber['jmag'][istar] = catdb['jmag'][ind1[0]]
                fiber['jerr'][istar] = catdb['e_jmag'][ind1[0]]
                fiber['hmag'][istar] = catdb['hmag'][ind1[0]]
                fiber['herr'][istar] = catdb['e_hmag'][ind1[0]]
                fiber['kmag'][istar] = catdb['kmag'][ind1[0]]
                fiber['kerr'][istar] = catdb['e_kmag'][ind1[0]]
                fiber['phflag'][istar] = catdb['twomflag'][ind1[0]]
                fiber['gaiadr2_sourceid'][istar] = catdb['gaia'][ind1[0]]
                fiber['gaiadr2_ra'][istar] = catdb['ra'][ind1[0]]
                fiber['gaiadr2_dec'][istar] = catdb['dec'][ind1[0]]
                fiber['gaiadr2_pmra'][istar] = catdb['pmra'][ind1[0]]
                fiber['gaiadr2_pmra_error'][istar] = catdb['e_pmra'][ind1[0]]
                fiber['gaiadr2_pmdec'][istar] = catdb['pmdec'][ind1[0]]
                fiber['gaiadr2_pmdec_error'][istar] = catdb['e_pmdec'][ind1[0]]
                fiber['gaiadr2_plx'][istar] = catdb['plx'][ind1[0]]
                fiber['gaiadr2_plx_error'][istar] = catdb['e_plx'][ind1[0]]
                fiber['gaiadr2_gmag'][istar] = catdb['gaiamag'][ind1[0]]
                fiber['gaiadr2_gerr'][istar] = catdb['e_gaiamag'][ind1[0]]
                fiber['gaiadr2_bpmag'][istar] = catdb['gaiabp'][ind1[0]]
                fiber['gaiadr2_bperr'][istar] = catdb['e_gaiabp'][ind1[0]]
                fiber['gaiadr2_rpmag'][istar] = catdb['gaiarp'][ind1[0]]
                fiber['gaiadr2_rperr'][istar] = catdb['e_gaiarp'][ind1[0]]
            else:
                print('no match catalogdb match for ',fiber['object'][istar])






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

    return platedata


