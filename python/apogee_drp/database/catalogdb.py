# Get catalogdb data for sources
import numpy as np
from . import apogeedb
from dlnpyutils import utils as dln
from astropy.coordinates import SkyCoord
import astropy.units as u
import traceback

#from sdssdb.peewee.sdss5db.catalogdb import SDSS_ID_flat    
#from sdssdb.peewee.sdss5db.targetdb import database
#import sdssdb
#test = database.set_profile('pipelines')
#database.connect()
#except:
#    traceback.print_exc()

lead_mjd = {
 'bhm_csc':0.0,
 'bhm_rm_v0':0.0,
 'gaia_dr2_source':0.0,
 'gaia_dr3_source':0.0,
 'gaia_qso':0.0,
 'guvcat':0.0,
 'legacy_survey_dr10':0.0,
 'legacy_survey_dr8':0.0,
 'panstarrs1':0.0,
 'ps1_g18':0.0,
 'sdss_dr13_photoobj':0,
 'sdss_dr13_photoobj_primary':0,
 'sdss_dr16_specobj':0,
 'skymapper_dr1_1':0,
 'skymapper_dr2':0,
 'supercosmos':0,
 'tic_v8':0,
 'tic_v8_extended':0,
 'twomass_psc':51544.0,
 'tycho2':48437.0,
 'unwise':0}
#'skies_v1':XXX,
#'skies_v2':XXX,

def get_ids(catalogid=None,sdss_id=None,apogee_id=None):
    """
    Get 2MASS, Gaia DR3 ID, sdss_id,  catalogids.  Deal with duplicate issues.
    # Only ONE star at a time.
    """

    # Get SDSS_IDs
    db = apogeedb.DBSession()
    
    # Use catalogid
    if catalogid is not None:
        sql = 'select s.*,f.catalogid,f.version_id from catalogdb.sdss_id_flat as f'
        sql += ' join catalogdb.sdss_id_stacked as s on f.sdss_id=s.sdss_id where f.catalogid='+str(catalogid)        
        data = db.query(sql=sql,fmt="table")
        if len(data)==0:
            print('No results found for catalogid'+str(catalogid))
            db.close()
            return []
    # Use sdss_id
    elif sdss_id is not None:
        sql = 'select * from catalogdb.sdss_id_stacked where sdss_id='+str(sdss_id)
        data = db.query(sql=sql,fmt="table")
        if len(data)==0:
            print('No results found for sdss_id'+str(sdss_id))
            db.close()
            return []
        data.sort('sdss_id')
        # Get latest catalogid for this sdss_id
        catids = np.array([data['catalogid31'][0],data['catalogid25'][0],data['catalogid21'][0]])
        gdcatid, = np.where(catids>0)
        data['catalogid'] = -999999
        if len(gdcatid)>0:
            data['catalogid'][0] = catids[gdcatid[0]]
        else:
            print('Weird! No catalogid for sdss_id='+str(sdss_id))
    else:
        raise ValueError('Need catalogid or sdss_id')


    # Get proper motion and "lead" from catalogdb.catalog
    data['pmra'] = np.nan
    data['pmdec'] = np.nan
    data['lead'] = 50*' '
    if data['catalogid'][0] > 0:
        sql = 'select * from catalogdb.catalog where catalogid='+str(data['catalogid'][0])
        cdata = db.query(sql=sql,fmt="table")
        if len(cdata)>0:
            data['pmra'][0] = cdata['pmra'][0]
            data['pmdec'][0] = cdata['pmdec'][0]
            data['lead'][0] = cdata['lead'][0]
        else:
            print('Weird! No catalogdb.catalog entry for catalogid='+str(data['catalogid'][0]))
    
    # Deal with duplicates
    #---------------------
    if len(data)>1:
        if len(data)==2 and data['catalogid31'][0]>0 and data['catalogid31'][0]==data['catalogid'][1]:
        # Do they have the same catalogid31?  if yes, then just pick the smaller sdss_id
            print('duplicates in sdss_id_flat. picking the smaller sdss_id')
            data = data[:1]
        else:
            print('Dealing with duplicates')
            import pdb; pdb.set_trace()
        
    # Get Gaia DR3 ID
    #----------------
    # First, check any existing cross-matches
    gdata = []
    for catname in ['catalogid31','catalogid25','catalogid21']:
        if catname in data.colnames and data[catname][0]>0:
            catid1 = data[catname][0]
            sql = 'select * from catalogdb.catalog_to_gaia_dr3_source where catalogid='+str(catid1)
            gxdata = db.query(sql=sql,fmt="table")
            if len(gxdata)>0:
                sourceid = gxdata['target_id'][0]
                gcols = 'source_id,ra,dec,pmra,pmdec,phot_g_mean_mag as gmag' 
                sql = 'select '+gcols+' from catalogdb.gaia_dr3_source where source_id='+str(sourceid)
                gdata = db.query(sql=sql,fmt="table")
                if len(gdata)>0:
                    gdata['xflag'] = 1
                    break                
    # Second, do our own crossmatch
    if len(gdata)==0:
        print('Do own gaia xmatch')
        ra = data['ra_sdss_id'][0]
        pmra = data['pmra'][0]
        dec = data['dec_sdss_id'][0]
        pmdec = data['pmdec'][0]
        # No proper motions, query gaia dr3 around the coordinates and
        #  use the gaia dr3 to propagate to common epoch
        if pmra < -999990 or pmdec < -999990:
            gcols = 'source_id,ra,dec,pmra,pmdec,phot_g_mean_mag as gmag'
            sql = 'select '+gcols+' from catalogdb.gaia_dr3_source'
            sql += ' where q3c_radial_query(ra,dec,{:.7f},{:.7f},1.0/60)'.format(ra,dec)
            gaiadata = db.query(sql=sql,fmt="table")
            if len(gaiadata)>0:
                gpmra = gaiadata['pmra']
                gpmra[~np.isfinite(gpmra)] = 0.0  # set NaNs to zero
                gpmdec = gaiadata['pmdec']
                gpmdec[~np.isfinite(gpmdec)] = 0.0
                gaiamjd = 57388.0   # 1/1/2016, J2016.0, EDR3, DR3
                leadepoch = 0
                tdelt = (mjd-gaiamjd)/365.242170 # convert to years 
                # convert from mas/yr->deg/yr and convert to angle in RA 
                gra_epoch = gaiadata['ra'] + tdelt*pmra/3600.0/1000.0/np.cos(np.deg2rad(gaiadata['dec'])) 
                gdec_epoch = gaiadata['dec'] + tdelt*pmdec/3600.0/1000.0
                # No crossmatch this to our coordinates
                dist = dln.sphdist(ra,dec,gra_epoch,gdec_epoch)*3600
                bestind = np.argmin(dist)
                mindist = np.min(dist)
                # Only keep matches within 1"
            else:
                print('No gaia sources with 1 arcmin.')
            import pdb; pdb.set_trace()
        # We have proper motions, use them to propagate to a common epoch
        else:
            import pdb; pdb.set_trace()
            gcols = 'source_id,ra,dec,pmra,pmdec,phot_g_mean_mag as gmag'
            sql = 'select '+gcols+' from catalogdb.gaia_dr3_source'
            sql += ' where q3c_radial_query(ra,dec,{:.7f},{:.7f},1.0/3600)'.format(ra,dec)
            gdata = db.query(sql=sql,fmt="table")
            gdata['xflag'] = 2
    # Deal with duplicates
    if len(gdata)>1:
        print('Need to deal with gaia duplicates')
        import pdb; pdb.set_trace()        
    # Add gaia data to output table
    data['gaiadr3_source_id'] = -1
    for name in ['gaiadr3_ra','gaiadr3_dec','gaiadr3_pmra','gaiadr3_pmdec','gaiadr3_gmag']:
        data[name] = np.nan
    data['gaiadr3_xflag'] = 0
    data['gaiadr3_match'] = False
    if len(gdata)>0:
        data['gaiadr3_source_id'] = gdata['source_id'][0]
        data['gaiadr3_ra'] = gdata['ra'][0]
        data['gaiadr3_dec'] = gdata['dec'][0]
        data['gaiadr3_pmra'] = gdata['pmra'][0]
        data['gaiadr3_pmdec'] = gdata['pmdec'][0]
        data['gaiadr3_gmag'] = gdata['gmag'][0]
        data['gaiadr3_xflag'] = gdata['xflag'][0]
        data['gaiadr3_match'] = True
    
        
    # Get 2MASS ID
    #-------------
    # First, check any existing cross-matches
    tdata = []
    for catname in ['catalogid31','catalogid25','catalogid21']:
        if catname in data.colnames and data[catname][0]>0:
            catid1 = data[catname][0]
            sql = 'select * from catalogdb.catalog_to_twomass_psc where catalogid='+str(catid1)
            txdata = db.query(sql=sql,fmt="table")
            if len(txdata)>0:
                pts_key = txdata['target_id'][0]
                tcols = 'designation,ra,decl,h_m as hmag,pts_key' 
                sql = 'select '+tcols+' from catalogdb.twomass_psc where pts_key='+str(pts_key)
                tdata = db.query(sql=sql,fmt="table")
                if len(tdata)>0:
                    tdata['xflag'] = 1
                    break
    # Second, check TIC_V8
    if len(tdata)==0:
        ticdata = []
        for catname in ['catalogid31','catalogid25','catalogid21']:
            if catname in data.colnames and data[catname][0]>0:
                catid1 = data[catname][0]
                tic_colarr = ['twomass','ra','decl','hmag']
                tic_cols = ','.join('t.'+np.char.array(tic_colarr))
                sql = "select "+tic_cols+" from catalogdb.tic_v8 as t join catalogdb.catalog_to_tic_v8 as x"
                sql += " on x.target_id=t.id join catalogdb.catalog as c on x.catalogid=c.catalogid where x.catalogid="+str(catid1)
                tdata = db.query(sql=sql,fmt="table")
                if len(tdata)>0:
                    tdata['xflag'] = 2                    
                    tdata['twomass'].name = 'designation'
                    break                
    # Third, do our own crossmatch with Gaia DR3 or CatalogID coords+proper motions
    if len(tdata)==0:
        print('Do own 2MASS xmatch')
        import pdb; pdb.set_trace()
        ra = data['ra_sdss_id']
        pmra = data['pmra']
        dec = data['dec_sdss_id']
        pmdec = data['pmdec']
        # No proper motions, query gaia dr3 around the coordinates and
        #  use the gaia dr3 to propagate to common epoch
        if pmra < -999990 or pmdec < -999990:
            tcols = 'designation,ra,decl,hmag'
            sql = 'select '+tcols+' from catalogdb.twomass_psc'
            sql += ' where q3c_radial_query(ra,decl,{:.7f},{:.7f},30.0/3600)'.format(ra,dec)
            tdata = db.query(sql=sql,fmt="table")
            import pdb; pdb.set_trace()
        # We have proper motions, use them to propagate to a common epoch
        else:
            import pdb; pdb.set_trace()
            tcols = 'designation,ra,decl,hmag'
            sql = 'select '+tcols+' from catalogdb.twomass_psc'
            sql += ' where q3c_radial_query(ra,decl,{:.7f},{:.7f},30.0/3600)'.format(ra,dec)
            tdata = db.query(sql=sql,fmt="table")
        tdata['xflag'] = 3
    # Deal with duplicates
    if len(tdata)>1:
        print('Need to deal with 2MASS duplicates')
        import pdb; pdb.set_trace()
        
    # Add 2MASS data to output table
    data['twomass'] = 18*' '
    data['twomass_ra'] = np.nan
    data['twomass_dec'] = np.nan
    data['twomass_hmag'] = np.nan
    data['twomas_xflag'] = 0    
    data['twomass_match'] = False
    if len(tdata)>0:
        data['twomass'] = tdata['designation'][0]
        data['twomass_ra'] = tdata['ra'][0]
        data['twomass_dec'] = tdata['decl'][0]
        data['twomass_hmag'] = tdata['hmag'][0]
        data['twomass_xflag'] = tdata['xflag'][0]
        data['twomass_match'] = True
        
    # SHOULD WE DO THIS ON OPERATIONS TO MAKE SURE WE HAVE ALL OF THE STARS!!!??
        
    # duplicates
    # Case 1: same catalogid31 but different sdss_ids (normally by 1)

    # Case 2: 

    # APOGEE_ID
    #----------
    # 1) 2MASS ID
    data['apogee_id'] = 18*' '
    if data['twomass_match']:
        data['apogee_id'][0] = '2M'+data['twomass'][0]
    # 2) Construct 2MASS ID from ra_sdss_id/dec_sdss_id, with AP prefix        
    else:
        # apogeetarget/pro/make_2mass_style_id.pro makes these
        # APG-Jhhmmss[.]ss±ddmmss[.]s
        # http://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec1_8a.html
        #  2M00034301-7717269        
        # RA: 00034301 = 00h 03m 43.02s
        # DEC: -7717269 = -71d 17m 26.9s
        ra = data['ra_sdss_id'][0]
        dec = data['dec_sdss_id'][0]
        coo = SkyCoord(ra,dec,unit='deg')
        apogee_id = 'AP'+coo.ra.to_string(unit=u.hourangle, sep="", precision=2, pad=True).replace('.','')
        apogee_id += coo.dec.to_string(sep="", precision=1, alwayssign=True, pad=True).replace('.','')
        data['apogee_id'][0] = apogee_id
        
    db.close()

    return data
    
def get_sdssid(catalogid):
    # Get SDSS_IDs
    db = apogeedb.DBSession()
    
    # Only returns unique catalogid values
    # i.e. if there are duplicates in catalogid, then the number of elements
    # in the returned results will not match the number in catalogid
    #tp = SDSS_ID_flat.select(SDSS_ID_flat.catalogid, SDSS_ID_flat.sdss_id)\
    #                 .where(SDSS_ID_flat.catalogid.in_(catalogid)).dicts()

    sql = 'select s.*,f.catalogid,f.version_id from catalogdb.sdss_id_flat as f'
    sql += ' join catalogdb.sdss_id_stacked as s on f.sdss_id=s.sdss_id where f.catalogid'
    # Multiple IDs
    if dln.size(catalogid)>1:
        ids = ','.join(np.array(catalogid).astype(str))
        sql += " in ("+ids+")"
    # Single ID
    else:
        if type(catalogid)==np.ndarray:
            ids = str(catalogid[0])
        else:
            ids = str(catalogid)
        sql += "="+ids
    
    data = db.query(sql=sql,fmt="table")    
    
    if len(data)==0:
        print('no matches')
    else:
        _,ind1,ind2 = np.intersect1d(catalogid,data['catalogid'],return_indices=True)
        out = np.zeros(dln.size(catalogid),dtype=data.dtype)
        if len(ind1)>0:
            for c in data.dtype.names:
                out[c][ind1] = data[c][ind2]
        if len(ind1) != dln.size(catalogid):
            print(len(catalogid)-len(ind1),' rows missing SDSS_IDs')
            
    db.close()
    
    return out
    
def getdata(catid=None,ra=None,dec=None,designid=None,dcr=1.0,
            table='tmass,gaiadr3',sdssid=True):
    """
    Get catalogdb data for objects.  You can either query by
    catalogid or by coordinates (ra/dec).

    Parameters
    ----------
    id : list
       List or array of catalogids.
    ra : numpy float array
       Array of RA values (need DEC as well). 
    dec : numpy float array
       Array of DEC values.
    dcr : float, optional
       Search radius in arcsec.  Default is 1". 
    table : str, optional
       Table to search.  Can be a comma-separated list.
         Options are "tic", "tmass", "gaiadr2", or "gaiadr3".
         Default is 'tmass,gaiadr3'.
    sdssid : bool, optional
       Return the sdssid.  Default is True.

    Returns
    -------
    Returns a catalog of catalogdb information.

    Examples
    --------
    Query using catalogids 
    catalogstr = getdata(catid=catalogids) 
    Query using coordinates
    catalogstr = getdata(ra=ra,dec=dec)

    """

    # ---- Multiple table queries -----
    if len(table.split(','))>1:
        tables = table.split(',')
        data = []
        # Use designid to cat catalogids
        if designid is not None:
            data = getdata(catid=catid,ra=ra,dec=dec,designid=designid,
                           dcr=dcr,table='tic_v8',sdssid=False)
            data = data[['catalogid','version_id','ra','dec','carton','program']]
            catid = data['catalogid']
            designid = None
        # Query all the tables and merge results
        for t in tables:
            o = getdata(catid=catid,ra=ra,dec=dec,dcr=dcr,table=t,sdssid=False)
            if len(o)==0:
                print('no results for '+t)
                continue
            if len(data) == 0:
                data = o
            else:
                # Add new columns to "data"
                newcols = [c for c in o.colnames if c not in data.colnames]
                for c in newcols:
                    data[c] = np.zeros(len(data),dtype=o[c].dtype)
                # Match using catalogid
                _,ind1,ind2 = np.intersect1d(data['catalogid'],o['catalogid'],return_indices=True)
                # Add information for matching columns
                if len(ind1)>0:
                    for c in newcols:
                        data[c][ind1] = o[c][ind2]
                        
        # Get SDSSID
        if sdssid and len(data)>0:
            sout = get_sdssid(data['catalogid'].tolist())
            data['sdss_id'] = 0            
            data['sdss_id'] = sout['sdss_id']

        return data

    # ---- Single table queries ----
    
    db = apogeedb.DBSession()

    xcols = 'c.catalogid,c.ra,c.dec,'
    tic_colarr = ['twomass','jmag','e_jmag','hmag','e_hmag','kmag','e_kmag','twomflag',
                  'gaia','pmra','e_pmra','pmdec','e_pmdec','plx','e_plx','gaiamag',
                  'e_gaiamag','gaiabp','e_gaiabp','gaiarp','e_gaiarp']
    tic_cols = ','.join('t.'+np.char.array(tic_colarr))
    tmass_colarr = ['designation as twomass','j_m as jmag','j_cmsig as e_jmag','h_m as hmag',
                    'h_cmsig as e_hmag','k_m as kmag','k_cmsig as e_kmag','ph_qual as twomflag']
    tmass_cols = ','.join('t.'+np.char.array(tmass_colarr))
    gaia_colarr = ['source_id as gaia','pmra','pmra_error as e_pmra','pmdec','pmdec_error as e_pmdec',
                   'parallax as plx','parallax_error as e_plx','phot_g_mean_mag as gaiamag',
                   '2.5*log(1+1/phot_g_mean_flux_over_error) as e_gaiamag','phot_bp_mean_mag as gaiabp',
                   '2.5*log(1+1/phot_bp_mean_flux_over_error) as e_gaiabp','phot_rp_mean_mag as gaiarp',
                   '2.5*log(1+1/phot_rp_mean_flux_over_error) as e_gaiarp']
    gaia_cols = ','.join('t.'+np.char.array(gaia_colarr[:8]))
    gaia_cols += ',2.5*log(1+1/t.phot_g_mean_flux_over_error) as e_gaiamag,t.phot_bp_mean_mag as gaiabp'
    gaia_cols += ',2.5*log(1+1/t.phot_bp_mean_flux_over_error) as e_gaiabp,t.phot_rp_mean_mag as gaiarp'
    gaia_cols += ',2.5*log(1+1/t.phot_rp_mean_flux_over_error) as e_gaiarp'    

    # Use catalogid
    if catid is not None:
        if table=='tic' or table=='ticv8' or table=='tic_v8':
            sql = "select "+xcols+tic_cols+" from catalogdb.tic_v8 as t join catalogdb.catalog_to_tic_v8 as x on x.target_id=t.id "+\
                "join catalogdb.catalog as c on x.catalogid=c.catalogid where x.catalogid"
        elif table=='tmass' or table=='twomass' or table=='twomass_psc':
            sql = "select "+xcols+tmass_cols+" from catalogdb.twomass_psc as t join catalogdb.catalog_to_twomass_psc as x on x.target_id=t.pts_key "+\
                  "join catalogdb.catalog as c on x.catalogid=c.catalogid where x.catalogid"
        elif table=='gaiadr2' or table=='gaia_dr2' or table=='gaia_dr2_source':
            sql = "select "+xcols+gaia_cols+",'dr2' as gaia_release from catalogdb.gaia_dr2_source as t join catalogdb.catalog_to_gaia_dr2_source as x on x.target_id=t.source_id "+\
                  "join catalogdb.catalog as c on x.catalogid=c.catalogid where x.catalogid"
        elif table=='gaiadr3' or table=='gaia_dr3' or table=='gaia_dr3_source':
            sql = "select "+xcols+gaia_cols+",'dr3' as gaia_release from catalogdb.gaia_dr3_source as t join catalogdb.catalog_to_gaia_dr3_source as x on x.target_id=t.source_id "+\
                  "join catalogdb.catalog as c on x.catalogid=c.catalogid where x.catalogid"
        else:
            raise ValueError(str(table)+' not supported')
            
        # Multiple IDs
        if dln.size(catid)>1:
            ids = ','.join(np.array(catid).astype(str))
            sql += " in ("+ids+")"

        # Single ID
        else:
            if type(catid)==np.ndarray:
                ids = str(catid[0])
            else:
                ids = str(catid)
            sql += "="+ids

    # Get all targets for a certain design
    elif designid is not None:
        sql = "select t.catalogid,cat.version_id,t.ra,t.dec,t.pmra,t.pmdec,t.epoch,m.bp as gaia_bpmag,m.rp as gaia_rpmag,m.gaia_g as gaia_gmag,"+\
              "m.j as jmag,m.h as hmag,m.k as kmag,c.carton,c.program "+\
              "from targetdb.carton_to_target as ct "+\
              "join targetdb.assignment as a on a.carton_to_target_pk=ct.pk "+\
              "join magnitude as m on m.carton_to_target_pk=a.carton_to_target_pk "+\
              "join target as t on t.pk=ct.target_pk "+\
              "join carton as c on c.pk=ct.carton_pk "+\
              "join catalogdb.catalog as cat on cat.catalogid=t.catalogid "+\
              "where a.design_id="+str(designid)+";"
        
    # Use coordinates
    else:
        radlim = str(dcr/3600.0)
        ra = np.atleast_1d(ra)
        dec = np.atleast_1d(dec)
        nra = len(ra)
        xcols = 'cat.catalogid,cat.q3c_dist,cat.ra,cat.dec,'
        coords = []
        for k in range(nra):
            coords.append( '('+str(ra[k])+','+str(dec[k])+')' )
        vals = ','.join(coords)
        ctable = '(VALUES '+vals+' ) as v'
        # Subquery makes a temporary table from q3c coordinate query with catalogdb.catalog
        sqlsub = "with r as (select c.catalogid,c.ra,c.dec,(q3c_dist(c.ra,c.dec,v.column1,v.column2)*3600.0) as q3c_dist "+\
                 " from "+ctable+",catalogdb.catalog as c "+\
                 "where q3c_join(v.column1,v.column2,c.ra,c.dec,"+radlim+") LIMIT 1000000)"
        if table=='tic' or table=='ticv8' or table=='tic_v8':
            # Use inline query for first join with catalogdb.catalog_to_tic_v8
            sqlinline = "( "+sqlsub+" select r.catalogid,r.ra,r.dec,r.q3c_dist,x.target_id from r "+\
                " inner join catalogdb.catalog_to_tic_v8 as x on x.catalogid=r.catalogid) as cat"
            # Final join with catalogdb.tic_v8
            sql = "select "+xcols+tic_cols+" from "+sqlinline+" inner join catalogdb.tic_v8 as t on cat.target_id=t.id"
        elif table=='tmass' or table=='twomass' or table=='twomass_psc':
            # Use inline query for first join with catalogdb.catalog_to_twomass_psc
            sqlinline = "( "+sqlsub+" select r.catalogid,r.ra,r.dec,r.q3c_dist,x.target_id from r "+\
                " inner join catalogdb.catalog_to_twomass_psc as x on x.catalogid=r.catalogid) as cat"
            # Final join with catalogdb.twomass_psc
            sql = "select "+xcols+tmass_cols+" from "+sqlinline+" inner join catalogdb.twomass_psc as t on cat.target_id=t.pts_key"
        elif table=='gaiadr2' or table=='gaia_dr2' or table=='gaia_dr2_source':
            # Use inline query for first join with catalogdb.catalog_to_gaia_dr2_source
            sqlinline = "( "+sqlsub+" select r.catalogid,r.ra,r.dec,r.q3c_dist,x.target_id from r "+\
                " inner join catalogdb.catalog_to_gaia_dr2_source as x on x.catalogid=r.catalogid) as cat"
            # Final join with catalogdb.gaia_dr2_source
            sql = "select "+xcols+gaia_cols+",'dr2' as gaia_release from "+sqlinline+" inner join catalogdb.gaia_dr2_source as t on cat.target_id=t.source_id"
        elif table=='gaiadr3' or table=='gaia_dr3' or table=='gaia_dr3_source':        
            # Use inline query for first join with catalogdb.catalog_to_gaia_dr3_source
            sqlinline = "( "+sqlsub+" select r.catalogid,r.ra,r.dec,r.q3c_dist,x.target_id from r "+\
                " inner join catalogdb.catalog_to_gaia_dr3_source as x on x.catalogid=r.catalogid) as cat"
            # Final join with catalogdb.gaia_dr3_source
            sql = "select "+xcols+gaia_cols+",'dr3' as gaia_release from "+sqlinline+" inner join catalogdb.gaia_dr3_source as t on cat.target_id=t.source_id"
        else:
            raise ValueError(str(table)+' not supported')
        
        # Turning this off improves q3c queries
        sql = 'set enable_seqscan=off; '+sql

    # Do the query
    data = db.query(sql=sql,fmt="table")
    
    db.close()

    # Get SDSS_IDs
    if sdssid and len(data)>0:
        sout = get_sdssid(data['catalogid'].tolist())
        data['sdss_id'] = 0
        data['sdss_id'] = sout['sdss_id']
        data['version_id'] = 0
        data['version_id'] = sout['version_id']
        
    return data
