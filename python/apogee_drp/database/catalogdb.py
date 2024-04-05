# Get catalogdb data for sources
import numpy as np
from . import apogeedb
from dlnpyutils import utils as dln
import traceback

#from sdssdb.peewee.sdss5db.catalogdb import SDSS_ID_flat    
#from sdssdb.peewee.sdss5db.targetdb import database
#import sdssdb
#test = database.set_profile('pipelines')
#database.connect()
#except:
#    traceback.print_exc()

def get_sdssid(catalogid):
    # Get SDSS_IDs
    db = apogeedb.DBSession()
    
    sid = np.zeros(len(catalogid),int)-999
    # Only returns unique catalogid values
    # i.e. if there are duplicates in catalogid, then the number of elements
    # in the returned results will not match the number in catalogid
    #tp = SDSS_ID_flat.select(SDSS_ID_flat.catalogid, SDSS_ID_flat.sdss_id)\
    #                 .where(SDSS_ID_flat.catalogid.in_(catalogid)).dicts()

    sql = 'select * from catalogdb.sdss_id_flat where catalogid'
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
        if len(ind1)>0:
            sid[ind1] = data['sdss_id'][ind2]
        if len(ind1) != len(catalogid):
            print(len(catalogid)-len(ind1),' rows missing SDSS_IDs')

    db.close()
    
    return sid
    
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
        data = None
        # Use designid to cat catalogids
        if designid is not None:
            data = getdata(catid=catid,ra=ra,dec=dec,designid=designid,
                           dcr=dcr,table='tic_v8',sdssid=False)
            data = data[['catalogid','ra','dec','carton','program']]
            catid = data['catalogid']
            designid = None
        # Query all the tables and merge results
        for t in tables:
            o = getdata(catid=catid,ra=ra,dec=dec,dcr=dcr,table=t,sdssid=False)
            if len(o)==0:
                print('no results for '+t)
                continue
            if data is None:
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
        if sdssid:
            sid = get_sdssid(data['catalogid'].tolist())
            data['sdss_id'] = sid
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
        sql = "select t.catalogid,t.ra,t.dec,t.pmra,t.pmdec,t.epoch,m.bp as gaia_bpmag,m.rp as gaia_rpmag,m.gaia_g as gaia_gmag,"+\
              "m.j as jmag,m.h as hmag,m.k as kmag,c.carton,c.program "+\
              "from targetdb.carton_to_target as ct "+\
              "join targetdb.assignment as a on a.carton_to_target_pk=ct.pk "+\
              "join magnitude as m on m.carton_to_target_pk=a.carton_to_target_pk "+\
              "join target as t on t.pk=ct.target_pk "+\
              "join carton as c on c.pk=ct.carton_pk "+\
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
    if sdssid:
        sid = get_sdssid(data['catalogid'].tolist())
        data['sdss_id'] = sid
    
    return data
