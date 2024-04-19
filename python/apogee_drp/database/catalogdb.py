# Get catalogdb data for sources
import numpy as np
from . import apogeedb
from dlnpyutils import utils as dln
from astropy.coordinates import SkyCoord
import astropy.units as u
import traceback

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

def getdata(catid=None,ra=None,dec=None,designid=None,dcr=1.0,sdssid=True):
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

    db = apogeedb.DBSession()
    
    colarr = ['twomass','jmag','e_jmag','hmag','e_hmag','kmag','e_kmag','twomflag',
              'gaia','pmra','e_pmra','pmdec','e_pmdec','plx','e_plx','gaiamag',
              'e_gaiamag','gaiabp','e_gaiabp','gaiarp','e_gaiarp']
    cols = 'c.catalogid,c.ra,c.dec,' + ','.join('t.'+np.char.array(colarr))
    cols += ",'dr2' as gaia_release"
    
    # Use catalogid
    if catid is not None:
        # Multiple IDs
        if dln.size(catid)>1:
            ids = ','.join(np.array(catid).astype(str))
            sql = "select "+cols+" from catalogdb.tic_v8 as t join catalogdb.catalog_to_tic_v8 as x on x.target_id=t.id "+\
                  "join catalogdb.catalog as c on x.catalogid=c.catalogid where x.catalogid in ("+ids+")"
        # Single ID
        else:
            if type(catid)==np.ndarray:
                ids = str(catid[0])
            else:
                ids = str(catid)
            sql = "select "+cols+" from catalogdb.tic_v8 as t join catalogdb.catalog_to_tic_v8 as x on x.target_id=t.id "+\
                  "join catalogdb.catalog as c on x.catalogid=c.catalogid where x.catalogid="+ids
    # Get all targets for a certain design
    elif designid is not None:
        sql = "select t.catalogid,t.ra,t.dec,t.pmra,t.pmdec,t.epoch,m.bp as gaia_bpmag,m.rp as gaia_rpmag,m.gaia_g as gaia_gmag,"+\
              "'dr2' as gaia_release, m.j as jmag,m.h as hmag,m.k as kmag,c.carton,c.program "+\
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
        cols = 'cat.catalogid,cat.q3c_dist,cat.ra,cat.dec,' + ','.join('t.'+np.char.array(colarr))
        coords = []
        for k in range(nra):
            coords.append( '('+str(ra[k])+','+str(dec[k])+')' )
        vals = ','.join(coords)
        ctable = '(VALUES '+vals+' ) as v'
        # Subquery makes a temporary table from q3c coordinate query with catalogdb.catalog
        sqlsub = "with r as (select c.catalogid,c.ra,c.dec,(q3c_dist(c.ra,c.dec,v.column1,v.column2)*3600.0) as q3c_dist "+\
                 " from "+ctable+",catalogdb.catalog as c "+\
                 "where q3c_join(v.column1,v.column2,c.ra,c.dec,"+radlim+") LIMIT 1000000)"
        # Use inline query for first join with catalogdb.catalog_to_tic_v8
        sqlinline = "( "+sqlsub+" select r.catalogid,r.ra,r.dec,r.q3c_dist,x.target_id from r "+\
              " inner join catalogdb.catalog_to_tic_v8 as x on x.catalogid=r.catalogid) as cat"
        # Final join with catalogdb.tic_v8
        sql = "select "+cols+" from "+sqlinline+" inner join catalogdb.tic_v8 as t on cat.target_id=t.id"
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
    
    return data
