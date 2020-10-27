# Get catalogdb data for sources
import numpy as np
from . import apogeedb
from dlnpyutils import utils as dln


def getdata(catid=None,ra=None,dec=None,dcr=1.0):
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

    # Use catalogid
    if catid is not None:
        # Multiple IDs
        if dln.size(catid)>1:
            ids = ','.join(np.array(catid).astype(str))
            sql = "select "+cols+" from catalogdb.tic_v8 as t join catalogdb.catalog_to_tic_v8 as x on x.target_id=t.id "+\
                  "join catalogdb.catalog as c on x.catalogid=c.catalogid where x.catalogid in ("+ids+")"
        # Single ID
        else:
            ids = str(catid)
            sql = "select "+cols+" from catalogdb.tic_v8 as t join catalogdb.catalog_to_tic_v8 as x on x.target_id=t.id "+\
                  "join catalogdb.catalog as c on x.catalogid=c.catalogid where x.catalogid="+ids
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

    return data
