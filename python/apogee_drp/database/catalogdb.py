# Get catalogdb data for sources
import numpy as np
from . import apogeedb
from dlnpyutils import utils as dln


def getdata(catalogid):
    """ Get catalogdb information for sources using catalogid."""
    db = apogeedb.DBSession()
    
    colarr = ['twomass','jmag','e_jmag','hmag','e_hmag','kmag','e_kmag','twomflag',
              'gaia','pmra','e_pmra','pmdec','e_pmdec','plx','e_plx','gaiamag',
              'e_gaiamag','gaiabp','e_gaiabp','gaiarp','e_gaiarp']
    cols = 'x.catalogid,c.ra,c.dec,' + ','.join('t.'+np.char.array(colarr))
    
    # Multiple IDs
    if dln.size(catalogid)>1:
        ids = ','.join(np.array(catalogid).astype(str))
        sql = "select "+cols+" from catalogdb.tic_v8 as t join catalogdb.catalog_to_tic_v8 as x on x.target_id=t.id "+\
        "join catalogdb.catalog as c on x.catalogid=c.catalogid where x.catalogid in ("+ids+")"
    # Single ID
    else:
        ids = str(catalogid)
        sql = "select "+cols+" from catalogdb.tic_v8 as t join catalogdb.catalog_to_tic_v8 as x on x.target_id=t.id "+\
        "join catalogdb.catalog as c on x.catalogid=c.catalogid where x.catalogid="+ids

    # Do the query
    data = db.query(sql=sql,fmt="table")


    db.close()

    return data
