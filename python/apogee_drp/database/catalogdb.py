# Get catalogdb data for sources
import numpy as np
from dlnpyutils import utils as dln
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table,vstack,hstack
from astropy.time import Time
from sdss_semaphore.targeting import TargetingFlags
import traceback
from . import apogeedb
from ..utils import apload

        
def getxmatchids(sdssid):
    """
    Get all of the crossmatched IDs from catalogdb.sdss_id_to_catalog.
    This follows all the paths to the main catalogs.
    """
    # This returns 2MASS, Gaia DR2, Gaia DR3 and Tycho_v8 IDs
    
    db = apogeedb.DBSession()
    if dln.size(sdssid)==1:
        tomatch = "="+str(np.atleast_1d(sdssid)[0])
    else:
        tomatch = " in ("+','.join(np.char.array(sdssid).astype(str))+")"
    sql = "select sdss_id,"+\
          "STRING_AGG(distinct catalogid::text,',') as catalogids,"+\
          "STRING_AGG(distinct version_id::text,',') as version_ids,"+\
          "STRING_AGG(distinct lead::text,',') as leads,"+\
          "STRING_AGG(distinct gaia_dr3_source__source_id::text,',') as gaia_dr3_source_id,"+\
          "STRING_AGG(distinct gaia_dr2_source__source_id::text,',') as gaia_dr2_source_id,"+\
          "STRING_AGG(distinct tic_v8__id::text,',') as tic_v8_id,"+\
          "STRING_AGG(distinct twomass_psc__pts_key::text,',') as twomass_psc_pts_key "+\
          "from catalogdb.sdss_id_to_catalog "+\
          "where sdss_id"+tomatch+" "+\
          "group by sdss_id"
    res = db.query(sql=sql,fmt='table')
    db.close()

    # The results from the query will be out of order and
    # might be missing some rows
    data = Table(np.zeros(len(sdssid),dtype=res.dtype))
    _,ind1,ind2 = np.intersect1d(sdssid,res['sdss_id'],return_indices=True)
    if len(ind1)==0:
        return data
    for c in data.colnames:
        data[c][ind1] = res[c][ind2]
    
    return data

def getsdssid(catalogid):
    # Get SDSS_IDs
    db = apogeedb.DBSession()
    
    # Only returns unique catalogid values
    # i.e. if there are duplicates in catalogid, then the number of elements
    # in the returned results will not match the number in catalogid
    #tp = SDSS_ID_flat.select(SDSS_ID_flat.catalogid, SDSS_ID_flat.sdss_id)\
    #                 .where(SDSS_ID_flat.catalogid.in_(catalogid)).dicts()

    sql = 'select s.*,f.catalogid,f.version_id from catalogdb.sdss_id_flat as f'
    sql += ' join catalogdb.sdss_id_stacked as s on f.sdss_id=s.sdss_id where f.catalogid'
    catalogid = np.atleast_1d(catalogid).tolist()
    # Multiple IDs
    if dln.size(catalogid)>1:
        ids = ','.join(np.array(catalogid).astype(str))
        sql += " in ("+ids+")"
    # Single ID
    else:
        if type(catalogid)==np.ndarray or type(catalogid)==list:
            ids = str(catalogid[0])
        else:
            ids = str(catalogid)
        sql += "="+ids

    data = db.query(sql=sql,fmt="table")    
    
    if len(data)==0:
        print('no matches')
        out = []
    else:
        _,ind1,ind2 = np.intersect1d(catalogid,data['catalogid'],return_indices=True)
        out = Table(np.zeros(dln.size(catalogid),dtype=data.dtype))
        if len(ind1)>0:
            for c in data.dtype.names:
                out[c][ind1] = data[c][ind2]
        if len(ind1) != dln.size(catalogid):
            print(len(catalogid)-len(ind1),' rows missing SDSS_IDs')
            
    db.close()
    
    return out

def gethealpix(ra,dec):
    """ Get healpix with coordinates, should use ra_sdss_id/dec_sdss_id. """
    rr = np.atleast_1d(ra)
    dd = np.atleast_1d(dec)
    hpix = np.zeros(len(rr),int)-1
    for i in range(len(rr)):
        if rr[i] < 0 or dd[i]<-100:
            continue
        hpix[i] = apload.coords2healpix(rr[i],dd[i])
    return hpix
        
def gettargeting(sdssid):
    """
    Get targeting information.
    """

    db = apogeedb.DBSession()
    if dln.size(sdssid)==1:
        tomatch = "="+str(np.atleast_1d(sdssid)[0])
    else:
        tomatch = " in ("+','.join(np.char.array(sdssid).astype(str))+")"
    sql = "select s.sdss_id,"+\
          "STRING_AGG(DISTINCT t.pk::text,',') as sdss5_target_pks,"+\
          "STRING_AGG(ct.carton_pk::text,',') as sdss5_target_carton_pks,"+\
          "STRING_AGG(DISTINCT c.carton::text,',') as sdss5_target_cartons,"+\
          "STRING_AGG(DISTINCT t.catalogid::text,',') as sdss5_target_catalogids "+\
          "from targetdb.target as t "+\
          "join targetdb.carton_to_target as ct on t.pk = ct.target_pk "+\
          "join targetdb.carton as c on c.pk = ct.carton_pk "+\
          "join catalogdb.sdss_id_flat as s on s.catalogid = t.catalogid "+\
          "where s.sdss_id"+tomatch+" "+\
          "group by s.sdss_id"
    res = db.query(sql=sql,fmt='table')
    db.close()

    # The results from the query will be out of order and
    # might be missing some rows
    data = Table(np.zeros(len(sdssid),dtype=res.dtype))
    _,ind1,ind2 = np.intersect1d(sdssid,res['sdss_id'],return_indices=True)
    if len(ind1)==0:
        return data
    for c in data.colnames:
        data[c][ind1] = res[c][ind2]
        
    # Use semaphore to create the targeting bitmasks
    data['sdss5_target_flagshex'] = np.zeros(len(data),(str,150))
    flags = TargetingFlags()
    all_carton_pks = [ attrs["carton_pk"] for bit, attrs in flags.mapping.items() ]
    for i in range(len(data)):
        carton_pks = np.array(data['sdss5_target_carton_pks'][i].split(',')).astype(int)
        flags = TargetingFlags()
        for k in carton_pks:
            if k in all_carton_pks:
                flags.set_bit_by_carton_pk(0,k)
        # the array is zero-padded at the end
        #data['sdss5_target_flags'][i,:flags.array.shape[1]] = flags.array[0,:]        
        # convert to hex string for easy storage in database
        flags_array = flags.array[0,:]
        flags_array_hex = ''.join('{:02x}'.format(x) for x in flags_array)
        data['sdss5_target_flagshex'][i] = flags_array_hex
        
    return data
    
def getdesign(designid):
    """
    Get information on a design
    """

    db = apogeedb.DBSession()
    sql = "select t.catalogid,cat.version_id,t.ra,t.dec,c.carton,c.program "+\
          "from targetdb.carton_to_target as ct "+\
          "join targetdb.assignment as a on a.carton_to_target_pk=ct.pk "+\
          "join targetdb.instrument as i on i.pk=a.instrument_pk "+\
          "join magnitude as m on m.carton_to_target_pk=a.carton_to_target_pk "+\
          "join target as t on t.pk=ct.target_pk "+\
          "join carton as c on c.pk=ct.carton_pk "+\
          "join catalogdb.catalog as cat on cat.catalogid=t.catalogid "+\
          "where a.design_id="+str(designid)+" and i.label='APOGEE';"
    data = db.query(sql=sql,fmt="table")
    db.close()
    
    return data

def coords2catid(ra,dec,dcr=1.0):
    """
    Get catalogids from ra/dec coordinates
    """

    db = apogeedb.DBSession()

    radlim = str(dcr/3600.0)
    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)
    nra = len(ra)
    coords = []
    for k in range(nra):
        coords.append( '('+str(ra[k])+','+str(dec[k])+')' )
    vals = ','.join(coords)
    ctable = '(VALUES '+vals+' ) as v'
    # Subquery makes a temporary table from q3c coordinate query with catalogdb.sdss_id_stacked
    sql = 'select s.* from '+ctable+',catalogdb.sdss_id_stacked as s'
    sql += ' where q3c_join(v.column1,v.column2,s.ra_sdss_id,s.dec_sdss_id,'+radlim+') LIMIT 1000000'    
    # Turning this off improves q3c queries
    sql = 'set enable_seqscan=off; '+sql
        
    data = db.query(sql=sql,fmt="table")    
    db.close()

    return data

def checkbrightneighbors(sdssid,ra,dec):
    """ Check for bright neighbors that pollute the 2" fibers."""

    # Can use this for 2MASS sources
    # select original_ext_source_id,count(*) as occurances
    # from catalogdb.gaia_edr3_tmass_psc_xsc_best_neighbour
    # group by original_ext_source_id;

    # but we actually want to check for any bright, close neighbor
    # even if we targeted with gaia

    # Initialize the output table
    nsdssid = len(np.atleast_1d(sdssid))
    dt = [('sdssid',int),('ra',float),('dec',float),('xmatchcount',np.int16),
          ('brightneicount',np.int16),('brightneiflag',np.int16),('fluxfrac',np.float32)]
    data = Table(np.zeros(nsdssid,dtype=np.dtype(dt)))
    data['sdssid'] = sdssid
    data['ra'] = ra
    data['dec'] = dec
    
    db = apogeedb.DBSession()

    # First, count how many Gaia DR3 sources there are within 1"
    # of this position

    dcr = 1.0
    radlim = str(dcr/3600.0)
    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)
    coords = []
    for k in range(nsdssid):
        coords.append( '('+str(sdssid[k])+','+str(ra[k])+','+str(dec[k])+')' )
    vals = ','.join(coords)
    ctable = '(VALUES '+vals+' ) as v'
    # Subquery makes a temporary table from q3c coordinate query with catalogdb.catalog
    sql = 'select v.column1 as sdssid,count(*) as xmatchcount from '+ctable+',catalogdb.gaia_dr3_source as g'
    sql += ' where q3c_join(v.column2,v.column3,g.ra,g.dec,'+radlim+')'
    sql += ' group by sdssid LIMIT 1000000'
    # Turning this off improves q3c queries
    sql = 'set enable_seqscan=off; '+sql
    res = db.query(sql=sql,fmt="table")

    # Put the information in the table
    _,ind1,ind2 = np.intersect1d(data['sdssid'],res['sdssid'],return_indices=True)
    if len(ind1)>0:
        data['xmatchcount'][ind1] = res['xmatchcount'][ind2]
    
    # Next, let's check each star with close neighbors in more detail
    bad, = np.where(data['xmatchcount']>1)

    for i in range(len(bad)):
        sql = 'select source_id,ra,dec,phot_g_mean_mag as gmag,'
        sql += 'q3c_dist(ra,dec,{:},{:})*3600 as dist '.format(data['ra'][bad[i]],data['dec'][bad[i]])
        sql += 'from catalogdb.gaia_dr3_source '
        sql += 'where q3c_radial_query(ra,dec,{:},{:},{:})'.format(data['ra'][bad[i]],data['dec'][bad[i]],dcr/3600.0)
        sql += ' order by dist'
        res = db.query(sql=sql,fmt='table')
        if len(res)<=1:
            continue
        res['gmag'][~np.isfinite(res['gmag'])] = 99.99  # deal with any NaNs
        data['brightneicount'][bad[i]] = len(res)-1        
        # Estimate the relative flux of the target and the neighbor in the fiber
        # Assume the first one (closest) is the source itself
        fratio = 10**((res['gmag'][0]-res['gmag'][1:])/2.5)
        # Take the spatial offset into account
        # If you calculate the intersection of a 2D Gaussian (sigma=1") with a 2" diameter circle
        # and the fraction of flux that enters the circle, this follows a Gaussian with
        # sigma=1.13".  This is *relative" to no offset which of course doesn't fully fill the circle
        fratio *= np.exp(-0.5*res['dist'][1:]**2/1.13**2)
        totfratio = np.sum(fratio)  # total flux fraction, in case there are multiple ones
        data['fluxfrac'][bad[i]] = totfratio
        # Flag based on flux fration
        # >=1     1
        # >=0.5   2
        # >=0.1   3
        # >=0.05  4
        # >=0.01  5
        brightneiflag = 0
        if totfratio>=1.0:
            brightneiflag = 1
        elif totfratio>=0.5:
            brightneiflag = 2
        elif totfratio>=0.1:
            brightneiflag = 3
        elif totfratio>=0.05:
            brightneiflag = 4
        elif totfratio>=0.01:
            brightneiflag = 5            
        data['brightneiflag'][bad[i]] = brightneiflag
    
    db.close()

    return data


def sdssidcatalogquery(sdssid,table):
    """
    Query catalogs using sdss_id_to_catalog
    """

    db = apogeedb.DBSession()

    tic_colarr = ['twomass','jmag','e_jmag','hmag','e_hmag','kmag','e_kmag','twomflag',
                  'gaia','pmra','e_pmra','pmdec','e_pmdec','plx','e_plx','gaiamag',
                  'e_gaiamag','gaiabp','e_gaiabp','gaiarp','e_gaiarp']
    tic_cols = ','.join(tic_colarr)
    tmass_colarr = ['designation as twomass','j_m as jmag','j_cmsig as e_jmag','h_m as hmag',
                    'h_cmsig as e_hmag','k_m as kmag','k_cmsig as e_kmag','ph_qual as twomflag']
    tmass_cols = ','.join(tmass_colarr)
    gaia_colarr = ['source_id as gaia','pmra','pmra_error as e_pmra','pmdec','pmdec_error as e_pmdec',
                   'parallax as plx','parallax_error as e_plx','phot_g_mean_mag as gaiamag',
                   '2.5*log(1+1/phot_g_mean_flux_over_error) as e_gaiamag','phot_bp_mean_mag as gaiabp',
                   '2.5*log(1+1/phot_bp_mean_flux_over_error) as e_gaiabp','phot_rp_mean_mag as gaiarp',
                   '2.5*log(1+1/phot_rp_mean_flux_over_error) as e_gaiarp']
    gaia_cols = ','.join(gaia_colarr[:8])
    gaia_cols += ',2.5*log(1+1/phot_g_mean_flux_over_error) as e_gaiamag,phot_bp_mean_mag as gaiabp'
    gaia_cols += ',2.5*log(1+1/phot_bp_mean_flux_over_error) as e_gaiabp,phot_rp_mean_mag as gaiarp'
    gaia_cols += ',2.5*log(1+1/phot_rp_mean_flux_over_error) as e_gaiarp'    

    # Get the xmatch IDs first
    #  this returns the results in the correct order
    idata = getxmatchids(sdssid)
    
    if table=='tic' or table=='ticv8' or table=='tic_v8':
        tablename = 'tic_v8'
        cols = tic_cols
        sqlidname = 'id'
        idname = 'tic_v8_id'
        ids = idata[idname]
        cols += ','+sqlidname+' as '+idname  # always return id/key        
        sql = "select "+cols+" from catalogdb.tic_v8"
    elif table=='tmass' or table=='twomass' or table=='twomass_psc':
        tablename = 'twomass_psc'
        cols = tmass_cols
        sqlidname = 'pts_key'
        idname = 'twomass_psc_pts_key'
        ids = idata[idname]
        cols += ','+sqlidname+' as '+idname  # always return id/key        
        sql = "select "+cols+" from catalogdb.twomass_psc"
    elif table=='gaiadr2' or table=='gaia_dr2' or table=='gaia_dr2_source':
        tablename = 'gaia_dr2_source'
        cols = gaia_cols
        sqlidname = 'source_id'
        idname = 'gaia_dr2_source_id'
        ids = idata[idname]
        cols += ','+sqlidname+' as '+idname  # always return id/key        
        sql = "select "+cols+",'dr2' as gaia_release from catalogdb.gaia_dr2_source"
    elif table=='gaiadr3' or table=='gaia_dr3' or table=='gaia_dr3_source':
        tablename = 'gaia_dr3_source'        
        cols = gaia_cols
        sqlidname = 'source_id'
        idname = 'gaia_dr3_source_id'
        ids = idata[idname]
        cols += ','+sqlidname+' as '+idname  # always return id/key
        sql = "select "+cols+",'dr3' as gaia_release from catalogdb.gaia_dr3_source"
    else:
        raise ValueError(str(table)+' not supported')

    # Initializing the output table
    nsdssid = len(np.atleast_1d(sdssid))
    coldata = db.query(sql=sql+' limit 1',fmt='table')  # Get column names/dtypes
    data = Table(np.zeros(nsdssid,dtype=coldata.dtype))
    # Initialize to nans
    for c in data.colnames:
        if data[c].dtype.kind == 'f':
            data[c] = np.nan
        elif data[c].dtype.kind == 'i':
            data[c] = -9999
    data['sdss_id'] = sdssid
    data[tablename+'_match'] = False
    # Add the id for this catalog to table
    data[idname] = ids
    
    # Need good ids
    #  all the IDs returned by getxmatchids() are strings
    gdid, = np.where(data[idname] != '')
    # No good ids to match
    if len(gdid)==0:
        db.close()
        return data
    
    # Multiple IDs
    sql += " where "+sqlidname
    if len(gdid)>1:
        sql += " in (" + ','.join(np.array(ids[gdid]).astype(str)) + ")"
    # Single ID
    else:
        sql += "=" + str(ids[gdid[0]])

    # Do the query
    res = db.query(sql=sql,fmt="table")
    db.close()
    
    # Insert the query results into the output table
    _,ind1,ind2 = np.intersect1d(data[idname][gdid],res[idname],return_indices=True)
    if len(ind1)==0:
        return data
    copycols = res.colnames  # columns to copy over
    copycols.remove(idname)  
    for c in copycols:
        data[c][gdid[ind1]] = res[c][ind2]
    data[tablename+'_match'][gdid[ind1]] = True
    
    return data


def getdata(catid=None,ra=None,dec=None,designid=None,dcr=1.0,
            table='tmass,gaiadr3'):
    """
    Get catalogdb data for objects.  You can either query by
    catalogid or by coordinates (ra/dec).

    Parameters
    ----------
    catid : list
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

    Returns
    -------
    data : table
       A table of catalogdb information.

    Examples
    --------

    Query using catalogids:
    catalogstr = getdata(catid=catalogids) 

    Query using coordinates:
    catalogstr = getdata(ra=ra,dec=dec)

    """

    # Always get version31 catalogids and sdss_id for all stars
    if catid is not None:
        catid = np.atleast_1d(catid).tolist()
        db = apogeedb.DBSession()
        sql = 'select * from catalogdb.catalog where catalogid in ('
        sql += ','.join(np.char.array(catid).astype(str))+')'
        ddata = db.query(sql=sql,fmt='table')
        db.close()
        sdata = getsdssid(catid)
        del sdata[['catalogid','version_id']]
        data = hstack((ddata,sdata))        
    elif designid is not None:
        ddata = getdesign(designid)
        if len(ddata)==0:
            return []
        sdata = getsdssid(ddata['catalogid'].tolist())
        del sdata[['catalogid','version_id']]
        data = hstack((ddata,sdata))
    elif ra is not None and dec is not None:
        data = coords2catid(ra,dec)
    else:
        raise ValueError('Need catid, designid, ra+dec or sdssid')
    
    # Get healpix
    data['healpix'] = gethealpix(data['ra_sdss_id'],data['dec_sdss_id'])
            
    # Get targeting information
    tout = gettargeting(data['sdss_id'].tolist())
    data['sdss5_target_pks'] = 1000*' '
    data['sdss5_target_carton_pks'] = 1000*' '
    data['sdss5_target_catalogids'] = 1000*' '
    data['sdss5_target_cartons'] = 1000*' '            
    data['sdss5_target_flagshex'] = np.zeros(len(data),(str,150))
    _,ind1,ind2 = np.intersect1d(data['sdss_id'],tout['sdss_id'],
                                 return_indices=True)
    if len(ind1)>0:
        data['sdss5_target_pks'][ind1] = tout['sdss5_target_pks'][ind2]
        data['sdss5_target_catalogids'][ind1] = tout['sdss5_target_catalogids'][ind2]
        data['sdss5_target_carton_pks'][ind1] = tout['sdss5_target_carton_pks'][ind2]
        data['sdss5_target_cartons'][ind1] = tout['sdss5_target_cartons'][ind2]
        data['sdss5_target_flagshex'][ind1] = tout['sdss5_target_flagshex'][ind2]

    # Now query the catalogs
    tables = table.split(',')
    # Query all the tables and merge results
    for t in tables:
        res = sdssidcatalogquery(data['sdss_id'],t)
        # the results are returned in the correct order
        # Add new columns to "data"
        newcols = [c for c in res.colnames if c not in data.colnames]
        for c in res.colnames:
            if c not in data.colnames:
                data[c] = res[c]

    # Check for close neighbors
    neidata = checkbrightneighbors(data['sdss_id'],data['ra_sdss_id'],data['dec_sdss_id'])
    data['brightneicount'] = neidata['brightneicount']
    data['brightneiflag'] = neidata['brightneiflag']
    data['brightneifluxfrac'] = neidata['fluxfrac']    
                
    return data
