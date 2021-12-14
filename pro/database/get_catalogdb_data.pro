;+
;
; GET_CATALOGDB_DATA
;
; Get catalogdb data for objects.  You can either query by
; catalogid or by coordinates (ra/dec).
;
; INPUTS:
;  =id    Array of catalogids.
;  =ra    Array of RA values (need DEC as well).
;  =dec   Array of DEC values.
;  =dcr   Search radius in arcsec.  Default is 1".
;  =designid  Design ID.
;
; OUTPUTS:
;  Returns a structure of catalogdb information.
;
; USAGE:
;  Query using catalogids
;  IDL>catalogstr = get_catalogdb_data(id=catalogids)
;  Query using coordinates
;  IDL>catalogstr = get_catalogdb_data(ra=ra,dec=dec)
;
; By D.Nidever, Oct 2020
;-
function get_catalogdb_data,id=catalogid,ra=ra,dec=dec,dcr=dcr,designid=designid

catalogstr = -1

if (n_elements(catalogid) eq 0) and (n_elements(ra) eq 0 or n_elements(dec) eq 0) and (n_elements(designid) eq 0) then begin
  print,'Syntax - catalogstr = get_catalogdb_data(id=id,ra=ra,dec=dec)'
  return,-1
endif

; use temporary files and symlinks
tbase = MKTEMP('catdb',outdir=getlocaldir())    ; create base, leave so other processes won't take it
tempfile = tbase+'.fits'
file_delete,tempfile,/allow

;; Call the python code
cmd = '#!/usr/bin/env python'
push,cmd,'from apogee_drp.database import apogeedb'
push,cmd,'db = apogeedb.DBSession()'
colarr = ['twomass','jmag','e_jmag','hmag','e_hmag','kmag','e_kmag','twomflag','gaia','pmra','e_pmra','pmdec','e_pmdec','plx','e_plx','gaiamag','e_gaiamag','gaiabp','e_gaiabp','gaiarp','e_gaiarp']
cols = 'c.catalogid,c.ra,c.dec,' + strjoin('t.'+colarr,',')

;; Use catalogid
if n_elements(catalogid) gt 0 then begin
  if n_elements(catalogid) gt 1 then begin
    ids = strjoin(strtrim(catalogid,2),',')
    sql = "select "+cols+" from catalogdb.tic_v8 as t join catalogdb.catalog_to_tic_v8 as x on x.target_id=t.id "+$
          "join catalogdb.catalog as c on x.catalogid=c.catalogid where x.catalogid in ("+ids+")"
  endif else begin
    ids = strtrim(catalogid[0],2)
    sql = "select "+cols+" from catalogdb.tic_v8 as t join catalogdb.catalog_to_tic_v8 as x on x.target_id=t.id "+$
          "join catalogdb.catalog as c on x.catalogid=c.catalogid where x.catalogid="+ids
  endelse

;; Get all targets for a certain design
endif else if keyword_set(designid) then begin
  sql = "select t.catalogid,t.ra,t.dec,t.pmra,t.pmdec,t.epoch,m.bp as gaia_bpmag,m.rp as gaia_rpmag,m.gaia_g as gaia_gmag,"+$
        "m.j as jmag,m.h as hmag,m.k as kmag,c.carton,c.program "+$
        "from targetdb.carton_to_target as ct "+$
        "join targetdb.assignment as a on a.carton_to_target_pk=ct.pk "+$
        "join magnitude as m on m.carton_to_target_pk=a.carton_to_target_pk "+$
        "join target as t on t.pk=ct.target_pk "+$
        "join carton as c on c.pk=ct.carton_pk "+$
        "where a.design_id="+strtrim(designid,2)+";"

;; Use coordinates
endif else begin
  if n_elements(dcr) eq 0 then dcr=1.0  ; search radius in arcsec
  radlim = strtrim(dcr/3600.0,2)
  nra = n_elements(ra)
  cols = 'cat.catalogid,cat.q3c_dist,cat.ra,cat.dec,' + strjoin('t.'+colarr,',')
  coords = strarr(nra)
  for k=0,n_elements(ra)-1 do coords[k] = '('+strtrim(ra[k],2)+','+strtrim(dec[k],2)+')'
  vals = strjoin(coords,',')
  ctable = '(VALUES '+vals+' ) as v'
  ;; Subquery makes a temporary table from q3c coordinate query with catalogdb.catalog
  sqlsub = "with r as (select c.catalogid,c.ra,c.dec,(q3c_dist(c.ra,c.dec,v.column1,v.column2)*3600.0) as q3c_dist "+$
           " from "+ctable+",catalogdb.catalog as c "+$
           "where q3c_join(v.column1,v.column2,c.ra,c.dec,"+radlim+") LIMIT 1000000)"
  ;; Use inline query for first join with catalogdb.catalog_to_tic_v8
  sqlinline = "( "+sqlsub+" select r.catalogid,r.ra,r.dec,r.q3c_dist,x.target_id from r "+$
              " inner join catalogdb.catalog_to_tic_v8 as x on x.catalogid=r.catalogid) as cat"
  ;; Final join with catalogdb.tic_v8
  sql = "select "+cols+" from "+sqlinline+" inner join catalogdb.tic_v8 as t on cat.target_id=t.id"

  ;sql += sqlsub+" select "+cols+" from r "+$
  ;       "inner join catalogdb.catalog_to_tic_v8 as x on r.catalogid=x.catalogid "+$
  ;       "inner join catalogdb.tic_v8 as t on x.target_id=t.id"
  ;; Turning this off improves q3c queries
  sql = 'set enable_seqscan=off; '+sql
  ; For single star
  ; sql = "select "+cols+" from catalogdb.tic_v8 as t join catalogdb.catalog_to_tic_v8 as x on x.target_id=t.id "+$
  ;        "join catalogdb.catalog as c on x.catalogid=c.catalogid where q3c_radial_query(c.ra,c.dec,"+strtrim(ra,2)+","+strtrim(dec,2)+",0.0002)"
endelse

push,cmd,'cat = db.query(sql="'+sql+'",fmt="table")'
push,cmd,'cat.write("'+tempfile+'")'
push,cmd,'db.close()'
scriptfile = tbase+'.py'
WRITELINE,scriptfile,cmd
FILE_CHMOD,scriptfile,'755'o
SPAWN,scriptfile,out,errout,/noshell

if errout[0] ne '' or n_elements(errout) gt 1 then begin
  print,'Problems inserting apVisitSum catalog into the database'
  for i=0,n_elements(errout)-1 do print,errout[i]
endif

;; Load the catalog
if file_test(tempfile) eq 1 then catalogstr = MRDFITS(tempfile,1,/silent)

FILE_DELETE,[tbase,tbase+'.fits',tbase+'.py'],/allow

return,catalogstr

end
