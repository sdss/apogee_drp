;+
;
; GET_CATALOGDB_DATA
;
; Get catalogdb data for objects
;
; INPUTS:
;  catalogid  List of catalogids.
;
; OUTPUTS:
;  Returns a structure of catalogdb information.
;
; USAGE:
;  IDL>catalogstr = get_catalogdb_data(catalogids)
;
; By D.Nidever, Oct 2020
;-
function get_catalogdb_data,catalogid

undefine,catalogstr

if n_elements(catalogid) eq 0 then begin
  print,'Syntax - catalogstr = get_catalogdb_data(catalogid)'
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
cols = 'x.catalogid,c.ra,c.dec,' + strjoin('t.'+colarr,',')
if n_elements(catalogid) gt 1 then begin
  ids = strjoin(strtrim(catalogid,2),',')
  sql = "select "+cols+" from catalogdb.tic_v8 as t join catalogdb.catalog_to_tic_v8 as x on x.target_id=t.id "+$
        "join catalogdb.catalog as c on x.catalogid=c.catalogid where x.catalogid in ("+ids+")"
endif else begin
  ids = strtrim(catalogid[0],2)
  sql = "select "+cols+" from catalogdb.tic_v8 as t join catalogdb.catalog_to_tic_v8 as x on x.target_id=t.id "+$
        "join catalogdb.catalog as c on x.catalogid=c.catalogid where x.catalogid="+ids
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
catalogstr = MRDFITS(tempfile,1,/silent)

FILE_DELETE,[tbase,tbase+'.fits',tbase+'.py'],/allow

return,catalogstr

end
