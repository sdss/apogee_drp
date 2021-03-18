;+
;
; DBQUERY
;
; Run a database query.
;
; INPUTS:
;  =sql   SQL query statement.
;
; OUTPUTS:
;  Returns a structure of outputs.
;
; USAGE:
;  IDL>out = dbquery('select * from catalogdb.catalog limit 10')
;
; By D.Nidever, Oct 2020
;-
function dbquery,sql,count=count

undefine,out

if n_elements(sql) eq 0 then begin
  print,'Syntax - out = dbquery(sql)'
  return,-1
endif

; use temporary files and symlinks
tbase = MKTEMP('dbquery',outdir=getlocaldir())    ; create base, leave so other processes won't take it
tempfile = tbase+'.fits'
file_delete,tempfile,/allow

;; Call the python code
cmd = '#!/usr/bin/env python'
push,cmd,'from apogee_drp.database import apogeedb'
push,cmd,'db = apogeedb.DBSession()'
push,cmd,'cat = db.query(sql="'+sql+'",fmt="table")'
push,cmd,'if len(cat)>0: cat.write("'+tempfile+'")'
push,cmd,'db.close()'
scriptfile = tbase+'.py'
WRITELINE,scriptfile,cmd
FILE_CHMOD,scriptfile,'755'o
SPAWN,scriptfile,out,errout,/noshell

if errout[0] ne '' or n_elements(errout) gt 1 then begin
  print,'Problems with the database query'
  for i=0,n_elements(errout)-1 do print,errout[i]
endif

;; Load the catalog
if file_test(tempfile) eq 1 then out = MRDFITS(tempfile,1,/silent) else out=-1
if size(out,/type) ne 8 then count=0 else count=n_elements(out)

FILE_DELETE,[tbase,tbase+'.fits',tbase+'.py'],/allow

return,out

end
