;+
;
; DBINGEST_VISIT
;
; Insert the apVisitSum file into the apogee_drp database.
;
; INPUTS:
;  allvisit  The apVisitSum structure from ap1dvisit.pro.
;
; OUTPUTS:
;  Not outputs.  The allvisit structure is loaded into the
;  apogee_drp.visit database table.
;
; USAGE:
;  IDL>dbingest_visit,allvisit
;
; By D.Nidever, Oct 2020
;-

pro dbingest_visit,allvisit

if n_elements(allvisit) eq 0 then begin
  print,'Syntax - dbingest_visit,allvisit'
  return
endif

; use temporary files and symlinks
tbase = MKTEMP('allvisit',outdir=getlocaldir())    ; create base, leave so other processes won't take it
tempfile = tbase+'.fits'
file_delete,tempfile,/allow
MWRFITS,allvisit,tempfile,/create

;; Call the python code
cmd = '#!/usr/bin/env python'
push,cmd,'from apogee_drp.database import apogeedb'
push,cmd,'from astropy.io import fits'
push,cmd,'db = apogeedb.DBSession()'
push,cmd,'cat = fits.getdata("'+tempfile+'",1)'
push,cmd,'db.ingest("visit",cat)'
push,cmd,'db.close()'
scriptfile = tbase+'.py'
WRITELINE,scriptfile,cmd
FILE_CHMOD,scriptfile,'755'o
SPAWN,scriptfile,out,errout,/noshell

if errout[0] ne '' or n_elements(errout) gt 1 then begin
  print,'Problems inserting apVisitSum catalog into the database'
  for i=0,n_elements(errout)-1 do print,errout[i]
endif

FILE_DELETE,[tbase,tbase+'.fits',tbase+'.py'],/allow

end
