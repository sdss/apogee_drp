;+
;
; MKMODELPSF
;
; Procedure to make an APOGEE model PSF master calibration file.
; This is a wrapper around the python mkmodelpsf program.
;
; INPUT:
;  modelpsf    The ID8 numbers of the model PSF.
;  =sparseid   Sparse frame to use.
;  =psfid      PSF frame to use.
;  /clobber    Overwrite existing files.
;  /unlock     Delete the lock file and start fresh.
;
; OUTPUT:
;  A set of apPSFModel-[abc]-ID8.fits files in the appropriate location
;   determined by the SDSS/APOGEE tree directory structure.
;
; USAGE:
;  IDL>mkmodelpsf,modelpsf,sparseid=sparseid,psfid=psfid,/clobber
;
; By D. Nidever, 2022
;  copied from mfpi.pro
;-

pro mkmodelpsf,modelpsf,sparseid=sparseid,psfid=psfid,$
               clobber=clobber,unlock=unlock

  if n_elements(name) eq 0 then name=string(modelpsf[0])
  dirs = getdir(apodir,caldir,spectrodir,vers)
  psfdir = apogee_filename('PSFModel',num=name,chip='a',/dir)
  file = dirs.prefix+string(format='("PSFModel-",i8.8)',name)
  lockfile = wavedir+file+'.lock'

  ;; If another process is alreadying make this file, wait!
  if not keyword_set(unlock) then begin
    while file_test(lockfile) do begin
      if keyword_set(nowait) then return
      apwait,file,10
    endwhile
  endif else begin
    if file_test(lockfile) then file_delete,lockfile,/allow
  endelse

  ;; Does product already exist?
  ;; check all three chip files
  smodelpsf = string(modelpsf,format='(i08)')
  cmjd = getcmjd(modelpsf)
  mjd = long(cmjd)
  chips = ['a','b','c']
  allfiles = wavedir+dirs.prefix+'PSFModel-'+chips+'-'+cmjd+'-'+smodelpsf+'.fits'
  if total(file_test(allfiles)) eq 3 and not keyword_set(clobber) then begin
    print,' modelpsf file: ', psfdir+file+'.fits', ' already made'
    return
  endif
  file_delete,allfiles,/allow  ;; delete any existing files to start fresh

  print,'Making modelpsf: ', modelpsf
  ;; Open .lock file
  openw,lock,/get_lun,lockfile
  free_lun,lock

  ;; New Python version! 
  cmd = ['mkmodelwave',strtrim(modelpsf,2),strtrim(sparseid,2),strtrim(psfid,2),dirs.apred,strmid(dirs.telescope,0,3),'--verbose']
  print,'Running: ',cmd
  spawn,cmd,/noshell

  ;; Check that the calibration file was successfully created
  outfile = wavedir+repstr(file,dirs.prefix+'PSFModel-',dirs.prefix+'PSFModel-a-')
  if file_test(outfile) then begin
    openw,lock,/get_lun,wavedir+file+'.dat'
    free_lun,lock
  endif

  file_delete,lockfile,/allow

end
