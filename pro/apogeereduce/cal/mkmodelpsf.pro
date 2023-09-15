;+
;
; MKMODELPSF
;
; Procedure to make an APOGEE model PSF master calibration file.
; This is a wrapper around the python mkmodelpsf program.
;
; INPUT:
;  modelpsf    The name of the model PSF file.
;  =sparseid   Sparse frame to use.
;  =psfid      PSF frame to use.
;  /clobber    Overwrite existing files.
;  /unlock     Delete the lock file and start fresh.
;
; OUTPUT:
;  A set of apPSFModel-[abc]-NAME.fits files in the appropriate location
;   determined by the SDSS/APOGEE tree directory structure.
;
; USAGE:
;  IDL>mkmodelpsf,modelpsf,sparseid=sparseid,psfid=psfid,/clobber
;
; By D. Nidever, 2022
;  copied from mkfpi.pro
;-

pro mkmodelpsf,modelpsf,sparseid=sparseid,psfid=psfid,$
               clobber=clobber,unlock=unlock

  if n_elements(name) eq 0 then name=string(modelpsf[0])
  dirs = getdir(apodir,caldir,spectrodir,vers)
  psfdir = apogee_filename('PSFModel',num=name,chip='a',/dir)
  file = dirs.prefix+'PSFModel-'+strtrim(name,2)
  ;;  string(format='("PSFModel-",i8.8)',name)
  psffile = psfdir+file

  ;; If another process is alreadying make this file, wait!
  aplock,psffile,waittime=10,unlock=unlock
  
  ;; Does product already exist?
  ;; check all three chip files
  smodelpsf = strtrim(modelpsf,2)
  ;;smodelpsf = string(modelpsf,format='(i08)')
  ;;cmjd = getcmjd(modelpsf)
  ;;mjd = long(cmjd)
  chips = ['a','b','c']
  allfiles = psfdir+dirs.prefix+'PSFModel-'+chips+'-'+smodelpsf+'.fits'
  if total(file_test(allfiles)) eq 3 and not keyword_set(clobber) then begin
    print,' modelpsf file: ', psfdir+file+'.fits', ' already made'
    return
  endif
  file_delete,allfiles,/allow  ;; delete any existing files to start fresh

  print,'Making modelpsf: ', modelpsf
  ;; Open .lock file
  aplock,psffile,/lock
  
  ;; New Python version! 
  cmd = ['mkmodelpsf',strtrim(modelpsf,2),strtrim(sparseid,2),strtrim(psfid,2),dirs.apred,dirs.telescope,'--verbose']
  print,'Running: ',cmd
  spawn,cmd,/noshell

  ;; Check that the calibration file was successfully created
  outfile = psfdir+repstr(file,dirs.prefix+'PSFModel-',dirs.prefix+'PSFModel-a-')
  if file_test(outfile) then begin
    openw,lock,/get_lun,psfdir+file+'.dat'
    free_lun,lock
  endif

  ;;file_delete,lockfile,/allow
  aplock,psffile,/clear
 
end
