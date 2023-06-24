;+
;
; MKFPI
;
; Procedure to make an APOGEE FPI wavelength calibration file from
; FPI arc lamp exposures.  This is a wrapper around the python
; apmultiwavecal program.
;
; INPUT:
;  fpiid       The ID8 numbers of the FPI arc lamp exposures to use.
;  =name       Output filename base.  By default fpiid is used.
;  =darkid     Dark frame to be used if images are reduced.
;  =flatid     Flat frame to be used if images are reduced.
;  =psfid      PSF frame to be used if images are reduced.
;  =modelpsf   Model PSF calibration frame to use.
;  =fiberid    ETrace frame to be used if images are reduced.
;  /clobber    Overwrite existing files.
;  /unlock     Delete the lock file and start fresh.
;  /psflibrary   Use PSF library to get PSF cal for images.
;
; OUTPUT:
;  A set of apWaveFPI-[abc]-ID8.fits files in the appropriate location
;   determined by the SDSS/APOGEE tree directory structure.
;
; USAGE:
;  IDL>mkfpi,ims,name=name,darkid=darkid,flatid=flatid,psfid=psfid,fiberid=fiberid,/clobber
;
; By D. Nidever, 2021
;  copied from mkwave.pro
;-

pro mkfpi,fpiid,name=name,darkid=darkid,flatid=flatid,psfid=psfid,$
          modelpsf=modelpsf,fiberid=fiberid,clobber=clobber,$
          unlock=unlock,psflibrary=psflibrary

  if n_elements(name) eq 0 then name=string(fpiid[0])
  dirs = getdir(apodir,caldir,spectrodir,vers)
  wavedir = apogee_filename('Wave',num=name,chip='a',/dir)
  file = dirs.prefix+string(format='("WaveFPI-",i8.8)',name)
  fpifile = wavedir+file
  ;;lockfile = wavedir+file+'.lock'

  ;; If another process is alreadying make this file, wait!
  ;;if not keyword_set(unlock) then begin
  ;;  while file_test(lockfile) do begin
  ;;    if keyword_set(nowait) then return
  ;;    apwait,file,10
  ;;  endwhile
  ;;endif else begin
  ;;  if file_test(lockfile) then file_delete,lockfile,/allow
  ;;endelse
  aplock,fpifile,waittime=10,unlock=unlock
  
  ;; Does product already exist?
  ;; check all three chip files
  sfpiid = string(fpiid,format='(i08)')
  cmjd = getcmjd(fpiid)
  mjd = long(cmjd)
  chips = ['a','b','c']
  allfiles = wavedir+dirs.prefix+'WaveFPI-'+chips+'-'+cmjd+'-'+sfpiid+'.fits'
  if total(file_test(allfiles)) eq 3 and not keyword_set(clobber) then begin
    print,' Wavecal file: ', wavedir+file+'.fits', ' already made'
    return
  endif
  file_delete,allfiles,/allow  ;; delete any existing files to start fresh

  print,'Making fpi: ', fpiid
  ;; Open .lock file
  ;;openw,lock,/get_lun,lockfile
  ;;free_lun,lock
  aplock,fpifile,/lock
  
  ;; Process the frames
  if keyword_set(psfid) then $
    MKPSF,psfid,darkid=darkid,flatid=flatid,fiberid=fiberid,unlock=unlock
  w = approcess(fpiid,dark=darkid,flat=flatid,psf=psfid,modelpsf=modelpsf,flux=0,/doproc,unlock=unlock)

  ;; Make sure the dailywave file is there
  ;;  it uses modelpsf by default now
  MAKECAL,dailywave=mjd,clobber=clobber,unlock=unlock,librarypsf=psflibrary

  ;; New Python version! 
  cmd = ['mkfpi',strtrim(cmjd,2),dirs.apred,strmid(dirs.telescope,0,3),'--num',sfpiid,'--verbose']
  print,'Running: ',cmd
  spawn,cmd,/noshell

  ;; Check that the calibration file was successfully created
  outfile = wavedir+repstr(file,dirs.prefix+'WaveFPI-',dirs.prefix+'WaveFPI-a-')
  if file_test(outfile) then begin
    openw,lock,/get_lun,wavedir+file+'.dat'
    free_lun,lock
  endif

  ;;file_delete,lockfile,/allow
  aplock,fpifile,/clear
  
end
