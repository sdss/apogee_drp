;+
;
; MKWAVE
;
; Procedure to make an APOGEE wavelength calibration file from
; arc lamp exposures.  This is a wrapper around the python
; apmultiwavecal program.
;
; INPUT:
;  waveid      The ID8 numbers of the arc lamp exposures to use.
;  =name       Output filename base.  By default waveid[0] is used.
;  =darkid     Dark frame to be used if images are reduced.
;  =flatid     Flat frame to be used if images are reduced.
;  =psfid      PSF frame to be used if images are reduced.
;  =fiberid    ETrace frame to be used if images are reduced.
;  /nowait     If file is already being made then don't wait
;                just return.
;  /clobber    Overwrite existing files.
;  /nofit      Skip fit (find lines only).
;  /unlock     Delete the lock file and start fresh.
;
; OUTPUT:
;  A set of apWave-[abc]-ID8.fits files in the appropriate location
;   determined by the SDSS/APOGEE tree directory structure.
;
; USAGE:
;  IDL>mkwave,ims,name=name,darkid=darkid,flatid=flatid,psfid=psfid,fiberid=fiberid,/clobber
;
; By J. Holtzman, 2011
;  Added doc strings, updates to use data model  D. Nidever, Sep 2020 
;-

pro mkwave,waveid,name=name,darkid=darkid,flatid=flatid,psfid=psfid,$
           fiberid=fiberid,clobber=clobber,nowait=nowait,nofit=nofit,$
           unlock=unlock

  if n_elements(name) eq 0 then name=string(waveid[0])
  dirs = getdir(apodir,caldir,spectrodir,vers)
  wavedir = apogee_filename('Wave',num=name,chip='a',/dir)
  file = dirs.prefix+string(format='("Wave-",i8.8)',name)
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
  if file_test(wavedir+file+'.dat') and not keyword_set(clobber) then begin
    print,' Wavecal file: ', wavedir+file+'.dat', ' already made'
    return
  endif

  print,'Making wave: ', waveid
  ;; Open .lock file
  openw,lock,/get_lun,lockfile
  free_lun,lock

  ;; Process the frames
  cmjd = getcmjd(psfid)
  MKPSF,psfid,darkid=darkid,flatid=flatid,fiberid=fiberid,unlock=unlock
  w = approcess(waveid,dark=darkid,flat=flatid,psf=psfid,flux=0,/doproc)

  ;; New Python version! 
  cmd = ['apmultiwavecal','--name',strtrim(name,2),'--vers',dirs.apred]
  if keyword_set(nofit) then cmd=[cmd,'--nofit']
  if keyword_set(plot) then cmd=[cmd,'--plot','--hard']
  cmd = [cmd,'--inst',dirs.instrument,'--verbose']
  for i=0,n_elements(waveid)-1 do cmd=[cmd,string(waveid[i])]
  spawn,cmd,/noshell

  openw,lock,/get_lun,wavedir+file+'.dat'
  free_lun,lock

  file_delete,lockfile,/allow

end
