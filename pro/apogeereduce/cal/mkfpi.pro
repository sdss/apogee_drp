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
;  =name       Output filename base.  By default waveid[0] is used.
;  =darkid     Dark frame to be used if images are reduced.
;  =flatid     Flat frame to be used if images are reduced.
;  =psfid      PSF frame to be used if images are reduced.
;  =fiberid    ETrace frame to be used if images are reduced.
;  /clobber    Overwrite existing files.
;  /unlock     Delete the lock file and start fresh.
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
          fiberid=fiberid,clobber=clobber,nowait=nowait,$
          unlock=unlock

  if n_elements(name) eq 0 then name=string(fpiid[0])
  dirs = getdir(apodir,caldir,spectrodir,vers)
  wavedir = apogee_filename('Wave',num=name,chip='a',/dir)
  file = dirs.prefix+string(format='("WaveFPI-",i8.8)',name)
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
  if file_test(wavedir+file+'.fits') and not keyword_set(clobber) then begin
    print,' Wavecal file: ', wavedir+file+'.fits', ' already made'
    return
  endif

  print,'Making fpi: ', fpiid
  ;; Open .lock file
  openw,lock,/get_lun,lockfile
  free_lun,lock

  ;; Process the frames
  cmjd = getcmjd(psfid)
  MKPSF,psfid,darkid=darkid,flatid=flatid,fiberid=fiberid,unlock=unlock
  w = approcess(fpiid,dark=darkid,flat=flatid,psf=psfid,flux=0,/doproc)

  ;; New Python version! 
  cmd = ['ap1dwavecal','--fpiid',strtrim(fpiid,2),'--vers',dirs.apred]
  cmd = [cmd,'--telescope',dirs.telescope,'--verbose']
  for i=0,n_elements(fpiid)-1 do cmd=[cmd,string(fpiid[i])]
  spawn,cmd,/noshell

  ;; Check that the calibration file was successfully created
  outfile = wavedir+repstr(file,'apWaveFPI-','apWaveFPI-a-')
  if file_test(outfile) then begin
    openw,lock,/get_lun,wavedir+file+'.dat'
    free_lun,lock
  endif

  file_delete,lockfile,/allow

end
