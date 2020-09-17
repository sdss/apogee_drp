;+
;
; MKMULTIWAVE
;
; Procedure to make an APOGEE wavelength calibration file from
; multiple arc lamp exposures from different nights.  This is
; a wrapper around the python apmultiwavecal program.
;
; INPUT:
;  waveid      Array of ID8 numbers of the arc lamp exposures to use.
;  =name       Output filename base.  By default waveid[0] is used.
;  /nowait     If file is already being made then don't wait
;                just return.
;  /clobber    Overwrite existing files.
;  =file       Depricated parameter.
;
; OUTPUT:
;  A set of apWave-[abc]-ID8.fits files in the appropriate location
;   determined by the SDSS/APOGEE tree directory structure.
;
; USAGE:
;  IDL>mkmultiwave,ims,name=name,/clobber
;
; By J. Holtzman, 2011
;  Added doc strings, updates to use data model  D. Nidever, Sep 2020 
;-

pro mkmultiwave,waveid,name=name,clobber=clobber,nowait=nowait,file=calfile

  if n_elements(name) eq 0 then name=string(waveid[0])
  dirs = getdir(apodir,caldir,spectrodir,vers)
  wavedir = apogee_filename('Wave',num=name,chip='a',/dir)
  file = dirs.prefix+string(format='("Wave-",i8.8)',name)
  ;; If another process is alreadying make this file, wait!
  while file_test(wavedir+file+'.lock') do begin
    if keyword_set(nowait) then begin
      print,' Wavecal file: ', wavedir+file, ' already being made (.lock file exists)'
      return
    endif
    apwait,file,10
  endwhile
  ;; Does product already exist?
  if file_test(wavedir+file+'py.dat') and not keyword_set(clobber) then begin
    print,' Wavecal file: ', wavedir+file+'py.dat', ' already made'
    return
  endif

  print,'Making wave: ', waveid
  ;; Open .lock file
  openw,lock,/get_lun,wavedir+file+'.lock'
  free_lun,lock

  ;; Process the frames and find lines
  for i=0,n_elements(waveid)-1,2 do $
    MAKECAL,wave=waveid[i],file=dirs.libdir+'cal/'+dirs.instrument+'-wave.par',/nofit

  ;; New Python version!
  cmd = ['apmultiwavecal','--name',name,'--vers',dirs.apred,'--plot','--hard','--inst',dirs.instrument]
  for i=0,n_elements(waveid)-1 do cmd=[cmd,string(waveid[i])]
  spawn,cmd,/noshell
  if file_test(apogee_filename('Wave',num=waveid[0],chip='a')) then begin
    openw,1,wavedir+file+'py.dat'
    close,1
  endif else stop,'HALT:  failed to make wavecal',waveid

  file_delete,wavedir+file+'.lock'

end
