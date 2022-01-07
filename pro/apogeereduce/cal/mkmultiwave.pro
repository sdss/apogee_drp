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
;  /unlock     Delete the lockfile and start fresh.
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

pro mkmultiwave,waveid,name=name,clobber=clobber,nowait=nowait,file=calfile,unlock=unlock

  if n_elements(name) eq 0 then name=string(waveid[0])
  dirs = getdir(apodir,caldir,spectrodir,vers)
  wavedir = apogee_filename('Wave',num=name,chip='a',/dir)
  file = dirs.prefix+string(format='("Wave-",i8.8)',name)
  lockfile = wavedir+file+'.lock'

  ;; If another process is alreadying make this file, wait!
  if not keyword_set(unlock) then begin
    while file_test(lockfile) do begin
      if keyword_set(nowait) then begin
        print,' Wavecal file: ', wavedir+file, ' already being made (.lock file exists)'
        return
      endif
      apwait,file,10
    endwhile
  endif else begin
    if file_test(lockfile) then file_delete,lockfile,/allow
  endelse

  ;; Does product already exist?
  if file_test(wavedir+file+'py.dat') and not keyword_set(clobber) then begin
    print,' Wavecal file: ', wavedir+file+'py.dat', ' already made'
    return
  endif

  print,'Making wave: ', waveid
  ;; Open .lock file
  openw,lock,/get_lun,lockfile
  free_lun,lock

  ;; Process the frames and find lines
  print,''
  print,'***** Processing the frames and finding the lines *****'
  print,''
  for i=0,n_elements(waveid)-1,2 do begin
    print,''
    print,'--- Frame ',strtrim(i+1,2),':  ',waveid[i],' ---'
    MAKECAL,wave=waveid[i],file=dirs.libdir+'cal/'+dirs.instrument+'-wave.par',/nofit,unlock=unlock
  endfor

  ;; New Python version!
  print,'Running apmultiwavecal'
  ;;cmd = ['apmultiwavecal','--name',name,'--vers',dirs.apred,'--plot','--hard','--inst',dirs.instrument]
  cmd = ['apmultiwavecal','--name',name,'--vers',dirs.apred,'--hard','--inst',dirs.instrument,'--verbose']
  for i=0,n_elements(waveid)-1 do cmd=[cmd,string(waveid[i])]
  spawn,cmd,/noshell
  if file_test(apogee_filename('Wave',num=waveid[0],chip='a')) then begin
    openw,1,wavedir+file+'py.dat'
    close,1
  endif else stop,'HALT:  failed to make wavecal',waveid

  file_delete,lockfile,/allow

end
