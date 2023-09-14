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
;  /psflibrary   Use PSF library to get PSF cal for images.
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

pro mkmultiwave,waveid,name=name,clobber=clobber,nowait=nowait,file=calfile,$
                unlock=unlock,psflibrary=psflibrary

  if n_elements(name) eq 0 then name=string(waveid[0])
  dirs = getdir(apodir,caldir,spectrodir,vers)
  wavedir = apogee_filename('Wave',num=name,chip='a',/dir)
  if file_test(wavedir,/directory) eq 0 then file_mkdir,wavedir
  file = dirs.prefix+string(format='("Wave-",i8.8)',name)
  wavefile = wavedir+file

  ;; If another process is alreadying make this file, wait!
  aplock,wavefile,waittime=10,unlock=unlock
  
  ;; Does product already exist?
  chips = ['a','b','c']
  swaveid = string(name,format='(i08)')
  allfiles = wavedir+dirs.prefix+'Wave-'+chips+'-'+swaveid+'.fits'
  allfiles = [allfiles,wavedir+dirs.prefix+'Wave-'+swaveid+'py.dat']
  if total(file_test(allfiles)) eq 4 and not keyword_set(clobber) then begin
    print,' Wavecal file: ', wavedir+file+'py.dat', ' already made'
    return
  endif

  print,'Making wave: ', waveid
  ;; Open .lock file
  aplock,wavefile,/lock
  
  ;; Process the frames and find lines
  print,''
  print,'***** Processing the frames and finding the lines *****'
  print,''
  for i=0,n_elements(waveid)-1,2 do begin
    print,''
    print,'--- Frame ',strtrim(i+1,2),':  ',strtrim(waveid[i],2),' ---'
    MAKECAL,wave=waveid[i],file=dirs.libdir+'cal/'+dirs.instrument+'-wave.par',$
            /nofit,unlock=unlock,librarypsf=psflibrary
  endfor

  ;; New Python version!
  print,'Running apmultiwavecal'
  cmd = ['apmultiwavecal','--name',name,'--vers',dirs.apred,'--hard','--inst',dirs.instrument,'--verbose']
  for i=0,n_elements(waveid)-1 do cmd=[cmd,string(waveid[i])]
  spawn,cmd,/noshell
  ;; Check that all three chip files exist
  swaveid = string(name,format='(i08)')
  allfiles = wavedir+dirs.prefix+'Wave-'+chips+'-'+swaveid+'.fits'
  if total(file_test(allfiles)) eq 3 then begin
    openw,1,wavedir+file+'py.dat'
    close,1
  endif else stop,'HALT:  failed to make wavecal',waveid

  aplock,wavefile,/clear
  
end
