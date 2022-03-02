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
          fiberid=fiberid,clobber=clobber,unlock=unlock,psflibrary=psflibrary

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
  ;; check all three chip files
  sfpiid = string(fpiid,format='(i08)')
  chips = ['a','b','c']
  allfiles = wavedir+dirs.prefix+'WaveFPI-'+chips+'-'+sfpiid+'.fits'
  if total(file_test(allfiles)) eq 3 and not keyword_set(clobber) then begin
    print,' Wavecal file: ', wavedir+file+'.fits', ' already made'
    return
  endif
  file_delete,allfiles,/allow  ;; delete any existing files to start fresh

  print,'Making fpi: ', fpiid
  ;; Open .lock file
  openw,lock,/get_lun,lockfile
  free_lun,lock

  ;; Process the frames
  cmjd = getcmjd(psfid)
  mjd = long(cmjd)
  MKPSF,psfid,darkid=darkid,flatid=flatid,fiberid=fiberid,unlock=unlock
  w = approcess(fpiid,dark=darkid,flat=flatid,psf=psfid,flux=0,/doproc,unlock=unlock)
  
  ;; Get the individual arclamp exposures and make sure they have been processed
  expinfo = dbquery("select * from apogee_drp.exposure where mjd>="+strtrim(mjd-7,2)+" and mjd<="+strtrim(mjd,2)+" and exptype='ARCLAMP'")
  expinfo.arctype = strtrim(expinfo.arctype,2)
  gdarc = where(expinfo.arctype eq 'UNE' or expinfo.arctype eq 'THAR',ngdarc)
  if ngdarc eq 0 then begin
    print,'No good arclamp exposures for '+strtrim(mjd-7,2)+'<=MJD<='+strtrim(mjd,2)
    return
  endif
  waveid = expinfo[gdarc].num
  print,strtrim(ngdarc,2),' good arclamp exposures'

  ;; Process the frames and find lines
  print,''
  print,'***** Processing the frames and finding the lines *****'
  print,''
  for i=0,n_elements(waveid)-1 do begin
    print,''
    print,'--- Frame ',strtrim(i+1,2),':  ',strtrim(waveid[i],2),' ---'
    ;; Check if they exist
    wavefiles = apogee_filename('Wave',num=waveid[i],chip=chips)
    if total(file_test(wavefiles)) lt 3 or keyword_set(clobber) then begin
      MAKECAL,wave=waveid[i],file=dirs.libdir+'cal/'+dirs.instrument+'-wave.par',/nofit,unlock=unlock,psflibrary=psflibrary
    endif else begin
      print,repstr(wavefiles[0],'-a-','-'),' made already'
    endelse
  endfor

  ;; New Python version! 
  cmd = ['mkfpiwave',strtrim(cmjd,2),dirs.apred,strmid(dirs.telescope,0,3),'--verbose']
  print,'Running: ',cmd
  spawn,cmd,/noshell

  ;; Check that the calibration file was successfully created
  outfile = wavedir+repstr(file,'apWaveFPI-','apWaveFPI-a-')
  if file_test(outfile) then begin
    openw,lock,/get_lun,wavedir+file+'.dat'
    free_lun,lock
  endif

  file_delete,lockfile,/allow

end
