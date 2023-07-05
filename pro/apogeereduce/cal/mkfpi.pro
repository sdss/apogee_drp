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

  common apver,ver,telescop,instrume
  obs = strmid(telescop,0,3)
  
  if n_elements(name) eq 0 then name=string(fpiid[0])
  dirs = getdir(apodir,caldir,spectrodir,vers)
  wavedir = apogee_filename('Wave',num=name,chip='a',/dir)
  file = dirs.prefix+string(format='("WaveFPI-",i8.8)',name)
  fpifile = wavedir+file

  ;; If another process is alreadying make this file, wait!
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
  aplock,fpifile,/lock
  
  ;; Make sure we have the PSF cal product
  if keyword_set(psfid) then $
    MKPSF,psfid,darkid=darkid,flatid=flatid,fiberid=fiberid,unlock=unlock

  ;; Get all FPI frames for this night and process them
  expinfo = dbquery("select * from apogee_drp.exposure where mjd="+strtrim(mjd,2)+" and observatory='"+strtrim(obs,2)+"' and exptype='FPI'")
  expinfo.exptype = strtrim(expinfo.exptype,2)
  gdfpi = where(expinfo.exptype eq 'FPI',ngdfpi)
  if ngdfpi eq 0 then begin
    print,'No FPI exposures for MJD ',strtrim(mjd,2)
    aplock,fpifile,/clear 
  endif
  allfpinum = expinfo[gdfpi].num
  print,'Found ',strtrim(allfpinum,2),' FPI exposures for MJD ',strtrim(mjd,2)
  
  ;; Process the frames
  for i=0,n_elements(allfpinum)-1 do $
    w = approcess(allfpinum[i],dark=darkid,flat=flatid,psf=psfid,modelpsf=modelpsf,flux=0,/doproc,unlock=unlock)

  ;; Make sure the dailywave file is there
  ;;  it uses modelpsf by default now
  MAKECAL,dailywave=mjd,unlock=unlock,librarypsf=psflibrary

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

  aplock,fpifile,/clear
  
end
