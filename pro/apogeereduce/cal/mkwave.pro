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
;  =modelpsf   Model PSF calibration frame to use.
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
           modelpsf=modelpsf,fiberid=fiberid,clobber=clobber,nowait=nowait,$
           nofit=nofit,unlock=unlock

  if n_elements(name) eq 0 then name=string(waveid[0])
  dirs = getdir(apodir,caldir,spectrodir,vers)
  wavedir = apogee_filename('Wave',num=name,chip='a',/dir)
  file = dirs.prefix+string(format='("Wave-",i8.8)',name)
  wavefile = wavedir+file

  ;; If another process is alreadying make this file, wait!
  aplock,wavefile,waittime=10,unlock=unlock
  
  ;; Does product already exist?
  ;; check all three chips and .dat file
  chips = ['a','b','c']
  swaveid = string(waveid[0],format='(i08)')
  allfiles = wavedir+dirs.prefix+'Wave-'+chips+'-'+swaveid+'.fits'
  if total(file_test(allfiles)) eq 3 and not keyword_set(clobber) then begin
    print,' Wavecal file: ', wavedir+file+'.dat', ' already made'
    return
  endif
  file_delete,allfiles,/allow  ;; delete any existing files to start fresh

  print,'Making wave: ', waveid
  ;; Open .lock file
  aplock,wavefile,/lock
  
  cmjd = getcmjd(waveid[0],mjd=mjd)
  ;; This code is not used !?
  ;;expinfo = dbquery("select * from apogee_drp.exposure where mjd>="+strtrim(mjd-7,2)+" and mjd<="+strtrim(mjd+7,2)+" and exptype='ARCLAMP'")
  ;;expinfo.arctype = strtrim(expinfo.arctype,2)
  ;;gdarc = where(expinfo.arctype eq 'UNE' or expinfo.arctype eq 'THAR',ngdarc)

  ;; Process the frame, if necessary
  chipfiles = apogee_filename('1D',num=waveid[0],chip=['a','b','c'])
  if total(file_test(chipfiles)) ne 3 then begin
    if keyword_set(psfid) then begin
      cmjd = getcmjd(psfid)
      MKPSF,psfid,darkid=darkid,flatid=flatid,fiberid=fiberid,unlock=unlock
    endif
    w = approcess(waveid[0],dark=darkid,flat=flatid,psf=psfid,modelpsf=modelpsf,flux=0,/doproc,unlock=unlock)
  endif

  ;; Check that the data is okay
  chfile = apogee_filename('2D',num=waveid[0],chip='b')
  if file_test(chfile) eq 0 then begin
    print,chfile+' NOT FOUND'
    aplock,wavefile,/clear
    return
  endif
  head0 = headfits(chfile,exten=0)
  FITS_READ,chfile,im1,head1
  ;; UNE, bright line at X=1452
  if sxpar(head0,'LAMPUNE') then begin
    sub = im1[1452-100:1452+100,*]
    thresh = 40
  ;; THARNE, bright line at X=1566 
  endif else if sxpar(head0,'LAMPTHAR') then begin
    sub = im1[1566-100:1566+100,*]
    thresh = 1000
  endif else begin
    sub = im1[900:1100,*]
    thresh = 10
  endelse
  smsub = medfilt2d(sub,7,dim=1)                            ;; smooth in spectral axis
  resmsub = REBIN(smsub[*,0:(2048/8)*8-1],[201,2048/8])*8  ;; rebin in spatial axis
  peakflux = MAX(resmsub,dim=1)                             ;; peak flux feature in spectral dim.
  avgpeakflux = median(peakflux)
  ;; Check the line flux
  if avgpeakflux/sxpar(head0,'nread') lt thresh then begin
    print,'Not enough flux in ',chfile
    aplock,wavefile,/clear     
    return
  endif
  
  ;; New Python version! 
  cmd = ['apmultiwavecal','--name',strtrim(name,2),'--vers',dirs.apred]
  if keyword_set(nofit) then cmd=[cmd,'--nofit']
  if keyword_set(plot) then cmd=[cmd,'--plot','--hard']
  if keyword_set(clobber) then cmd=[cmd,'--clobber']
  cmd = [cmd,'--inst',dirs.instrument,'--verbose']
  for i=0,n_elements(waveid)-1 do cmd=[cmd,string(waveid[i])]
  spawn,cmd,/noshell

  ;; Check that the calibration file was successfully created
  outfile = wavedir+repstr(file,'apWave-','apWave-a-')
  if file_test(outfile) then begin
    openw,lock,/get_lun,wavedir+file+'.dat'
    free_lun,lock
  endif

  ;;file_delete,lockfile,/allow
  aplock,wavefile,/clear
  
end
