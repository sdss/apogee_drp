;+
;
; GETCAL
;
; Given a cal file and MJD, returns info from appropriate
; MJD for requested date.
;
; INPUTS:
;  mjd              MJD for which calibrations are requested.    
;  file             Master calibration file list/index.
;
; OUTPUTS:
;  =darkid          Dark calibration frame number.
;  =flatid          Flat calibration frame number.
;  =sparseid        Sparsepak PSF calibration frame number.
;  =bpmid           Bad pixel mask calibration frame number.
;  =waveid          Wavelength calibration frame number.
;  =multiwaveid     Multiple-night wavelength calibration number.
;  =lsfid           LSF calibration frame number.
;  =fluxid          Flux calibration frame number.
;  =detid           Detector calibration frame number.
;  =fiberid         Fiber calibration frame number.
;  =badfiberid      BadFiber calibration frame number.
;  =fixifberid      FixFiber calibration frame number.
;  =littrowid       Littrow calibrationf frame number.
;  =persistid       Persistence calibration frame number.
;  =persistmodelid  Persistesnce model calibration frame number.
;  =responseid      Response calibration frame number.
;
; USAGE:
;  IDL>getcal,mjd,calfile,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid,littrowid=littrowid
;
; Written by J.Holtzman Aug 2011
;  Add doc strings, general cleanup by D. Nidever, Sep 2020
;-


pro getcal,mjd,file,darkid=darkid,flatid=flatid,sparseid=sparseid,bpmid=bpmid,$
           waveid=waveid,multiwaveid=multiwaveid,lsfid=lsfid,fluxid=fluxid,$
           detid=detid,fiberid=fiberid,badfiberid=badfiberid,fixfiberid=fixfiberid,$
           littrowid=littrowid,persistid=persistid,persistmodelid=persistmodelid,$
           responseid=responseid

  ;; Get the calibration files for desired date (mjd) from master calibration index (file)
  readcal,file,darkstr,flatstr,sparsestr,fiberstr,badfiberstr,fixfiberstr,wavestr,$
          lsfstr,bpmstr,fluxstr,detstr,littrowstr,persiststr,persistmodelstr,$
          responsestr,multiwavestr
  darkid = readcalstr(darkstr,mjd)
  flatid = readcalstr(flatstr,mjd)
  sparseid = readcalstr(sparsestr,mjd)
  fiberid = readcalstr(fiberstr,mjd)
  badfiberid = getnums(readcalstr(badfiberstr,mjd))
  fixfiberid = getnums(readcalstr(fixfiberstr,mjd))
  bpmid = readcalstr(bpmstr,mjd)
  waveid = readcalstr(wavestr,mjd)
  multiwaveid = readcalstr(multiwavestr,mjd)
  lsfid = readcalstr(lsfstr,mjd)
  fluxid = readcalstr(fluxstr,mjd)
  detid = readcalstr(detstr,mjd)
  littrowid = readcalstr(littrowstr,mjd)
  persistid = readcalstr(persiststr,mjd)
  persistmodelid = readcalstr(persistmodelstr,mjd)
  responseid = readcalstr(responsestr,mjd)
  print,'  dark: ', darkid
  print,'  flat: ', flatid
  print,'  bpm: ', bpmid
  print,'  sparse: ', sparseid
  print,'  fiber: ', fiberid
  print,'  badfiber: ', badfiberid
  print,'  fixfiber: ', fixfiberid
  print,'  wave: ', waveid
  print,'  multiwave: ', multiwaveid
  print,'  lsf: ', lsfid
  print,'  flux: ', fluxid
  print,'  det: ', detid
  print,'  littrow: ', littrowid
  print,'  persist: ', persistid
  print,'  persistmodel: ', persistmodelid
  print,'  response: ', responseid
end

