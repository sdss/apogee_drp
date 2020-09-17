;+
;
; GETDIR
;
; Returns structure with directories given current environment
; and loads some important parameters in a common block.
;
; INPUTS:
;  none
;  /onem        Depricated parameter for 1m.
;  /apogees     Depricated parameter for apogee south.  
;  =v           Depricated parameter to input the reduction version.
;
; OUTPUTS:
;  apogeedir    The main reduction directory.
;  caldir       The main calibration directory.
;  spectrodir   The reduction directory for the current version.
;  vers         The reduction version (e.g., r13).
;  prefix       The prefix 'ap' for apogee-n and 'as' for apogee-s.
;  =apred_vers  The reduction version. 
;  =datadir     The raw data directory.
;
; USAGE:
;  IDL>dirs = getdir(apogeedir,spectrodir,caldir,vers,libdir,prefix)
;
; By J. Holtzman, 2011
;  Added doc strings, updates to use data model  D. Nidever, Sep 2020 
;-

function getdir,apogeedir,caldir,spectrodir,vers,libdir,prefix,$
                apred_vers=apred_vers,datadir=datadir,$
                v=v,onem=onem,apogees=apogees

  common apver,ver,telescop,instrume

  ;; Override
  if n_elements(ver) gt 0 then apred_vers=ver else begin
    ;; Strict versioning
    vers = getenv('APOGEE_VERS')
    apred_vers = vers
  endelse
  vers = apred_vers

  apogeedir = getenv('APOGEE_REDUX')
  ;speclib = getenv('APOGEE_SPECLIB')
  ;aspcap = getenv('APOGEE_ASPCAP')
  pipedir = getenv('APOGEE_DRP_DIR')
  libdir = pipedir+'/data/'
  if telescop eq 'apo1m' then begin
    datadir = getenv('APOGEE_DATA_1M')+'/'
  endif else if telescop eq 'lco25m' then begin
    datadir = getenv('APOGEE_DATA_2S')+'/' 
  endif else begin
    datadir = getenv('APOGEE_DATA')+'/' 
  endelse
  if instrume eq 'apogee-n' then begin
    prefix = 'ap'
    mapperdir = getenv('MAPPER_DATA')+'/'
    calfile = libdir+'cal/apogee-n.par'
  endif else begin
    prefix = 'as'
    mapperdir = getenv('MAPPER_DATA_2S')+'/'
    calfile = libdir+'cal/apogee-s.par'
  endelse
  spectrodir = apogeedir+'/'+apred_vers+'/'
  caldir = spectrodir+'cal/'+instrume+'/'
  expdir = spectrodir+'exposures/'+instrume+'/'

  out = {datadir:datadir, apogeedir:apogeedir, expdir:expdir, caldir:caldir, spectrodir:spectrodir,$
         libdir:libdir, prefix:prefix, calfile:calfile, mapperdir:mapperdir, apred:apred_vers,$
         telescope:telescop, instrument:instrume, redux:apogeedir}
         ;telescope:telescop, instrument:instrume, redux:apogeedir,speclib:speclib, aspcap:aspcap}
  return,out
end
