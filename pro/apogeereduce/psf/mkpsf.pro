;+
;
; MKPSF
;
; Make an APOGEE PSF calibration file.  This is a wrapper around APMKPSF.PRO
; but ensures that the necessary 3D->2D and 2D->1D steps have been performed
; with APPROCESS.PRO.
;
; INPUTS:
;  psfid       ID8 number of the exposure to use.
;  =darkid     ID8 number of the dark/bpm calibration to use.
;  =flatid     ID8 number of the flat calibration to use.
;  =sparseid   ID8 number of the sparse calibration to use.
;  =littrowid  ID8 number of the littrow calibration to use.
;  =fiberid    ID8 number for the ETrace calibration file to use.
;  =yshift     Trace shifts in the y-direction (scalar or 3-element array).
;                Used when making FIBER calibration product
;  /average    Find the fibers    
;  /clobber    Overwrite any existing files.
;  /unlock     Remove any existing lock file and start fresh.
;  /etraceonly Only make fiber ETrace files, not the full EPSF.
;
; OUTPUTS:
;  A set of apPSF-[abc]-ID8.fits files in the appropriate location      
;   determined by the SDSS/APOGEE tree directory structure.
;
; USAGE:
;  IDL>mkpsf,littrowid,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid,average=200,/clobber
;
; By J. Holtzman, 2011
;  Added doc strings, updates to use data model  D. Nidever, Sep 2020
;-

pro mkpsf,psfid,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid,$
          littrowid=littrowid,yshift=yshift,average=average,clobber=clobber,$
          unlock=unlock,etraceonly=etraceonly

  dirs = getdir(apodir,caldir,spectrodir,vers)
  caldir = dirs.caldir
  
  psfdir = apogee_filename('PSF',num=psfid[0],chip='c',/dir)
  file = apogee_filename('PSF',num=psfid[0],chip='c',/base)
  ;; If another process is alreadying make this file, wait!
  ;;  don't use chip in name, apmkpsf.pro makes chip-specific lock files
  if file_test(psfdir,/directory) eq 0 then file_mkdir,psfdir
  aplock,file,waittime=10,unlock=unlock
  
  ;; Does product already exist?
  ;; check all three chips and the EPSF and ETrace files
  tracedir = apogee_filename('ETrace',num=psfid[0],chip='c',/dir)
  chips = ['a','b','c']
  spsfid = string(psfid[0],format='(i08)')
  allfiles = psfdir+[dirs.prefix+'PSF-'+chips+'-'+spsfid,dirs.prefix+'EPSF-'+chips+'-'+spsfid]+'.fits'
  allfiles = [allfiles,tracedir+'apETrace-'+chips+'-'+spsfid+'.fits']
  if total(file_test(allfiles)) eq 9 and not keyword_set(clobber) then begin
    print,' PSF file: ', psfdir+file, ' already made'
    return
  endif
  file_delete,allfiles,/allow  ;; delete any existing files to start fresh
  if not keyword_set(fiberid) then fiberid=0
  if not keyword_set(sparseid) then sparseid=0

  print,'Making PSF: ', psfid[0]
  ;; Open .lock file
  aplock,file,/lock
  
  cmjd = getcmjd(psfid)
  print,'mkpsf approcess...'
  d = approcess(psfid,darkid=darkid,flatid=flatid,littrowid=littrowid,/nocr,nfs=1,/doap3dproc,unlock=unlock)
  psffile = apogee_filename('2D',num=psfid[0],chip='c',/dir)+'/'+string(format='(i8.8)',psfid)
  APMKPSF,psffile,psfdir,sparseid=sparseid,fiberid=fiberid,yshift=yshift,average=average,$
          clobber=clobber,unlock=unlock,etraceonly=etraceonly

  ;;file_delete,lockfile,/allow
  aplock,file,/clear
  
end

