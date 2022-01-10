;+
;
; MKLSF
;
; Procedure to make an APOGEE LSF calibration file.  This is a wrapper
; around APLSF.PRO but this ensures that the necessary calibration
; files and basic processing steps have already been performed.
;
; INPUT:
;  lsfid       The ID8 number of the exposure to use.
;  waveid      ID8 number of the wave calibration to use. 
;  =darkid     ID8 number of the dark calibration to use.
;  =flatid     ID8 number of the flat calibration to use.
;  =psfid      ID8 number of the psf calibration to use.
;  =fiberid    ID8 number for the ETrace calibration file to use. 
;  =fibers     An array if fibers to fit.  By default all 300 are fit.
;  /clobber    Overwrite any existing files.
;  /full       Perform full Gauss-Hermite fitting, otherwise just
;                Gaussian fitting is performed by default.
;  /pl         Make plots.
;  /nowait     If LSF file is already being made then don't wait
;                just return.
;  /newwave    Depricated parameter.
;  /unlock     Delete the lock file and start fresh.
;
; OUTPUT:
;  A set of apLSF-[abc]-ID8.fits files in the appropriate location
;   determined by the SDSS/APOGEE tree directory structure.
;
; USAGE:
;  IDL>mklsf,ims,waveid,darkid=darkid,flatid=flatid,psfid=psfid,fiberid=fiberid,/full,/clobber,/pl
;
; By J. Holtzman, 2011
;  Added doc strings, updates to use data model  D. Nidever, Sep 2020 
;-

pro mklsf,lsfid,waveid,darkid=darkid,flatid=flatid,psfid=psfid,fiberid=fiberid,$
          clobber=clobber,full=full,newwave=newwave,pl=pl,fibers=fibers,nowait=nowait,$
          unlock=unlock

  if not keyword_set(newwave) then newwave=0

  dirs = getdir(apodir,caldir,spectrodir,vers)
  caldir = dirs.caldir
  file = apogee_filename('LSF',num=lsfid[0],/nochip)
  file = file_dirname(file)+'/'+file_basename(file,'.fits')
  lockfile = file+'.lock'

  ;; If another process is alreadying make this file, wait!
  if not keyword_set(unlock) then begin
    while file_test(lockfile) do begin
      if keyword_set(nowait) then begin
        print,' LSF file: ', file, ' already being made (.lock file exists)'
        return
      endif
      apwait,file,10
    endwhile
  endif else begin
    if file_test(lockfile) then file_delete,lockfile,/allow
  endelse

  ; does product already exist?
  ;; check all three chip files and .sav file exist
  slsfid = string(lsfid[0],format='(i08)')
  chips = ['a','b','c']
  lsfdir = apogee_filename('LSF',num=lsfid[0],chip='c',/dir)
  allfiles = lsfdir+dirs.prefix+'LSF-'+chips+'-'+slsfid+'.fits'
  allfiles = [allfiles,lsfdir+dirs.prefix+'LSF-'+slsfid+'.sav']
  if total(file_test(allfiles)) eq 4 and not keyword_set(clobber) then begin
    print,' LSF file: ',file+'.sav',' already made'
    return
  endif
  file_delete,allfiles,/allow  ;; delete any existing files to start fresh

  ;; Open .lock file
  openw,lock,/get_lun,lockfile
  free_lun,lock

  cmjd = getcmjd(psfid)

  lsffile = apogee_filename('1D',num=lsfid[0],chip='c')

  MKPSF,psfid,darkid=darkid,flatid=flatid,fiberid=fiberid,/clobber,/unlock
  w = approcess(lsfid,dark=darkid,flat=flatid,psf=psfid,flux=0,/doproc,/skywave,/clobber)
  cmd = ['apskywavecal','dummy','--frameid',string(lsfid),'--waveid',string(waveid),'--apred',dirs.apred,'--telescope',dirs.telescope]
  spawn,cmd,/noshell

  lsffile = file_dirname(lsffile)+'/'+string(format='(i8.8)',lsfid)
  if size(waveid,/type) eq 7 then wavefile = caldir+'wave/'+waveid else $
    wavefile = caldir+'wave/'+string(format='(i8.8)',waveid)
  psffile = caldir+'/psf/'+string(format='(i8.8)',psfid)
  APLSF,lsffile,wavefile,psf=psffile,/gauss,pl=pl
  if keyword_set(full) then APLSF,lsffile,wavefile,psf=psffile,/clobber,pl=pl,fibers=fibers

  file_delete,lockfile,/allow
end
