;+
;
; MKLITTROW
;
; Procedure to derive the APOGEE Littrow calibration file.
;
; INPUT:
;  littrowid   The ID8 number of the exposure to use.
;  =darkid     ID8 number of the dark calibration to use.
;  =flatid     ID8 number of the flat calibration to use.
;  =sparseid   ID8 number of the sparse calibration to use.
;  =fiberid    ID8 number for the ETrace calibration file to use. 
;  /clobber    Overwrite any existing files.
;  =cmjd       Depricated parameter.
;  /unlock     Delete lock file and start fresh 
;
; OUTPUT:
;  A set of apLittrow-[abc]-ID8.fits files in the appropriate location
;   determined by the SDSS/APOGEE tree directory structure.
;
; USAGE:
;  IDL>mklittrow,littrowid
;
; By J. Holtzman, 2011
;  Added doc strings, updates to use data model  D. Nidever, Sep 2020 
;-

pro mklittrow,littrowid,darkid=darkid,flatid=flatid,sparseid=sparseid,$
              fiberid=fiberid,clobber=clobber,cmjd=cmjd,unlock=unlock

  dirs = getdir()
  caldir = dirs.caldir

  litdir = apogee_filename('Littrow',num=littrowid,chip='b',/dir)
  if file_test(litdir,/directory) eq 0 then file_mkdir,litdir
  file = apogee_filename('Littrow',num=littrowid,chip='b',/base)
  lockfile = litdir+file+'.lock'

  ;; If another process is alreadying making this file, wait!
  if not keyword_set(unlock) then begin
    while file_test(lockfile) do apwait,lockfile,10
  endif else begin
    if file_test(lockfile) then file_delete,lockfile,/allow
  endelse

  ;; Does product already exist?
  ;;  we only use the b detector file
  if file_test(litdir+file) and not keyword_set(clobber) then begin
    print,' littrow file: ',litdir+file,' already made'
    return
  endif
  file_delete,allfiles,/allow  ;; delete any existing files to start fresh
  ;; Open .lock file
  openw,lock,/get_lun,lockfile
  free_lun,lock

  ;; Make empirical PSF with broader smoothing in columns so that Littrow ghost is not incorporated as much
  MKPSF,littrowid,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid,average=200,/clobber,unlock=unlock
  ;; Process the frame with this PSF to get model that does not have Littrow ghost
  psfdir = apogee_filename('PSF',num=littrowid,chip='b',/dir)
  wavefile = 0
  indir = apogee_filename('2D',num=littrowid,chip='b',/dir)
  AP2DPROC,indir+'/'+string(format='(i8.8)',littrowid),$
           psfdir+'/'+string(format='(i8.8)',littrowid),4,wavefile=wavefile,/clobber

  ;; Read in the 2D file and the model, and use them to find the Littrow ghost
  im2 = apread('2D',num=littrowid,chip='b')
  im2mod = apread('2Dmodel',num=littrowid,chip='b')
  im = im2.flux
  immask = im2.mask
  scat_remove,im,scat=1
  immod = im2mod.flux
  bad = where((immask and badmask()) gt 0)
  im[bad] = !values.f_nan
  l = where(median(im[1200:1500,*]-immod[1200:1500,*],20) gt 10,complement=nl)

  ;; Write out an integer mask
  litt = intarr(2048,2048)
  tmp = im[1200:1500,*]*0 & tmp[l]=1 & tmp[nl]=0
  litt[1250:1450,*] = tmp[50:250,*]
  file = apogee_filename('Littrow',num=littrowid,chip='b')

  MKHDR,head,litt   ;,/image
  leadstr = 'MKLITTROW: '
  sxaddhist,leadstr+systime(0),head
  info = GET_LOGIN_INFO()
  sxaddhist,leadstr+info.user_name+' on '+info.machine_name,head
  sxaddhist,leadstr+'IDL '+!version.release+' '+!version.os+' '+!version.arch,head
  sxaddhist,leadstr+' APOGEE Reduction Pipeline Version: '+getvers(),head
  MWRFITS,litt,file,head,/create

  ;; Move PSFs to littrow directory since they are not a standard PSF!
  outdir = litdir
  if file_test(outdir,/directory) eq 0 then file_mkdir,outdir

  files = file_search(psfdir+'/*'+string(format='(i8.8)',littrowid)+'*.fits')
  file_move,files,outdir,/over
  files = file_search(apogee_filename('1D',num=littrowid,chip='b',/dir)+'/*1D*'+string(format='(i8.8)',littrowid)+'*.fits')
  file_move,files,outdir,/over
  files = file_search(apogee_filename('2Dmodel',num=littrowid,chip='b',/dir)+'/*2Dmodel*'+string(format='(i8.8)',littrowid)+'*.fits')
  file_move,files,outdir,/over

  file = apogee_filename('Littrow',num=littrowid,chip='b',/base,/nochip)
  file_delete,lockfile,/allow

end

