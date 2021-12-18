;+
;
; MKPERSIST
;
; Procedure to make an APOGEE persistence calibration file from
; a dark and flat exposure.
;
; INPUT:
;  persistid   The frame name for the output apPersist file.
;  dark        The dark frame to use to derive the persistence.
;  flat        The flat frame to use to derive the persistence.
;  =cmjd       The MJD directory to put the ap2D/ap1D files in.
;  =darkid     Dark frame to be used if images are reduced.
;  =flatid     Flat frame to be used if images are reduced.
;  =sparseid   Sparse frame to be used if images are reduced.
;  =fiberid    ETrace frame to be used if images are reduced.
;  =thresh     Threshold to use for persistence.  Default is 0.1.
;  /clobber    Overwrite existing files.
;  /unlock     Delete lockfile and start fresh.
;
; OUTPUT:
;  A set of apPersist-[abc]-ID8.fits files in the appropriate location
;   determined by the SDSS/APOGEE tree directory structure.
;
; USAGE:
;  IDL>mkpersist,persist,darkid,flatid,thresh=thresh,cmjd=cmjd,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid,/clobber
;
; By J. Holtzman, 2011
;  Added doc strings, updates to use data model  D. Nidever, Sep 2020 
;-

pro mkpersist,persistid,dark,flat,cmjd=cmjd,darkid=darkid,flatid=flatid,$
              sparseid=sparseid,fiberid=fiberid,clobber=clobber,thresh=thresh,$
              unlock=unlock

  if not keyword_set(thresh) then thresh=0.1

  dirs = getdir()

  perdir = apogee_filename('Persist',num=persistid,chip='c',/dir)
  file = apogee_filename('Persist',num=persistid,chip='c',/base)
  lockfile = perdir+file+'.lock'

  ;; If another process is alreadying making this file, wait!
  if not keyword_set(unlock) then begin
    while file_test(lockfile) do apwait,lockfile,10
  endif else begin
    if file_test(lockfile) then file_delete,lockfile,/allow
  endelse

  ;; Does product already exist?
  if file_test(perdir+file) and not keyword_set(clobber) then begin
    print,' persist file: ',perdir+file,' already made'
    return
  endif
  ;; Open .lock file
  openw,lock,/get_lun,lockfile
  free_lun,lock

  if keyword_set(cmjd) then begin
    d = approcess([dark,flat],cmjd=cmjd,darkid=darkid,flatid=flatid,psfid=psfid,nfs=1,/doap3dproc) 
  endif else begin
    d = approcess([dark,flat],darkid=darkid,flatid=flatid,psfid=psfid,nfs=1,/doap3dproc)
  endelse

  d = apread('2D',num=dark)
  f = apread('2D',num=flat)

  ;; Write out an integer mask
  chip = ['a','b','c']
  for ichip=0,2 do begin
    persist = intarr(2048,2048)
    r = d[ichip].flux/f[ichip].flux
    bad = where(d[ichip].mask and badmask() or f[ichip].mask and badmask(),nbad)
    if nbad gt 0 then r[bad]=0.
    rz = zap(r,[10,10])
    print,median(rz)
    bad = where(rz gt thresh/4.,nbad)
    if nbad gt 0 then persist[bad]=4
    bad = where(rz gt thresh/2.,nbad)
    if nbad gt 0 then persist[bad]=2
    bad = where(rz gt thresh,nbad)
    if nbad gt 0 then persist[bad]=1
    file = apogee_filename('Persist',num=persistid,chip=chip[ichip])
    MWRFITS,persist,file,/create
    MWRFITS,rz,file
  endfor

  file = apogee_filename('Persist',num=persistid,chip='c',/base)
  file_delete,lockfile,/allow
end
