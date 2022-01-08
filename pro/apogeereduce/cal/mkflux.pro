;+
;
; MKFLUX
;
; Makes APOGEE flux calibration file.
;
; INPUT:
;  ims: list of image numbers to include in flux calibration file.
;  cmjd=cmjd : (optional,obsolete) gives MJD directory name if not encoded in file number
;  darkid=darkid : dark frame to be used if images are reduced
;  flatid=flatid : flat frame to be used if images are reduced
;  psfid=psfid : psf frame to be used if images are reduced
;  waveid=waveid : wave frame to be used if images are reduced
;  littrowid=littrowid : littrow frame to be used if images are reduced
;  persistid=persistid : persist frame to be used if images are reduced
;  /clobber : rereduce images even if they exist
;  /onedclobber : overwrite the 1D files
;  /unlock : delete lock file and start fresh
;
; OUTPUT:
;  A set of apFlux-[abc]-ID8.fits files in the appropriate location
;   determined by the SDSS/APOGEE tree directory structure.
;
; USAGE:
;  IDL>mkflux,ims,cmjd=cmjd,darkid=darkid,flatid=flatid,/clobber
;
; By J. Holtzman, 2011?
;  Added doc strings, updates to use data model  D. Nidever, Sep 2020 
;-

pro mkflux,ims,cmjd=cmjd,darkid=darkid,flatid=flatid,psfid=psfid,waveid=waveid,littrowid=littrowid,$
           persistid=persistid,clobber=clobber,onedclobber=onedclobber,bbtemp=bbtemp,plate=plate,$
           plugid=plugid,holtz=holtz,temp=temp,unlock=unlock

  dirs = getdir(apodir,caldir,spectrodir,vers)
  caldir = dirs.caldir

  file = apogee_filename('Flux',num=ims[0],chip='c',/base)
  fluxdir = apogee_filename('Flux',num=ims[0],chip='c',/dir)
  if file_test(fluxdir,/directory) eq 0 then file_mkdir,fluxdir
  fluxlockfile = fluxdir+file+'.lock'

  ;; If another process is alreadying making this file, wait!
  if not keyword_set(unlock) then begin
    while file_test(fluxlockfile) do apwait,fluxlockfile,10
  endif else begin
    if file_test(fluxlockfile) then file_delete,fluxlockfile,/allow
  endelse

  ;; Does product already exist?
  if file_test(fluxdir+file) and not keyword_set(clobber) then begin
    print,' flux file: ',fluxdir+file,' already made'
    if n_elements(temp) ne 0 then goto,response
    return
  endif
  ;; Open .lock file
  openw,lock,/get_lun,fluxlockfile
  free_lun,lock

  if not keyword_set(plate) then plate=0

  ;; Need to make sure extraction is done without flux calibration
  i1 = ims[0]
  files = apogee_filename('1D',num=i1,chip=['a','b','c'])
  if total(file_test(files),/int) gt 0 then file_delete,files,/allow
  if keyword_set(cmjd) then begin
    d = approcess(ims,cmjd=cmjd,darkid=darkid,flatid=flatid,psfid=psfid,littrowid=littrowid,$
                  persistid=persistid,/nocr,nfs=1,/doproc,unlock=unlock)
  endif else begin
    d = approcess(ims,darkid=darkid,flatid=flatid,psfid=psfid,littrowid=littrowid,$
                  persistid=persistid,/nocr,nfs=1,/doproc,unlock=unlock)
  endelse
  cmjd = getcmjd(i1)
  inpfile = apogee_filename('1D',num=i1,chip='a',/dir)+string(format='(i8.8)',i1)
  ;inpfile = dirs.expdir+cmjd+'/'+string(format='(i8.8)',i1)
  APMKFLUXCAL,inpfile,outdir=fluxdir,/clobber

  ;; Clean up in case someone might want to reduce these files with flux calibration
  ;files = apogee_filename('1D',num=i1,chip=['a','b','c'])
  ;if total(file_test(files),/int) gt 0 then file_delete,files,/allow
  ;files = file_search(dirs.expdir+getcmjd(i1)+'/'+dirs.prefix+'1D-?-'+string(format='(i8.8)',i1)+'.fits')
  ;if files[0] ne '' then file_delete,files

  ;; Holtz's flux calibration method
  if keyword_set(holtz) then begin

    nframes = n_elements(ims)
    for ii=0,nframes-1 do begin
      i = ims[ii]
      ;if keyword_set(cmjd) then frame=apread(i,err,mask,head,cmjd=cmjd,/oned) $
      ;else frame=apread(i,err,mask,head,/oned) 
      frame = apread('1D',num=i)
      if ii eq 0 then begin
        head0 = frame[0].hdr
        sz = size(frame[0].flux)
        flux = fltarr(sz[1],sz[2],3)
      endif
      for ichip=0,2 do flux[*,*,ichip]+=frame[ichip].flux
    endfor
  
    bad =- 1
    if keyword_set(plate) then begin
      if not keyword_set(cmjd) then cmjd=getcmjd(ims[0])
      fiber = getfiber(plate,cmjd,plugid=plugid)
      bad = where(fiber.fiberid lt 0)
    endif

    sz = size(flux)
    chips = ['a','b','c']
    resp = fltarr(2048,300,3)
    for ichip=0,2 do begin
      if keyword_set(bbtemp) then begin
        wavedir = apogee_filename('Wave',num=waveid,chip=chips[ichip],/dir)
        file = apogee_filename('Wave',num=waveid,chip=chips[ichip],/base)
        ;file = dirs.prefix+string(format='("Wave-",a,"-",i8.8)',chips[ichip],waveid)
        wavetab = mrdfits(wavedir+file,1)
        refspec = fltarr(sz[1],sz[2])
        for ifiber=0,sz[1]-1 do refspec[ifiber,*]=planck(wavetab[ifiber,*],bbtemp)
      endif else begin
        refflux = reform(flux[*,150,ichip],sz[1],1)
        refspec = refflux/refflux
      endelse
      rows = intarr(sz[2])+1
      refimg = rows##refspec
      tmp = zap(flux[*,*,ichip],[100,1])
      if ichip eq 1 then norm=tmp[1024,150]
      resp[*,*,ichip] = refimg/tmp
      if (bad[0] ge 0) then for i=0,n_elements(bad)-1 do resp[*,bad[i]]=0.
    endfor
    ;; Normalize to center of green chip
    for ichip=0,2 do begin
      resp[*,*,ichip] *= norm
      file = apogee_filename('Flux',num=i1,chip=chips[ichip],/base)
      ;file = dirs.prefix+string(format='("Flux-",a,"-",i8.8)',chips[ichip],i1)
      MWRFITS,resp[*,*,ichip],fluxdir+file,head0,/create
    endfor
  endif

  file_delete,fluxlockfile,/allow  ;; delete lock file

  response:
  ;; Extra block if we are calculating response function 
  if n_elements(temp) gt 0 then begin
    file = apogee_filename('Response',num=ims[0],chip='c',/base,/nochip)
    ;file = dirs.prefix+string(format='("Response-c-",i8.8)',ims[0])
    responselockfile = fluxdir+file+'.lock'
    
    ;; If another process is alreadying making this file, wait!
    if not keyword_set(unlock) then begin
      while file_test(responselockfile) do apwait,responselockfile,10
    endif else begin
      if file_test(responselockfile) then file_delete,responselockfile,/allow
    endelse
    ;; Does product already exist?
    if file_test(fluxdir+file) and not keyword_set(clobber) then begin
      print,' flux file: ',fluxdir+file,' already made'
      return
    endif
    ;; Open .lock file
    openw,lock,/get_lun,responselockfile
    free_lun,lock

    chips = ['a','b','c']
    wave = mrdfits(apogee_filename('Wave',num=waveid,chip=chips[1]),2)
    flux = mrdfits(apogee_filename('Flux',num=ims[0],chip=chips[1]),3)
    ;wave = mrdfits(caldir+'/wave/'+dirs.prefix+'Wave-'+chips[1]+'-'+string(format='(i8.8)',waveid)+'.fits',2)
    ;flux = mrdfits(caldir+'/flux/'+dirs.prefix+'Flux-'+chips[1]+'-'+string(format='(i8.8)',ims[0])+'.fits',3)
    bbnorm = flux[1024] / PLANCK(wave[1024,150],temp)
    for i=0,2 do begin
      wave = mrdfits(apogee_filename('Wave',num=waveid,chip=chips[i]),2)
      flux = mrdfits(apogee_filename('Flux',num=ims[0],chip=chips[i]),3)
      ;wave = mrdfits(caldir+'/wave/'+dirs.prefix+'Wave-'+chips[i]+'-'+string(format='(i8.8)',waveid)+'.fits',2)
      ;flux = mrdfits(caldir+'/flux/'+dirs.prefix+'Flux-'+chips[i]+'-'+string(format='(i8.8)',ims[0])+'.fits',3)
      bbflux = PLANCK( wave[*,150], temp) * bbnorm
      mkhdr,head,bbflux/flux
      leadstr = 'APMKFLAT: '
      sxaddhist,leadstr+systime(0),head
      info = GET_LOGIN_INFO()
      sxaddhist,leadstr+info.user_name+' on '+info.machine_name,head
      sxaddhist,leadstr+'IDL '+!version.release+' '+!version.os+' '+!version.arch,head
      sxaddhist,leadstr+' APOGEE Reduction Pipeline Version: '+getvers(),head
      file = apogee_filename('Response',num=ims[0],chip=chipos[i])
      MWRFITS,bbflux/flux,fluxdir+file,head
    endfor

    file_delete,responselockfile,/allow   ;; delete lock file
  endif
 
end
