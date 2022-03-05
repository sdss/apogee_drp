;+
;
; MAKECAL
;
; This will make one or ALL of the specified calibration product types
; listed in the master calibration index file.
;
; INPUT:
;  =file         Name of master calibration index file, if not
;                  specified use default cal.par in calibration directory
;  /dark         Make all of the darks in the file
;  dark=darkid   Make the dark with name=darkid 
;  /flat         Make all of the flats in the file
;  flat=flatid   Make the flat with name=flatid 
;  /wave         Make all of the wavecals in the file
;  wave=waveid   Make the wavecal with name=waveid 
;  /lsf          Make all of the lsfs in the file
;  lsf=lsfid     Make the lsf with name=lsfid 
;  fpi=fpiid     Make the FPI with name=fpiid
;  /librarypsf   Use PSF library to get PSF cal for images.
;
; OUTPUT:
;  Calibration products are generated in places specified by the
;  SDSS/APOGEE directory tree.
;
; USAGE:
;  IDL>makecal,file=file,/dark,/flat,/wave,/lsf
;        OR
;  IDL>makecal,file=file,dark=darkid,flat=flatid,wave=waveid,lsf=lsfid
;
; Written by J.Holtzman Aug 2011
;  Added doc strings and general cleanup by D. Nidever, Sep 2020
;-


pro makecal,file=file,det=det,dark=dark,flat=flat,wave=wave,multiwave=multiwave,$
            lsf=lsf,bpm=bpm,psf=psf,flux=flux,sparse=sparse,fiber=fiber,$
            littrow=littrow,persist=persist,modelpersist=modelpersist,$
            response=response,mjd=mjd,full=full,newwave=newwave,nskip=nskip,$
            average=average,clobber=clobber,vers=vers,telescope=telescope,$
            nofit=nofit,pl=pl,unlock=unlock,fpi=fpi,librarypsf=librarypsf

  if keyword_set(vers) and keyword_set(telescope) then apsetver,vers=vers,telescope=telescope
  dirs = getdir(apo_dir,cal_dir,spectro_dir,apo_vers,lib_dir)

  ;; Get default file name if file not specified
  if keyword_set(file) then begin
    if  strpos(file,'/') lt 0 then file=file_dirname(dirs.calfile)+'/'+file 
  endif else file=dirs.calfile
  calfile = dirs.calfile

  if not keyword_set(full) then full=0
  if not keyword_set(newwave) then newwave=0
  if not keyword_set(nskip) then nskip=1
  chips = ['a','b','c']

  ;; Read calibration master file into calibration structures
  READCAL,file,darkstr,flatstr,sparsestr,fiberstr,badfiberstr,fixfiberstr,wavestr,lsfstr,bpmstr,$
          fluxstr,detstr,littrowstr,persiststr,persistmodelstr,responsestr,multiwavestr

  ;; Make Detector calibration files
  ;;--------------------------------
  if keyword_set(det) then begin
    print,'makecal det: ', det
    if det gt 1 then begin
      file = apogee_filename('Detector',num=det,/nochip)
      file = file_dirname(file)+'/'+file_basename(file,'.fits')
      detdir = apogee_filename('Detector',num=det,chip='a',/dir)
      sdetid = string(det,format='(i08)')
      allfiles = detdir+'/'+dirs.prefix+'Detector-'+chips+'-'+sdetid+'.fits'
      if total(file_test(allfiles)) eq 3 and not keyword_set(clobber) then begin
        print,' detector file: ',file,' already made'
        return
      endif
      i = where(detstr.name eq det)
      if i lt 0 then begin
        print,'No matching calibration line for ', det
        stop
      endif
      MKDET,detstr[i].name,detstr[i].linid,unlock=unlock
    endif 
  endif

  ;; Make Dark calibration files
  ;;----------------------------
  if keyword_set(dark) then begin
    print,'makecal dark: ', dark
    if dark gt 1 then begin
      file = apogee_filename('Dark',num=dark,/nochip)
      file = file_dirname(file)+'/'+file_basename(file,'.fits')
      darkdir = apogee_filename('Dark',num=dark,chip='a',/dir)
      sdarkid = string(dark,format='(i08)')
      allfiles = darkdir+'/'+[dirs.prefix+'Dark-'+chips+'-'+sdarkid+'.fits',dirs.prefix+'Dark-'+sdarkid+'.tab']
      if total(file_test(allfiles)) eq 4 and not keyword_set(clobber) then begin
        print,' dark file: ',file+'.tab',' already made'
        return
      endif
      i = where(darkstr.name eq dark)
      if i lt 0 then begin
        print,'No matching calibration line for ', dark
        stop
      endif
      ims = getnums(darkstr[i].frames)
      cmjd = getcmjd(ims[0],mjd=mjd)
      GETCAL,mjd,calfile,detid=detid
      MAKECAL,det=detid,unlock=unlock
      MKDARK,ims,clobber=clobber,unlock=unlock
    endif else begin
      if keyword_set(mjd) then  begin
        num = getnum(mjd) 
        red = where(darkstr.frames/10000L eq num)
      endif else red=indgen(n_elements(darkstr))
      if (red[0] ge 0) then begin
        for i=0,n_elements(red)-1 do begin
          ims = getnums(darkstr[red[i]].frames)
          cmjd = getcmjd(ims[0],mjd=mjd)
          GETCAL,mjd,calfile,detid=detid
          MAKECAL,det=detid,unlock=unlock
          MKDARK,ims,clobber=clobber,unlock=unlock
        endfor
      endif
    endelse
  endif

  ;; Make Flat calibration files
  ;;----------------------------
  if keyword_set(flat) then begin
    print,'makecal flat: ', flat
    if flat gt 1 then begin
      file = apogee_filename('Flat',num=flat,/nochip)
      file = file_dirname(file)+'/'+file_basename(file,'.fits')
      sflatid = string(flat,format='(i08)')
      flatdir = apogee_filename('Flat',num=flat,chip='c',/dir)
      allfiles = flatdir+dirs.prefix+'Flat-'+chips+'-'+sflatid+'.fits'
      allfiles = [allfiles,flatdir+dirs.prefix+'Flat-'+sflatid+'.tab']
      if total(file_test(allfiles)) eq 4 and not keyword_set(clobber) then begin
        print,' flat file: ',file+'.tab',' already made'
        return
      endif
      i = where(flatstr.name eq flat)
      if i lt 0 then begin
        print,'No matching calibration line for ', flat
        stop
      endif
      ims = getnums(flatstr[i].frames)
      cmjd = getcmjd(ims[0],mjd=mjd)
      GETCAL,mjd,calfile,darkid=darkid
      MAKECAL,dark=darkid,unlock=unlock
      MKFLAT,ims,darkid=darkid,nrep=flatstr[i].nrep,dithered=flatstr[i].dithered,clobber=clobber,unlock=unlock
    endif else begin
      if keyword_set(mjd) then  begin
        num = getnum(mjd) 
        red = where(flatstr.frames/10000L eq num)
      endif else red=indgen(n_elements(darkstr))
      if (red[0] ge 0) then begin
        for i=0,n_elements(red)-1 do begin
          ims = getnums(flatstr[red[i]].frames)
          cmjd = getcmjd(ims[0],mjd=mjd)
          GETCAL,mjd,calfile,darkid=darkid
          MAKECAL,dark=darkid,unlock=unlock
          MKFLAT,ims,darkid=darkid,nrep=flatstr[i].nrep,dithered=flatstr[i].dithered,clobber=clobber,unlock=unlock
        endfor
      endif
    endelse
  endif

  ;; Make BPM calibration files
  ;;---------------------------
  if keyword_set(bpm) then begin
    print,'makecal bpm: ', bpm
    if bpm gt 1 then begin
      sbpmid = string(bpm,format='(i08)')
      bpmdir = apogee_filename('BPM',num=bpm,chip='c',/dir)
      allfiles = bpmdir+dirs.prefix+'BPM-'+chips+'-'+sbpmid+'.fits'
      file = apogee_filename('BPM',num=bpm,chip='c')
      if total(file_test(allfiles)) eq 3 and not keyword_set(clobber) then begin
        print,' bpm file: ',file, ' already made'
        return
      endif
      i = where(bpmstr.name eq bpm)
      if i lt 0 then begin
        print,'No matching calibration line for ', bpm
        stop
      endif
      MAKECAL,dark=bpmstr[i].darkid,unlock=unlock
      MAKECAL,flat=bpmstr[i].flatid,unlock=unlock
      MKBPM,bpmstr[i].name,darkid=bpmstr[i].darkid,flatid=bpmstr[i].flatid,clobber=clobber,unlock=unlock
    endif else begin
      if keyword_set(mjd) then  begin
        num = getnum(mjd) 
        red = where(bpmstr.frames/10000L eq num)
      endif else red=indgen(n_elements(bpmstr))
      if (red[0] ge 0) then begin
        for i=0,n_elements(red)-1 do begin
          MAKECAL,dark=bpmstr[i].darkid,clobber=clobber,unlock=unlock
          MAKECAL,flat=bpmstr[i].flatid,clobber=clobber,unlock=unlock
          MKBPM,bpmstr[red[i]].name,darkid=bpmstr[i].darkid,flatid=bpmstr[i].flatid,clobber=clobber,unlock=unlock
        endfor
      endif
    endelse
  endif

  ;; Make Sparsepak PSF calibration product
  ;;---------------------------------------
  if keyword_set(sparse) then begin
    print,'makecal sparse: ', sparse
    if sparse gt 1 then begin
      file = apogee_filename('Sparse',num=sparse,chip='c')
      psfdir = file_dirname(file)
      sparseid = string(sparse,format='(i08)')
      allfiles = psfdir+'/'+[dirs.prefix+'EPSF-'+chips+'-'+sparseid+'.fits',dirs.prefix+'Sparse-'+sparseid+'.fits']
      if total(file_test(allfiles)) eq 4 and not keyword_set(clobber) then begin
        print,' sparse file: ',file,' already made'
        return
      endif
      i = where(sparsestr.name eq sparse)
      if i lt 0 then begin
        print,'No matching calibration line for ', sparse
        stop
      endif
      ims = getnums(sparsestr[i].frames)
      cmjd = getcmjd(ims[0],mjd=mjd)
      GETCAL,mjd,calfile,darkid=darkid,flatid=flatid,bpmid=bpmid
      MAKECAL,dark=darkid,unlock=unlock
      MAKECAL,flat=flatid,unlock=unlock
      MAKECAL,bpm=bpmid,unlock=unlock
      darkims = getnums(sparsestr[i].darkframes)
      maxread = getnums(sparsestr[i].maxread)
      if n_elements(maxread) ne 3 then begin
        print,'sparse maxread does not have 3 elements! '
        stop
      endif 
      MKEPSF,ims,darkid=darkid,flatid=flatid,darkims=darkims,dmax=sparsestr[i].dmax,$
             maxread=maxread,clobber=clobber,/filter,thresh=0.2,scat=2,unlock=unlock
    endif
  endif

  ;; Make fiber calibration file
  ;;----------------------------
  if keyword_set(fiber) then begin
    print,'makecal fiber: ', fiber
    if fiber gt 1 then begin
      file = apogee_filename('PSF',num=fiber,chip='c')
      psfdir = file_dirname(file)
      sfiberid = string(fiber,format='(i08)')
      allfiles = psfdir+'/'+[dirs.prefix+'EPSF-'+chips+'-'+sfiberid+'.fits',dirs.prefix+'PSF-'+chips+'-'+sfiberid+'.fits']
      if total(file_test(allfiles)) eq 6 and not keyword_set(clobber) then begin
        print,' psf file: ',file, ' already made'
        return
      endif
      cmjd = getcmjd(fiber,mjd=mjd)
      GETCAL,mjd,calfile,darkid=darkid,flatid=flatid,sparseid=sparseid
      MKPSF,fiber,darkid=darkid,flatid=flatid,sparseid=sparseid,unlock=unlock
    endif
  endif

  ;; Make PSF calibration file
  ;;--------------------------
  if keyword_set(psf) and ~keyword_set(flux) then begin
    print,'makecal psf: ', psf
    if psf gt 1 then begin
      file = apogee_filename('PSF',num=psf,chip='c')
      psfdir = file_dirname(file)
      spsfid = string(psf,format='(i08)')
      allfiles = psfdir+'/'+[dirs.prefix+'EPSF-'+chips+'-'+spsfid+'.fits',dirs.prefix+'PSF-'+chips+'-'+spsfid+'.fits']
      if total(file_test(allfiles)) eq 6 and not keyword_set(clobber) then begin
        print,' psf file: ',file, ' already made'
        return
      endif
      cmjd = getcmjd(psf,mjd=mjd)
      GETCAL,mjd,calfile,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid,littrowid=littrowid
      MAKECAL,littrow=littrowid,unlock=unlock
      MKPSF,psf,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid,littrowid=littrowid,clobber=clobber,unlock=unlock
    endif
  endif

  ;; Make FPI calibration file
  ;;--------------------------
  if keyword_set(fpi) then begin
    print,'makecal fpi: ', fpi
    if fpi gt 1 then begin
      file = apogee_filename('WaveFPI',num=fpi,chip='c')
      wavedir = file_dirname(file)
      sfpiid = string(fpi,format='(i08)')
      allfiles = wavedir+'/'+dirs.prefix+'WaveFPI-'+chips+'-'+sfpiid+'.fits'
      if total(file_test(allfiles)) eq 3 and not keyword_set(clobber) then begin
        print,' fpi file: ',file, ' already made'
        return
      endif
      cmjd = getcmjd(fpi[0],mjd=mjd)
      ;; What PSF to use
      if keyword_set(psf) then begin
        psfid = psf
      ;; Try to find a PSF from this day
      endif else begin
        print,'Trying to automatically find a PSF calibration file'
        psfid = GETPSFCAL(fpi[0],psflibrary=librarypsf)
      endelse
      MAKECAL,psf=psfid,unlock=unlock
      GETCAL,mjd,calfile,darkid=darkid,flatid=flatid,bpmid=bpmid,fiberid=fiberid
      MAKECAL,fiber=fiberid,unlock=unlock
      MKFPI,fpi,name=name,darkid=darkid,flatid=flatid,psfid=psfid,$
            fiberid=fiberid,clobber=clobber,unlock=unlock,psflibrary=librarypsf
    endif
  endif

  ;; Make Littrow calibration file
  ;;------------------------------
  if keyword_set(littrow) then begin
    print,'makecal littrow: ', littrow
    if littrow gt 1 then begin
      file = apogee_filename('Littrow',num=littrow,chip='b')
      if file_test(file) and not keyword_set(clobber) then begin
        print,' littrow file: ',file, ' already made'
        return
      endif
      cmjd = getcmjd(littrow,mjd=mjd)
      GETCAL,mjd,calfile,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid
      MAKECAL,flat=flatid,unlock=unlock
      MKLITTROW,littrow,cmjd=cmjd,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid,clobber=clobber,unlock=unlock
    endif
  endif

  ;; Make Persistence calibration file
  ;;----------------------------------
  if keyword_set(persist) then begin
    print,'makecal persist: ', persist
    if persist gt 1 then begin
      file = apogee_filename('Persist',num=persist,chip='c')
      perdir = file_dirname(file)
      sperid = string(persist,format='(i08)')
      allfiles = perdir+'/'+dirs.prefix+'Persist-'+chips+'-'+sperid+'.fits'
      if total(file_test(allfiles)) eq 3 and not keyword_set(clobber) then begin
        print,' persist file: ',file, ' already made'
        return
      endif
      i = where(persiststr.name eq persist)
      if i lt 0 then begin
        print,'No matching calibration line for ', persist
        stop
      endif
      cmjd = getcmjd(persist,mjd=mjd)
      GETCAL,mjd,calfile,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid
      MKPERSIST,persist,persiststr[i].darkid,persiststr[i].flatid,thresh=persiststr[i].thresh,$
                cmjd=cmjd,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid,$
                clobber=clobber,unlock=unlock
    endif
  endif

  ;; Make Persistence model calibration file
  ;;----------------------------------------
  if keyword_set(modelpersist) then begin
    print,'makecal modelpersist: ', modelpersist
    if modelpersist gt 1 then begin
      file = apogee_filename('PersistModel',num=modelpersist,chip='c')
      perdir = file_dirname(file)
      sperid = string(modelpersist,format='(i08)')
      allfiles = perdir+'/'+dirs.prefix+'PersistModel-'+chips+'-'+sperid+'.fits'
      if total(file_test(allfiles)) eq 3 and not keyword_set(clobber) then begin
        print,' modelpersist file: ',file, ' already made'
        return
      endif
      i = where(persistmodelstr.name eq modelpersist)
      if i lt 0 then begin
        print,'No matching calibration line for ', modelpersist
        stop
      endif
      cmjd = getcmjd(modelpersist,mjd=mjd)
      GETCAL,mjd,calfile,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid
      MKPERSISTMODEL,modelpersist
    endif
  endif

  ;; Make Flux calibration file
  ;;---------------------------
  if keyword_set(flux) then begin
    print,'makecal flux: ', flux
    if flux gt 1 then begin
      file = apogee_filename('Flux',num=flux,chip='c')
      fluxdir = file_dirname(file)
      sfluxid = string(flux,format='(i08)')
      allfiles = fluxdir+'/'+dirs.prefix+'Flux-'+chips+'-'+sfluxid+'.fits'      
      if total(file_test(allfiles)) eq 3 and not keyword_set(clobber) then begin
        print,' flux file: ',file, ' already made'
        return
      endif
      cmjd = getcmjd(flux[0],mjd=mjd)
      ;; What PSF to use
      if keyword_set(psf) then begin
        psfid = psf
      ;; Try to find a PSF from this day
      endif else begin
        print,'Trying to automatically find a PSF calibration file'
        psfid = GETPSFCAL(flux[0],psflibrary=librarypsf)
      endelse
      cmjd = getcmjd(flux,mjd=mjd)
      MAKECAL,psf=psfid,unlock=unlock
      GETCAL,mjd,calfile,darkid=darkid,flatid=flatid,littrowid=littrowid,waveid=waveid
      MAKECAL,littrow=littrowid,unlock=unlock
      MKFLUX,flux,darkid=darkid,flatid=flatid,psfid=psfid,littrowid=littrowid,waveid=waveid,$
             clobber=clobber,unlock=unlock
    endif
  endif

  ;; Make Response calibration file
  ;;-------------------------------
  if keyword_set(response) then begin
    print,'makecal response: ', response
    if response gt 1 then begin
      file = apogee_filename('Response',num=response,chip='c')
      resdir = file_dirname(file)
      sresid = string(response,format='(i08)')
      allfiles = resdir+'/'+dirs.prefix+'Response-'+chips+'-'+sresid+'.fits'
      if total(file_test(allfiles)) eq 3 and not keyword_set(clobber) then begin
        print,' response file: ',file, ' already made'
        return
      endif
      i = where(responsestr.name eq response,nres)
      if nres eq 0 then begin
        print,'No matching calibration line for ', response
        stop
      endif else if nres gt 1 then i=i[0]
      cmjd = getcmjd(response,mjd=mjd)
      GETCAL,mjd,calfile,darkid=darkid,flatid=flatid,littrowid=littrowid,waveid=waveid,fiberid=fiberid
      MAKECAL,psf=responsestr[i].psf,unlock=unlock
      MAKECAL,wave=waveid,unlock=unlock
      MAKECAL,fiber=fiberid,unlock=unlock
      MAKECAL,littrow=littrowid,unlock=unlock
      MKFLUX,response,darkid=darkid,flatid=flatid,psfid=responsestr[i].psf,littrowid=littrowid,$
             waveid=waveid,temp=responsestr[i].temp,clobber=clobber,unlock=unlock
    endif
  endif

  ;; Make Wavelength calibration file
  ;;---------------------------------
  if keyword_set(wave) then begin
    print,'makecal wave: ', wave
    if wave gt 1 then begin
      file = apogee_filename('Wave',num=wave,chip='c')
      wavedir = file_dirname(file)
      swaveid = string(wave,format='(i08)')
      allfiles = wavedir+'/'+dirs.prefix+'Wave-'+chips+'-'+swaveid+'.fits'
      if total(file_test(allfiles)) eq 3 and not keyword_set(clobber) then begin
        print,' wave file: ',file, ' already made'
        return
      endif
      i = where(wavestr.name eq wave,nwave)
      if nwave gt 0 then begin
        ims = getnums(wavestr[i[0]].frames)
        name = wavestr[i[0]].name
        psfid = wavestr[i[0]].psfid
      ;; Use the input filename
      endif else begin
        ims = wave
        name = ims[0]
        cmjd = getcmjd(ims[0],mjd=mjd)
        ;; What PSF to use
        if keyword_set(psf) then begin
          psfid = psf
        ;; Try to find a PSF from this day
        endif else begin
          print,'Trying to automatically find a PSF calibration file'
          psfid = GETPSFCAL(ims[0],psflibrary=librarypsf)
        endelse
      endelse
      cmjd = getcmjd(ims[0],mjd=mjd)
      GETCAL,mjd,calfile,darkid=darkid,flatid=flatid,bpmid=bpmid,fiberid=fiberid
      MAKECAL,bpm=bpmid,unlock=unlock
      MAKECAL,fiber=fiberid,unlock=unlock
      MKWAVE,ims,name=name,darkid=darkid,flatid=flatid,psfid=psfid,$
             fiberid=fiberid,clobber=clobber,nofit=nofit,unlock=unlock
    endif else begin
      if keyword_set(mjd) then  begin
        num = getnum(mjd) 
        red = where(wavestr.frames/10000L eq num)
      endif else red=indgen(n_elements(wavestr))
      if (red[0] ge 0) then begin
        for i=0,n_elements(red)-1,nskip do begin
          ims = getnums(wavestr[red[i]].frames)
          cmjd = getcmjd(ims[0],mjd=mjd)
          GETCAL,mjd,calfile,darkid=darkid,flatid=flatid,bpmid=bpmid,fiberid=fiberid
          MAKECAL,bpm=bpmid,unlock=unlock
          MAKECAL,fiber=fiberid,unlock=unlock
          MKWAVE,ims,name=wavestr[red[i]].name,darkid=darkid,flatid=flatid,psfid=wavestr[red[i]].psfid,$
                 fiberid=fiberid,clobber=clobber,/nowait,nofit=nofit,unlock=unlock
        endfor
      endif
    endelse
  endif

  ;; Make multi-night wavelength calibration file
  ;;---------------------------------------------
  if keyword_set(multiwave) then begin
    print,'makecal multiwave: ', multiwave
    if multiwave gt 1 then begin
      file = apogee_filename('Wave',num=multiwave,/nochip)
      file = file_dirname(file)+'/'+file_basename(file,'.fits')
      swaveid = string(multiwave,format='(i08)')
      wavedir = apogee_filename('Wave',num=multiwave,chip='a',/dir)
      allfiles = wavedir+dirs.prefix+'Wave-'+chips+'-'+swaveid+'.fits'
      allfiles = [allfiles,wavedir+dirs.prefix+'Wave-'+swaveid+'py.dat']
      if total(file_test(allfiles)) eq 4 and not keyword_set(clobber) then begin
        print,' multiwave file: ',file+'.dat',' already made'
        return
      endif
      i = where(multiwavestr.name eq multiwave,nwave)
      if nwave le 0 then begin
        print,'No matching calibration line for ', multiwave
        stop
      endif
      ims = getnums(multiwavestr[i[0]].frames)
      MKMULTIWAVE,ims,name=multiwavestr[i[0]].name,clobber=clobber,file=file,unlock=unlock,$
                  psflibrary=librarypsf
    endif else begin
      if keyword_set(mjd) then  begin
        num = getnum(mjd) 
        red = where(multiwavestr.frames/10000L eq num)
      endif else red=indgen(n_elements(multiwavestr))
      if (red[0] ge 0) then begin
       for i=0,n_elements(red)-1,nskip do begin
        ims = getnums(multiwavestr[red[i]].frames)
        MKMULTIWAVE,ims,name=multiwavestr[red[i]].name,clobber=clobber,file=file,unlock=unlock,$
                    /nowait,psflibrary=librarypsf
       endfor
      endif
    endelse
  endif

  ;; Make LSF calibration file
  ;;--------------------------
  if keyword_set(lsf) then begin
    print,'makecal lsf: ', lsf
    if lsf gt 1 then begin
      file = apogee_filename('LSF',num=lsf,/nochip)
      file = file_dirname(file)+'/'+file_basename(file,'.fits')
      slsfid = string(lsf,format='(i08)')
      lsfdir = apogee_filename('LSF',num=lsf,chip='c',/dir)
      allfiles = lsfdir+dirs.prefix+'LSF-'+chips+'-'+slsfid+'.fits'
      allfiles = [allfiles,lsfdir+dirs.prefix+'LSF-'+slsfid+'.sav']
      if total(file_test(allfiles)) eq 4 and not keyword_set(clobber) then begin
        print,' lsf file: ',file+'.sav',' already made'
        return
      endif
      i = where(lsfstr.name eq lsf,nlsf)
      if nlsf le 0 then begin
        print,'No matching calibration line for ', lsf
        stop
      endif
      ims = getnums(lsfstr[i[0]].frames)
      cmjd = getcmjd(ims[0],mjd=mjd)
      GETCAL,mjd,calfile,darkid=darkid,flatid=flatid,multiwaveid=waveid,fiberid=fiberid
      MAKECAL,multiwave=waveid,unlock=unlock
      MKLSF,ims,waveid,darkid=darkid,flatid=flatid,psfid=lsfstr[i[0]].psfid,fiberid=fiberid,$
            full=full,newwave=newwave,clobber=clobber,pl=pl,unlock=unlock
    endif else begin
      if keyword_set(mjd) then  begin
        num = getnum(mjd) 
        red = where(lsfstr.frames/10000L eq num)
      endif else red=indgen(n_elements(lsfstr))
      if (red[0] ge 0) then begin
        for i=0,n_elements(red)-1,nskip do begin
          ims = getnums(lsfstr[red[i]].frames)
          cmjd = getcmjd(ims[0],mjd=mjd)
          GETCAL,mjd,calfile,darkid=darkid,flatid=flatid,multiwaveid=waveid,fiberid=fiberid
          MAKECAL,multiwave=waveid,unlock=unlock,librarypsf=librarypsf
          print,'calling mklsf'
          MKLSF,ims,waveid,darkid=darkid,flatid=flatid,psfid=lsfstr[i].psfid,fiberid=fiberid,$
                full=full,newwave=newwave,clobber=clobber,pl=pl,unlock=unlock,/nowait
        endfor
      endif
    endelse
  endif

end
