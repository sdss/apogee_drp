;+
;
; GETPSFCAL
;
; Find the best PSF calibration file an exposure, or make
; a new one.
;
; INPUTS:
;  num          Eight digit Exposure number.
;  /psflibrary  Use the PSF library to find the best PSF to use.
;                 Used for FPS data by default.  Otherwise it
;                 searches through the apPSF files that are on disk.
;
; OUTPUTS:
;  psfid  Exposure number of the PSF calibration file.
;
; USAGE:
;  IDL>psfid = getpsfcal(40620001)
;
; By D.Nidever  Feb 2022
;-

function getpsfcal,num,psflibrary=psflibrary
  common apver,ver,telescop,instrume

  ;; Not enough inputs
  if n_elements(num) eq 0 then begin
    print,'Syntax - psfid=getpsfcal(num)'
    return,-1
  endif

  cmjd = getcmjd(num[0],mjd=mjd)
  if mjd ge 59556 then fps=1 else fps=0
  ;; Use PSF library for FPS by default
  if n_elements(psflibrary) eq 0 and keyword_set(fps) then psflibrary=1

  ;; Use the PSF library
  if keyword_set(psflibrary) then begin
    print,'Using PSF Library'
    ;; You can do "domeflattrace --mjdplate" where mjdplate could be
    ;; e.g. 59223-9244, or "domeflattrace --planfile", with absolute
    ;; path of planfile
    observatory = strlowcase(strmid(telescop,0,3))
    spawn,['psflibrary',observatory,'--ims',strtrim(num,2)],out,errout,/noshell
    nout = n_elements(out)
    ;;for f=0,nout-1 do print,out[f]
    ;; Parse the output
    lo = where(stregex(out,'^PSF FLAT RESULTS:',/boolean) eq 1,nlo)
    hi = first_el(where(strtrim(out,2) eq '' and lindgen(nout) gt lo[0]))
    if lo ne -1 and hi ne -1 then begin
      outarr = strsplitter(out[lo+1:hi-1],' ',/extract)
      ims = reform(outarr[0,*])
      psfflatims = reform(outarr[1,*])
      psfid = psfflatims[0]
      print,'psf: ',psfid
      return,psfid
    endif else begin
      print,'Problem running psflibrary for ',strtrim(num,2)
    endelse
  endif


  ;; Try to find a PSF from this day
  pname = apogee_filename('PSF',num=strmid(strtrim(num[0],2),0,4)+'????',chip='a')
  psffiles = file_search(pname,count=npsffiles)

  ;; Some apPSF files found
  ;;-----------------------
  if npsffiles gt 0 then begin

    ;; Find closest one
    ;;  make sure it's from a QUARTZFLAT for FPS
    psfids = lonarr(npsffiles)
    for j=0,npsffiles-1 do psfids[j]=strmid(file_basename(psffiles[j]),8,8)
    if keyword_set(fps) then begin
      exptype = strarr(npsffiles)
      for i=0,npsffiles-1 do begin
        if file_test(psffiles[i]) eq 1 then begin
          head = headfits(psffiles[i],exten=0)
          exptype[i] = sxpar(head,'exptype')
        endif
      endfor
      gd = where(exptype eq 'QUARTZFLAT',ngd)
      if ngd eq 0 then begin
        print,'No QUARTZFLAT PSF calibration file found for MJD=',cmjd
        return,-1
      endif
      psffiles = psffiles[gd]
      psfids = psfids[gd]
    endif
    si = sort(abs(long(psfids-long(num[0]))))
    psfid = psfids[si[0]]

  ;; None found, try to make one
  ;;----------------------------
  endif else begin
    print,'No PSF calibration file found for MJD=',cmjd,'. Trying to make one.'
    ;; Try to make a PSF file
    psfinfo = dbquery("select * from apogee_drp.exposure where mjd="+cmjd+$
                      " and exptype='DOMEFLAT' or exptype='QUARTZFLAT'",count=npsfinfo)
    if npsfinfo eq 0 then begin
      print,'No DOMEFLAT or QUARTZFLAT exposure for MJD=',cmjd
      return, -1
    endif
    ;;  Make sure it's from a QUARTZFLAT for FPS
    gd = where(psfinfo.exptype eq 'QUARTZFLAT',ngd)
    if ngd eq 0 then begin
      print,'No QUARTZFLAT PSF calibration file found for MJD=',cmjd
      return,-1
    endif
    psfinfo = psfinfo[gd]

    ;; Find quartzflat/domeflat that is closest to the waveid
    si = sort(abs(long(psfinfo.num-long(num))))
    psfid = psfinfo[si[0]].num
    MAKECAL,psf=psfid,unlock=unlock

    ;; Check that the apPSF files exist
    pname = apogee_filename('PSF',num=psfid,chip=['a','b','c'])
    if total(file_test(pname)) ne 3 then begin
      print,repstr(apogee_filename('PSF',num=psfid,chip='a'),'-a-','-'),' NOT FOUND'
      return,-1
    endif

  endelse

  print,'psf: ',psfid
  return,psfid

end
