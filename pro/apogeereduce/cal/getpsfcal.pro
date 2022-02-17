;+
;
; GETPSFCAL
;
; Find the best PSF calibration file an exposure.
;
; INPUTS:
;  num    Eight digit Exposure number.
;
; OUTPUTS:
;  psfid  Exposure number of the PSF calibration file.
;
; USAGE:
;  IDL>psfid = getpsfcal(40620001)
;
; By D.Nidever  Feb 2022
;-

function getpsfcal,num

  ;; Not enough inputs
  if n_elements(num) eq 0 then begin
    print,'Syntax - psfid=getpsfcal(num)'
    return,-1
  endif

  cmjd = getcmjd(num[0],mjd=mjd)
  if mjd ge 59556 then fps=1 else fps=0

  ;; Try to find a PSF from this day
  pname = apogee_filename('PSF',num=strmid(strtrim(num[0],2),0,4)+'????',chip='a')
  psffiles = file_search(pname,count=npsffiles)
  if npsffiles eq 0 then begin
    print,'No PSF calibration file found for MJD=',cmjd
    return,-1
  endif
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

  return,psfid

end
