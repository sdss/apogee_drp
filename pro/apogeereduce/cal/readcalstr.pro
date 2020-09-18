;+
;
; READCALSTR
;
; Returns the name of calibration product in input structure to
; be used for specified date.

; INPUT
;   str  Name of a calibration structure with minimum tags
;          mjd1, mjd2, name.
;   mjd  Desired MJD
; 
; OUTPUT:
;   name  Name of calibration product valid for input MJD
;
; USAGE:
;  IDL>name=readcalstr(str,mjd)
;
; Written by J.Holtzman Aug 2011
;  Broke into separate file and cleaned up docs by D. Nidever, Sep 2020
;-

function readcalstr,str,mjd

  n=0
  for i=0,n_elements(str)-1 do begin
    if mjd ge str[i].mjd1 and mjd le str[i].mjd2 then begin
      ret=i & n+=1
    endif
  endfor
  if n eq 0 then begin
;    print,'No cal product found for mjd ', mjd
;    stop
    return,0L
  endif else if n gt 1 then begin
    print,'Multiple cal products found for mjd ', mjd, ' will use last: ',str[ret].name
    stop
  endif
  return,str[ret].name
end

