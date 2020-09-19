;+
;
; READCALSTR
;
; Returns the name of calibration product in input structure to
; be used for a specified MJD/date.
;
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

  ;; Check for rows that are in the right date range
  gd = where(mjd ge str.mjd1 and mjd le str.mjd2,ngd)
  ;; Some found
  if ngd gt 0 then begin
    ;; More than one, use the last
    if ngd gt 1 then begin
      gd = gd[ngd-1]
      print,'Multiple cal products found for mjd ', mjd, ' will use last: ',str[gd[0]].name      
    endif
    return,str[gd[0]].name

  ;; None found, return 0L
  endif else begin
    return,0L
  endelse

;; Old code
;  n = 0
;  for i=0,n_elements(str)-1 do begin
;    if mjd ge str[i].mjd1 and mjd le str[i].mjd2 then begin
;      ret=i & n+=1
;    endif
;  endfor
;  if n eq 0 then begin
;;    print,'No cal product found for mjd ', mjd
;;    stop
;    return,0L
;  endif else if n gt 1 then begin
;    print,'Multiple cal products found for mjd ', mjd, ' will use last: ',str[ret].name
;    stop
;  endif
;  return,str[ret].name

end

