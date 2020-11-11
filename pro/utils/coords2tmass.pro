;+
;
; COORDS2TMASS
;
; Construct 2MASS name from the RA/DEC coordinates
;
; INPUTS:
;  RA      RA in degrees.
;  DEC     DEC in degrees.
;
; OUTPUTS:
;  name    The 2MASS-style name
;
; USAGE:
;  IDL>name = coord2tmass(ra,dec)
;
; By D.Nidever,  Nov 2020
;-

function coords2tmass,ra,dec

  if n_elements(ra) eq 0 or n_elements(dec) eq 0 then begin
    print,'Syntax - name = coords2tmass(ra,dec)'
    return,-1
  endif

  ;; apogeetarget/pro/make_2mass_style_id.pro makes these
  ;; APG-Jhhmmss[.]ss±ddmmss[.]s
  ;; http://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec1_8a.html

  ;; 2M00034301-7717269
  ;; RA: 00034301 = 00h 03m 43.01s
  ;; DEC: -7717269 = -71d 17m 26.9s
  ;; 8 digits for RA and 7 digits for DEC
  raarr = sixty(ra/15.0)
  name = string(raarr[0],raarr[1],round(raarr[2]*100),format='(i02,i02,i04)')
  if dec lt 0 then name+='-' else name+='+'  ;; add sign
  decarr = sixty(abs(dec))
  name += string(decarr[0],decarr[1],round(decarr[2]*10),format='(i02,i02,i03)')

  return,name

end
