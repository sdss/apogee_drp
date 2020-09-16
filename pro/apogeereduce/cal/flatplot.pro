;+
;
; FLATPLOT
;
; Create diagnostic plot for APOGEE flat calibration files.
;
; INPUTS:
;  flat  The flat 2D image.
;  file  The name of the output file (no suffix/extension).
;
; OUTPUTS:
;  A diagnostic plot is saved to file.jpg
;
; USAGE:
;  IDL>flatplot,flat,file
;
;-

pro flatplot,flat,file

 ;; Make a jpg of the flat
 set_plot,'ps'
 device,file='a.eps',/encap,xsize=15,ysize=15
 low = 0.5
 high = 1.5
 disp = flat>low<high
 disp = 255*(disp-low)/(high-low)
 tv,nint(disp)
 device,/close
 spawn,'convert a.eps '+file+'.jpg'
 file_delete,'a.eps'
 set_plot,'X'

end
