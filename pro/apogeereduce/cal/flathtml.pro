;+
;
; FLATHTML
;
; Makes a simple diagnostic HTML page for an APOGEE superflat
; calibration product.
;
; INPUTS:
;  flatdir  The flat calibration directory.
;
; OUTPUTS:
;  An html file is created with name flatdir/html/flats.html
;
; USAGE:
;  IDL>flathtml,flatdir
;
; By J. Holtzman, 2011?
; Added doc strings and cleanup by D. Nidever, Sep 2020
;-

pro flathtml,flatdir,plots=plots

flats = file_search(flatdir+'/*.tab')
if not file_test(flatdir+'/html') then file_mkdir,flatdir+'/html'
openw,lun,/get_lun,flatdir+'/html/flats.html'
printf,lun,'<HTML><BODY><TABLE BORDER=1>'
printf,lun,'<TR><TD>ID'
printf,lun,'<TD>NFRAMES'
printf,lun,'<TD>A<TD> B<TD> C'
chips = ['a','b','c']
for i=0,n_elements(flats)-1 do begin
  flatlog = mrdfits(flats[i],1)
  for ichip=0,2 do begin
    if keyword_set(plots) then begin
      file = flatdir+'/'+flats[i].name
      flat = mrdfits(file+'.fits',1)
      FLATPLOT,flat,file
    endif
    if ichip eq 0 then begin
      printf,lun,'<TR><TD>',flatlog[ichip].num 
      printf,lun,'<TD><center>',flatlog[ichip].nframes
    endif
    file = string(format='("apFlat-",a,"-",i8.8)',chips[ichip],flatlog[ichip].num) 
    printf,lun,'<TD><center><a href=../plots/'+file+'.jpg><img src=../plots/'+file+'.jpg width=100></a>'
  endfor
endfor
printf,lun,'</table></body></html>'
free_lun,lun

end
