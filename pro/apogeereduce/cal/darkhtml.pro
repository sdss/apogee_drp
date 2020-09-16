;+
;
; DARKHTML
;
; Makes a simple diagnostic HTML page for an APOGEE superdark
; calibration product.
;
; INPUTS:
;  darkdir  The dark calibration directory.
;
; OUTPUTS:
;  An html file is created with name darkdir/html/darks.html
;
; USAGE:
;  IDL>darkhtml,darkdir
;
; By J. Holtzman, 2011?
; Added doc strings and cleanup by D. Nidever, Sep 2020
;-

pro darkhtml,darkdir

darks = file_search(darkdir+'/*.tab')
if not file_test(darkdir+'/html',/dir) then file_mkdir,darkdir+'/html'
openw,lun,/get_lun,darkdir+'/html/darks.html'
printf,lun,'<HTML><BODY><TABLE BORDER=1>'
printf,lun,'<TR><TD>ID'
printf,lun,'<TD>CHIP'
printf,lun,'<TD>NREADS'
printf,lun,'<TD>NFRAMES'
printf,lun,'<TD>MEDIAN RATE'
printf,lun,'<TD>NSAT'
printf,lun,'<TD>NHOT'
printf,lun,'<TD>NHOTNEIGH'
printf,lun,'<TD>NBAD'
printf,lun,'<TD>NNEG'
chips=['a','b','c']
for i=0,n_elements(darks)-1 do begin
  darklog = mrdfits(darks[i],1)
  for ichip=0,2 do begin
    if ichip eq 0 then printf,lun,'<TR><TD>',darklog[ichip].num else printf,lun,'<TR><TD>'
    printf,lun,'<TD><center>',chips[ichip]
    printf,lun,'<TD><center>',darklog[ichip].nreads
    printf,lun,'<TD><center>',darklog[ichip].nframes
    printf,lun,'<TD><center>',darklog[ichip].medrate
    printf,lun,'<TD><center>',darklog[ichip].nsat
    printf,lun,'<TD><center>',darklog[ichip].nhot
    printf,lun,'<TD><center>',darklog[ichip].nhotneigh
    printf,lun,'<TD><center>',darklog[ichip].nbad
    printf,lun,'<TD><center>',darklog[ichip].nneg
    file = string(format='("apDark-",a,"-",i8.8)',chips[ichip],darklog[ichip].num) 
    printf,lun,'<TD><center><a href=../plots/'+file+'.jpg>Image</a>'
    printf,lun,'<TD><center><a href=../plots/'+file+'.gif>Plots</a>'
  endfor
endfor
printf,lun,'</table></body></html>'
free_lun,lun

end
