;+
;
; MKBPM
;
; Create an APOGEE bad pixel mask calibration file.
;
; INPUTS:
;  bpmid     ID8 number for this bad pixel mask.
;  =darkid   ID8 number for the dark to use.
;  =flatid   ID8 number for the flat to use.
;  =badrow   Array of known bad rows
;  /clobber  Overwrite existing files.
;  /unlock   Delete lock file and start fresh 
;
; OUTPUTS:
;  BPM files (e.g. apBPM-a-ID8.fits) are created in the appropriate
;  directory using the SDSS/APOGEE tree structure.
;
; USAGE:
;  IDL>mkbpm,12345678,darkid=00110012,flatid=00110013
;
; By J. Holtzman, 2011?
;  Added doc strings and cleanup.  D. Nidever, Sep 2020
;-

pro mkbpm,bpmid,darkid=darkid,flatid=flatid,badrow=badrow,clobber=clobber,unlock=unlock

 dirs = getdir()
 file = apogee_filename('BPM',num=bpmid,chip='c')
 lockfile = file+'.lock'

 ;; If another process is alreadying make this file, wait!
 if not keyword_set(unlock) then begin
   while file_test(lockfile) do apwait,file,10
 endif else begin
   if file_test(lockfile) then file_delete,lockfile,/allow
 endelse

 ;; Does product already exist?
 if file_test(file) and not keyword_set(clobber) then begin
   print,' BPM file: ', file, ' already made'
   return
 endif

 print,'Making BPM: ', bpmid
 ;; Open .lock file
 openw,lock,/get_lun,lockfile
 free_lun,lock

 chips = ['a','b','c']

 for ichip=0,2 do begin
   chip = chips[ichip]

   mask = intarr(2048,2048)

   ;; Bad pixels from dark frame
   file = apogee_filename("Dark",chip=chip,num=darkid)
   darkmask = mrdfits(file,3)
   bad = where(darkmask gt 0)
   mask[bad] = mask[bad] or maskval('BADDARK')

   ;; Bad pixels from flat frame
   if flatid gt 0 then begin
     file = apogee_filename("Flat",chip=chip,num=flatid)
     flatmask = mrdfits(file,3)
     bad = where(flatmask gt 0,nbad)
     if nbad gt 0 then mask[bad]=mask[bad] or maskval('BADFLAT')
   endif else flatmask=darkmask*0

   ;; Flag them both as bad pixel (for historical compatibility?)
   bad = where((darkmask or flatmask) gt 0,nbad)
   if nbad gt 0 then mask[bad] = mask[bad] or maskval('BADPIX')
   if keyword_set(badrow) then begin
     for i=0,n_elements(badrow)-1 do begin
       if badrow.chip eq ichip then mask[*,badrow.row]=mask[*,badrow.row] or maskval('BADPIX')
     endfor
   endif
   file = apogee_filename('BPM',chip=chip,num=bpmid)
   MWRFITS,mask,file,/create
 endfor

 file_delete,lockfile

end
