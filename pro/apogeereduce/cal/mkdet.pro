;+
;
; MKDET
;
; Make an APOGEE detector calibration product.
;
; INPUTS:
;  detid    ID8 number for the detector file.
;  linid    ID8 number for the linearity file.
;  /unlock  Delete lock file and start fresh 
;
; OUTPUTS:
;  Detector files (e.g. apDetector-a-ID8.fits) are created in the
;  appropriate directory using the SDSS/APOGEE tree structure. 
;
; USAGE:
;  IDL>mkdet,detid,linid
;
; By J. Holtzman, 2011?
;  Added doc strings, updates to use data model  D. Nidever, Sep 2020
;-

pro mkdet,detid,linid,unlock=unlock

 dirs = getdir()
 caldir = dirs.caldir
 detfile = apogee_filename('Detector',num=detid,chip='c')
 lockfile = detfile+'.lock'
 ;; If another process is already making this file, wait!
 if not keyword_set(unlock) then begin
   while file_test(lockfile) do apwait,lockfile,10
 endif else begin
   if file_test(lockfile) then file_delete,lockfile,/allow
 endelse

 ;; Does product already exist?
 print,'testing detector file: ',detfile
 if file_test(detfile) and not keyword_set(clobber) then begin
   print,' Detector file: ', detfile, ' already made'
   return
 endif

 print,'Making Detector: ', detid
 ; open .lock file
 openw,lock,/get_lun,lockfile
 free_lun,lock

 lincorr = fltarr(4,3)
 for iquad=0,3 do lincorr[iquad,*]=[1.,0.,0.]
 if n_elements(linid) gt 0 then if linid gt 0 then begin
   linpar = mklinearity(linid)
   for iquad=0,3 do lincorr[iquad,*]=linpar
 endif

 if dirs.instrument eq 'apogee-n' then begin
   g = 1.9
   ;; These are supposed to be CDS DN!
   r = [20.,11.,16.] /sqrt(2.)
   ;; 10/17 analysis get 12, 9, 10 single read in DN
   r = [12,8,8] ; actually achieved CDS
   r = [13,11,10] ; equivalent single read in UTR analysis (which gives lower rn overall)
 endif else begin
   g = 3.0
   ;; These are supposed to be CDS DN!
   r = [15.,15.,15.] /sqrt(2.)
   ;; 10/17, get 6, 8, 4.5 single read in DN
   r = [4,5,3]   ; actually achieved CDS
   r = [7,8,4]   ; equivalent single read in UTR analysis (which gives lower rn overall)
   ;; JCW 2/28/17 email
   ;;Our current measurements are (blue, green, red):
   ;;
   ;;gain (e-/ADU): 3.0, 3.3, 2.7
   ;;read noise (e-): 9.3, 15.2, 8.6
   ;;dark current (e-/sec): 0.011, 0.014, 0.008
 endelse

 chips = ['a','b','c']
 for ichip=0,2 do begin
   gain = [g,g,g,g]
   rn = [r[ichip],r[ichip],r[ichip],r[ichip]]*gain
   file = apogee_filename('Detector',num=detid,chip=chips[ichip])
   mkhdr,head,0
   MWRFITS,0,file,head,/create
   MWRFITS,rn,file
   MWRFITS,gain,file
   MWRFITS,lincorr,file
 endfor

 file_delete,lockfile,/allow
end
