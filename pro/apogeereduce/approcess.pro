;+
;
; APPROCESS
;
; A wrapper around AP3DPROC.PRO and AP2DPROC.PRO.  Mostly used for
; calibration products.
;
; INPUTS:
;  nums            Array of exposure ID8 numbers.
;  =cmjd           The MJD of the output directory to use.
;  /clobber        Overwrite Existing files.
;  /onedclobber    Overwrite any existing 1D files.  If /clobber is
;                    set then /onedclobber is set automatically. 
;  =detid          Detector calibration frame to be used in reduction.
;  =darkid         Dark calibration frame to be used in reduction.
;  =flatid         Flat calibration frame to be used in reduction.
;  =traceid        Trace calibration frame to be used in reduction.
;  =psfid          PSF calibration frame to be used in reduction.
;  =fluxid         Flux calibration frame to be used in reduction.
;  =waveid         Wavelength calibration frame to be used in reduction.
;  =littrowid      Littrow calibration frame to be used in reduction.
;  =persistid      Persistence calibration frame to be used in reduction.
;  /nocr           Input to ap3dproc.pro.  Do not detect and fix CRs.
;  /stp            Stop at the end of the program.
;  =jchip          Only process this chip (0, 1 or 2).
;  /nfs            Input to ap3dproc.pro.  Number of samples to use
;                    for Fowler sampling.  Default is to use
;                    up-the-ramp sampling.
;  /doproc         Perform both 3D->2D and 2D->1D processing.
;  /doap3dproc     Perform 3D->2D processing.
;  /doap2dproc     Perform 2D->1D processing.
;  =logfile        Name of a logfile to use.
;  =outdir         The output directory.
;  =maxread        The maximum number of reads to use in ap3dproc.
;                    Can be 3-element array for the three chips.
;  /skywave        Input to ap2dproc.pro. Enable a pixel-shift to
;                    wavelength solution based on sky lines.
;  =step           Obsolete parameter.
;  /nofs           Obsolete parameter.
;
; OUTPUTS:
;  none
;
; USAGE:
;  IDL>im=approcess(nums)
;
; By J. Holtzman, 2011
;  Doc string updates and cleanup, used new data model  by D. Nidever, Sep 2020
;-

; approcess reduces a sequence of images, all 3 chips, and writes out
function approcess,nums,cmjd=cmjd,clobber=clobber,onedclobber=onedclobber,detid=detid,$
                   darkid=darkid,flatid=flatid,traceid=traceid,psfid=psfid,fluxid=fluxid,$
                   waveid=waveid,littrowid=littrowid,persistid=persistid,nocr=nocr,stp=stp,$
                   jchip=jchip,nfs=nfs,nofs=nofs,doproc=doproc,doap3dproc=doap3dproc,$
                   doap2dproc=doap2dproc,logfile=logfile,outdir=outdir,maxread=maxread,$
                   skywave=skywave,step=step,unlock=unlock

common savedepsf, savedepsffiles, epsfchip
savedepsffiles = [' ',' ',' ']
epsfchip = 0

if n_elements(step) eq 0 then step=1
if n_elements(stp) eq 0 then stp=0
if n_elements(nocr) eq 0 then nocr=0
if n_elements(nofs) eq 0 then nofs=0
if n_elements(nfs) eq 0 then nfs=0
if n_elements(detid) eq 0 then detid=0
if n_elements(darkid) eq 0 then darkid=0
if n_elements(flatid) eq 0 then flatid=0
if n_elements(littrowid) eq 0 then littrowid=0
if n_elements(persistid) eq 0 then persistid=0
if n_elements(traceid) eq 0 then traceid=0
if n_elements(psfid) eq 0 then psfid=0
if n_elements(fluxid) eq 0 then fluxid=0
if n_elements(waveid) eq 0 then waveid=0
if keyword_set(clobber) then onedclobber=1
if n_elements(maxread) eq 0 then maxread=[0,0,0]

dirs = getdir(apodir,caldir,spectrodir,vers,datadir=datadir)
leadstr = 'APPROCESS: '

print,'Using detector: ',detid,' dark: ', darkid,' flat: ', flatid, ' trace: ', traceid, ' psf: ',psfid
if keyword_set(jchip) then begin
  j1=jchip & j2=jchip
endif else begin
  j1=0 & j2=2
endelse
chip = ['a','b','c']


;; Perform Processing
;;-------------------
if keyword_set(doproc) or keyword_set(doap3dproc) then begin 
  ;; 3D -> 2D Processing
  ;;--------------------
  ; use approcess as front end for ap3dproc and ap2dproc only 
  for ichip=j1,j2 do begin
    ; set up calibration file names
    if darkid gt 0 then $
      bpmcorr = apogee_filename('BPM',num=darkid,chip=chip[ichip])
    if darkid gt 0 then $
      darkcorr = apogee_filename('Dark',num=darkid,chip=chip[ichip])
    if flatid gt 0 then $
      flatcorr = apogee_filename('Flat',num=flatid,chip=chip[ichip])
    apgundef,littrowcorr
    if littrowid gt 0 and ichip eq 1 then $
      littrowcorr = apogee_filename('Littrow',num=littrowid,chip=chip[ichip])
    if persistid gt 0 then $
      persistcorr = apogee_filename('Persist',num=persistid,chip=chip[ichip])
    ;; Exposure loop
    for inum=0,n_elements(nums)-1 do begin
      num = nums[inum]
      if n_elements(cmjd) eq 0 then cmjd=getcmjd(long(num))
      if n_elements(outdir) eq 0 then outdir=dirs.expdir+cmjd+'/'
      ifile = apogee_filename('R',num=num,chip=chip[ichip],mjd=cmjd)
      ofile = apogee_filename('2D',num=num,chip=chip[ichip])
      if file_test(outdir,/directory) eq 0 then file_mkdir,outdir
      if nfs eq 0 then uptheramp=1 else uptheramp=0
      print,'Calling ap3dproc...'
      AP3DPROC,ifile,ofile,flatcorr=flatcorr,darkcorr=darkcorr,bpmcorr=bpmcorr,$
               littrowcorr=littrowcorr,persistcorr=persistcorr,nocr=nocr,$
               uptheramp=uptheramp,nfowler=nfs,fitsdir=getlocaldir(),$
               clobber=clobber,maxread=maxread[ichip],unlock=unlock
    endfor
  endfor

  ;; Perform 2D -> 1D Processing
  ;;----------------------------
  ; ap2dproc does all 3 chips together
  if keyword_set(doproc) then begin
    ;; Exposure loop
    for inum=0,n_elements(nums)-1 do begin
      num = nums[inum]
      tracefile = apogee_filename('PSF',num=psfid,chip='c',/dir)+string(format='(i8.8)',psfid) 
      wavefile = 0
      if keyword_set(waveid) then wavefile=apogee_filename('Wave',num=waveid,chip='c',/dir)+string(format='(i8.8)',waveid)
      wavedir = apogee_filename('Wave',num=0,chip='a',/dir)
      expdir = apogee_filename('2D',num=num,chip='a',/dir)
      AP2DPROC,expdir+'/'+string(format='(i8.8)',num),$
               tracefile,4,outdir=outdir,wavefile=wavefile,$
               clobber=clobber,skywave=skywave,unlock=unlock

      chiptag = ['a','b','c']
      files = apogee_filename('2D',num=num,chip=chiptag)
      modfiles = apogee_filename('2Dmodel',num=num,chip=chiptag)
      for jj=0,n_elements(files)-1 do begin
         if file_test(files[jj]) then begin
           file_delete,files[jj]+'.fz',/allow_nonexistent
           ;SPAWN,['fpack','-D','-Y',files[jj]],/noshell
         endif
         if file_test(modfiles[jj]) then begin
           file_delete,modfiles[jj]+'.fz',/allow_nonexistent
           SPAWN,['fpack','-D','-Y',modfiles[jj]],/noshell
         endif
      endfor

    endfor
  endif  ; /doproc

endif

end


