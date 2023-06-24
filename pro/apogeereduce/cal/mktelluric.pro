;+
;
; MKTELLURIC
;
; Procedure to make an APOGEE daily telluric calibration file.
;
; INPUT:
;  tellid       Telluric ID, WAVEID-LSFID.
;  /nowait      If file is already being made then don't wait
;                 just return.
;  /clobber     Overwrite existing files.
;  /unlock      Delete the lock file and start fresh.
;
; OUTPUT:
;  A set of apTelluric-[abc]-WAVEID-LSFID.fits files in the appropriate location
;   determined by the SDSS/APOGEE tree directory structure.
;
; USAGE:
;  IDL>mktelluric,tellid,/clobber
;
; Made from mkwave.pro by D.Nidever, March 2022
;-

pro mktelluric,tellid,clobber=clobber,nowait=nowait,unlock=unlock

  name = strtrim(tellid,2)
  dirs = getdir(apodir,caldir,spectrodir,vers)
  telldir = apogee_filename('Telluric',num=0,chip='a',/dir)
  file = dirs.prefix+'Telluric-'+name
  tellfile = telldir+file
  ;;lockfile = telldir+file+'.lock'
  
  ;; If another process is alreadying make this file, wait!
  ;;if not keyword_set(unlock) then begin
  ;;  while file_test(lockfile) do begin
  ;;    if keyword_set(nowait) then return
  ;;    apwait,file,10
  ;;  endwhile
  ;;endif else begin
  ;;  if file_test(lockfile) then file_delete,lockfile,/allow
  ;;endelse
  aplock,tellfile,waittime=10,unlock=unlock
  
  ;; Does product already exist?
  ;; check all three chips and .dat file
  chips = ['a','b','c']
  allfiles = telldir+dirs.prefix+'Telluric-'+chips+'-'+name+'.fits'
  if total(file_test(allfiles)) eq 3 and not keyword_set(clobber) then begin
    print,' Telluric file: ', telldir+file, ' already made'
    return
  endif
  file_delete,allfiles,/allow  ;; delete any existing files to start fresh

  print,'Making telluric: ', name
  ;; Open .lock file
  ;;openw,lock,/get_lun,lockfile
  ;;free_lun,lock
  aplock,tellfile,/lock
  
  ;; Get the wavelength calibration files
  waveid = long((strsplit(tellid,'-',/extract))[0])
  wavedir = apogee_filename('Wave',num=0,chip='a',/dir)
  wavefiles = wavedir+'/'+dirs.prefix+'Wave-'+chips+'-'+strtrim(waveid,2)+'.fits'
  if total(file_test(wavefiles),/int) ne 3 then begin
    print,'Wave '+strtrim(waveid,2)+' files not all found'
    file_delete,lockfile,/allow
    return
  endif
  ;; Get the LSF calibration file
  lsfid = long((strsplit(tellid,'-',/extract))[1])  
  lsffiles = apogee_filename('LSF',num=lsfid,chip=chips)
  if total(file_test(lsffiles),/int) ne 3 then begin
    print,'LSF '+strtrim(lsfid,2)+' files not all found'
    file_delete,lockfile,/allow
    return
  endif
  
  ;; aptelluric_convolve will return the array of LSF-convolved telluric spectra appropriate
  ;;   for the specific wavelength solutions of this frame
  ;; There are 3 species, and there may be models computed with different "scale" factor, i.e.
  ;;   columns and precipitable water values. If this is the case, we fit each star not
  ;;   only for a scaling factor of the model spectrum, but also for which of the models
  ;;   is the best fit. For self-consistency, we adopt the model for each species that provides
  ;;   the best fit for the largest number of stars, and then go back and refit all stars
  ;;   with this model
  ;; The telluric array is thus 4D: [npixels,nfibers,nspecies,nmodels]

  ;; Construct fake frame with wavefile, lsffile, flux, and lsfcoef
  flux = fltarr(2048,300)
  FITS_READ,lsffiles[0],lsfcoef1
  aframe = {wavefile:wavefiles[0],lsffile:lsffiles[0],lsfcoef:lsfcoef1,flux:flux}
  FITS_READ,lsffiles[1],lsfcoef2
  bframe = {wavefile:wavefiles[1],lsffile:lsffiles[1],lsfcoef:lsfcoef2,flux:flux}
  FITS_READ,lsffiles[2],lsfcoef3
  cframe = {wavefile:wavefiles[2],lsffile:lsffiles[2],lsfcoef:lsfcoef3,flux:flux}
  frame = {chipa:aframe,chipb:bframe,chipc:cframe}

  ;; Make the telluric calibration file
  convolved_telluric = APTELLURIC_CONVOLVE(frame,fiber=lindgen(300),/convonly,unlock=unlock,clobber=clobber)
  
  ;; Check that the calibration file was successfully created
  tellfiles = apogee_filename('Telluric',num=tellid,chip=chips)
  if total(file_test(outfile),/int) eq 3 then begin
    print,'Telluric file '+dirs.prefix+'Telluric-'+strtrim(tellid,2)+' completely successfully'
    openw,lock,/get_lun,telldir+file+'.dat'
    free_lun,lock
  endif else print,'PROBLEMS with apTelluric-'+strtrim(tellid,2)+' files'

  ;;file_delete,lockfile,/allow
  aplock,tellfile,/clear
  
end
