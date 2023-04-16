;+
;
; APMKPSF
;
; This program makes apTrace files from regular flat frames.
; It fits polynomials to the fiber traces and the flat flux
; and creates a normalized PSF 2D image.
;
; INPUTS:
;  flatframe   The directory and ID8 of the 2D flat frame
;                concatenated.
;  outdir      The directory to write the apTrace files to.
;                This should normally be SPECTRO_DIR/cal/trace/
;  =fiberid    ID8 number for the ETrace calibration file to use. 
;  /no_epsf    Don't create the empirical PSF.
;  =peakthresh The threshold to use for finding peaks/fibers.
;  /pl         Plot the fits
;  /clobber    Overwrite the output file if it already exists
;  /unlock     Delete lockfile and start fresh.
;  /verbose    Verbose output to the screen.
;  /silent     Don't print anything to the screen
;  /stp        Stop at the end of the program
;
; OUTPUTS:
;  The apTrace files are written to the "outdir" directory.
;
; USAGE:
;  IDL>apmkpsf,flatframe,outdir
;
; By D.Nidever  July 2010
;-

pro apmkpsf,flatframe,outdir,no_epsf=no_epsf,pl=pl,clobber=clobber,$
            peakthresh=peakthresh,verbose=verbose,silent=silent,stp=stp,$
            sparseid=sparseid,fiberid=fiberid,average=average,unlock=unlock

t0 = systime(1)
if not keyword_set(sparseid) then sparseid=0 else makecal,sparse=sparseid
if not keyword_set(fiberid) then fiberid=0 else makecal,fiber=fiberid

; Get APOGEE directories
dirs = getdir(apogee_dir,cal_dir,spectro_dir,apogee_vers)

; Not enough inputs
nflatframe = n_elements(flatframe)
noutdir = n_elements(outdir)
if nflatframe eq 0 or noutdir eq 0 then begin
  print,'Syntax - apmkpsf,flatframe,outdir,no_epsf=no_epsf,pl=pl,clobber=clobber,verbose=verbose,silent=silent,stp=stp'
  return
endif

print,'Processing ',strtrim(nflatframe,2),' Flat Frames'

chiptag = ['a','b','c']

;; Loop through the flat frames
FOR i=0,nflatframe-1 do begin

  iflatframe = flatframe[i]
  flatdir = file_dirname(iflatframe)+'/'
  flatid = file_basename(iflatframe)
  flatframeid = string(long(flatid),format='(I08)')

  print,''
  print,'------------------------------------------------------'
  print,strtrim(i+1,2),'/',strtrim(nflatframe,2),'  Creating Trace/PSF for Frame Number >>',strtrim(flatframeid,2),'<<'
  print,'------------------------------------------------------'

  ;; Double-check that the file exists in the SP2D directory
  files = apogee_filename('2D',num=flatframeid,chip=['a','b','c'])   ; chip filenames
  info = APFILEINFO(files,/silent)
  okay = (info.exists AND info.sp2dfmt AND info.allchips AND ((info.naxis eq 3) or (info.exten eq 1)))
  if min(okay) lt 1 then begin
    bd = where(okay eq 0,nbd)
    print,'Not halted: There is a problem with files: ',strjoin((files)(bd),' ')
    stop
    goto,BOMB
  endif

  ;; Load the frame
  APLOADFRAME,files,str

  szflux = size(str.(0).flux)
  nx = szflux[1]
  ny = szflux[2]

  ;; Loop through the chips
  For j=0,2 do begin

    ;; Does the output file already exist?
    outfile = apogee_filename('PSF',chip=chiptag[j],num=flatframeid)
    lockfile = outfile+'.lock'
    ;; if another process is creating file, wait
    if not keyword_set(unlock) then begin
      while file_test(lockfile) do apwait,lockfile,10
    endif else begin
      if file_test(lockfile) then file_delete,lockfile,/allow  
    endelse

    ;; check all three chips and the EPSF and ETrace files
    psfdir = apogee_filename('PSF',chip=chiptag[j],num=flatframeid,/dir)
    tracedir = apogee_filename('ETrace',num=flatframeid,chip=chiptag[j],/dir)
    sflatframeid = string(flatframeid,format='(i08)')
    allfiles = psfdir+[dirs.prefix+'PSF-'+chiptag[j]+'-'+sflatframeid,dirs.prefix+'EPSF-'+chiptag[j]+'-'+sflatframeid]+'.fits'
    allfiles = [allfiles,tracedir+dirs.prefix+'ETrace-'+chiptag[j]+'-'+sflatframeid+'.fits']
    if total(file_test(allfiles)) eq 3 and not keyword_set(clobber) then begin
      print,outfile,' already exists and CLOBBER=0'
      goto,BOMB1
    endif

    ;; Create a lock file
    openw,lock,/get_lun,lockfile
    free_lun,lock

    ;; Kludge for 379 c
    ;;  1842 is a bad row
    if j eq 2 then $
      str.(j).flux[*,1842]=(str.(j).flux[*,1841]+str.(j).flux[*,1843])*0.5

    ;; Trace the fibers
    APFINDTRACE,str.(j),tracestr,pl=pl,nthreshsig=20,sigkern=1.2,thresh=peakthresh
    if n_elements(tracestr) eq 0 then begin
      print,'No traces found for ',outfile
      file_delete,lockfile,/allow
      goto,BOMB1
    endif

    ;; Extract the spectra
    print,'Extracting the spectra'
    APEXTRACT,str.(j),tracestr,outstr

    ;; Fit polynomial to the flux and add it to TRACESTR
    ;;--------------------------------------------------
    npoly = 4
    ADD_TAG,tracestr,'FLUXCOEF',fltarr(npoly+1),tracestr  ; add fluxcoef
    nfibers = n_elements(tracestr)
    sz = size(outstr.flux)
    npix = sz[1]
    x = findgen(npix)
    modelflux = fltarr(npix,nfibers)
    medflux = fltarr(npix,nfibers)  ; median-filtered flux
    for k=0,nfibers-1 do begin
      coef = ROBUST_POLY_FITQ(x,outstr.flux[*,k],npoly)
      tracestr[j].fluxcoef = coef
      modelflux[*,k] = poly(x,coef)
      ;; Get median-filtered flux
      medbin = 11
      medbin2 = medbin/2+1
      medflux1 = MEDIAN(outstr.flux[*,k],medbin)
      medflux1[0:medbin2] = modelflux[0:medbin2,k]          ; use poly-fit to fix the ends
      medflux1[nx-medbin2-1:nx-1] = modelflux[nx-medbin2-1:nx-1,k]
      medflux[*,k] = medflux1
    endfor

    ;; Now create the normalized PSF image
    ;;------------------------------------
    print,'Creating normalized PSF image'
    psfim = str.(j).flux*0.
    For k=0,nfibers-1 do begin
      fwhm = tracestr[k].fwhm * 0.8
      coef = tracestr[k].coef
      ymid = POLY(x,coef)
      ylo = MIN(floor(ymid-fwhm) > 0)
      yhi = MAX(ceil(ymid+fwhm) < (ny-1))
      num = yhi-ylo+1

      ;; Make a MASK based on the trace and FWHM
      yy = replicate(1.0,nx)#(lindgen(num)+ylo)
      ymid2d = ymid#replicate(1.0,num)
      mask = long(yy ge floor(ymid2d-fwhm) and yy le ceil(ymid2d+fwhm))
  
      flux2d = modelflux[*,k]#(fltarr(num)+1) > 1.0

      ;; Make normalized PSF image
      normpsf1 = (str.(j).flux[*,ylo:yhi] > 0) * mask/flux2d

      ;; Use median smoothed version of PSF image to mask out any
      ;;  bad pixels
      ;; I think we can median smooth MORE!!! maybe even 51 pixels
      nmedsmooth = 51 ; 7
      med_normpsf1 = MEDFILT2D(normpsf1,nmedsmooth,dim=1,/edge)

      ;; Make sure it's normalized
      totflux1 = TOTAL(med_normpsf1,2)

      ;; Look for bad pixels
      bd = where(totflux1 lt 0.8,nbd)
      if nbd gt 0 then begin
        med101_normpsf1 = MEDFILT2D(normpsf1,101,dim=1,/edge)
        for l=0,nbd-1 do med_normpsf1[bd[l],*]=med101_normpsf1[bd[l],*]
        totflux1 = TOTAL(med_normpsf1,2)  ; redo total flux
      endif

      ;; Now normalize the smoothed PSF image
      totflux2d = totflux1#(fltarr(num)+1)
      finalpsf = med_normpsf1 / totflux2d   ; normalize

      psfim[*,ylo:yhi] += finalpsf
    Endfor

    ;; Initialize the Output header
    ;;------------------------------
    head = str.(j).header
    sxaddpar,head,'PSFFILE',iflatframe,' Flat file used for Trace/PSF'
    leadstr = 'APMKPSF: '
    sxaddhist,leadstr+systime(0),head
    info = GET_LOGIN_INFO()
    sxaddhist,leadstr+info.user_name+' on '+info.machine_name,head
    sxaddhist,leadstr+'IDL '+!version.release+' '+!version.os+' '+!version.arch,head
    sxaddhist,leadstr+'Output File:',head
    sxaddhist,leadstr+' HDU1 - Trace structure for Extract_type=1',head
    sxaddhist,leadstr+' HDU2 - PSF image for Extract_type=2',head
    sxaddhist,leadstr+' HDU3 - Trace information for Extract_type=3',head
    sxaddhist,leadstr+' HDU4 - Width information for Extract_type=3',head
    sxaddhist,leadstr+' HDU5 - Empirical PSF structure for Extract_type=4',head
    sxaddpar,head,'NTRACES',nfibers

    ;; Save the trace structure
    ;;-------------------------
    if file_test(outdir,/directory) eq 0 then FILE_MKDIR,outdir  ; check that the dir exists
    outfile=apogee_filename('PSF',chip=chiptag[j],num=flatframeid)
    print,'Writing trace information to ',outfile
    FITS_WRITE,outfile,0,head         ; write the header
    MWRFITS,tracestr,outfile,/silent  ; save trace structure as FITS binary file
    MWRFITS,psfim,outfile,/silent     ; save normalized PSF image as 2nd ext.


    ;; IDLSPEC2D tracing steps
    ;;-------------------------
    ;; CODE FROM /net/home/dln5q/sdss3/Linux64/idlspec2d/trunk/pro/spec2d/SPCALIB.PRO

    ;;------------------------------------------------------------------
    ;; Create spatial tracing from flat-field image
    ;;------------------------------------------------------------------

    flatimg = float( str.(j).flux )
    flativar = 1.0/float(str.(j).err)^2
    flatmask = str.(j).mask
    flatimg = transpose(flatimg)
    flativar = transpose(flativar)
    flatmask = transpose(flatmask)
    mask = 1-(flatmask eq 1 or flatmask eq 8)

    ;; aptrace300crude can give crazy traces for the red chip
    ;;  and crashing on the middle and blue.
    ;;  apfindtrace.pro does better.
    x = findgen(npix)
    ycen2 = findgen(npix)#replicate(1,nfibers)
    xsol2 = fltarr(npix,nfibers)
    for k=0,nfibers-1 do xsol2[*,k]=poly(x,tracestr[k].coef)
    ;; Using apfindtrace traces

    ;; Need to refit the traces using -1<x<1 which is
    ;; the format used by traceset2xy.pro
    tset = {func:'poly',xmin:0.0,xmax:float(npix)-1,coeff:double(tracestr.coef*0.0)}
    npoly = n_elements(tracestr[0].coef)-1
    xmid = 0.5 * (tset.xmin + tset.xmax)
    xrange = tset.xmax - tset.xmin
    for k=0,nfibers-1 do begin
      xvec = 2.0 * (x-xmid)/xrange
      coef = poly_fit(xvec,xsol2[*,k],npoly)
      tset.coeff[*,k] = coef
    endfor
    apgundef,ycen,xsol
    traceset2xy, tset, ycen, xsol

    ;;---------------------------------------------------------------------
    ;; Extract the flat-field image to obtain width and flux
    ;;---------------------------------------------------------------------
    sigma = 1.0 ; Initial guess for gaussian width
    highrej = 15
    lowrej = 15
    ;npoly = 10 ; Fit 1 terms to background
    ;; since the Gaussian is NOT a good fit use a lower
    ;;  order background
    npoly = 3
    wfixed = [1,1] ; Fit the first gaussian term + gaussian width
    mask1 = mask

    print,'Extracting Flux with sigma=1'
    AP_EXTRACT_IMAGE, flatimg, flativar, xsol, sigma, flux, fluxivar, $
          proftype=3, wfixed=wfixed, highrej=highrej, lowrej=lowrej, $
          npoly=npoly, relative=1, ansimage=ansimage, reject=[0.1, 0.6, 0.6], $
          chisq=chisq3,ymodel=ymodel,mask=mask1

    print,'Fitting width'
    widthset3 = APFITFLATWIDTH(flux, fluxivar, ansimage, tmp_fibmask, $
                    ncoeff=5, sigma=sigma, medwidth=medwidth)
    ansimage = 0

    proftype = 3           ; |x|^3
    widthset = widthset3

    ;; Make sure the widths are positive
    apgundef,xx,sigma2
    traceset2xy,widthset,xx,sigma2
    dum = where(sigma2 le 0.0,nbad)
    if nbad gt 0 then print,'ERROR: Some negative widths'

    ;; KLUDGE, fiber 300 is messed up on red chip because it
    ;;  falls off the edge
    if file_basename(flatdir) eq '55648' and nfibers eq 299 and j eq 0 then $
      widthset.coeff[*,nfibers-1] = widthset.coeff[*,nfibers-2]

    ;; Fiber 90 is a bit messed up
    if strmid(flatid,0,4) eq '0126' or strmid(flatid,0,4) eq '0127' then $
      widthset.coeff[*,90] = widthset.coeff[*,89]
    ;; Fiber 238 is messed up in red
    if strmid(flatid,0,4) eq '0126' and j eq 2 then $
      widthset.coeff[*,238] = widthset.coeff[*,237]

    ;;---------------------------------------------------------------------
    ;; Final extraction
    ;;---------------------------------------------------------------------
    
    ;; Save the information in the output file
    ;; Save the trace information
    MKHDR,tset_head,tset.coeff,/image
    sxaddpar,tset_head,'FUNC',tset.func
    sxaddpar,tset_head,'XMIN',tset.xmin
    sxaddpar,tset_head,'XMAX',tset.xmax
    MWRFITS,tset.coeff,outfile,tset_head,/silent
    ;; Save the widthset information
    MKHDR,wset_head,widthset.coeff,/image
    sxaddpar,wset_head,'FUNC',widthset.func
    sxaddpar,wset_head,'XMIN',widthset.xmin
    sxaddpar,wset_head,'XMAX',widthset.xmax
    sxaddpar,wset_head,'PROFTYPE',proftype
    sxaddpar,wset_head,'MEDWIDTH1',medwidth[0]
    sxaddpar,wset_head,'MEDWIDTH2',medwidth[1]
    sxaddpar,wset_head,'MEDWIDTH3',medwidth[2]
    sxaddpar,wset_head,'MEDWIDTH4',medwidth[3]
    MWRFITS,widthset.coeff,outfile,wset_head,/silent

    ; Create Jon's Empirical PSF
    ;---------------------------
    if not keyword_set(no_epsf) then $
      APMKPSF_EPSF,str.(j),strmid(outdir,0,strlen(outdir)-4),flatid,j,sparseid=sparseid,fiberid=fiberid,/scat,average=average

    outfile = apogee_filename('PSF',chip=chiptag[j],num=flatframeid)
    file_delete,lockfile

    BOMB1:

    print,''
  Endfor ; chip loop

  BOMB:
ENDFOR  ; flat frame loop

print,'APMKPSF FINISHED'
dt = systime(1)-t0
print,'Time elapsed = ',strtrim(dt/60.,2),' min.'

if keyword_set(stp) then stop

end
