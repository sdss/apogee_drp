;+
;
; APMKFLUXCAL
;
; This programs makes the APOGEE 1D relative flux calibration
; file.
;
; INPUTS:
;  flatid     The name of the RAW lamp image to use (flat lamp,
;               domeflat or blackbody).  This should be the directory and ID8
;               concatenated.
;  =outdir    The output directory.  The default is SPECTRO_DIR/cal/flux/
;  /collapse  Collapse from 3D.
;  =bbtemp    The blackbody temperature in Kelvin (only for Blackbody
;               exposures).
;  =waveid    The wavelength directory and ID concatenated.  This is
;               needed for blackbody exposures.
;  /absolute  Use absolute normalization.  For Blackbody exposures only.
;  /reproc    Reprocess the raw and 2D files if they already exist.
;  /clobber   Overwrite the previous apFlux calibration file if it
;               exists.
;  /unlock    Delete lock file and start fresh.
;  /pl        Plot the data.
;
; OUTPUTS:
;  The output flux calibration file is written to the output
;  directory with filename of apFlux-MJD5-ID8.fits
;  =error   The error message if one occured.
;
; By D. Nidever   Mar 2011
;-

pro apmkfluxcal,flatid,outdir=outdir0,bbtemp=bbtemp,waveid=waveid,reproc=reproc,$
                clobber=clobber,absolute=absolute,pl=pl,error=error,collapse=collapse,$
                unlock=unlock
  
t0 = systime(1)

;; Not enough inputs
nflatid = n_elements(flatid)
if nflatid eq 0 then begin
  error = 'Syntax - apmkfluxcal,flatid,outdir=outdir0,bbtemp=bbtemp,waveid=waveid,'+$
          'absolute=absolute,reproc=reproc,clobber=clobber,pl=pl,error=error,collapse=collapse'
  print,error
  return
endif

;; Get APOGEE directories
dirs = getdir(apogee_dir,cal_dir,spectro_dir,apogee_vers)
red_dir = dirs.expdir
flux_dir = cal_dir+'flux/'

;; Default values
if n_elements(outdir0) eq 0 then outdir=flux_dir else outdir=outdir0

chiptag = ['a','b','c']

print,''
print,'Running APMKFLUXCAL'
print,''

;; Get MJD5 from ID
daynum = strmid(string(long(file_basename(flatid)),format='(I08)'),0,4)
mjd5 = long(daynum)+55562
red_dir = dirs.expdir+strtrim(mjd5,2)+'/'

;; Do the 3D->2D processing
;;------------------------------
if keyword_set(collapse) then begin

  ;; Make the raw chip files
  flatframenum = file_basename(flatid)
  flatframeid = string(long(flatframenum),format='(I08)')
  rawfiles = file_dirname(flatid)+'/'+dirs.prefix+'R-'+chiptag+'-'+flatframeid+'.fits'
  cmpfiles = file_dirname(flatid)+'/'+dirs.prefix+'R-'+chiptag+'-'+flatframeid+'.apz'

  ;; Check that they exist and are okay
  rawinfo = APFILEINFO(rawfiles,/silent)
  rawokay = (rawinfo.exists AND (rawinfo.filesize gt 0) AND rawinfo.rawfmt AND rawinfo.allchips AND ((rawinfo.naxis eq 3) OR (rawinfo.exten eq 1)))
  ;;  must be 3D or have extensions
  if min(rawokay) lt 1 then begin
    bd = where(rawokay eq 0,nbd)
    print,'There is a problem with the raw files: ',strjoin((rawfiles)(bd),' ')
    print,'Checking the compressed files'

    cmpinfo = APFILEINFO(cmpfiles,/silent)
    cmpokay = (cmpinfo.exists AND (cmpinfo.filesize gt 0) AND cmpinfo.rawfmt AND cmpinfo.allchips AND ((cmpinfo.naxis eq 3) OR (cmpinfo.exten eq 1)))
    if min(cmpokay) lt 1 then begin
      bd = where(cmpokay eq 0,nbd)
      print,'There is a problem with the compressed files: ',strjoin((cmpfiles)(bd),' ')
      return
    endif

    files = cmpfiles
    info = cmpinfo

  endif else begin
    files = rawfiles
    info = rawinfo
  endelse

  ;; Does the output file already exist?
  outfile = addslash(outdir)+dirs.prefix+'Flux-'+strtrim(info[0].mjd5,2)+'-'+info[0].fid8+'.fits'
  if file_test(outfile) eq 1 and not keyword_set(clobber) then begin
    print,outfile,' already exists and CLOBBER=0'
    return
  endif

  print,flatid

  ;; Dimensions
  nx = 2048L
  ny = 2048L


  ;==================================
  ; Step 1 - Collapse the datacubes
  ;==================================

  ; Loop over chips
  ;-----------------
  For i=0,2 do begin

    file = files[i]


    ;; Get calibration files
    ;;----------------------

    ;; Detector
    APGETCALIB,file,'DETECTOR',detfile,error=error
    if n_elements(error) gt 0 then begin
      print,'No DETECTOR calibration file'
      return
    endif
    detcorr = file_dirname(detfile)+'/'+dirs.prefix+'Detector-'+chiptag[i]+'-'+file_basename(detfile)+'.fits'
    ;; BPM
    APGETCALIB,file,'BPM',bpmfile,error=error
    if n_elements(error) gt 0 then begin
      print,'No BPM calibration file'
      return
    endif
    bpmcorr = file_dirname(bpmfile)+'/'+dirs.prefix+'BPM-'+chiptag[i]+'-'+file_basename(bpmfile)+'.fits'
    ;; Superflat
    APGETCALIB,file,'FLAT',flatfile,error=error
    if n_elements(error) gt 0 then begin
      print,'No DARK calibration file'
      return
    endif
    flatcorr = file_dirname(flatfile)+'/'+dirs.prefix+'Flat-'+chiptag[i]+'-'+file_basename(flatfile)+'.fits'

    ;; Fix saturated pixels for Nread=3 for now
    rd3satfix = 1

    print,''
    print,'-----------------------------------------'
    print,' Processing chip '+chiptag[i]+' - '+file_basename(file)
    print,'-----------------------------------------'
    print,''


    ;; Output file
    outdir2d = red_dir
    if file_test(outdir2d,/directory) eq 0 then file_mkdir,outdir2d
    outfile = outdir2d+dirs.prefix+'2D-'+chiptag[i]+'-'+info[i].fid8+'.fits'

    ;; Process the raw file
    if file_test(outfile) eq 0 or keyword_set(reproc) then begin

      ;; Process the datacube with AP3DPROC
      ;;------------------------------------
      ;;  up the ramp sampling
      apgundef,output,head
      AP3DPROC,file,outfile,detcorr=detcorr,bpmcorr=bpmcorr,darkcorr=darkcorr,flatcorr=flatcorr,$
               /crfix,criter=0,/satfix,/uptheramp,error=procerror,/clobber,rd3satfix=rd3satfix,$
               unlock=unlock

    ;; Already processed
    endif else begin
      print,file,' Already processed'
    endelse

  Endfor ; chip loop

  outfiles2d = outdir2d+dirs.prefix+'2D-'+chiptag+'-'+file_basename(flatid)+'.fits'

;; Don't collapse from 3D
Endif else begin

  outdir2d = red_dir
  outfiles2d = outdir2d+dirs.prefix+'2D-'+chiptag+'-'+file_basename(flatid)+'.fits'

Endelse  ; collapse


;==================================
; Step 2 - Extract the spectra
;==================================


;outdir1d = outdir+strtrim(info[0].mjd5,2)+'/'
info = apfileinfo(outfiles2d[0],/silent)
outdir1d = outdir2d
outfiles1d = outdir1d+'/'+dirs.prefix+'1D-'+chiptag+'-'+info[0].fid8+'.fits'
if total(file_test(outfiles1d)) ne 3 or keyword_set(reproc) then begin
  ;; Trace
  APGETCALIB,outfiles2d[0],'PSF',psffile,error=error
  if n_elements(error) gt 0 then begin
    print,'No PSF calibration file'
    return
  endif

  inpfile = outdir1d+'/'+info[0].fid8

  extract_type = 3  ; Gaussian PSF fitting for now
  AP2DPROC,inpfile,psffile,extract_type,outdir=outdir1d,unlock=unlock

;; Previously processed
endif else begin
  print,outdir1d+'/'+info[0].fid8,' previously extracted'
endelse


print,''
print,'Making RELATIVE FLUX CALIBRATION FILE'
print,''

;; Load the 1D extracted spectra
APLOADFRAME,outdir1d+dirs.prefix+'1D-'+info[0].fid8,frame
sz = size(frame.(0).flux)
npix = sz[1]
nfibers = sz[2]

;; What type of lamp is it
exptype = sxpar(frame.(0).header,'EXPTYPE',count=nexptype)
if nexptype eq 0 then exptype = 'UNKNOWN'

; Get the reference spectrum for each chip
CASE exptype of

  ;; Blackbody
  'BLACKBODY': begin

    ;; Need wavelength array
    if n_elements(waveid) eq 0 then begin
      error = 'Need WAVEID for Blackbody exposure'
      print,error
      return
    endif
    ;; Need BB temp
    if n_elements(bbtemp) eq 0 then begin
      error = 'Need BBTEMP for Blackbody exposure'
      print,error
      return
    endif

    wavefiles = file_dirname(waveid)+'/'+dirs.prefix+'Wave-'+chiptag+'-'+file_basename(waveid)+'.fits'
    if total(file_test(wavefiles)) ne 3 then begin
      bd = where(file_test(wavefiles) eq 0,nbd)
      error = 'WAVELENGTH files '+strjoin(wavefiles[bd],' ')+' NOT FOUND'
      print,error
      return
    endif

    refspec0 = fltarr(npix,3)
    for i=0,2 do begin
       ;;  Load the wavelength array
       FITS_READ,wavefiles[i],wim,whead,exten=1
       szwim = size(wim)
       wave = reform(wim[szwim[1]/2,*])
       ;; To calculate the Planck function in units of ergs/cm2/s/A
       bbflux = PLANCK( wave, bbtemp)
       refspec0[*,i] = bbflux
    endfor

    ;; Normalize the flux
    if not keyword_set(absolute) then $
      refspec = refspec0 / refspec0[npix/2,1]

  end ; blackbody

  ;; Flat, Quartz, Dome, etc.
  else: begin

    ;; Create "reference" spectrum from polynomial fit to smoothed, median spectrum
    refspec0 = fltarr(npix,3)
    for i=0,2 do begin
      ;; Next lines not used?
      temp = MEDFILT2D(frame.(i).flux,31,dim=1,/edge)
      maxspec = max(temp,dim=2)
      smmaxspec = medfilt1d(maxspec,21,/edge)

      ;; Skip fibers that have zero flux for median
      medlevel = median(frame.(i).flux,dim=1)
      gd = where(medlevel gt 0 and finite(medlevel) eq 1)
      refspec1 = median(frame.(i).flux[*,gd],dim=2)

      tmpflux = frame.(i).flux[*,gd]
      bd = where(frame.(i).mask[*,gd] and badmask())
      tmpflux[bd] = !values.f_nan
      for j=0,n_elements(gd)-1 do tmpflux[*,j]=medfilt1d(tmpflux[*,j],41,/edge)
      refspec1 = median(tmpflux,dim=2)

      refspec1[0:3] = !values.f_nan
      refspec1[npix-4:npix-1] = !values.f_nan
      smrefspec1 = MEDFILT1D(refspec1,41,/edge)
      refspec0[*,i] = smrefspec1 
    endfor


    ;; We want to take out spectral structure of the lamp, which we
    ;; do by fitting a global polynomial. However, at LCO there is structure
    ;; in the red chip that is apparently from the lamp/screen, so for that particular
    ;; chip, we want to preserve that feature in the reference spectrum, so it
    ;; is not propagated

    ;; Fit a polynomial to the spectra, all chips together, avoiding LCO red dip
    x = [ [findgen(npix)-1023.5-2048-150], [findgen(npix)-1023.5], [findgen(npix)-1023.5+2048+150] ]
    if dirs.telescope eq 'lco25m' then begin
      pix = indgen(3*npix)
      gd = where(pix lt 700 or pix gt 1900 and finite(refspec0) eq 1)
      coef = robust_poly_fit(x[gd],refspec0[gd],4)
    endif else begin
      gd = where(finite(refspec0) eq 1)
      coef = robust_poly_fit(x[gd],refspec0[gd],4)
    endelse
    ;; We're using the polynomial fit
    refspec = poly(x,coef)

    ;; Try to get the LCO dip from the ratio of red chip flux to a low order fit
    ;; and multiply that back into the reference spectrum
    if dirs.telescope eq 'lco25m' then begin
      pix = indgen(2048)
      gd = where(pix lt 700 or pix gt 1900)
      fit = robust_poly_fit(pix[gd],refspec0[gd,0],2)
      dip = refspec0[*,0]/poly(pix,fit)
      refspec[*,0] *= dip
    endif

    if keyword_set(pl) then begin
      plot,x,refspec0,/nodata,tit='Reference Spectrum'
      for i=0,2 do begin
        oplot,x[*,i],refspec0[*,i]
        oplot,x[*,i],refspec[*,i],co=250
      endfor
    endif
  end ; flat, quartz, dome

ENDCASE


; Calculate the relative flux calibration
;----------------------------------------

;; Loop through the chips
allfluxcal = fltarr(2048,300,3)
allthru = fltarr(300,3)
For i=0,2 do begin
  flux = frame.(i).flux

  ;; Now divide each fiber by the median spectrum
  ratio = flux / (refspec[*,i] # replicate(1,nfibers) )
  bd = where(ratio lt 1e-3,nbd)
  if nbd gt 0 then ratio[bd]=!values.f_nan
  ratio[0:3,*] = !values.f_nan              ; fix ref pixels
  ratio[npix-4:npix-1,*] = !values.f_nan    ; fix ref pixels

  ;; Set bad pixels to NaN
  bd = where(frame.(i).mask and badmask())
  ratio[bd] = !values.f_nan

  ;; Use average of neighbors for FPI fibers 75 and 225
  ;;   and broken fibers
  medratio = median(ratio,dim=1)
  broken = where(finite(medratio) eq 0,nbroken,comp=good,ncomp=ngood)
  if mjd5 ge 59556 then begin
    dointerp = [75,225]
    if dirs.telescope eq 'lco25m' then dointerp = [87,218]
    if nbroken gt 0 then dointerp=[dointerp,broken]
  endif else begin
    if nbroken gt 0 then dointerp=broken
  endelse

  ;; Fix fibers with issues
  avgratio = median(ratio,dim=2)
  for k=0,n_elements(dointerp)-1 do begin
    ind1 = dointerp[k]
    ;; Get closest good fibers 
    dist = abs(good-ind1)
    si = sort(dist)
    bestnei = good[si[0:1]]
    ;; Take average of two closest fibers
    if abs(ind1-bestnei[0]) lt 10 then begin
      ratio[*,ind1] = 0.5*(ratio[*,bestnei[0]]+ratio[*,bestnei[1]])
    ;; No good fibers close by, use global median
    endif else begin
      ratio[*,ind1] = avgratio
    endelse
  endfor

  ;; Interpolate over the Littrow ghost using a low order polynomial fit to the region around it
  for j=0,nfibers-1 do begin
    bd = where(frame.(i).mask[*,j] and maskval('LITTROW_GHOST'),nbd)
    if nbd gt 0 then begin
      ratio[bd,j] = !values.f_nan
      ;; Do a fit to region +100 surrounding pixels
      is = bd[0]-50
      ie = bd[nbd-1]+50
      x = findgen(n_elements(ratio[*,j]))
      xl = x[is:ie]
      yl = ratio[is:ie,j]
      ;; poly_fit doesn't want to have NaNs fed to it, so just feed good values
      gd = where(finite(yl) eq 1,ngd)
      if ngd gt 2 then begin
        xfit = xl[gd]
        yfit = yl[gd]
        coef = robust_poly_fit(xfit,yfit,2)
        ;; Use fit to fill in ghost region
        ratio[bd,j] = poly(x[bd],coef) 
      endif
    endif
  endfor

  sm_ratio_med = MEDFILT2D(ratio,51,dim=1,/edge)  ;; 21

  ;; Fix bad pixels
  for j=0,nfibers-1 do begin
    bd = where(finite(sm_ratio_med[*,j]) eq 0,nbd)
    gd = where(finite(sm_ratio_med[*,j]) eq 1,ngd)
    if ngd gt 0 and nbd gt 0 then begin
      sm_ratio1 = MEDFILT1D(sm_ratio_med[*,j],201,/edge)
      sm_ratio_med[bd,j] = sm_ratio1[bd]
    endif
  endfor

  ;; Smooth it a little bit
  sm_ratio = SMOOTH(sm_ratio_med,[100,1],/edge_truncate)

  ;; Keep the flux calibration images
  fluxcal = sm_ratio
  allfluxcal[*,*,i] = fluxcal
  ;; Calculate the throughput
  thru = median(sm_ratio,dim=1)
  thru /= median(thru)
  allthru[*,i] = thru
  ;; Plotting
  if keyword_set(pl) then $
    displayc,sm_ratio,/z,xtit='X',ytit='Y',tit='Relative Flux Calibration for Chip '+chiptag[i]
  
  ;; Output the Flux calibration to file
  ;;-------------------------------------
  ;;outfile = addslash(outdir)+dirs.prefix+'Flux-'+strtrim(info[0].mjd5,2)+'-'+info[0].fid8+'.fits'
  outfile = addslash(outdir)+dirs.prefix+'Flux-'+chiptag[i]+'-'+info[0].fid8+'.fits'
  print,'Writing FLUX file to ',outfile
  head = frame.(0).header
  sxaddpar,head,'OBSTYPE','FLUXCORR'

  ;; The output names should probably have the PLATEID and CARTID in it
  ;; if this is a DOMEFLAT.

  ;; Update header
  leadstr = 'APMKFLUXCAL: '
  sxaddhist,leadstr+systime(0),head
  login_info = GET_LOGIN_INFO()
  sxaddhist,leadstr+login_info.user_name+' on '+login_info.machine_name,head
  sxaddhist,leadstr+'IDL '+!version.release+' '+!version.os+' '+!version.arch,head
  sxaddhist,leadstr+' APOGEE Reduction Pipeline Version: '+getvers(),head
  sxaddhist,leadstr+'Output File:',head
  sxaddhist,leadstr+' HDU1 - Relative Flux Calibration [Npix,Nfibers,3]',head
  sxaddhist,leadstr+' HDU2 - Throughput [Nfibers,3]',head
  maxlen = 72-strlen(leadstr)
  line = 'File="'+info[0].dir+'/'+info[0].fid8+'"'
  if strlen(line) gt maxlen then begin
    line1 = strmid(line,0,maxlen)
    line2 = strmid(line,maxlen,100)
    sxaddhist,leadstr+line1,head
    sxaddhist,leadstr+line2,head
  endif else sxaddhist,leadstr+line,head
  sxaddhist,'LAMPTYPE='+exptype,head
  sxaddpar,head,'V_APRED',getgitvers(),'apogee software version'
  sxaddpar,head,'APRED',getvers(),'apogee reduction version'
  
  ;; HDU0 - header only
  FITS_WRITE,outfile,0,head,/no_abort     ; write the header
  ;; HDU1 - flux calibration
  MKHDR,head1,fluxcal,/image
  sxaddpar,head1,'CTYPE1','Pixel'
  sxaddpar,head1,'CTYPE2','Fiber'
  sxaddpar,head1,'CTYPE3','Chip'
  sxaddpar,head1,'BUNIT','Relative Flux'
  if exptype eq 'BLACKBODY' and keyword_set(absolute) then $
    sxaddpar,head1,'BUNIT','Absolute Flux (ergs/cm2/s/A)'
  sxaddpar,head1,'EXTNAME','RELATIVE FLUX'
  MWRFITS,fluxcal,outfile,head1,/silent ; Flux calibration array
  ;; HDU2 - throughput
  MKHDR,head2,thru,/image
  sxaddpar,head2,'CTYPE1','Fiber'
  sxaddpar,head2,'CTYPE2','Chip'
  sxaddpar,head2,'BUNIT','Throughput'
  sxaddpar,head2,'EXTNAME','THROUGHPUT'
  MWRFITS,thru,outfile,head2,/silent      ; throughput values

  ;; HDU3 - reference spectrum, polynomial fit
  MKHDR,head3,refspec[*,i],/image
  sxaddpar,head3,'CTYPE1','Pixel'
  sxaddpar,head3,'CTYPE3','Chip'
  sxaddpar,head3,'BUNIT','Relative Flux'
  sxaddpar,head3,'EXTNAME','REFERENCE SPECTRUM'
  MWRFITS,refspec[*,i],outfile,head3,/silent    

  ;; HDU4 - reference spectrum, original soothed median spectrum
  MKHDR,head4,refspec0[*,i],/image
  sxaddpar,head4,'CTYPE1','Pixel'
  sxaddpar,head4,'CTYPE3','Chip'
  sxaddpar,head4,'BUNIT','Relative Flux'
  sxaddpar,head4,'EXTNAME','MEDIUM REFERENCE SPECTRUM'
  MWRFITS,refspec0[*,i],outfile,head4,/silent    

Endfor ; chip loop

;; Check missing fibers
;;  sometimes fibers will be missing in some chips but not others
;;  this can cause "jumps" in the spectra because the flux corrections
;;   are "missing" for some chips
fmed = fltarr(300,3)
for i=0,2 do fmed[*,i] = median(frame.(i).flux,dim=1)
goodchip = total(fmed ne 0,2,/integer)
fiberstofix = where(goodchip lt 3 and goodchip gt 0,nfiberstofix)
if nfiberstofix gt 0 then $
  print,'Fixing ',strtrim(nfiberstofix,2),' fibers that have missing chips: ',strjoin(strtrim(fiberstofix,2),',')
for i=0,nfiberstofix-1 do begin
  ifiber = fiberstofix[i]
  gdchip = where(fmed[ifiber,*] ne 0,ngdchip)
  bdchip = where(fmed[ifiber,*] eq 0,nbdchip)
  medfluxcal = median([allfluxcal[*,ifiber,gdchip]])
  allthru[ifiber,bdchip] = medfluxcal
  for j=0,nbdchip-1 do begin
    allfluxcal[*,ifiber,bdchip[j]] *= medfluxcal/median(allfluxcal[*,ifiber,bdchip[j]])
  endfor
endfor
if nfiberstofix gt 0 then begin
  for i=0,2 do begin
    outfile = addslash(outdir)+dirs.prefix+'Flux-'+chiptag[i]+'-'+info[0].fid8+'.fits'
    header1 = headfits(outfile,exten=1)
    MODFITS,outfile,allfluxcal[*,*,i],header1,exten_no=1
    header2 = headfits(outfile,exten=2)
    MODFITS,outfile,allthru[*,i],header2,exten_no=2    
  endfor
endif

print,''
print,'APMKFLUXCAL Finished'
print,'dt = ',strtrim(systime(1)-t0,2),' sec'

end

