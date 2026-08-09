;+
;
; APMKPSF_EPSF
;
; This creates an empirical PSF for APMKPSF.
;
; INPUTS
;  frame     A structure containing the 2D chip image of
;              the flat field image from which to make the PSF.
;              This should include FLUX, ERR and MASK.
;  =fiberid  ID8 number for the ETrace calibration file to use.
;  =yshift   Trace pixel shift in y-direction.  Used when making
;              FIBER calibration file.
;  /silent   Don't print anything to the screen
;
; OUTPUTS
;  outpsf   An Ntrace structure giving the empirical PSF
;             for each trace.
;
; USAGE:
;  IDL>apmkpsf_epsf,frame,outpsf
;
; By J. Holtzman 2011
; Incorporated into APMKPSF by D. Nidever  May 2011
;-

pro apmkpsf_epsf,frame,caldir,im,ichip,silent=silent,sparseid=sparseid,dmax=dmax,sdmax=sdmax,$
                 thresh=thresh,smooth=smooth,fiberid=fiberid,scat=scat,average=average,yshift=yshift,$
                 etraceonly=etraceonly

; Not enough inputs
if n_elements(frame) eq 0 then begin
  if not keyword_set(silent) then $
    print,'Syntax - apmkpsf_epsf,frame,outfile'
  return
endif
if not keyword_set(fiberid) then fiberid=0
if not keyword_set(yshift) then yshift=0.0

dirs = getdir()
chip = ['a','b','c']

if not keyword_set(silent) then print,'Creating Empirical PSF'

if size(frame,/type) eq 8 then begin
  red = frame.flux 
  mask = frame.mask
  head = frame.header
endif else begin
  red = frame
  mask = fix(frame*0.)
  bd = where(finite(red) eq 0,nbd)
  if nbd gt 0 then mask[bd]=1
  mkhdr,head,0
endelse

if keyword_set(scat) then scat_remove,red,scat=scat

wind = 2
if not keyword_set(average) then average=50
bad = where((mask and badmask()) gt 0,nbad)
if nbad gt 0 then red[bad]=!values.f_nan
;; Use zap to filt out bad pixels and reject remnant CRs, use [50,1] median filter to construct PSF
psf = zap(red,[average,1])

time0 = systime(/seconds)
;; First find the fibers using average in central region by looking for peaks
p = fltarr(2048)
for i=0,2047 do begin
  robust_mean,psf[900:1100,i],tmp
  p[i] = tmp*201
endfor
if not keyword_set(smooth) then smooth=4
ps = smooth(double(p),smooth,/nan)
; shifted to peak in smoothed spectrum 1/13
high = MAX(ps,/nan)
if not keyword_set(thresh) then thresh=0.05
ploc = findpeak(ps,level=max([thresh*high,5000.]))
ntrace = n_elements(ploc)
print,'Found ',strtrim(ntrace,2),' traces'

;; Getting reference positions from FIBER calibration file
if fiberid gt 0 then begin
  file = apogee_filename('ETrace',chip=chip[ichip],num=fiberid)
  ref = mrdfits(file)
  fibers = ref[1000,*]

;; Use default reference trace positions and yshift
;; Probably making fiber calibration file
endif else begin
  reftracefile = dirs.libdir+'/cal/'+dirs.instrument+'_fiber_positions.fits'
  if file_test(reftracefile) eq 0 then begin
    print,'Fiber trace reference file not found'
    print,reftracefile
    return
  endif
  ref = mrdfits(reftracefile)
  fibers = ref[ichip,*] + yshift  ;; apply trace yshift for this exposure
  print,'Using yshift = ',yshift
endelse

;; Determine which fiber each trace corresponds to
fiber = intarr(ntrace)-1
dmin = fltarr(n_elements(fiber))
offset = fltarr(n_elements(fiber))  
nmatched = 0
pmid = []
;; Loop over traces and find the best matching fiber
for i=0,n_elements(fiber)-1 do begin
  ;; Do a 3-point quadratic fit to get center a bit better than peak
  nfit = 3

  fit = poly_fit(-nfit/2+indgen(nfit),ps[ploc[i]-nfit/2:ploc[i]+nfit/2],2)
  pcen = ploc[i]-fit[1]/(2*fit[2])
  if pcen lt ploc[i]-nfit/2 or pcen gt ploc[i]+nfit/2 then pcen=ploc[i]     
  dist = abs(pcen-fibers)
  mindist = min(dist,imin)
  dmin[i] = mindist               ;; absolute distance
  offset[i] = pcen-fibers[imin]   ;; distance with sign +/-
  ;;dmin[nmatched] = min(dist,imin)
  ;;if dmin[nmatched] lt 2 then begin
  ;;  fiber[nmatched] = imin 
  ;;  pmid=[pmid,ploc[i]*1.]
  if mindist lt 2 then begin
    fiber[i] = imin
    pmid = [pmid,ploc[i]*1.]       
    nmatched += 1
  endif else begin
    ;; If can't find a match, call this trace bad, but move on
    print,'not halted: cant find corresponding fiber for trace', i, ploc[i]
  endelse
endfor

;; Some traces didn't find a matching fiber
;;  trim to the ones that did have matches
bad = where(fiber lt 0,nbad,comp=good,ncomp=ngood)
if nbad gt 0 then begin
  ntrace = ngood
  fiber = fiber[good]
  dmin = dmin[good]
  offset = offset[good]
  ploc = ploc[good]
endif
  
;endif else begin
;  fiber = indgen(ntrace)
;  dmin = fltarr(n_elements(fiber))
;  pmid = ploc
;endelse

print,'Mean distance: ',MEAN(dmin)

;; Deal with duplicates
if ntrace gt 300 then begin
  fiberindex = create_index(fiber)
  bad = where(fiberindex.num gt 1,nbad)
  for i=0,nbad-1 do begin
    ;; Multiple traces are matched with the same fiber
    ind = fiberindex.index[fiberindex.lo[bad[i]]:fiberindex.hi[bad[i]]]
    ;; Pick the one with the closer match
    offset1 = offset[ind]-mean(offset)
    flux1 = ps[ploc[ind]]
    ;; Mark the rest as bad
    bad1 = (sort(abs(offset1)))[1:*]  
    fiber[ind[bad1]] = -1
    dmin[ind[bad1]] = -1
    offset[ind[bad1]] = -1
    pmid[ind[bad1]] = -1
  endfor
  ;; Remove the "bad" traces with no matches
  torem = where(fiber lt 0,ntorem,comp=tokeep,ncomp=ntokeep)
  if ntorem gt 0 then begin
    fiber = fiber[tokeep]
    dmin = dmin[tokeep]
    offset = offset[tokeep]
    pmid = pmid[tokeep]
  endif
  ntrace = ntokeep
endif

if ntrace gt 300 then begin
  print, 'halted: chip ', ichip,' has ',ntrace, ' traces!'
  stop
endif

;; If we have an entire bad row, set it to zero so it doesn't kill the nearby traces
for j=4,2043 do begin
  bad = where(mask[*,j] and badmask() gt 0,nbad)
  if nbad eq 2048 then psf[*,j]=0.
endfor

;; Now march to higher columns using locations from last column to get location in next column 
;; Get centroids using pixels within cent-wind < y < cent+wind, and save in trace[ncol,ntrace]
y = indgen(2048)
trace = fltarr(2048,ntrace)
ptmp = pmid
for i=1024,2047-average do begin
  yp = y*psf[i,*]
  for j=0,ntrace-1 do begin
    pc = (nint(ptmp[j]) > 0) < 2047
    lo = (pc-wind) > 0
    hi = (pc+wind) < 2047
    trace[i,j] = TOTAL(yp[lo:hi],/nan)/TOTAL(psf[i,lo:hi],/nan)
  endfor
  new = trace[i,*]
  good = where(finite(new),ngood)
  if ngood gt 0 then ptmp[good] = new[good]
endfor

;; Now march to lower columns using locations from last column to get location in next column 
ptmp = pmid
for i=1023,average,-1 do begin
  yp = y*psf[i,*]
  for j=0,ntrace-1 do begin
    pc = (nint(ptmp[j]) > 0) < 2047
    lo = (pc-wind) > 0
    hi = (pc+wind) < 2047
    trace[i,j] = TOTAL(yp[lo:hi],/nan)/TOTAL(psf[i,lo:hi],/nan)
  endfor
  new = trace[i,*]
  good = where(finite(new),ngood)
  if ngood gt 0 then ptmp[good] = new[good]
endfor

;; Do polynomical fit to traces, and replace into trace
ftrace = trace*0.
for j=0,ntrace-1 do begin
  gd=where(y gt average and y lt 2047-average and finite(trace[*,j]) eq 1,ngd)
  if ngd gt 3 then begin
    coef = poly_fit(y[gd],trace[gd,j],2,yfit=yfit)
    ftrace[*,j] = coef[0]+coef[1]*y+coef[2]*y*y
  ;; problematic fiber
  endif else begin
    print,'trace=',j,' has no good pixels'
    ftrace[*,j] = !values.f_nan 
  endelse
endfor
trace = ftrace
sxaddpar,head,'NTRACE',ntrace
sxaddpar,head,'AVGDIST',MEAN(dmin)
file=apogee_filename('ETrace',chip=chip[ichip],num=im)
if file_test(file_dirname(file)) eq 0 then file_mkdir,file_dirname(file)
MWRFITS,trace,file,head,/create
print,'Writing ',file

;; When making the fiber calibration, only the trace positions are needed.
;; Do not continue with the empirical-PSF construction, which may require
;; the sparse calibration and introduce a circular dependency.
if keyword_set(etraceonly) then return

;; Get sparse pack PSF if desired and available
if keyword_set(sparseid) then begin
  file = apogee_filename('EPSF',chip=chip[ichip],num=sparseid)
  tmp = mrdfits(file,0,phead)
  nsparse = sxpar(phead,'NTRACE')
  img = ptrarr(nsparse,/allocate_heap)
  for i=0,nsparse-1 do begin
    ptmp = mrdfits(file,i+1,/silent)
    *img[i] = ptmp.img
    s = {cent: ptmp.cent, lo: ptmp.lo, hi: ptmp.hi, img: img[i]}
    if i eq 0 then spsf=replicate(s,nsparse)
    spsf[i] = s
  endfor
endif

;; Assign observed PSF to separate fibers
;; Flux from each pixel will be split between the two neighboring traces.
;; With sparseid, use sparse pack image to determine the weights.
;; If no sparse id, use gaussian with sigma=2 to set weights.
;;
;; IMPORTANT MEMORY NOTE:
;; The old version allocated
;;
;;    bpsf = fltarr(2048,2048,ntrace)
;;
;; which is ~4.7 GiB for ntrace=300.  That cube is mostly empty because each
;; fiber only occupies a narrow band around its trace.  The code below keeps
;; the same pixel-splitting logic, but stores each fiber in a compact vertical
;; strip using pointer arrays.
sig2 = 2*2^2
if not keyword_set(dmax) then dmax = 7

;; Pre-compute the vertical range needed for each trace.  These ranges are
;; intentionally a little conservative: trace +/- dmax over all columns.
loarr = intarr(ntrace)
hiarr = intarr(ntrace)
imgptr = ptrarr(ntrace)
for k=0,ntrace-1 do begin
  goodcen = where(finite(trace[*,k]),ngoodcen)
  if ngoodcen gt 0 then begin
    loarr[k] = floor(min(trace[goodcen,k])-dmax) > 0
    hiarr[k] = ceil(max(trace[goodcen,k])+dmax) < 2047
  endif else begin
    loarr[k] = 0
    hiarr[k] = 0
  endelse
  imgptr[k] = ptr_new(fltarr(2048,hiarr[k]-loarr[k]+1))
endfor

;; Column loop
for i=0,2047 do begin
  if not keyword_set(silent) then $
    print,format='(%"PSF column: %5d\r",$)',i

  k2 = 0
  tmp = fltarr(2048,ntrace)

  ;; Check once per column whether this column has any finite PSF values.
  gd = where(finite(psf[i,*]),ngd)

  ;; Row loop
  if ngd gt 0 then begin
    for j=4,2043 do begin
      ;; Find the two traces on either side of this pixel.
      while trace[i,k2] lt j and k2 lt ntrace-1 do k2+=1
      k1 = k2-1 lt 0 ? 0 : k2-1

      ;; Find the distance from the surrounding traces.
      ;; If this is the first trace, only use one.
      if k1 ne k2 then d1 = j-trace[i,k1] else d1=dmax+1
      d2 = j-trace[i,k2]

      ;; Determine the weight to be given to each neighboring trace.
      w1=0 & w2=0
      if abs(d1) lt dmax or abs(d2) lt dmax then begin
        if keyword_set(sparseid) then begin
          ;; Find the nearest sparse trace.
          d = abs(spsf.cent[i]-j)
          dmin = min(d,is)
          sd1 = spsf[is].cent[i]+d1-spsf[is].lo
          sd2 = spsf[is].cent[i]+d2-spsf[is].lo
          a = spsf[is].img
          sz = size(*a)
          if abs(d1) lt dmax and nint(sd1) ge 0 and fix(sd1)+1 lt sz[2] then begin
            wlo = (*a)[i,fix(sd1)]
            whi = (*a)[i,fix(sd1)+1]
            w1 = wlo+(sd1-fix(sd1))*(whi-wlo)
          endif
          if abs(d2) lt dmax and nint(sd2) ge 0 and fix(sd2)+1 lt sz[2] then begin
            wlo = (*a)[i,fix(sd2)]
            whi = (*a)[i,fix(sd2)+1]
            w2 = wlo+(sd2-fix(sd2))*(whi-wlo)
          endif
        endif else begin
          if (abs(d1) lt dmax) then w1=exp(-d1^2/sig2)
          if (abs(d2) lt dmax) then w2=exp(-d2^2/sig2)
        endelse
      endif

      wtot = w1+w2

      ;; Add this pixel into the empirical PSFs for the neighboring traces with
      ;; appropriate weights.  This preserves the original overlap treatment.
      ;; Only distribute finite image pixels.  Test each weight separately because
      ;; NaN*0 is still NaN and would contaminate a zero-weight neighboring trace.
      if wtot gt 0 and finite(psf[i,j]) then begin
        if w1 gt 0 then tmp[j,k1] = psf[i,j]*w1/wtot
        if w2 gt 0 then tmp[j,k2] = psf[i,j]*w2/wtot
      endif
    endfor
  endif

  ;; Normalize each trace contribution in this column, as before, but store it
  ;; in the compact strip rather than in bpsf[2048,2048,ntrace].
  norm = TOTAL(tmp,1,/nan)
  for k=0,ntrace-1 do begin
    bp = where(tmp[*,k] gt 0,nbp)
    if nbp gt 0 and norm[k] gt 0 then begin
      yy = bp - loarr[k]
      good = where(yy ge 0 and yy le (hiarr[k]-loarr[k]),ngood)
      if ngood gt 0 then begin
        p = imgptr[k]
        (*p)[i,yy[good]] = tmp[bp[good],k]/norm[k]
      endif
    endif
  endfor

endfor

file = apogee_filename('EPSF',chip=chip[ichip],num=im)
sxdelpar,head,'NAXIS1'
sxdelpar,head,'NAXIS2'
MWRFITS,0,file,head,/create
print,'Writing ',file

;; Put the PSFs in the output structure.  Trim each compact strip down to the
;; actually non-zero rows before writing, preserving the original LO/HI meaning.
print,'ntrace=',ntrace
for k=0,ntrace-1 do begin
  p = imgptr[k]
  m = TOTAL(*p,1,/nan)
  ind = where(finite(m) and m ne 0,nind)
  if nind gt 0 then begin
    i1 = MIN(ind)
    i2 = MAX(ind)
    outpsf = {fiber: fiber[k], cent: trace[*,k], lo: loarr[k]+i1, hi: loarr[k]+i2, img: (*p)[*,i1:i2]}
    MWRFITS,outpsf,file,/silent
  endif else begin
    print,'not halted, but bad PSF at: ',k
  endelse
  ptr_free,p
endfor

end

