;+
;
; MKFLAT
;
; Makes APOGEE superflat calibration files from dithered individual frames.
;
; INPUT:
;  ims: list of image numbers to include in superflat
;  cmjd=cmjd : (optional,obsolete) gives MJD directory name if not encoded in file number
;  darkid=darkid : dark frame to be used if images are reduced
;  /clobber : rereduce images even if they exist
;  /kludge : set bottom and top non-illuminated pixels to unity
;  nrep=nrep : median filters each batch of nrep frames before combining
;  /unlock : delete lock file and start fresh 
;
; OUTPUT:
;  A set of apFlat-[abc]-ID8.fits files in the appropriate location
;   determined by the SDSS/APOGEE tree directory structure.
;
; USAGE:
;  IDL>mkflat,ims,cmjd=cmjd,darkid=darkid,/clobber,/kludge,nrep=nrep
;
; By J. Holtzman, 2011?
;  Added doc strings, updates to use data model  D. Nidever, Sep 2020 
;-

pro mkflat,ims,cmjd=cmjd,darkid=darkid,clobber=clobber,kludge=kludge,nrep=nrep,dithered=dithered,unlock=unlock

  i1 = ims[0]
  nframes = n_elements(ims)
  if not keyword_set(nrep) then nrep=1

  dirs = getdir(apodir,caldir,specdir,apovers,libdir,datadir=datadir)

  flatdir = apogee_filename('Flat',num=i1,chip='c',/dir)
  flatfile = flatdir+dirs.prefix+string(format='("Flat-",i8.8)',i1)+'.tab'
  lockfile = flatfile+'.lock'
  ;; Is another process already creating file?
  if not keyword_set(unlock) then begin
    while file_test(lockfile) do apwait,flatfile,10
  endif else begin
    if file_test(lockfile) then file_delete,lockfile,/allow
  endelse

  ;: Does file already exist?
  ;; check all three chip files
  sflatid = string(ims[0],format='(i08)')
  chip = ['a','b','c']
  allfiles = flatdir+dirs.prefix+'Flat-'+chip+'-'+sflatid+'.fits'
  allfiles = [allfiles,flatdir+dirs.prefix+'Flat-'+sflatid+'.tab']
  if total(file_test(allfiles)) eq 4 and not keyword_set(clobber) then begin
    print,' Flat file: ', flatfile, ' already made'
    return
  endif
  file_delete,allfiles,/allow  ;; delete any existing files to start fresh

  ;; Open lock file
  openw,lock,/get_lun,lockfile
  free_lun,lock

  sum = {name: '',num: i1, nframes: 0}
  flatlog = REPLICATE(sum,3)

  perclow = 0.85              ; fraction for rejecting pixels
  nperclow = 0.95             ; fraction for rejecting neighbor pixels 
  perchi = 1.25               ; fraction for rejecting pixels
  nperchi = 1.05              ; fraction for rejecting neighbor pixels 
  x1norm = 800 & x2norm=1200  ; region for getting normalization 
  y1norm = 800 & y2norm=1000
  filter = [50,1]             ; filter size for smoothing for large scale structure
 
  outdir = flatdir
  if file_test(outdir,/directory) eq 0 then file_mkdir,outdir
  nfs = 1
  uptheramp = 0
  nocr = 1
  cmjd = getcmjd(ims[0],mjd=mjd)
  GETCAL,mjd,dirs.calfile,dark=darkid,bpm=bpmid,det=detid

  ;; Read and process frames to 2D
  for ichip=0,2 do begin
    if darkid gt 0 then darkcorr=apogee_filename('Dark',num=darkid,chip=chip[ichip])
    if detid gt 0 then detcorr=apogee_filename('Detector',num=detid,chip=chip[ichip])
    for inum=0,n_elements(ims)-1 do begin
      num = ims[inum]
      ifile = apogee_filename('R',num=num,chip=chip[ichip])
      ofile = apogee_filename('2D',num=num,chip=chip[ichip],/base)
      AP3DPROC,ifile,outdir+ofile,detcorr=detcorr,darkcorr=darkcorr,$
               nocr=nocr,uptheramp=uptheramp,nfowler=nfs,fitsdir=getlocaldir()
    endfor
  endfor
  
  ;; Sum up all of the individual flats
  ;;  Median nrep frames before summing if requested
  flats = fltarr(2048,2048,3,nframes)
  flatmasks = intarr(2048,2048,3)
  flatsum = fltarr(2048,2048,3)
  for ii=0,nframes-1,nrep do begin
    for irep=0,nrep-1 do begin
      i = ims[ii+irep]
      for ichip=0,2 do begin
        ofile = apogee_filename('2D',num=i,chip=chip[ichip],/base)
        f = mrdfits(outdir+ofile,0,head)
        flats[*,*,ichip,ii+irep] = mrdfits(outdir+ofile,1)
        flatmasks[*,*,ichip] = mrdfits(outdir+ofile,3)
      endfor
    endfor
    if ii eq 0 then head0=head
    if nrep gt 1 then flatsum+=median(flats[*,*,*,ii:ii+nrep-1],dimension=4) $
    else flatsum+=flats[*,*,*,ii]
  endfor

  ;; Normalize the flatsums to roughly avoid discontinuities across chips
  ;; Normalize center of middle chip to unity
  norm = median(flatsum[x1norm:x2norm,y1norm:y2norm,1])
  flatsum[*,*,1] /= norm
  flatsum[*,*,0] /= median(flatsum[1950:2044,500:1500,0])/median(flatsum[5:100,500:1500,1])
  flatsum[*,*,2] /= median(flatsum[5:100,500:1500,2])/median(flatsum[1950:2044,500:1500,1])

  ;; Create the superflat 
  for ichip=0,2 do begin
    flat = flatsum[*,*,ichip]

    ;; Create mask
    sz = size(flat)
    mask = bytarr(sz[1],sz[2])

    ;; Mask from reductions, using last frame read
    bad = where((flatmasks[*,*,ichip] and badmask()) gt 0,nbad)
    if nbad gt 0 then flat[bad]=!values.f_nan
    if nbad gt 0 then mask[bad]=mask[bad] or 1

    ;; Set pixels to bad when below some fraction 
    localflat = flat/zap(flat,[100,10])
    ;; Relative to neighbors
    low = where(localflat lt perclow,nlow)
    if nlow gt 0 then mask[low]=mask[low] or 2
    ;; Absolute
    low = where(flat lt 0.1,nlow)
    if nlow gt 0 then mask[low]=mask[low] or 2
    ;; High pixels
    hi = where(localflat gt perchi,nhi)
    if nhi gt 0 then mask[hi]=mask[hi] or 2

    ;; Set neighboring pixels to bad at slightly lower threshold, iteratively
    for iter=0,10  do begin
      low = where(mask gt 0)
      n = [-1,1,-2049,-2048,-2047,2047,2048,2049]
      for in=0,n_elements(n)-1 do begin
        neigh = low+n[in]
        off = where(neigh lt 0 or neigh gt 2048L*2048,noff)
        if noff gt 0 then neigh[off]=0
        lowneigh = where(localflat[neigh] lt nperclow,nlow)
        if nlow gt 0 then mask[neigh[lowneigh]]=mask[neigh[lowneigh]] or 4
        hineigh = where(localflat[neigh] gt nperchi,nhi)
        if nhi gt 0 then mask[neigh[hineigh]]=mask[neigh[hineigh]] or 4
      endfor
    endfor

    ;; Mask any zero values
    bad = where(flat eq 0.,nbad)
    if nbad gt 0 then mask[bad]=mask[bad] or 8
    if nbad gt 0 then flat[bad]=!values.f_nan
 
    if keyword_set(dithered) then begin
      ;; Get the large scale structure from smoothing, avoiding bad pixels (NaNs)
      sm = smooth(flat,100,/nan,/edge_truncate)
      rows = intarr(2048)+1
      smrows = (total(sm,1,/nan)/2048)##rows
      smcols = rows##(total(sm,2,/nan)/2048)
   
      ;; Median filter the median flat with a rectangular filter and 
      ;;   divide flat by this to remove horizontal structure
      sflat = zap(flat,filter)
      flat /= sflat
  
      ;; Now put the large scale structure in apart from
      ;;  structure that is horizontal (fibers) or vertical (SED)
      flat *= sm/smrows/smcols
  
      ;; Kludge to set unilluminated pixels to 1
      for i=0,13 do  begin
        dark = where(flat[*,i] lt -99, ndark)
        if ndark gt 0 then flat[dark,i]=1.
        if ndark gt 0 then mask[dark,i]=0
        dark = where(flat[*,2047-i] lt -99, ndark)
        if ndark gt 0 then flat[dark,2047-i]=1.
        if ndark gt 0 then mask[dark,2047-i]=0
      endfor
    endif else begin
      ;; If not dithered, still take out spectral signature
      rows = intarr(2048)+1
      cols = fltarr(2048)

      ;; Spectral signature from median of each column
      for icol=0,2047 do cols[icol]=median(flat[icol,*])
      ;; Medfilt doesn't do much if intensity is varying across cols
      smrows = rows##medfilt1d(cols,100)
      sflat = smrows

      ;; Dec 2018: don't take out median spectral signature, this leaves 
      ;;  structure in spectra that is hard to normalize out
      ;; instead, take out a low order polynomial fit to estimate spectral signature
      x = indgen(2048)
      gd = where(finite(cols))
      coef = robust_poly_fit(x[gd],cols[gd],2)
      smrows = rows##poly(x,coef)
      sflat = smrows

      ;; Feb 2019: polynomial fit introduces spurious signal, so just don't bother with spectral signature!
      ;; divide out estimate of spectral signature
      ;;flat/=smrows

    endelse
    ;; Set bad values to -100 before writing to avoid NaNs in output file
    bad = where(finite(flat) eq 0,nbad)
    if nbad gt 0 then flat[bad]=0.

    ;; Write it out!
    file = apogee_filename('Flat',num=i1,chip=chip[ichip])
    leadstr = 'APMKFLAT: '
    sxaddhist,leadstr+systime(0),head0
    info = GET_LOGIN_INFO()
    sxaddhist,leadstr+info.user_name+' on '+info.machine_name,head0
    sxaddhist,leadstr+'IDL '+!version.release+' '+!version.os+' '+!version.arch,head0
    sxaddhist,leadstr+' APOGEE Reduction Pipeline Version: '+getvers(),head0
    MWRFITS,0,file,head0,/create
    MWRFITS,flat,file
    MWRFITS,sflat,file
    MWRFITS,mask,file

    ;; Make a jpg of the flat
    if not file_test(flatdir+'plots',/dir) then file_mkdir,flatdir+'plots'
    FLATPLOT,flat,flatdir+'plots/'+file_basename(file,'.fits')

    flatlog[ichip].name = file
    flatlog[ichip].num = i1
    flatlog[ichip].nframes = nframes
  endfor

  ;; Write out flat summary information
  file = dirs.prefix+string(format='("Flat-",i8.8)',i1)+'.tab'
  MWRFITS,flatlog,flatdir+file,/create

  ;; Wemove lock file
  file_delete,lockfile,/allow_non

  ;; Cmpile summary web page
  FLATHTML,caldir

end
