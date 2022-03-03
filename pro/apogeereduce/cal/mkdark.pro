;+
;
; MKDARK
;  
; Makes APOGEE superdark calibration product.
;
; INPUT:
;   ims: list of image numbers to include in superdark
;   cmjd=cmjd : (optional,obsolete) gives MJD directory name if not encoded in file number
;   step=step : (optional,obsolete) process every step image in UTR
;   psfid=psfid : (optional) EPSF id to use to try to subtract off thermal traces
;  /unlock : delete lock file and start fresh 
;
; OUTPUT:
;   A set of apDark-[abc]-ID8.fits files.
;
; USAGE:
;   IDL>mkdark,ims,cmjd=cmjd,step=step,psfid=psfid
;
; By J. Holtzman, 2011??
;  Updates, added doc strings, and cleanup by D. Nidever, Sep 2020
;-

pro mkdark,ims,cmjd=cmjd,step=step,psfid=psfid,clobber=clobber,unlock=unlock

i1 = ims[0]
nframes = n_elements(ims)

dirs = getdir()
caldir = dirs.caldir

adarkfile = apogee_filename('Dark',num=i1,chip='a')
darkdir = file_dirname(adarkfile)
prefix = strmid(file_basename(adarkfile),0,2)
darkfile = darkdir+'/'+prefix+string(format='("Dark-",i8.8)',i1) +'.tab'
lockfile = darkfile+'.lock'
chips = ['a','b','c']

;; Does file already exist?
;; check all three chip files and the .tab file
sdarkid = string(ims[0],format='(i08)')
allfiles = darkdir+'/'+[dirs.prefix+'Dark-'+chips+'-'+sdarkid+'.fits',dirs.prefix+'Dark-'+sdarkid+'.tab']
if total(file_test(allfiles)) eq 4 and not keyword_set(clobber) then begin
  print,' Dark file: ', darkfile, ' already made'
  return
endif
file_delete,allfiles,/allow  ;; delete any existing files to start fresh

;; Is another process already creating file
if not keyword_set(unlock) then begin
  while file_test(lockfile) do apwait,lockfile,10
endif else begin
  if file_test(lockfile) then file_delete,lockfile,/allow
endelse

;; Open lock file
openw,lock,/get_lun,lockfile
free_lun,lock

;; Initialize summary structure
sum = {num:i1, nframes:0, nreads:0, nsat:0L, nhot:0L, nhotneigh:0L, nbad:0L, medrate:0., psfid:0L, nneg:0L}
darklog = REPLICATE(sum,3)

if not keyword_set(step) then step=0
;; Loop over the chips
for ichip=0,n_elements(chips)-1 do begin
  chip = chips[ichip]

  time0 = systime(/seconds)
  ii = 0
  for jj=0,n_elements(ims)-1 do begin
    i = ims[jj]
    if not keyword_set(cmjd) then cm=getcmjd(i) else cm=cmjd
    print,strtrim(jj,2)+'/'+strtrim(n_elements(ims),2),chip,i

    ;; Process (bias-only) each individual frame
    d = process(cm,i,chip,head,r,step=step,/nofs,/nofix,/nocr)
    print,'Done process'
    sz = size(d)
    if sz[1] ne 2048 then stop,sz
    mask = bytarr(sz[1],sz[2])

    ;; Construct cube of reads minus second read
    if jj eq 0 then head0=head
    sz = size(r)
    if jj eq 0 then begin
      if ichip eq 0 then red=fltarr(2048,2048,sz[3],nframes) else red*=0.
    endif
    red[*,*,*,ii] = r
    apgundef,r
    for iread=sz[3]-1,1,-1 do begin
      red[*,*,iread,ii] -= red[*,*,1,ii]
    endfor
    ii = ii+1
    help,/mem
  endfor
 
  ;; Median them all
  print,'Median...'
  dark = median(red,dimension=4)

  ;; Option to remove any trace of spectral traces
  if keyword_set(psfid) then begin
    darklog[ichip].psfid = psfid
    print,'reading epsf '
    epsffile = apogee_filename('EPSF',psfid,chip=chip)
    ;file = dirs.prefix+string(format='("EPSF-",a,"-",i8.8)',chip,psfid)
    tmp = mrdfits(epsffile,0,head)
    ntrace = sxpar(head,'NTRACE')
    img = ptrarr(ntrace,/allocate_heap)
    for i=0,ntrace-1 do begin
      ptmp = mrdfits(epsffile,i+1,/silent)
      *img[i] = ptmp.img
      p = {lo: ptmp.lo, hi: ptmp.hi, img: img[i]}
      if i eq 0 then psf=replicate(p,ntrace)
      psf[i] = p
    endfor
    nread = sz[3]
    for iread=2,nread-1 do begin
      var = dark[*,*,iread]
      ;; Want to subtract off mean background dark level before fitting traces
      ;; Iterate once for this
      back = median(dark[*,*,iread],10)
      niter = 2
      for iter=0,niter-1 do begin
        print,iread,iter
        d = dark[*,*,iread]-back
        spec = extract(d,ntrace,psf,var)
        sspec = zap(spec,[200,1])
        d *= 0
        for k=0,ntrace-1 do begin
          p1 = psf[k]
          lo = psf[k].lo & hi=psf[k].hi
          img = *p1.img
          r = intarr(hi-lo+1)+1
          sub = sspec[*,k]#r
          bad = where(sub lt 0,nbad)
          if nbad gt 0 then sub[bad]=0
          d[*,lo:hi] += sub*img
        endfor
        if iter lt niter-1 then back=median(dark[*,*,iread]-d,10)
      endfor
      dark[*,*,iread] -= d
    endfor
  endif

  ;; Flag "hot" pixels in mask image
  nread = sz[3]
  rate = (dark[*,*,nread-1]-dark[*,*,1])/(nread-2)

  ;; Create mask array
  ;; NaN is bad!
  bad = where(finite(rate) eq 0,nsat) 
  if nsat gt 0 then mask[bad]=mask[bad] or 1

  ;; Flux accumulating very fast is bad!
  maxrate = 10.
  hot = where(rate gt maxrate,nhot)
  if nhot gt 0 then mask[hot]=mask[hot] or 2
  ;; Flag adjacent pixels to hot pixels as bad at 1/4 the maximum rate
  n = [-1,1,-2048,2048]
  nhotneigh = 0
  for in=0,3 do begin
    ;; Only consider neighbors on the chip!
    neigh = hot+n[in]
    on = where(neigh ge 0 and neigh lt 2048L*2048L)
    nlow = where(rate[neigh[on]] gt maxrate/4.,nbad)
    if nbad ge 0 then mask[neigh[on[nlow]]]=mask[neigh[on[nlow]]] or 4
    nhotneigh += n_elements(hot)
    ;; Same for bad
    neigh = bad+n[in]
    on = where(neigh ge 0 and neigh lt 2048L*2048L)
    nlow = where(rate[neigh[on]] gt maxrate/4.,nbad)
    if nbad gt 0 then mask[neigh[on[nlow]]]=mask[neigh[on[nlow]]] or 4
    nhotneigh += n_elements(hot)
  endfor

  print,'Creating chi2 array ....'
  chi2 = fltarr(2048L*2048*nread)
  n = intarr(2048L*2048*nread)
  dark = reform(dark,2048L*2048*nread,/overwrite)
  for ii=0,nframes-1 do begin
    tmp = reform(red[*,*,*,ii],2048L*2048*nread)
    good = where(finite(tmp),ngood)
    if ngood gt 0 then chi2[good]+=(tmp[good]-dark[good])^2/apvariance(dark[good],1)
    n[good] += 1
  endfor
  chi2 /= n
  dark = reform(dark,2048,2048,nread,/overwrite)
  chi2 = reform(chi2,2048,2048,nread,/overwrite)

  ;; Set nans to 0 before writing
  bad = where(finite(dark) eq 0,nbad)
  if nbad gt 0 then dark[bad]=0.
  medrate = median(rate)

  ;; Median filter along reads dimenstion
  for i=0,2047 do begin
    slice = reform(dark[i,*,*])
    dark[i,*,*] = medfilt2d(slice,7,dim=2)
  endfor 

  ;; Set negative pixels to zero
  neg = where(dark lt -10,nneg)
  if nneg gt 0 then dark[neg]=0.

  ;; Write them out
  if step gt 1 then  $
    file = prefix+string(format='("Dark",i1,"-",a,"-",i8.8)',step,chip,i1) $
  else $
    file = prefix+string(format='("Dark-",a,"-",i8.8)',chip,i1) 

  leadstr = 'APMKDARK: '
  sxaddhist,leadstr+systime(0),head0
  info = GET_LOGIN_INFO()
  sxaddhist,leadstr+info.user_name+' on '+info.machine_name,head0
  sxaddhist,leadstr+'IDL '+!version.release+' '+!version.os+' '+!version.arch,head0
  sxaddhist,leadstr+' APOGEE Reduction Pipeline Version: '+getvers(),head0
  MWRFITS,0,darkdir+'/'+file+'.fits',head0,/create
  MWRFITS,dark,darkdir+'/'+file+'.fits'
  MWRFITS,chi2,darkdir+'/'+file+'.fits'
  MWRFITS,mask,darkdir+'/'+file+'.fits'

  ;; Make some plots/images
  if not file_test(darkdir+'/plots',/dir) then file_mkdir,darkdir+'/plots'
  DARKPLOT,dark,mask,darkdir+'/plots/'+file,/hard
 
  ;; Summary data table
  darklog[ichip].num = i1
  darklog[ichip].nframes = nframes
  darklog[ichip].nreads = nread
  darklog[ichip].nsat = nsat
  darklog[ichip].nhot = nhot
  darklog[ichip].nhotneigh = nhotneigh
  darklog[ichip].nbad = nbad
  darklog[ichip].medrate = medrate
  darklog[ichip].nneg = nneg
 
  ;; Save the rate file
  file = prefix+string(format='("DarkRate-",a,"-",i8.8)',chip,i1) 
  MWRFITS,rate,darkdir+'/'+file+'.fits',/create

  dark = 0
  time = systime(/seconds)
  print,'Done '+chip,time-time0

endfor

apgundef,red

;; Write the summary log information
file = prefix+string(format='("Dark-",i8.8)',i1) 
MWRFITS,darklog,darkdir+'/'+file+'.tab',/create

;; Remove lock file
file_delete,lockfile,/allow

;; Compile summary web page
DARKHTML,darkdir

end
