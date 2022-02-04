;+
;
; AP2D
;
; This program processes 2D APOGEE spectra.  It extracts the
; spectra.
;
; INPUTS:
;  planfiles     Input list of plate plan files
;  =exttype      
;  =mapper_data  Directory for mapper data.
;  /verbose      Print a lot of information to the screen
;  /clobber      Overwrite existing files (ap1D).
;  /calclobber   Overwrite existing daily calibration files (apPSF, apFlux).
;  /domelibrary  Use the domeflat library.
;  /stp          Stop at the end of the prrogram
;  /unlock      Delete lock file and start fresh
;
; OUTPUTS:
;  1D extracted spectra are output.  One file for each frame.
;
; USAGE:
;  IDL>ap2d
;
; Written by D.Nidever  Mar. 2010
; Modifications: J. Holtzman 2011+
;-

pro ap2d,planfiles,verbose=verbose,stp=stp,clobber=clobber,exttype=exttype,mapper_data=mapper_data,$
         calclobber=calclobber,domelibrary=domelibrary,unlock=unlock

common savedepsf, savedepsffiles, epsfchip

savedepsffiles = [' ',' ',' ']
epsfchip = 0

;; Default parameters
if n_elements(verbose) eq 0 then verbose=0  ; NOT verbose by default
; calclobber will redo PSF, Flux and 1D frames (but not other fundamental calibration frames)
if not keyword_set(calclobber) then calclobber=0  ; NOT calclobber by default,
; clobber will redo 1D frames
if not keyword_set(clobber) then clobber=0  ; NOT clobber by default,
if not keyword_set(exttype) then exttype=4
if n_elements(domelibrary) eq 0 then domelibrary=0

t0 = systime(1)

nplanfiles = n_elements(planfiles)
; Not enough inputs
if nplanfiles eq 0 then begin
  print,'Syntax - ap2d,planfiles'
  return
endif

print,''
print,'RUNNING AP2D'
print,''
print,strtrim(nplanfiles,2),' PLAN files'

chiptag = ['a','b','c']
apgundef,wavefile,responsefile

;--------------------------------------------
; Loop through the unique PLATE Observations
;--------------------------------------------
FOR i=0L,nplanfiles-1 do begin

  t1 = systime(1)
  planfile = planfiles[i]

  print,''
  print,'========================================================================='
  print,strtrim(i+1,2),'/',strtrim(nplanfiles,2),'  Processing Plan file ',planfile
  print,'========================================================================='

  ; Load the plan file
  ;--------------------
  print,'' & print,'Plan file information:'
  APLOADPLAN,planfile,planstr,/verbose,error=planerror
  if n_elements(planerror) gt 0 then goto,BOMB
  logfile = apogee_filename('Diag',plate=planstr.plateid,mjd=planstr.mjd)

  ;; Add PSFID tag to planstr APEXP structure
  if tag_exist(planstr.apexp,'psfid') eq 0 then begin
    apexp = planstr.apexp
    add_tag,apexp,'psfid',0L,apexp
    old = planstr
    oldtags = tag_names(old)
    planstr = create_struct(oldtags[0],old.(0))
    for j=1,n_elements(oldtags)-1 do begin
      if oldtags[j] eq 'APEXP' then begin
        planstr = create_struct(planstr,'APEXP',apexp)
      endif else begin
        planstr = create_struct(planstr,oldtags[j],old.(j))
      endelse
    endfor
    undefine,old,oldtags
  endif

  ;; Use domeflat library
  ;;---------------------
  ;; if (1) no domeflat ID set in planfile, or (2) domelibrary parameter
  ;; set in planfile, or (3) /domelibrary keyword is set.
  if tag_exist(planstr,'domelibrary') eq 1 then plandomelibrary=planstr.domelibrary else plandomelibrary=0
  if keyword_set(domelibrary) or tag_exist(planstr,'psfid') eq 0 or keyword_set(plandomelibrary) then begin
    print,'Using domeflat library'
    ;; You can do "domeflattrace --mjdplate" where mjdplate could be
    ;; e.g. 59223-9244, or "domeflattrace --planfile", with absolute
    ;; path of planfile
    ;; Force single domeflat if a short visit or domelibrary=='single'
    if planstr.telescope eq 'apo25m' or planstr.telescope eq 'apo1m' then observatory='apo' else observatory='lco'
    if strtrim(domelibrary,2) eq 'single' or strtrim(plandomelibrary,2) eq 'single' or n_elements(planstr.apexp) le 3 then begin
      spawn,['domeflattrace',observatory,'--planfile',planfile,'--s'],out,errout,/noshell
    endif else begin
      spawn,['domeflattrace',observatory,'--planfile',planfile],out,errout,/noshell
    endelse
    nout = n_elements(out)
    for f=0,nout-1 do print,out[f]
    ;; Parse the output
    lo = where(stregex(out,'^DOME FLAT RESULTS:',/boolean) eq 1,nlo)
    hi = first_el(where(strtrim(out,2) eq '' and lindgen(nout) gt lo[0]))
    if lo eq -1 or hi eq -1 then begin
      print,'Problem running domeflattrace for ',planfile,'.  Skipping this planfile.'
      continue
    endif
    outarr = strsplitter(out[lo+1:hi-1],' ',/extract)
    ims = reform(outarr[0,*])
    domeflatims = reform(outarr[1,*])
    ;; Update planstr
    match,apexp.name,ims,ind1,ind2,/sort
    planstr.apexp[ind1].psfid = domeflatims[ind2]
  endif else begin
    planstr.apexp.psfid = planstr.psfid
  endelse

  ; Don't extract dark frames
  if tag_exist(planstr,'platetype') then $
    if planstr.platetype eq 'dark' or planstr.platetype eq 'intflat' then goto,BOMB

  ; Try to make the required calibration files (if not already made)
  ; Then check if the calibration files exist
  ;--------------------------------------

  ; apPSF files 
  if planstr.sparseid ne 0 then makecal,sparse=planstr.sparseid
  if planstr.fiberid ne 0 then makecal,fiber=planstr.fiberid
  if tag_exist(planstr,'psfid') then begin
    MAKECAL,psf=planstr.psfid,clobber=calclobber
    tracefiles = apogee_filename('PSF',num=planstr.psfid,chip=chiptag)
    tracefile = file_dirname(tracefiles[0])+'/'+string(format='(i8.8)',planstr.psfid)
    tracetest = FILE_TEST(tracefiles)  
    if min(tracetest) eq 0 then begin
      bd1 = where(tracetest eq 0,nbd1)
      if nbd1 gt 0 then stop,'halt: ',tracefiles[bd1],' NOT FOUND'
      for ichip=0,2 do begin
        p = mrdfits(tracefiles[ichip],1,/silent)
        if n_elements(p) ne 300 then begin
          print, 'halt: tracefile ', tracefiles[ichip],' does not have 300 traces'
        endif
      endfor
    endif 
  endif

  ; apWave files : wavelength calibration
  waveid = planstr.waveid
  if tag_exist(planstr,'platetype') then if planstr.platetype eq 'cal' or planstr.platetype eq 'extra' then waveid=0
  if waveid gt 0 then MAKECAL,multiwave=waveid

  ; FPI calibration file
  if tag_exist(planstr,'fpi') then fpiid = planstr.fpi else fpiid=0

  ; apFlux files : since individual frames are usually made per plate
  if planstr.fluxid ne 0 then begin
    MAKECAL,flux=planstr.fluxid,psf=planstr.psfid,clobber=calclobber
    fluxfiles = apogee_filename('Flux',chip=chiptag,num=planstr.fluxid)
    fluxfile = file_dirname(fluxfiles[0])+'/'+string(format='(i8.8)',planstr.fluxid)
    fluxtest = FILE_TEST(fluxfiles)  
    if min(fluxtest) eq 0 then begin
      bd1 = where(fluxtest eq 0,nbd1)
      if nbd1 gt 0 then stop,'halt: ',fluxfiles[bd1],' NOT FOUND'
    endif
  endif else fluxtest=0

  ; apResponse files 
  ;  these aren't used anymore
  if tag_exist(planstr,'responseid') eq 0 then add_tag,planstr,'responseid',0,planstr
  if planstr.responseid ne 0 then begin
    MAKECAL,response=planstr.responseid
    responsefiles = apogee_filename('Response',chip=chiptag,num=planstr.responseid)
    responsefile = file_dirname(responsefiles[0])+'/'+string(format='(i8.8)',planstr.responseid)
    responsetest = FILE_TEST(responsefiles)  
    if min(responsetest) eq 0 then begin
      bd1 = where(responsetest eq 0,nbd1)
      if nbd1 gt 0 then stop,'halt: ',responsefiles[bd1],' NOT FOUND'
    endif
  endif

  ; Load the Plug Plate Map file
  ;------------------------------
  if tag_exist(planstr,'platetype') then if planstr.platetype eq 'cal' or planstr.platetype eq 'extra' or $
     planstr.platetype eq 'single' then plugmap=0 else begin
    print,'' & print,'Plug Map file information:'
    plugfile = planstr.plugmap
    if tag_exist(planstr,'fixfiberid') then fixfiberid=planstr.fixfiberid
    if size(fixfiberid,/type) eq 7 and n_elements(fixfiberid) eq 1 then $
      if (strtrim(fixfiberid,2) eq 'null' or strtrim(strlowcase(fixfiberid),2) eq 'none') then undefine,fixfiberid  ;; null/none  
    if tag_exist(planstr,'badfiberid') then badfiberid=planstr.badfiberid
    if size(badfiberid,/type) eq 7 and n_elements(badfiberid) eq 1 then $
      if (strtrim(badfiberid,2) eq 'null' or strtrim(strlowcase(badfiberid),2) eq 'none') then undefine,badfiberid  ;; null/none  
    ;; we only need the information on sky fibers
    plugmap = getplatedata(planstr.plateid,string(planstr.mjd,format='(i5.5)'),plugid=planstr.plugmap,fixfiberid=fixfiberid,$
                           badfiberid=badfiberid,mapper_data=mapper_data,/noobject)
    if n_elements(plugerror) gt 0 then stop,'halt: error with plugmap: ',plugfile
    plugmap.mjd = planstr.mjd   ; enter MJD from the plan file
  endelse

  ; Are there enough files
  nframes = n_elements(planstr.apexp)
  if nframes lt 1 then begin
    print,'No frames to process'
    goto,BOMB
  endif

  ; Process each frame
  ;-------------------
  For j=0L,nframes-1 do begin

    ;; Get trace files
    tracefiles = apogee_filename('PSF',num=planstr.apexp[i].psfid,chip=chiptag)
    tracefile = file_dirname(tracefiles[0])+'/'+string(format='(i8.8)',planstr.apexp[i].psfid)
    tracetest = FILE_TEST(tracefiles)  
    if min(tracetest) eq 0 then begin
      bd1 = where(tracetest eq 0,nbd1)
      if nbd1 gt 0 then stop,'halt: ',tracefiles[bd1],' NOT FOUND'
      for ichip=0,2 do begin
        p = mrdfits(tracefiles[ichip],1,/silent)
        if n_elements(p) ne 300 then begin
          print, 'halt: tracefile ', tracefiles[ichip],' does not have 300 traces'
        endif
      endfor
    endif 

    ; Make the filenames and check the files
    rawfiles = apogee_filename('R',chip=chiptag,num=planstr.apexp[j].name)
    rawinfo = APFILEINFO(rawfiles,/silent)        ; this returns useful info even if the files don't exist
    framenum = rawinfo[0].fid8   ; the frame number
    files = apogee_filename('2D',chip=chiptag,num=framenum)
    inpfile = file_dirname(files[0])+'/'+framenum
    info = APFILEINFO(files,/silent)
    okay = (info.exists AND info.sp2dfmt AND info.allchips AND (info.mjd5 eq planstr.mjd) AND $
            ((info.naxis eq 3) OR (info.exten eq 1)))
    if min(okay) lt 1 then begin
      bd = where(okay eq 0,nbd)
      stop,'halt: There is a problem with files: ',strjoin((files)(bd),' ')
    endif

    print,''
    print,'-----------------------------------------'
    print,strtrim(j+1,2),'/',strtrim(nframes,2),'  Processing Frame Number >>',strtrim(framenum,2),'<<'
    print,'-----------------------------------------'

    ; Run AP2DPROC
    if tag_exist(planstr,'platetype') then if planstr.platetype eq 'cal' then skywave=0 else skywave=1
    if tag_exist(planstr,'platetype') then if planstr.platetype eq 'sky' then plugmap=0
    outdir=apogee_filename('1D',num=framenum,chip='a',/dir)
    if file_test(outdir,/directory) eq 0 then FILE_MKDIR,outdir
    if min(fluxtest) eq 0 or planstr.apexp[j].flavor eq 'flux' then $
      AP2DPROC,inpfile,tracefile,exttype,outdir=outdir,unlock=unlock,$
               wavefile=wavefile,skywave=skywave,plugmap=plugmap,clobber=clobber,/compress $
    else if waveid gt 0 then begin
      AP2DPROC,inpfile,tracefile,exttype,outdir=outdir,unlock=unlock,$
               fluxcalfile=fluxfile,responsefile=responsefile,$
               wavefile=wavefile,skywave=skywave,plugmap=plugmap,clobber=clobber,/compress 
    endif else $
      AP2DPROC,inpfile,tracefile,exttype,outdir=outdir,unlock=unlock,$
               fluxcalfile=fluxfile,responsefile=responsefile,$
               clobber=clobber,/compress 

    BOMB1:

  Endfor ; frame loop

  ;; Now add in wavelength calibration information, with shift from
  ;;  fpi or sky lines
  ;; This used to call "apskywavecal", "ap1dwavecal" now handles
  ;; both cases (sky lines and FPI lines)
  if waveid gt 0 or fpiid gt 0 then begin
    cmd = ['ap1dwavecal',planfile]

    ;;;; Check if there is FPI flux in the 2 fibers
    ;;if fpiid gt 0 then begin
    ;;  outfile1 = apogee_filename('1D',num=framenum,chip='b')
    ;;  if file_test(outfile1) eq 0 then begin
    ;;    print,outfile1,' NOT FOUND'
    ;;    return
    ;;  endif
    ;;  ;;fits_read,outfile1,flux,head,exten=0
    ;;  ;;stop
    ;;endif

    ;; Don't use FPI fibers until we are using it routinely!!!
    ;;if fpiid gt 0 then begin  ;; use FPI lines
    ;;   cmd = [cmd,'--fpiid',strtrim(fpiid,2)]
    ;;endif else begin  ;; use sky lines
    if not keyword_set(skywave) then cmd=[cmd,'--nosky']
    ;;endelse
    spawn,cmd,/noshell
    ;; if skywave then spawn,['apskywavecal',planfile],/noshell $
    ;; else  spawn,['apskywavecal',planfile,'--nosky'],/noshell
  endif

  BOMB:

  ; Compress 2D files
  nframes = n_elements(planstr.apexp)
  for j=0L,nframes-1 do begin
    files = apogee_filename('2D',num=planstr.apexp[j].name,chip=chiptag)
    modfiles = apogee_filename('2Dmodel',num=planstr.apexp[j].name,chip=chiptag)
    for jj=0,n_elements(files)-1 do begin
      if file_test(files[jj]) then begin
        file_delete,files[jj]+'.fz',/allow_nonexistent
 ;       SPAWN,['fpack','-D','-Y',files[jj]],/noshell
      endif
      if file_test(modfiles[jj]) then begin
        file_delete,modfiles[jj]+'.fz',/allow_nonexistent
        SPAWN,['fpack','-D','-Y',modfiles[jj]],/noshell
      endif
    endfor
  endfor

  writelog,logfile,'AP2D: '+file_basename(planfile)+string(format='(f8.2)',systime(1)-t1)

ENDFOR  ; plan file loop

apgundef,epsfchip

print,'AP2D finished'
dt = systime(1)-t0
print,'dt = ',strtrim(string(dt,format='(F10.1)'),2),' sec'

if keyword_set(stp) then stop

end
