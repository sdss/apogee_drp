;+
;
; MKDAILYWAVE
;
; Procedure to make an APOGEE daily wavelength calibration file
; from 8 nearby days of arclamp exposures.  This is a wrapper
; around the python apmultiwavecal program.
;
; INPUT:
;  mjd          MJD of the night for which to make the daily
;                 wavelength solution.
;  =darkid      Dark frame to be used if images are reduced.
;  =flatid      Flat frame to be used if images are reduced.
;  =psfid       PSF frame to be used if images are reduced.
;  =modelpsf    Model PSF calibration frame to use.
;  =fiberid     ETrace frame to be used if images are reduced.
;  /nowait      If file is already being made then don't wait
;                 just return.
;  /clobber     Overwrite existing files.
;  /nofit       Skip fit (find lines only).
;  /unlock      Delete the lock file and start fresh.
;  /psflibrary  Use PSF library for extraction.
;
; OUTPUT:
;  A set of apWave-[abc]-ID8.fits files in the appropriate location
;   determined by the SDSS/APOGEE tree directory structure.
;
; USAGE:
;  IDL>mkdailywave,mjd,darkid=darkid,flatid=flatid,psfid=psfid,fiberid=fiberid,/clobber
;
; Made from mkwave.pro by D.Nidever, March 2022
;-

pro mkdailywave,mjd,darkid=darkid,flatid=flatid,psfid=psfid,$
                modelpsf=modelpsf,fiberid=fiberid,clobber=clobber,$
                nowait=nowait,nofit=nofit,unlock=unlock,psflibrary=psflibrary

  name = strtrim(mjd,2)
  dirs = getdir(apodir,caldir,spectrodir,vers)
  wavedir = apogee_filename('Wave',num=0,chip='a',/dir)
  wavebase = dirs.prefix+'Wave-'+name
  wavefile = wavedir+wavebase
  ;;lockfile = wavedir+file+'.lock'
  
  ;; If another process is alreadying make this file, wait!
  ;;if not keyword_set(unlock) then begin
  ;;  while file_test(lockfile) do begin
  ;;    if keyword_set(nowait) then return
  ;;    apwait,file,10
  ;;  endwhile
  ;;endif else begin
  ;;  if file_test(lockfile) then file_delete,lockfile,/allow
  ;;endelse
  aplock,wavefile,waittime=10,unlock=unlock
  
  ;; Does product already exist?
  ;; check all three chips and .dat file
  chips = ['a','b','c']
  allfiles = wavedir+dirs.prefix+'Wave-'+chips+'-'+name+'.fits'
  if total(file_test(allfiles)) eq 3 and not keyword_set(clobber) then begin
    print,' Wavecal file: ', wavefile, ' already made'
    return
  endif
  file_delete,allfiles,/allow  ;; delete any existing files to start fresh

  print,'Making dailywave: ', name
  ;; Open .lock file
  ;;openw,lock,/get_lun,lockfile
  ;;free_lun,lock
  aplock,wavefile,/lock
  
  ;; Get the arclamps that we need for the daily cal
  expinfo = dbquery("select * from apogee_drp.exposure where mjd>="+strtrim(long(mjd)-10,2)+$
                    " and mjd<="+strtrim(long(mjd)+10,2)+" and exptype='ARCLAMP' and "+$
                    "observatory='"+strmid(dirs.telescope,0,3)+"'")
  for i=0,n_elements(expinfo)-1 do begin
    ;; arctype info is missing in the db for early SDSS-V dates
    if strtrim(expinfo[i].arctype,2) eq '' then begin
      fil = apogee_filename('R',num=expinfo[i].num,chip='a')  
      expinfo2 = apfileinfo(fil,/silent)
      if expinfo2.lampune eq 1 then expinfo[i].arctype='UNE'
      if expinfo2.lampthar eq 1 then expinfo[i].arctype='THAR'      
    endif
  endfor
  expinfo.arctype = strtrim(expinfo.arctype,2)
  gdarc = where(expinfo.arctype eq 'UNE' or expinfo.arctype eq 'THAR',ngdarc)
  if ngdarc eq 0 then begin
    print,'No arclamps for these nights'
    return
  endif
  arcinfo = expinfo[gdarc]
  ;; Figure out which nights to use
  ui = uniq(arcinfo.mjd,sort(arcinfo.mjd))
  mjds = long(arcinfo[ui].mjd)
  nmjds = n_elements(mjds)
  si = sort(abs(mjds-mjd))
  keep = si[0:(nmjds-1)<7]
  mjds = mjds[keep]
  mjds = mjds[sort(mjds)]
  ;; Only keep the arclamps for these nights
  undefine,keep
  for i=0,n_elements(mjds)-1 do begin
    gd = where(arcinfo.mjd eq mjds[i],ngd)
    if ngd gt 0 then push,keep,gd
  endfor
  if n_elements(keep) eq 0 then begin
    print,'No arclamps for these nights'
    return
  endif
  arcinfo = arcinfo[keep]
  waveid = long(arcinfo.num)
  print,strtrim(n_elements(arcinfo),2),' arclamps'
  
  ;; Process the frames and find lines
  print,''
  print,'***** Processing the frames and finding the lines *****'
  print,''
  for i=0,n_elements(waveid)-1 do begin
    print,''
    print,'--- Frame ',strtrim(i+1,2),':  ',strtrim(waveid[i],2),' ---'
    ;; Check if it exists already
    file1 = apogee_filename('Wave',num=waveid[i],chip='c')
    wavedir1 = file_dirname(file1)
    swaveid1 = string(waveid[i],format='(i08)')
    allfiles1 = wavedir1+'/'+[dirs.prefix+'Wave-'+chips+'-'+swaveid1+'.fits',dirs.prefix+'Wave-'+swaveid1+'.dat']
    if total(file_test(allfiles1)) eq 4 then begin
      print,' wave file: ',dirs.prefix+'Wave-'+swaveid1, ' already made'
      continue
    endif

    ;; Check that the data is okay
    chfile = apogee_filename('2D',num=waveid[i],chip='b')
    if file_test(chfile) eq 0 then begin
      print,chfile,' NOT FOUND'
      continue
    endif
    head0 = headfits(chfile,exten=0)
    FITS_READ,chfile,im1,head1
    ;; UNE, bright line at X=1452
    if sxpar(head0,'LAMPUNE') then begin
      sub = im1[1452-100:1452+100,*]
      thresh = 40
    ;; THARNE, bright line at X=1566 
    endif else if sxpar(head0,'LAMPTHAR') then begin
      sub = im1[1566-100:1566+100,*]
      thresh = 1000
    endif else begin
      sub = im1[900:1100,*]
      thresh = 10
    endelse
    smsub = medfilt2d(sub,7,dim=1)                            ;; smooth in spectral axis
    resmsub = REBIN(smsub[*,0:(2048/8)*8-1],[201,2048/8])*8   ;; rebin in spatial axis
    peakflux = MAX(resmsub,dim=1)                             ;; peak flux feature in spectral dim.
    avgpeakflux = median(peakflux)
    ;; Check the line flux
    if avgpeakflux/sxpar(head0,'nread') lt thresh then begin
      print,'Not enough flux in ',chfile
      continue
    endif
      
    MAKECAL,wave=waveid[i],file=dirs.libdir+'cal/'+dirs.instrument+'-wave.par',$
            /nofit,unlock=unlock,librarypsf=psflibrary,modelpsf=modelpsf
  endfor

  ;; New Python version! 
  cmd = ['apdailywavecal','--apred',dirs.apred]
  if keyword_set(clobber) then cmd=[cmd,'--clobber']
  cmd = [cmd,'--observatory',strmid(dirs.telescope,0,3),'--verbose']
  cmd = [cmd,name]
  spawn,cmd,/noshell
  
  ;; Check that the calibration file was successfully created
  outfile = wavedir+repstr(wavefile,'apWave-','apWave-a-')
  if file_test(outfile) then begin
    openw,lock,/get_lun,wavedir+wavefile+'.dat'
    free_lun,lock
  endif

  ;;file_delete,lockfile,/allow
  aplock,wavefile,/clear
  
end
