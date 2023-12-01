;+
;
; MKFPIPEAKS
;
; Makes APOGEE FPI Peaks master calibration.
; This is NOT run as part of the regular pipeline processing, but
; rather just once by hand.
;
; INPUT:
;  ims: list of FPI full-frame image numbers to include in FPI peaks file.
;
; OUTPUT:
;  An fpi_peaks-s/n.fits file.
;
; USAGE:
;  IDL>mkfpipeaks
;
; By D. Nidever, 2023
;-

pro mkfpipeaks,ims,clobber=clobber

  redux_dir = getenv('APOGEE_REDUX')+'/daily/'
  chips = ['a','b','c']
  
  ;; North/South loop
  for i=0,1 do begin
    obs = (['apo','lco'])[i]
    instrument = (['apogee-n','apogee-s'])[i]
    telescope = (['apo25m','lco25m'])[i]
    prefix = (['ap','as'])[i]
    obstag = (['n','s'])[i]

    print,'Making FPI Peaks file for ',strupcase(obs)
    print,'------------------------------'
    
    ;; Get all of the FPILines files
    fpifiles = file_search(redux_dir+'cal/'+instrument+'/fpi/'+prefix+'FPILines-*.fits',count=nfpifiles)
    print,strtrim(nfpifiles,2),' FPILines files'
    
    ;; Create giant structure to hold all the line information
    schema = {x:0.0,row:0.0,height:0.0,flux:0.0,sigma:0.0,chip:'',expnum:0L,wave:0.0d0}
    str = replicate(schema,nfpifiles*150000)
    
    ;; Load FPI lines files
    count = 0LL
    ;;for j=0,nfpifiles-1 do begin
    for j=0,50 do begin
      base = file_basename(fpifiles[j],'.fits')
      print,j+1,' ',base 
      num = (strsplit(base,'-',/extract))[1]
      mjd = getcmjd(long(num))
      ;; Check for daily wavelength calibration files
      wavefiles = redux_dir+'cal/'+instrument+'/wave/'+prefix+'Wave-'+chips+'-'+strtrim(mjd,2)+'.fits'
      wtest = file_test(wavefiles)
      if total(wtest) lt 3 then begin
        print,'no daily wave cal. skipping'
        continue
      endif
      fpistr = MRDFITS(fpifiles[j],1,/silent)
      add_tag,fpistr,'wave',0.0d0,fpistr
      add_tag,fpistr,'sigma',0.0,fpistr
      ;;add_tag,fpistr,'expnum',0L,fpistr
      fpistr.expnum = num
      ;; Get Corresponding daily wave file and add wavelengths
      x = lindgen(2048)
      for c=0,2 do begin
        fits_read,wavefiles[c],wave,exten=2
        ind = where(fpistr.chip eq chips[c],nind)
        fpistr1 = fpistr[ind]
        index = create_index(fpistr1.row)
        rows = index.value
        for k=0,n_elements(rows)-1 do begin
          irow = rows[k]
          rind = index.index[index.lo[k]:index.hi[k]]
          dwave = abs(slope(wave[*,irow]))
          dwave = [dwave,dwave[-1]]
          dw = dwave[fpistr1[rind].pix0]
          ww = interpol(wave[*,irow],x,fpistr1[rind].pars[1])
          fpistr1[rind].wave = ww
          fpistr1[rind].sigma = fpistr1[rind].pars[2]*dw
        endfor
        fpistr[ind] = fpistr1  ;; stuff it back in
      endfor

      ;; Only keep successful lines
      gd = where(fpistr.success eq 'T',ngd)
      fpistr = fpistr[gd]
      nfpistr = n_elements(fpistr)
      
      ;; Plug into big structure
      str[count:count+nfpistr-1].x = fpistr.pars[1]
      str[count:count+nfpistr-1].row = fpistr.row
      str[count:count+nfpistr-1].flux = fpistr.sumflux
      str[count:count+nfpistr-1].height = fpistr.pars[0]      
      str[count:count+nfpistr-1].sigma = fpistr.sigma
      str[count:count+nfpistr-1].chip = fpistr.chip
      str[count:count+nfpistr-1].expnum = fpistr.expnum
      str[count:count+nfpistr-1].wave = fpistr.wave      
      count += nfpistr
      
    endfor   ;; FPILines file loop
    str = str[0:count-1]  ;; trim the big structure
    print,strtrim(count,2),' line measurements'
    
    ;; Find unique lines for each chip
    print,'Finding unique peaks'
    undefine,peakstr
    for c=0,2 do begin
      cind = where(str.chip eq chips[c],ncind)
      cstr = str[cind]
      hist = histogram(cstr.wave,binsize=1.0,locations=xhist)
      xhist += 0.5*1.0    ;; center of pixel
      ;; Find peaks
      thresh = max(hist)/5. > 1000
      ;;thresh = 100
      gdpeaks = where(hist gt thresh,ngdpeaks)
      xpeaks = []
      wpeaks = []
      for p=0,ngdpeaks-1 do begin
        wpk = xhist[gdpeaks[p]]
        if p gt 0 then begin
          lastind = n_elements(wpeaks)-1
          xdiff = xpeaks[lastind]-gdpeaks[p]
          ;; merge neighboring pixels and replace with pixel of highest value          
          if abs(xdiff) le 1 then begin
            xpeaks[lastind] = gdpeaks[p]                 ;; last value, not average
            wpeaks[lastind] = (wpeaks[lastind]+wpk)*0.5  ;; average the wavelengths
          endif else begin
            xpeaks = [xpeaks,gdpeaks[p]]
            wpeaks = [wpeaks,wpk]             
          endelse
        endif else begin
          xpeaks = [xpeaks,gdpeaks[p]]
          wpeaks = [wpeaks,wpk]
        endelse
      endfor  ;; peaks in histogram
      npeaks = n_elements(wpeaks)
      print,'chip ',chips[c],'  ',strtrim(npeaks,2),' unique peaks'
      ;; Now get all of the individual line measurements for each
      ;; unique line
      pkschema = {id:0L,chip:'',x:0.0d0,wave:0.0d0,height:0.0,flux:0.0,sigma:0.0,nfibers:0L,nlines:0L}
      peakstr1 = replicate(pkschema,npeaks)
      peakstr1.chip = chips[c]
      for p=0,npeaks-1 do begin
        pind = where(abs(cstr.wave-wpeaks[p]) lt 1,npind)
        med = median(cstr[pind].wave)
        sig = mad(cstr[pind].wave)
        pind = where(abs(cstr.wave-med) lt (3*sig>0.1),npind)
        nfibers = n_elements(uniq(cstr[pind].row,sort(cstr[pind].row)))
        peakstr1[p].id = p+1
        peakstr1[p].x = mean(cstr[pind].x)
        peakstr1[p].wave = mean(cstr[pind].wave)
        peakstr1[p].height = median(cstr[pind].height)
        peakstr1[p].flux = median(cstr[pind].flux)        
        peakstr1[p].sigma = median(cstr[pind].sigma)
        peakstr1[p].nfibers = nfibers
        peakstr1[p].nlines = npind
      endfor
      push,peakstr,peakstr1
      wslp = slope(peakstr1.wave)
      print,'min/median/max wave steps: ',min(wslp),median(wslp),max(wslp)
      print,'min/median/max height: ',min(peakstr1.height),median(peakstr1.height),max(peakstr1.height)
      print,'min/median/max nfibers: ',min(peakstr1.nfibers),median(peakstr1.nfibers),max(peakstr1.nfibers)
      print,'min/median/max nlines: ',min(peakstr1.nlines),median(peakstr1.nlines),max(peakstr1.nlines)      
    endfor  ;; chip loop
    npeakstr = n_elements(peakstr)
    print,strtrim(npeakstr,2),' unique FPI lines'
    
    ;; Save the table
    outfile = 'fpi_peaks-'+obstag+'.fits'
    print,'Writing to ',outfile
    MWRFITS,peakstr,outfile,/create
    head0 = headfits(outfile,exten=0)
    sxaddpar,head0,'V_APRED',getgitvers(),'apogee software version'
    sxaddpar,head0,'APRED',getvers(),'apogee reduction version'
    MODFITS,outfile,0,head0
    
    stop
    
  endfor  ;; north/south loop
  
  stop
  
end
