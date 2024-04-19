;+
;
; GETPLATEDATA
;
; Getplatedata loads up a structure with plate information and information about the 300 APOGEE fibers
;  This is obtained from a plPlugMapA file or from a 
;  plPlugMapM+plateHolesSorted combination
; returned structure includes:
;    fiberid, ra, dec, eta, zeta, hmag, objtype, obj (name)
;  for each of 300 APOGEE (spectrographid=2) files
;
; INPUTS:
;  plate          ID for the desired plate.
;  cmjd           MJD for the plugmap information.
;  =plugid        Name of plugmap file.
;  =asdaf         Array of fiberIDs for stars in a ASDAF
;                   (Any-Star-Down-Any Fiber) observation.
;  /mapa          Use "plPlugMapA" file, otherwise "plPlugMapM".
;  /obj1m         APO-1m observation.
;  /fixfiberid    Fix issues with fibers.
;  /noobject      Don't load the apogeeObject targeting information.
;  /twilight      This is a twilight observation, no stars.
;  =badiberid     Array of fiberIDs of bad fibers.
;  =mapper_data   Directory for mapper information (optional).
;  =starfiber     FiberID of the star for APO-1m observations.
;  /skip          Don't load the plugmap information.
;  /stp           Stop at the end of the program.
;  /apogees       Obsolete parameter.
;
; OUTPUTS:
;  plandata       Targeting information for an APOGEE plate.
;
; USAGE:
;  IDL>plugmap=getplatedata(planstr.plateid,string(planstr.mjd,format='(i5.5)'),plugid=planstr.plugmap,$
;                           fixfiberid=fixfiberid,badfiberid=badfiberid,mapper_data=mapper_data)
;
; By J. Holtzman, 2011?
;  Doc updates by D. Nidever, Sep 2020
;-

function catalog_info_blank
  ;; Info from apogeeObject
  cat0 = create_struct('alt_id',' ',$
                       'twomass_designation','',$
                       'jmag', -9999.99, $
                       'jerr', -9999.99, $
                       'hmag', -9999.99, $
                       'herr', -9999.99, $
                       'kmag', -9999.99, $
                       'kerr', -9999.99, $
                       'phflag','',$
                       'src_h', ' ', $
                       'pmra', 0., $
                       'pmdec', 0., $
                       'pm_src', ' ')
  return, cat0
end

;; Information in common between apogeeObject and apogee2Object
function catalog_info_common

  cat0 = create_struct('apogee_id',' ',$
                       'ra', 0.d0,$
                       'dec', 0.d0,$
                       'jmag', 0.,$
                       'jerr', 0.,$
                       'hmag', 0.,$
                       'herr', 0.,$
                       'kmag', 0.,$
                       'kerr', 0.,$
                       'alt_id',' ',$
                       'src_h', ' ', $
                       ;'wash_m', 0.,$
                       ;'wash_m_err', 0.,$
                       ;'wash_t2', 0., $
                       ;'wash_t2_err', 0., $
                       ;'ddo51', 0., $
                       ;'ddo51_err', 0., $
                       ;'irac_3_6', 0., $
                       ;'irac_3_6_err', 0., $
                       ;'irac_4_5', 0., $
                       ;'irac_4_5_err', 0., $
                       ;'irac_5_8', 0., $
                       ;'irac_5_8_err', 0., $
                       ;'irac_8_0', 0., $
                       ;'irac_8_0_err', 0., $
                       ;'wise_4_5', 0., $
                       ;'wise_4_5_err', 0., $
                       ;'targ_4_5', 0., $
                       ;'targ_4_5_err', 0., $
                       ;'ak_targ', -9999.99, $
                       ;'ak_targ_method', '', $
                       ;'ak_wise', -9999.99, $
                       ;'sfd_ebv', -9999.99, $
                       ;'wash_ddo51_giant_flag', 0, $
                       ;'wash_ddo51_star_flag', 0, $
                       'pmra', 0., $
                       'pmdec', 0., $
                       'pm_src', ' ')
  return, cat0
end

pro saveplatedata,filename,plug

  ;; Save plugmap data to a PlateData file.
  ;;
  ;; Inputs:
  ;;  filename  Filename to save the plugmap data to.
  ;;  plug      Plugmap data as returned from getdata.
  ;;
  ;; Outputs:
  ;;  The plugmap data is saved to disk.  Nothing is returned.
  ;;
  ;; Usage
  ;;  IDL>saveplatedata,'apPlateData-3745-59636.fits',plug

  ;; Save the data to file
  MWRFITS,plug.fiberdata,filename,/create
  if size(plug.guidedata,/type) eq 8 then begin
    MWRFITS,plug.guidedata,filename,/silent  ;; add extension
  endif else begin
    MWRFITS,0,filename,/silent
  endelse
  ;; Add header keywords
  MKHDR,head,1,0   ;; make basic header
  sxaddpar,head,'plate',plug.plate
  sxaddpar,head,'mjd',plug.mjd
  sxaddpar,head,'location',plug.locationid
  sxaddpar,head,'field',plug.field
  sxaddpar,head,'program',plug.programname
  sxaddpar,head,'ha1',plug.ha[0]
  sxaddpar,head,'ha2',plug.ha[1]
  sxaddpar,head,'ha3',plug.ha[2]
  MODFITS,filename,0,head  ;; add header to HDU0
  
  ;  hdu = fits.HDUList()
  ;  hdu.append(fits.PrimaryHDU())
  ;  hdu[0].header['plate'] = plug.get('plate')
  ;  hdu[0].header['mjd'] = plug.get('mjd')
  ;  hdu[0].header['location'] = plug.get('locationid')
  ;  hdu[0].header['field'] = plug.get('field')
  ;  hdu[0].header['program'] = plug.get('programname')
  ;  if plug.get('ha') is not None:
  ;      hdu[0].header['ha1'] = plug['ha'][0]
  ;      hdu[0].header['ha2'] = plug['ha'][1]
  ;      hdu[0].header['ha3'] = plug['ha'][2]
  ;  fiber = plug.get('fiberdata')
  ;  if fiber is not None:
  ;      hdu.append(fits.table_to_hdu(Table(fiber)))
  ;  else:
  ;      hdu.append(fits.ImageHDU())
  ;  guidedata = plug.get('guidedata')
  ;  if guidedata is not None:
  ;      hdu.append(fits.table_to_hdu(Table(guidedata)))
  ;  else:
  ;      hdu.append(fits.ImageHDU())
  ;  hdu.writeto(filename,overwrite=True)
  ;  hdu.close()

end


function readplatedata,filename
  ;; Read in plugmap data from apPlateData file.
  ;;
  ;; Inputs:
  ;;   filename  PlateData filename to load.
  ;;
  ;; Outputs:
  ;;   plug      Plugmap data.
  ;;
  ;; Usage:
  ;;  IDL>plug = readplatedata('apPlateData-3745-59636.fits')

  if file_test(filename) eq 0 then begin
    print,filename,' NOT FOUND'
    return,-1 
  endif

  ;; Get number of HDUs
  nextend = 0
  fits_open,filename,fcb
  nextend = fcb.nextend
  fits_close,fcb
  
  head = HEADFITS(filename,exten=0)
  if nextend gt 0 then begin
    fiber = MRDFITS(filename,1,/silent)
  endif else begin
    fiber = -1
  endelse
  if nextend gt 1 then begin
    guide = MRDFITS(filename,2,/silent)
  endif else begin
    guide = -1
  endelse
  
  ;plug = {}
  ;plug['plate'] = head.get('plate')
  ;plug['mjd'] = head.get('mjd')
  ;plug['locationid'] = head.get('location')
  ;plug['field'] = head.get('field')
  ;plug['programname'] = head.get('program')
  ;plug['ha'] = [head.get('ha1'),head.get('ha2'),head.get('ha3')]
  ;plug['fiberdata'] = fiberdata
  ;plug['guidedata'] = guidedata
  
  plug = {plate:0L, mjd:0L, locationid:0L, field:' ', programname:'',$
          ha:[-99.,-99.,-99.], fiberdata:fiber, guidedata:guide}
  plug.plate = sxpar(head,'plate')
  plug.mjd = sxpar(head,'mjd')
  plug.locationid = sxpar(head,'location')
  plug.field = strtrim(sxpar(head,'field'),2)
  plug.programname = strtrim(sxpar(head,'program'),2)
  ha1 = sxpar(head,'ha1',count=nha1)
  if nha1 gt 0 and strtrim(ha1,2) ne '-' then plug.ha[0] = float(ha1)
  ha2 = sxpar(head,'ha2',count=nha2)
  if nha2 gt 0 and strtrim(ha2,2) ne '-' then plug.ha[1] = float(ha2)  
  ha3 = sxpar(head,'ha3',count=nha3)
  if nha3 gt 0 and strtrim(ha3,2) ne '-' then plug.ha[2] = float(ha3)  
  
  return,plug

end


;; Main function
function getplatedata,plate,cmjd,plugid=plugid,asdaf=asdaf,mapa=mapa,obj1m=obj1m,fixfiberid=fixfiberid,$
                      noobject=noobject,stp=stp,skip=skip,twilight=twilight,badfiberid=badfiberid,$
                      apogees=apogees,mapper_data=mapper_data,starfiber=starfiber

dirs = getdir(apodir,datadir=datadir)
if n_elements(mapper_data) eq 0 then mapper_data=dirs.mapperdir
observatory = strmid(dirs.telescope,0,3)

mjd = 0L
reads,cmjd,mjd
if long(mjd) ge 59556 then begin
  fps = 1
  sdss5 = 1
  apogee2 = 0
endif else fps=0
if size(plate,/type) eq 7 then begin
  cplate = plate 
  platenum = 0L
endif else begin
  cplate = strtrim(string(format='(i6.4)',plate),2)
  platenum = long(plate)
endelse
if keyword_set(fps) then cplate=strtrim(long(plate),2)  ;; no zero padding for FPS config


;; Check if the apPlateData file already exists
;;  if yes, load it unless clobber is set
;; Use planfile name to construct PlateData filename
if keyword_set(obj1m) then begin
  planfile = apogee_filename('Plan',plate=plate,reduction=objid,mjd=mjd)
endif else begin
  if long(mjd) ge 59556 then begin  ;; FPS
    ;;;; get fieldid from confSummary file
    ;;;; at least 6 characters (0036XX), but potentially more 10036XX
    ;;if plate gt 999999 then begin
    ;;  configgrp = string(plate/100,format='(I0)')+'XX'
    ;;endif else begin
    ;;  configgrp = string(plate/100,format='(I04)')+'XX'
    ;;endelse 
    ;;plugdir = getenv('SDSSCORE_DIR')+'/'+observatory+'/summary_files/'+configgrp+'/'
    ;;configfile = 'confSummary-'+strtrim(plate,2)+'.par'

    ;; Use python code to get the confSummary file
    temp = MKTEMP('cfgfl',outdir='/tmp/')
    cmd = '#!/usr/bin/env python'
    push,cmd,'from apogee_drp.utils import apload'
    push,cmd,'load = apload.ApLoad(apred="'+dirs.apred+'",telescope="'+dirs.telescope+'")'
    push,cmd,'filename = load.filename("confSummary",configid='+strtrim(plate,2)+')'
    push,cmd,'with open("'+temp+'.txt","w") as f:'
    push,cmd,'    f.write(filename+"\n")'
    writeline,temp+'.py',cmd
    file_chmod,temp+'.py','755'o
    spawn,temp+'.py',/noshell
    if file_test(temp+'.txt') eq 0 then begin
      print,'Problem getting the confSummary filename'
      return,-1
    endif
    readline,temp+'.txt',configfile
    configfile = configfile[0]
    file_delete,[temp+'.py',temp+'.txt'],/allow
    readline,configfile,configlines
    ind = where(stregex(configlines,'field_id',/boolean) eq 1,nind)
    fieldid = (strsplit(configlines[ind[0]],/extract))[1]
    planfile = apogee_filename('Plan',plate=plate,mjd=mjd,field=strtrim(fieldid,2))
  endif else begin
    planfile = apogee_filename('Plan',plate=plate,mjd=mjd)
  endelse
  platedatafile = repstr(planfile,'Plan','PlateData')
  platedatafile = repstr(platedatafile,'.yaml','.fits')
endelse
if file_test(platedatafile) eq 1 and not keyword_set(clobber) then begin
  print,'Reading platedata from '+platedatafile
  return,readplatedata(platedatafile)
endif


;; -------- CALL THE PYTHON ROUTINE ----------
print,'Running Python platedata.py'
; use temporary files and symlinks
tbase = MKTEMP('pltdata',outdir=getlocaldir())    ; create base, leave so other processes won't take it
tempfile = tbase+'.fits'
file_delete,tempfile,/allow
;; defaults
if keyword_set(mapa) then mapa='True' else mapa='False'
if keyword_set(fixfiberid) then fixfiberid='True' else fixfiberid='False'
if keyword_set(noobject) then noobject='True' else noobject='False'
if keyword_set(skip) then skip='True' else skip='False'
if keyword_set(twilight) then twilight='True' else twilight='False'
if keyword_set(clobber) then clobber='True' else clobber='False'
if n_elements(plugid) eq 0 then plugid='None' else plugid='"'+strtrim(plugid,2)+'"'
if n_elements(asdaf) eq 0 then asdaf='None' else asdaf='"'+strtrim(asdaf,2)+'"'
if n_elements(obj1m) eq 0 then obj1m='None' else obj1dm='"'+strtrim(obj1m,2)+'"'
if n_elements(badfiberid) eq 0 then badfiberid='None' else badfiberid='"'+strtrim(badfiberid,2)+'"'
if n_elements(mapper_data) eq 0 then mapper_data='None' else mapper_data='"'+strtrim(mapper_data,2)+'"'
if n_elements(starfiber) eq 0 then starfiber='None' else starfiber='"'+strtrim(starfiber,2)+'"'
;; Call the python code
cmd = '#!/usr/bin/env python'
push,cmd,'from astropy.table import Table'
push,cmd,'from apogee_drp.utils import platedata'
pars = strtrim(plate,2)+','+strtrim(cmjd,2)+',"'+strtrim(dirs.apred,2)+'","'+strtrim(dirs.telescope,2)+'"'
pars += ',plugid='+plugid+',asdaf='+asdaf+',mapa='+mapa
pars += ',obj1m='+obj1m+',fixfiberid='+fixfiberid+',noobject='+noobject
pars += ',skip='+skip+',twilight='+twilight+',badfiberid='+badfiberid
pars += ',mapper_data='+mapper_data+',starfiber='+starfiber+',clobber='+clobber
push,cmd,'tab=platedata.getdata('+pars+')'
;;push,cmd,'if len(tab)>0: Table(tab).write("'+tempfile+'")'
scriptfile = tbase+'.py'
WRITELINE,scriptfile,cmd
FILE_CHMOD,scriptfile,'755'o
SPAWN,scriptfile,out,errout,/noshell
if errout[0] ne '' or n_elements(errout) gt 1 then begin
  print,'Problems with the platedata.py call'
  for i=0,n_elements(errout)-1 do print,errout[i]
endif
;; The python code will save it to the apPlateData file
if file_test(platedatafile) eq 1 then begin
  return,readplatedata(platedatafile)
endif else begin
  print,platedatafile,' NOT FOUND'
  return,-1
endelse

  
;;;---------------------------- IDL code ------------------------------  



;; Deal with null values from yaml file
if size(fixfiberid,/type) eq 7 and n_elements(fixfiberid) eq 1 then $
  if (strtrim(fixfiberid,2) eq 'null' or strtrim(strlowcase(fixfiberid),2) eq 'none') then undefine,fixfiberid  ;; null/none  
if size(badfiberid,/type) eq 7 and n_elements(badfiberid) eq 1 then $
  if (strtrim(badfiberid,2) eq 'null' or strtrim(strlowcase(badfiberid),2) eq 'none') then undefine,badfiberid  ;; null/none  

;; Create the output fiber structure
tmp = create_struct('fiberid',-1, 'ra',999999.d0, 'dec',999999.d0, 'eta',999999.d0, 'zeta',999999.d0, 'objtype','none', $
                    'holetype','OBJECT', 'object','', 'assigned',0, 'on_target',0, 'valid',0, 'tmass_style','', 'target1',0L, 'target2',0L,$
                    'target3',0L, 'target4',0L, 'spectrographid',2, 'mag',fltarr(5), catalog_info_blank(),$
                    'catalogid',-1LL, 'sdssv_apogee_target0',0LL, 'firstcarton','', 'cadence','','program','','category','',$
                    'gaia_g',-9999.99, 'gaia_bp',-9999.99, 'gaia_rp',-9999.99,$
                    'gaiadr2_sourceid','', 'gaiadr2_ra',-9999.99d0, 'gaiadr2_dec',-9999.99d0,$
                    'gaiadr2_plx',-9999.99, 'gaiadr2_plx_error',-9999.99, 'gaiadr2_pmra',-9999.99,$
                    'gaiadr2_pmra_error',-9999.99, 'gaiadr2_pmdec',-9999.99, 'gaiadr2_pmdec_error',-9999.99, 'gaiadr2_gmag',-9999.99,$
                    'gaiadr2_gerr',-9999.99, 'gaiadr2_bpmag',-9999.99, 'gaiadr2_bperr',-9999.99, 'gaiadr2_rpmag',-9999.99,$
                    'gaiadr2_rperr',-9999.99)
tmp.mag = 99.99  ;; default to faint magnitudes
tmp.jmag = 99.99 ;; default to faint magnitudes
tmp.hmag = 99.99 ;; default to faint magnitudes
tmp.kmag = 99.99 ;; default to faint magnitudes

guide = replicate(tmp,16)
loc = 0L
;; APO-1M observations
if keyword_set(obj1m) then begin
  if keyword_set(fixfiberid) then begin
    if fixfiberid eq 1 then begin
      fiberid = [218,220,222,223,226,227,228,229,230,231]
      if ~keyword_set(starfiber) then starfiber=229
    endif
  endif else begin
    fiberid = [218,219,221,223,226,228,230]
    if ~keyword_set(starfiber) then starfiber=223
  endelse
  fiber = replicate(tmp,n_elements(fiberid))
  platedata = {plate:platenum, mjd:mjd, plateid:cplate, locationid:1L, field:' ', programname:'', $
               cmjd:cmjd, ha:[-99.,-99.,-99.], fiberdata:fiber, guidedata:guide}
  platedata.field = cplate
  fiber.fiberid = fiberid
  fiber.objtype = replicate('SKY',n_elements(fiberid))
  j = where(fiber.fiberid eq starfiber)
  fiber[j].objtype = 'STAR'
  j = where(fiber.objtype eq 'SKY')
  fiber[j].target2 = 2L^4
  obj = mrdfits(getenv('APOGEEREDUCEPLAN_DIR')+'/data/1m/'+cplate+'.fits',1,status=status)

  if status eq 0 then begin
    j = where(strtrim(obj.name,2) eq strtrim(obj1m,2),nj)
    if nj gt 0 then begin
      ifiber = where(fiber.fiberid eq starfiber)
      fiber[ifiber].object = obj[j].name
      fiber[ifiber].tmass_style = strtrim(obj[j].name,2)
      fiber[ifiber].hmag = obj[j].h
      fiber[ifiber].mag[1] = obj[j].h
      fiber[ifiber].ra = obj[j].ra
      fiber[ifiber].dec = obj[j].dec
      fiber[ifiber].target2 = 2L^22
    endif
  endif else stop,'halt: no file found with object information!'
  platedata.fiberdata = fiber  
  return,platedata
endif
;; Twilight observations
if keyword_set(twilight) then begin
  fiber = replicate(tmp,300)
  platedata = {plate:platenum, mjd:mjd, plateid:cplate, locationid:1L, field:' ', programname:'', cmjd:cmjd,$
               ha:[-99.,-99.,-99.], fiberdata:fiber, guidedata:guide}
  platedata.field = cplate
  fiber.hmag = 10.
  fiber.mag = [10.,10.,10.,10.,10.]
  fiber.objtype = 'STAR'
  fiber.fiberid = indgen(300)+1
  platedata.fiberdata = fiber  
  return,platedata
endif
fiber = replicate(tmp,300)
reads,cplate,platenum
platedata = {plate:platenum, mjd:mjd, plateid:cplate, locationid:0L, field:' ', programname:'', cmjd:cmjd,$
             ha:[-99.,-99.,-99.], fiberdata:fiber, guidedata:guide}
if not keyword_set(fps) then begin
  platedata.field = apogee_field(loc,platenum,survey,programname)
  platedata.locationid = loc
  platedata.programname = programname
endif else begin
  ;stop
  ;platedata.locationid = fieldid  ;; fieldid
  ;platedata.programname = carton ;; carton
endelse

;; Do we want to use a plPlugMapA file with the matching already done?
havematch = 0
if keyword_set(mapa) then root = 'plPlugMapA' else root='plPlugMapM'
if keyword_set(plugid) then begin
  tmp = file_basename(plugid,'.par')
  if strpos(tmp,'plPlug') ge 0 then tplugid=strmid(tmp,11) else tplugid=tmp
  plugfile = root+'-'+tplugid+'.par'
endif else begin
  tplugid = root+'-'+cplate
  plugfile = root+'-'+cplate+'.par'
endelse
if keyword_set(mapa) then plugdir=datadir+cmjd+'/' else begin
  tmp = strsplit(tplugid,'-',/extract)
  plugmjd = tmp[1]
  plugdir = mapper_data+'/'+plugmjd+'/'
endelse
;; Rare cases where the plugmap exists in the data directory but NOT in the mapper directory                          
if file_test(plugdir+'/'+plugfile) eq 0 then begin
  if file_test(datadir+cmjd+'/'+repstr(plugfile,'MapM','MapA')) eq 1 then begin
    print,'Cannot find plPlugMapM file in mapper directory.  Using plPlugMapA file in data directory instead.'
    plugdir = datadir+cmjd+'/'
    root = 'plPlugMapA'
    plugfile = repstr(plugfile,'MapM','MapA')
 endif
endif
if keyword_set(fps) then begin
  ;; at least 6 characters (0036XX), but potentially more 10036XX
  if plate gt 999999 then begin
    configgrp = string(plate/100,format='(I0)')+'XX'
  endif else begin
    configgrp = string(plate/100,format='(I04)')+'XX'
  endelse
  plugdir = getenv('SDSSCORE_DIR')+'/'+observatory+'/summary_files/'+configgrp+'/'
  plugfile = 'confSummary-'+strtrim(plate,2)+'.par'
endif

;; Does the plugfile exist? If so, load it
if file_test(plugdir+'/'+plugfile) then APLOADPLUGMAP,plugdir+'/'+plugfile,plugmap,fixfiberid=fixfiberid else $
   if keyword_set(skip) then return,0 else stop,'halt: cannot find plugmap file '+plugdir+'/'+plugfile

platedata.locationid = plugmap.locationid
if not keyword_set(fps) then begin
  platedata.ha[0] = plugmap.ha[0]
  platedata.ha[1] = plugmap.ha_observable_min[0]
  platedata.ha[2] = plugmap.ha_observable_max[0]
endif
if not keyword_set(mapa) and not keyword_set(fps) then begin
  ;; Get the plateHolesSorted file for thie plate and read it
  platestr = string(format='(i6.6)',platenum)
  platedir = getenv('PLATELIST_DIR')+'/plates/'+strmid(platestr,0,4)+'XX/'+platestr
  holefile = 'plateHolesSorted-'+platestr+'.par'
  print,'yanny_read,'+platedir+'/'+holefile
  YANNY_READ,platedir+'/'+holefile,pdata,/anon,hdr=hdr
  p =* pdata
  YANNY_FREE,pdata
  ;; Use locationid from plateHoles files as there are a few cases
  ;;  where plugmapM is wrong
  j = where(strpos(strupcase(hdr),'LOCATIONID') ge 0)
  tmp = strsplit(hdr[j],/ext)
  loc = 0L
  reads,tmp[1],loc
  platedata.locationid = loc

  ;; Sometimes TMASS_ID in "p" is 2MASS-J23580454-0007415
  if tag_exist(p,'tmass_id') then $
    p.tmass_id = repstr(p.tmass_id,'2MASS-J','2M')
  
  ;; Fix telluric catalogIDs
  ;; There is a problem with some of the telluric catalogIDs due to
  ;; overflow.  We need to add 2^32 to them.
  ;; Use the minimum catalogid in the v0 cross-match
  ;; (version_id=21). That number is 4204681993.
  if tag_exist(p,'catalogid') then begin
    bdcatid = where(stregex(p.holetype,'APOGEE',/boolean,/fold_case) eq 1 and $
                    p.catalogid gt 0 and p.catalogid lt 4204681993LL,nbdcatid)
    if nbdcatid gt 0 then begin
      print,'KLUDGE!!!  Fixing overflow catalogIDs for ',strtrim(nbdcatid,2),' telluric stars'
      print,p[bdcatid].catalogid
      p[bdcatid].catalogid += 2LL^32
    endif
  endif

  ;; Read flag correction data
  have_flag_changes = 0
  print,platedir+'/flagModifications-'+platestr+'.txt'
  if file_test(platedir+'/flagModifications-'+platestr+'.txt') then begin
    print,'Reading flagModifications file: ','flagModifications-'+platestr+'.txt'
    flag_changes = IMPORTASCII(platedir+'/flagModifications-'+platestr+'.txt',/header)
    if size(flag_changes,/type) ne 8 then stop,'Error reading flagModifications file'
    have_flag_changes = 1
  endif
endif

;; Get SDSS-V FPS photometry from targetdb                                                                                                               
if keyword_set(fps) then begin
  print,'Querying targetdb/catalogdb'
  p = get_catalogdb_data(designid=plugmap.design_id)
  add_tag,p,'target_ra',0.0d0,p
  p.target_ra = p.ra
  add_tag,p,'target_dec',0.0d0,p
  p.target_dec = p.dec
  add_tag,p,'tmass_j',0.0,p
  p.tmass_j = p.jmag
  add_tag,p,'tmass_h',0.0,p
  p.tmass_h = p.hmag
  add_tag,p,'tmass_k',0.0,p
  p.tmass_k = p.kmag
  ;; get 2MASS IDs and other info from catalogdb
  gdid = where(plugmap.fiberdata.catalogid gt 0 and plugmap.fiberdata.spectrographid eq 2 and $
               plugmap.fiberdata.objtype ne 'SKY',ngdid)
  if ngdid gt 0 then begin
    catinfo = get_catalogdb_data(id=plugmap.fiberdata[gdid].catalogid)
    match,p.catalogid,catinfo.catalogid,ind1,ind2,/sort,coun=nmatch
    if nmatch gt 0 then begin
       orig = p
       undefine,p
       tagnames = ['tmass_id','e_jmag','e_hmag','e_kmag','phflag','gaiadr2_sourceid','gaiadr2_ra','gaiadr2_dec',$
                   'gaiadr2_pmra','gaiadr2_pmdec','gaiadr2_pmra_error','gaiadr2_pmdec_error','gaiadr2_plx',$
                   'gaiadr2_plx_error','gaiadr2_gmag','gaiadr2_gerr','gaiadr2_bpmag','gaiadr2_bperr',$
                   'gaiadr2_rpmag','gaiadr2_rperr']
       tagvals = ['" "','0.0','0.0','0.0','" "','0LL','0.0d0','0.0d0','0.0','0.0','0.0','0.0','0.0','0.0','0.0',$
                  '0.0','0.0','0.0','0.0','0.0']
       add_tags,orig,tagnames,tagvals,p
       undefine,orig
       p[ind1].tmass_id = catinfo[ind2].twomass
       p[ind1].e_jmag = catinfo[ind2].e_jmag
       p[ind1].e_hmag = catinfo[ind2].e_hmag
       p[ind1].e_kmag = catinfo[ind2].e_kmag
       p[ind1].phflag = catinfo[ind2].twomflag
       ; Deal with instances of Gaia source ID = 'None'
       if typename(catinfo[ind2].gaia) eq 'STRING' then begin
          g = where(strtrim(catinfo[ind2].gaia,2) ne 'None',ng)
          p[ind1[g]].gaiadr2_sourceid = catinfo[ind2[g]].gaia
       endif else begin
          p[ind1].gaiadr2_sourceid = catinfo[ind2].gaia
       endelse
       p[ind1].gaiadr2_ra = catinfo[ind2].ra
       p[ind1].gaiadr2_dec = catinfo[ind2].dec
       p[ind1].gaiadr2_pmra = catinfo[ind2].pmra
       p[ind1].gaiadr2_pmdec = catinfo[ind2].pmdec
       p[ind1].gaiadr2_pmra_error = catinfo[ind2].e_pmra
       p[ind1].gaiadr2_pmdec_error = catinfo[ind2].e_pmdec
       p[ind1].gaiadr2_plx = catinfo[ind2].plx
       p[ind1].gaiadr2_plx_error = catinfo[ind2].e_plx
       p[ind1].gaiadr2_gmag = catinfo[ind2].gaiamag
       p[ind1].gaiadr2_gerr = catinfo[ind2].e_gaiamag
       p[ind1].gaiadr2_bpmag = catinfo[ind2].gaiabp
       p[ind1].gaiadr2_bperr = catinfo[ind2].e_gaiabp
       p[ind1].gaiadr2_rpmag = catinfo[ind2].gaiarp
       p[ind1].gaiadr2_rperr = catinfo[ind2].e_gaiarp
    endif
  endif
endif

;; Load guide stars
if not keyword_set(fps) then begin
  for i=0,15 do begin
    m = where(plugmap.fiberdata.holetype eq 'GUIDE' and $
              plugmap.fiberdata.fiberid eq i,nm)
    guide[i].fiberid = plugmap.fiberdata[m].fiberid
    guide[i].ra = plugmap.fiberdata[m].ra 
    guide[i].dec = plugmap.fiberdata[m].dec 
    guide[i].eta = plugmap.fiberdata[m].eta
    guide[i].zeta = plugmap.fiberdata[m].zeta
    guide[i].spectrographid = plugmap.fiberdata[m].spectrographid
  endfor
  platedata.guidedata = guide
endif

;; Find matching plugged entry for each spectrum and load up the output information from correct source(s)
for i=0,299 do begin
  fiber[i].spectrographid = -1
  if keyword_set(fps) then begin
    m = where(plugmap.fiberdata.spectrographid eq 2 and $
              plugmap.fiberdata.fiberid eq 300-i,nm)
  endif else begin
    m = where(plugmap.fiberdata.holetype eq 'OBJECT' and $
              plugmap.fiberdata.spectrographid eq 2 and $
              plugmap.fiberdata.fiberid eq 300-i,nm)
  endelse
  if keyword_set(badfiberid) then begin
    j = where(badfiberid eq 300-i,nbad)
    if nbad gt 0 then begin
      print,'fiber index ',i,' declared as bad'
      nm = 0
    endif 
  endif
  if nm gt 1 then begin
    print,'halt: more than one match for fiber id !! MARVELS??'
    print,plugmap.fiberdata[m].fiberid,plugmap.fiberdata[m].primtarget,plugmap.fiberdata[m].sectarget
    stop
  endif
  if nm eq 1 then begin
    fiber[i].fiberid = plugmap.fiberdata[m].fiberid
    fiber[i].ra = plugmap.fiberdata[m].ra 
    fiber[i].dec = plugmap.fiberdata[m].dec 
    fiber[i].eta = plugmap.fiberdata[m].eta
    fiber[i].zeta = plugmap.fiberdata[m].zeta
    if keyword_set(fps) then begin
      fiber[i].assigned = plugmap.fiberdata[m].assigned
      fiber[i].on_target = plugmap.fiberdata[m].on_target
      fiber[i].valid = plugmap.fiberdata[m].valid
      fiber[i].sdssv_apogee_target0 = plugmap.fiberdata[m].sdssv_apogee_target0
      fiber[i].catalogid = plugmap.fiberdata[m].catalogid
    endif else begin
      fiber[i].target1 = plugmap.fiberdata[m].primTarget
      fiber[i].target2 = plugmap.fiberdata[m].secTarget
    endelse
    fiber[i].spectrographid = plugmap.fiberdata[m].spectrographid

    ;; Unassigned, off target, invalid targets
    if keyword_set(fps) then begin
      ;; ASSIGNED: assigned=1 indicates a science/sky target is
      ;;   assigned to that fiber in robostrategy. assigned=0 means no
      ;;   target was assigned for this fiber (likely used for BOSS instead).
      ;;   There's no targeting information for this fiber.
      ;; ON_TARGET: on_target=1 indicates that the fiber actually was
      ;;   placed on the target. In some cases you'll find that
      ;;   on_target=0 although assigned=1; that usually happens when
      ;;   there are collisions or deadlocks that we need to solve.
      ;; VALID: valid=0 means there was a problem converting from
      ;;   on-sky coordinates to alpha/beta for the robots. Position
      ;;   was unreachable.
      if fiber[i].assigned eq 0 or fiber[i].on_target eq 0 or fiber[i].valid eq 0 then begin
        cmt = 'FiberID '+strtrim(300-i,2)+' is '
        badcmt = []
        if fiber[i].assigned eq 0 then push,badcmt,'UNASSIGNED'
        if fiber[i].on_target eq 0 then push,badcmt,'NOT on target'
        if fiber[i].valid eq 0 then push,badcmt,'INVALID'
        cmt += strjoin(badcmt,', ')
        print,cmt
      endif
      ;; Unassigned, no information for this target
      if fiber[i].assigned eq 0 then begin
        fiber[i].hmag = 99.99
        fiber[i].mag = 99.99
        continue
      endif
    endif

    ;; Special for asdaf object plates
    if keyword_set(asdaf) then begin
      ;; ASDAF fiber
      if 300-i eq asdaf then begin
        fiber[i].objtype = 'STAR'
        fiber[i].hmag = 0.
      endif else begin
        fiber[i].objtype = 'SKY'
        fiber[i].hmag = -99.999
      endelse

    ;; Normal plate
    endif else begin
      fiber[i].objtype = plugmap.fiberdata[m].objtype
      ;; Fix up objtype
      fiber[i].objtype = 'STAR'
      if not keyword_set(fps) then $
        fiber[i].holetype = plugmap.fiberdata[m].holetype
      if keyword_set(mapa) then begin
        ;; HMAG's are correct from plPlugMapA files
        fiber[i].hmag = plugmap.fiberdata[m].mag[1]
        fiber[i].object = strtrim(plugmap.fiberdata[m].tmass_style,2)
        fiber[i].tmass_style = strtrim(plugmap.fiberdata[m].tmass_style,2)
        if is_bit_set(fiber[i].sectarget,9) eq 1 then fiber[i].objtype='HOT_STD'
        if is_bit_set(fiber[i].sectarget,4) eq 1 then fiber[i].objtype='SKY'
      endif else begin
        ;; FPS object type from CATEGORY
        if keyword_set(fps) then begin
          ;; objtype: STAR, HOT_STD, or SKY
          objtype = 'STAR'
          if strupcase(plugmap.fiberdata[m].category) eq 'SCIENCE' then objtype='STAR'
          if strupcase(plugmap.fiberdata[m].category) eq 'SKY_APOGEE' then objtype='SKY'
          if strupcase(plugmap.fiberdata[m].category) eq 'SKY_BOSS' then objtype='SKY'
          if strupcase(plugmap.fiberdata[m].category) eq 'STANDARD_APOGEE' then objtype='HOT_STD'             
          fiber[i].objtype = objtype
          if objtype eq 'SKY' then continue
        endif

        ;; Use CatalogID for FPS
        if keyword_set(fps) then begin
          match = where(p.catalogid eq fiber[i].catalogid,nmatch)
        ;; Get matching stars from coordinate match
        endif else begin
          match = where(abs(p.target_ra-fiber[i].ra) lt 0.00002 and $
                        abs(p.target_dec-fiber[i].dec) lt 0.00002,nmatch)
        endelse
        if nmatch gt 0 then begin
          ;; APOGEE-2 plate
          if tag_exist(p,'apogee2_target1') and platenum gt 7500 and platenum lt 15000 and not keyword_set(fps) then begin
            fiber[i].target1 = p[match].apogee2_target1
            fiber[i].target2 = p[match].apogee2_target2
            fiber[i].target3 = p[match].apogee2_target3
            apogee2 = 1
            sdss5 = 0
            if have_flag_changes then begin
              jj = where(flag_changes.PlateID eq platenum and flag_changes.TARGETID eq p[match].targetids, njj)
              if njj gt 0 then begin
                print,'modifying flags for',p[match].targetids
                fiber[i].target1 = flag_changes[jj].at1
                fiber[i].target2 = flag_changes[jj].at2
                fiber[i].target3 = flag_changes[jj].at3
                fiber[i].target4 = flag_changes[jj].at4
              endif
            endif
          endif
          ;; APOGEE-1 plate
          if not tag_exist(p,'apogee2_target1') and platenum le 7500 and not keyword_set(fps) then begin
            fiber[i].target1 = p[match].apogee_target1
            fiber[i].target2 = p[match].apogee_target2
            apogee2 = 0
            sdss5 = 0
          endif
          ;; SDSS-V plate
          if platenum ge 15000 or keyword_set(fps) then begin
            sdss5 = 1
            apogee2 = 0
            if not keyword_set(fps) then begin  ;; SDSS-V plate data
              fiber[i].catalogid = p[match].catalogid
              fiber[i].gaia_g = p[match].gaia_g
              fiber[i].gaia_bp = p[match].gaia_bp
              fiber[i].gaia_rp = p[match].gaia_rp
              fiber[i].sdssv_apogee_target0 = p[match].sdssv_apogee_target0
              fiber[i].firstcarton = p[match].firstcarton
              fiber[i].pmra = p[match].pmra
              fiber[i].pmdec = p[match].pmdec
              ;; objtype: STAR, HOT_STD, or SKY
              objtype = 'STAR'
              if is_bit_set(fiber[i].sdssv_apogee_target0,0) then objtype='SKY'
              if is_bit_set(fiber[i].sdssv_apogee_target0,1) then objtype='HOT_STD'
            ;; SDSS-V FPS data
            endif else begin
              fiber[i].catalogid = p[match].catalogid
              fiber[i].twomass_designation = p[match].tmass_id
              fiber[i].gaia_g = p[match].gaia_gmag
              fiber[i].gaia_bp = p[match].gaia_bpmag
              fiber[i].gaia_rp = p[match].gaia_rpmag
              fiber[i].jmag = p[match].jmag
              fiber[i].jerr = p[match].e_jmag
              fiber[i].hmag = p[match].hmag
              fiber[i].herr = p[match].e_hmag
              fiber[i].kmag = p[match].kmag
              fiber[i].kerr = p[match].e_kmag
              fiber[i].firstcarton = plugmap.fiberdata[m].firstcarton
              fiber[i].cadence = plugmap.fiberdata[m].cadence
              fiber[i].program = plugmap.fiberdata[m].program
              fiber[i].category = plugmap.fiberdata[m].category
              fiber[i].pmra = p[match].pmra
              fiber[i].pmdec = p[match].pmdec
              fiber[i].gaiadr2_sourceid = p[match].gaiadr2_sourceid
              fnames = ['gaiadr2_ra','gaiadr2_dec','gaiadr2_pmra','gaiadr2_pmdec','gaiadr2_pmra_error',$
                        'gaiadr2_pmdec_error','gaiadr2_plx','gaiadr2_plx_error','gaiadr2_gmag','gaiadr2_gerr',$
                        'gaiadr2_bpmag','gaiadr2_bperr','gaiadr2_rpmag','gaiadr2_rperr']
              for k=0,n_elements(fnames)-1 do begin
                mind1 = where(tag_names(fiber) eq strupcase(fnames[k]),nmind1)
                mind2 = where(tag_names(p) eq strupcase(fnames[k]),nmind2)
                if nmind1 eq 0 or nmind2 eq 0 then stop,'problem with names'
                fiber[i].(mind1[0]) = p[match].(mind2[0])
              endfor
            endelse
          endif
          ;; APOGEE-1/2 target types
          if platenum lt 15000 and not keyword_set(fps) then begin
            objtype = 'OBJECT'
            if is_bit_set(fiber[i].target2,9) eq 1 then objtype='HOT_STD'
            if is_bit_set(fiber[i].target2,4) eq 1 then objtype='SKY'
          endif
          ;; SKY's
          if objtype eq 'SKY' then begin
            objname = 'SKY' 
            hmag = 99.99
            fiber[i].mag = [hmag,hmag,hmag,hmag,hmag]
            fiber[i].objtype = 'SKY'
          endif else begin
            fiber[i].objtype = objtype
            if platenum lt 15000 and not keyword_set(fps) then begin
              tmp = strtrim(p[match].targetids,2)
            endif else begin
              tmp = strtrim(p[match].tmass_id,2)
            endelse
            len = strlen(tmp)
            objname = tmp
            if len ne 16 then begin
              ;; use +/- sign to figure out what characters to use  
              posind = strpos(tmp,'+')
              negind = strpos(tmp,'-')
              signind = posind
              if posind eq -1 then signind = negind
              if signind ne -1 then begin
                objname = strmid(tmp,signind-8,16)  ;; trim any extra characters
              endif else begin
                objname = strmid(tmp,strlen(tmp)-16)   ;; not sure what else to do
              endelse
            endif
            if strpos(tmp,'A') eq 0 then begin
              objname = 'AP'+objname
            endif else begin
              objname = '2M'+objname
            endelse
            hmag = p[match].tmass_h
            fiber[i].mag = [p[match].tmass_j,p[match].tmass_h,p[match].tmass_k,0.,0.]
            ;; Adopt PM un-adjusted  coordinates
            ;fiber[i].ra -= p[match].pmra/1000./3600./cos(fiber[i].dec*!pi/180.)*(p[match].epoch-2000.)
            ;fiber[i].dec -= p[match].pmdec/1000./3600.*(p[match].epoch-2000.)
          endelse
          fiber[i].hmag = hmag
          fiber[i].object = objname
          fiber[i].tmass_style = objname
        endif else print,'no match found in plateHoles!',fiber[i].ra,fiber[i].dec, 300-i
      endelse
    endelse
  endif else begin
    fiber[i].fiberid = -1
    print,'no match for fiber index',i
  endelse
endfor

;; SDSS-V, get catalogdb information
;;----------------------------------
if platenum ge 15000 then begin
  print,'Getting catalogdb information'
  objind = where(fiber.objtype eq 'OBJECT' or fiber.objtype eq 'HOT_STD',nobjind)
  objdata = fiber[objind]
  add_tag,objdata,'dbresults',0,objdata
  gdid = where(objdata.catalogid gt 0,ngdid,comp=bdid,ncomp=nbdid)
  ;; Get catalogdb information using catalogID
  undefine,catalogdb
  if ngdid gt 0 then begin
    print,'Querying catalogdb using catalogID for ',strtrim(ngdid,2),' stars'
    catalogdb1 = get_catalogdb_data(id=objdata[gdid].catalogid)
    ;; Got some results
    if size(catalogdb1,/type) eq 8 then begin
      print,'Got results for ',strtrim(n_elements(catalogdb1),2),' stars'
      push,catalogdb,catalogdb1
      match,catalogdb1.catalogid,objdata.catalogid,ind1,ind2,/sort,count=nmatch
      if nmatch gt 0 then objdata[ind2].dbresults=1
    endif else begin
      print,'No results'
    endelse
  endif

  ;; Get catalogdb information using coordinates (tellurics don't have IDs)    
  if nbdid gt 0 then begin
    print,'Querying catalogdb using coordinates for ',strtrim(nbdid,2),' stars'
    if total(objdata[bdid].catalogid ne -999) gt 0 then begin
        catalogdb2 = get_catalogdb_data(ra=objdata[bdid].ra,dec=objdata[bdid].dec)
        ;; this returns a "q3c_dist" column that we don't want to keep
        if size(catalogdb2,/type) eq 8 then begin
          print,'Got results for ',strtrim(n_elements(catalogdb2),2),' stars'
          catalogdb2 = REMOVE_TAG(catalogdb2,'q3c_dist')
          push,catalogdb,catalogdb2
        endif else begin
          print,'No results'
        endelse
    endif else begin
      print,'No results'
    endelse
  endif

  ;; Add catalogdb information
  for i=0,nobjind-1 do begin
    istar = objind[i]
    MATCH,catalogdb.catalogid,fiber[istar].catalogid,ind1,ind2,/sort,count=nmatch
    ;; some stars are missing ids, use coordinate matching instead
    if nmatch eq 0 then begin
      dist = sphdist(catalogdb.ra,catalogdb.dec,fiber[istar].ra,fiber[istar].dec,/deg)*3600.0
      ind1 = where(dist lt 0.5,nmatch)
      if nmatch gt 1 then ind1 = first_el(minloc(dist))
    endif
    ;; Still no match, just get catalogdb.catalog information 
    if nmatch eq 0 then begin
      ;; We have catalogid 
      if fiber[istar].catalogid gt 0 then begin
         cat = dbquery("select catalogid,ra,dec,version_id from catalogdb.catalog where catalogid="+strtrim(fiber[istar].catalogid,2),count=ncat)
      ;; Use coordinates instead
      endif else begin
         cat = dbquery("select catalogid,ra,dec,version_id from catalogdb.catalog where q3c_radial_query(ra,dec,"+$
                       strtrim(fiber[istar].ra,2)+","+strtrim(fiber[istar].dec,2)+",0.0001)",count=ncat)
         if size(cat,/type) ne 8 then $
           cat = dbquery("select catalogid,ra,dec,version_id from catalogdb.catalog where q3c_radial_query(ra,dec,"+$
                         strtrim(fiber[istar].ra,2)+","+strtrim(fiber[istar].dec,2)+",0.0002)",count=ncat)
      endelse
      ;; Still no match, try using 2MASS ID in TIC
      if ncat eq 0 then begin
        tmass = strmid(fiber[istar].tmass_style,2)
        sql = "select c2t.* from catalog_to_tic_v8 c2t join tic_v8 t on t.id=c2t.target_id join twomass_psc tm on "+$
              "tm.designation = t.twomass_psc join catalogdb.version v on v.id = c2t.version_id where tm.designation = '"+tmass+"'"+$
              " and v.plan = '0.1.0' and c2t.best is true"
        cat = dbquery(sql,count=ncat)
        if ncat gt 0 then begin
          catalogid = cat[0].catalogid
          cat = dbquery("select catalogid,ra,dec,version_id from catalogdb.catalog where catalogid="+strtrim(catalogid,2),count=ncat)
        endif
      endif
      ;; If there are multiple results, pick the closest
      if ncat gt 1 then begin
         dist = sphdist(fiber[istar].ra,fiber[istar].dec,cat.ra,cat.dec,/deg)*3600
         minind = first_el(minloc(dist))
         cat = cat[minind]
      endif
      ;; Add to catalogdb
      if ncat gt 0 then begin
         addcat = catalogdb[0]
         struct_assign,{dum:''},addcat
         for k=0,n_tags(addcat)-1 do begin
           if size(addcat.(k),/type) eq 4 or size(addcat.(k),/type) eq 5 then addcat.(k)=!values.f_nan
           if size(addcat.(k),/type) eq 2 or size(addcat.(k),/type) eq 3 then addcat.(k)=-1
           if size(addcat.(k),/type) eq 7 then addcat.(k)='None'
         endfor
         addcat.catalogid = cat.catalogid
         addcat.ra = cat.ra
         addcat.dec = cat.dec
         addcat.hmag = 99.99
         catalogdb = [catalogdb,addcat]
         ind1 = n_elements(catalogdb)-1   ;; the match for this star, last one in catalogdb
         nmatch = 1
       endif
    endif
    if nmatch gt 0 then begin
      ;; Sometimes the plateHoles tmass_style are "None", try to fix with catalogdb information
      if fiber[istar].tmass_style eq '2MNone' then begin
        ;; Use catalogdb.tic_v8 twomass name
        if catalogdb[ind1[0]].twomass ne 'None' then begin
          fiber[istar].tmass_style = '2M'+catalogdb[ind1[0]].twomass
        ;; Construct 2MASS-style name from GaiaDR2 RA/DEC
        endif else begin
          fiber[istar].tmass_style = '2M'+coords2tmass(catalogdb[ind1[0]].ra,catalogdb[ind1[0]].dec)
        endelse
        print,'Fixing tmass_style ID for ',fiber[istar].tmass_style
      endif
      if fiber[istar].catalogid lt 0 then $
        fiber[istar].catalogid=catalogdb[ind1[0]].catalogid
      fiber[istar].twomass_designation = catalogdb[ind1[0]].twomass
      fiber[istar].jmag = catalogdb[ind1[0]].jmag
      fiber[istar].jerr = catalogdb[ind1[0]].e_jmag
      fiber[istar].hmag = catalogdb[ind1[0]].hmag
      fiber[istar].herr = catalogdb[ind1[0]].e_hmag
      fiber[istar].kmag = catalogdb[ind1[0]].kmag
      fiber[istar].kerr = catalogdb[ind1[0]].e_kmag
      fiber[istar].phflag = catalogdb[ind1[0]].twomflag
      fiber[istar].gaiadr2_sourceid = catalogdb[ind1[0]].gaia
      fiber[istar].gaiadr2_ra = catalogdb[ind1[0]].ra
      fiber[istar].gaiadr2_dec = catalogdb[ind1[0]].dec
      fiber[istar].gaiadr2_pmra = catalogdb[ind1[0]].pmra
      fiber[istar].gaiadr2_pmra_error = catalogdb[ind1[0]].e_pmra
      fiber[istar].gaiadr2_pmdec = catalogdb[ind1[0]].pmdec
      fiber[istar].gaiadr2_pmdec_error = catalogdb[ind1[0]].e_pmdec
      fiber[istar].gaiadr2_plx = catalogdb[ind1[0]].plx
      fiber[istar].gaiadr2_plx_error = catalogdb[ind1[0]].e_plx
      fiber[istar].gaiadr2_gmag = catalogdb[ind1[0]].gaiamag
      fiber[istar].gaiadr2_gerr = catalogdb[ind1[0]].e_gaiamag
      fiber[istar].gaiadr2_bpmag = catalogdb[ind1[0]].gaiabp
      fiber[istar].gaiadr2_bperr = catalogdb[ind1[0]].e_gaiabp
      fiber[istar].gaiadr2_rpmag = catalogdb[ind1[0]].gaiarp
      fiber[istar].gaiadr2_rperr = catalogdb[ind1[0]].e_gaiarp
    endif else begin
      print,'WARNING: no catalogdb match for ',fiber[istar].object
    endelse
  endfor ; object loop

endif  ; get catalogdb info

;; SDSS-V FPS data
if keyword_set(fps) then begin
  platedata.field = plugmap.field_id
  if n_elements(p) gt 0 then platedata.programname=p[0].carton                
endif

;; Load apogeeObject file to get proper name and coordinates
;; Get apogeeObject catalog info for this field
;; No apogeeObject files for SDSS-V   
if apogee2 then apogeeobject='apogee2Object' else apogeeobject='apogeeObject'
if not keyword_set(noobject) and not keyword_set(sdss5) then begin
  targetdir = getenv('APOGEE_TARGET')

  ;; Get apogeeObject catalog info for this field
  ;; Find all matching apogeeObject files and loop through them looking for matches
  field=strtrim(apogee_field(platedata.locationid,platenum),2)
  files=file_search(targetdir+'/apogee*Object/*'+field+'*')
  if files[0] eq '' then begin
    stop,'cant find apogeeObject file: '+field
    return,field
  endif else begin
    if n_elements(files) gt 1 then print,'using multiple apogeeObject files: '+ files

    ;; We will only save tags we will use, to avoid conflict between apogeeObject and apogee2Object
    objects = []
    for ifile=0,n_elements(files)-1 do begin
      print,files[ifile]
      tmpobject = mrdfits(files[ifile],1)
      tmp_cat = replicate(catalog_info_common(),n_elements(tmpobject))
      struct_assign, tmpobject, tmp_cat
      print,n_elements(tmpobject)
      objects = [objects,tmp_cat]
    endfor
    ;; Fix NaNs, etc.
    aspcap_fixobject,objects

    ;; Match stars with coordinates
    ;;  deal with fiber RA/DEC = 999999.
    bdra = where(fiber.ra lt -0.1 or fiber.ra gt 360.1,nbdra)
    if nbdra gt 0 then begin
      tra = fiber.ra
      tra[bdra] = 0.0
      tdec = fiber.dec
      tdec[bdra] = -89.9999
      spherematch,objects.ra,objects.dec,tra,tdec,10./3600.,match1,match2,dist,maxmatch=1
    endif else begin
      spherematch,objects.ra,objects.dec,fiber.ra,fiber.dec,10./3600.,match1,match2,dist,maxmatch=1
    endelse
    for i=0,299 do begin
      if fiber[i].objtype eq 'STAR' or fiber[i].objtype eq 'HOT_STD' then begin
        j = where(match2 eq i,nj)
        if nj gt 0 then begin
          if strtrim(fiber[i].object,2) ne strtrim(objects[match1[j]].apogee_id) then begin
            print,'apogeeObject differs from plateHoles: '
            print,fiber[i].object+' '+objects[match1[j]].apogee_id
            print,fiber[i].ra,' ',objects[match1[j]].ra
            print,fiber[i].dec,' ',objects[match1[j]].dec
          endif
          fiber[i].tmass_style = objects[match1[j]].apogee_id
          fiber[i].ra = objects[match1[j]].ra
          fiber[i].dec = objects[match1[j]].dec
          ;if finite(objects[match1[j]].ak_targ) then fiber[i].ak_targ=objects[match1[j]].ak_targ
          ;fiber[i].ak_targ_method = objects[match1[j]].ak_targ_method
          ;if finite(objects[match1[j]].ak_wise) then fiber[i].ak_wise=objects[match1[j]].ak_wise
          ;if finite(objects[match1[j]].sfd_ebv) then fiber[i].sfd_ebv=objects[match1[j]].sfd_ebv
          fiber[i].jmag = objects[match1[j]].jmag
          fiber[i].jerr = objects[match1[j]].jerr
          fiber[i].hmag = objects[match1[j]].hmag
          fiber[i].herr = objects[match1[j]].herr
          fiber[i].kmag = objects[match1[j]].kmag
          fiber[i].kerr = objects[match1[j]].kerr
          fiber[i].alt_id = objects[match1[j]].alt_id
          fiber[i].src_h = objects[match1[j]].src_h
          ;fiber[i].wash_m = objects[match1[j]].wash_m
          ;fiber[i].wash_m_err = objects[match1[j]].wash_m_err
          ;fiber[i].wash_t2 = objects[match1[j]].wash_t2
          ;fiber[i].wash_t2_err = objects[match1[j]].wash_t2_err
          ;fiber[i].ddo51 = objects[match1[j]].ddo51
          ;fiber[i].ddo51_err = objects[match1[j]].ddo51_err
          ;fiber[i].irac_3_6 = objects[match1[j]].irac_3_6
          ;fiber[i].irac_3_6_err = objects[match1[j]].irac_3_6_err
          ;fiber[i].irac_4_5 = objects[match1[j]].irac_4_5
          ;fiber[i].irac_4_5_err = objects[match1[j]].irac_4_5_err
          ;fiber[i].irac_5_8 = objects[match1[j]].irac_5_8
          ;fiber[i].irac_5_8_err = objects[match1[j]].irac_5_8_err
          ;fiber[i].irac_8_0 = objects[match1[j]].irac_8_0
          ;fiber[i].irac_8_0_err = objects[match1[j]].irac_8_0_err
          ;fiber[i].wise_4_5 = objects[match1[j]].wise_4_5
          ;fiber[i].wise_4_5_err = objects[match1[j]].wise_4_5_err
          ;fiber[i].targ_4_5 = objects[match1[j]].targ_4_5
          ;fiber[i].targ_4_5_err = objects[match1[j]].targ_4_5_err
          ;fiber[i].wash_ddo51_giant_flag = objects[match1[j]].wash_ddo51_giant_flag
          ;fiber[i].wash_ddo51_star_flag = objects[match1[j]].wash_ddo51_star_flag
          fiber[i].pmra = objects[match1[j]].pmra
          fiber[i].pmdec = objects[match1[j]].pmdec
          fiber[i].pm_src = objects[match1[j]].pm_src
        endif else print,'not halted: no match in object file for ',fiber[i].object
      endif 
    endfor
  endelse
endif

platedata.fiberdata = fiber

;; Save the platedata to file
print,'Saving platedata to '+platedatafile
SAVEPLATEDATA,platedatafile,platedata


if keyword_set(stp) then stop

return,platedata

end




