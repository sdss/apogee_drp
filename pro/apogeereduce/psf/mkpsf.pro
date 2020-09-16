;======================================================================
pro mkpsf,psfid,darkid=darkid,flatid=flatid,sparseid=sparseid,fiberid=fiberid,littrowid=littrowid,average=average,clobber=clobber

  dirs = getdir(apodir,caldir,spectrodir,vers)
  caldir = dirs.caldir

  file = apogee_filename('PSF',num=psfid[0],chip='c')

  ;; If another process is alreadying make this file, wait!
  while file_test(file+'.lock') do apwait,file,10
  ;; Does product already exist?
  if file_test(file) and not keyword_set(clobber) then begin
    print,' PSF file: ', file+'.fits', ' already made'
    return
  endif
  if not keyword_set(fiberid) then fiberid=0
  if not keyword_set(sparseid) then sparseid=0

  print,'Making PSF: ', psfid[0]
  ;; Open .lock file
  openw,lock,/get_lun,file+'.lock'
  free_lun,lock

  cmjd = getcmjd(psfid)
  print,'mkpsf approcess...'
  d = approcess(psfid,darkid=darkid,flatid=flatid,littrowid=littrowid,/nocr,nfs=1,/doap3dproc)
  psffile = file_dirname(apogee_filename('2D',num=psfid[0],chip='c'))+'/'+string(format='(i8.8)',psfid)
  outdir = file_dirname(file)+'/'   ;; PSF directory
  APMKPSF,psffile,outdir,sparseid=sparseid,fiberid=fiberid,average=average,clobber=clobber

  file_delete,caldir+'psf/'+file+'.lock'
end

