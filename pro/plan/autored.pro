;+
; AUTORED
;      automated reduction procedure for APOGEE data
;      creates MJD5auto.pro files
;      runs runapred if plate not already done
;      runs mkhtml,mkhtmlsum,mkmonitor when all plates for a given MJD are done
;-
pro autored,mjds,vers=vers,norun=norun,apogees=apogees,override=override,aspcap=aspcap,suffix=suffix

; setup version and directories
if keyword_set(vers) then apsetver,vers=vers else stop,'need to set vers'
if ~keyword_set(aspcap) then aspcap='a'

if keyword_set(apogees) then begin
  prefix = 'as' 
  telescope = 'lco25m'
  instrument = 'apogee-s'
  apogee_data = getenv('APOGEE_DATA_S')
  mapper_data = getenv('MAPPER_DATA_S')
endif else begin
  prefix = 'ap'
  telescope = 'apo25m'
  instrument = 'apogee-n'
  apogee_data = getenv('APOGEE_DATA_N')
  mapper_data = getenv('MAPPER_DATA_N')
endelse
apsetver,vers=vers,telescope=telescope

dirs = getdir(a,c,s)
dir = s+'/autored/'+telescope+'/'
file_mkdir,dir

; Loop over each MJD
dosum = 0
for i=0,n_elements(mjds)-1 do begin
  mjd = mjds[i]
  cmjd = string(format='(i5.5)',mjd)
  print,mjd,cmjd
  ; If this MJD is done, we are done
  if ~file_test(dir+cmjd+'.done') then begin
    ; If not done has it been started?
    if file_test(dir+cmjd+'.plans') then begin
      ; If it has been started, are all the jobs done? If so, run the MJD summary
      readcol,dir+cmjd+'.plans',format='(a)',plans
      done = 1
      for j=0,n_elements(plans)-1 do begin
        junk = strsplit(file_basename(plans[j],'.par'),'-',/ext)
        cplate = junk[1]
        if strpos(plans[j],'Dark') ge 0 or strpos(plans[j],'Cal') ge 0 then $
           qafile = apogee_filename('QAcal',plate=cplate,mjd=cmjd) else $
           qafile = apogee_filename('QA',plate=cplate,mjd=cmjd)
        if ~file_test(qafile) and $
           strpos(file_basename(plans[j]),'sky') lt 0 and $
           strpos(file_basename(plans[j]),'dark') lt 0 $
           then done=0
        print,plans[j]
        print,cplate
        print,qafile
        print,file_test(qafile), done
      endfor
      ; This MJD is done, running apMJD
      if done then begin
        print,'all reductions complete'
        if ~file_test(s+'/exposures/'+instrument+'/'+cmjd+'/html/'+cmjd+'.html') then begin
          file_mkdir,s+'exposures/'+instrument+'/'+cmjd+'/plan'
          openw,plan,s+'exposures/'+instrument+'/'+cmjd+'/plan/'+prefix+'MJD-'+cmjd+'.par',/get_lun
          printf,plan,'apred_vers  '+vers
          printf,plan,'telescope  '+telescope
          printf,plan,'mjd  '+cmjd
          free_lun,plan
          openw,out,dir+cmjd+'.csh',/get_lun
          printf,out,'#!/bin/csh'
          printf,out,'cd $APOGEE_REDUX/'+vers
          printf,out,'apred exposures/'+instrument+'/'+cmjd+'/plan/'+prefix+'MJD*.par'
          free_lun,out
          print,'running apMJD...'
          if ~keyword_set(norun) then spawn,'csh '+dir+cmjd+'.csh >&'+dir+cmjd+'.log'
          dosum=1
        endif

        ;file_delete,dir+cmjd+'.csh',/allow
        ;file_delete,dir+cmjd+'.plans',/allow
        openw,out,dir+cmjd+'.done',/get_lun
        free_lun,out
      endif else begin
        print,'reductions still running'
      endelse

    ; Has not been started yet
    endif else begin

      ; If not started, make the plan files and start the reductions if transfer is complete
      if file_test(apogee_data+'/'+cmjd+'/'+cmjd+'.log') then begin
        ; Check if the data transfer is complete
        readcol,apogee_data+'/'+cmjd+'/'+cmjd+'.log',n,f,skip=3,format='(i,a)'
        complete = 1
        for j=0,n_elements(f)-1 do begin
          if ~file_test(apogee_data+'/'+cmjd+'/'+f[j]) then complete=0 else begin
            ; Check to see if plugmap is available
            h=headfits(apogee_data+'/'+cmjd+'/'+f[j],ext=1)
            exptype=strtrim(sxpar(h,'exptype'),2)
            if exptype eq 'OBJECT' then begin
              plugid=strtrim(sxpar(h,'name'),2)
              tmp=strsplit(plugid,'-',/extract)
              if ~file_test(mapper_data+'/'+tmp[1]+'/plPlugMapM-'+plugid+'.par') then begin
                print,'No plugmap found for: ', f[j],' ',plugid
                if ~keyword_set(override) then complete=0
              endif
            endif
          endelse
        endfor
        ; Data transfer is complete, make plan files and start reductions
        if complete then begin
          print,cmjd+' not done and transferred, creating plan files and running'
          undefine,planfiles
          ; Make the automatic reduction file and copy to MJD5.pro if it doesn't exist
          apmkplan,mjd,planfiles=planfiles,apogees=apogees,vers=vers
          if ~file_test(getenv('APOGEEREDUCEPLAN_DIR')+'/pro/'+telescope+'/'+telescope+'_'+cmjd+'.pro') then begin
            file_copy,getenv('APOGEEREDUCEPLAN_DIR')+'/pro/'+telescope+'/'+telescope+'_'+cmjd+'auto.pro',$
                      getenv('APOGEEREDUCEPLAN_DIR')+'/pro/'+telescope+'/'+telescope+'_'+cmjd+'.pro' 
          endif
          openw,out,dir+cmjd+'.plans' ,/get_lun
          for j=0,n_elements(planfiles)-1 do printf,out,planfiles[j]
          free_lun,out
          openw,out,dir+cmjd+'.slurm',/get_lun
          printf,out,'#!/bin/csh'
          printf,out,'#SBATCH --account=sdss-kp-fast'
          printf,out,'#SBATCH --partition=sdss-kp'
          printf,out,'#SBATCH --ntasks=16'
          printf,out,'#SBATCH --time=240:00:00'
          printf,out,'#SBATCH --nodes=1'
          printf,out,'#SBATCH -o '+cmjd+'.out'
          printf,out,'#SBATCH -e '+cmjd+'.out'
          printf,out,'setenv QUERYHOST apogee'
          printf,out,'setenv QUERYPORT 1050'
          printf,out,'setenv APOGEE_MAXRUN 8'
          printf,out,'setenv APOGEE_FLAG 1111111'
          printf,out,'setenv IDL_CPU_TPOOL_NTHREADS 4'
          printf,out,'cd $APOGEE_REDUX/'+vers
          printf,out,'idl << endidl'
          printf,out," vers='"+vers+"'"
          printf,out,' apsetver,vers=vers'
          printf,out,' @'+telescope+'_'+cmjd+'.pro'
          printf,out,'endidl'
          if ~keyword_set(norun) then printf,out,'runplans apred visit/'+telescope+'/*/*/'+cmjd+'/a?Plan*.par'+' cal/'+instrument+'/'+cmjd+'/*Plan*.par'
          printf,out,'wait'
          printf,out,'echo DONE'
          free_lun,out
          spawn,'sbatch '+dir+cmjd+'.slurm'

        endif else print,cmjd+' not done, but still transferring'
      endif else print,'no data log file yet...'
    endelse
  endif else print,' already done'
endfor  ; mjd loop

help,dosum

; Making summary/monitoring pages
if dosum then begin
  mkmonitor
  mkhtmlsum,/nocheck,apred=vers,apstar='stars',aspcap=aspcap,results='v',suffix=suffix
endif

end
