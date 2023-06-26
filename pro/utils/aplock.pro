;+
;
; APLOCK
;
; Procedure to handle using lock files.  The default behavior is to
; check for a lock file and wait until it no longer exists.
;
; This is generally used for APOGEE calibration files.
; The standard usage is:
;   aplock,calfile          ;; wait on lock file
;     check if the calibration already exists. if it does, then return
;   aplock,calfile,/lock    ;; create the lock file
;     make the calibration file
;   aplock,calfile,/clear   ;; clear the lock file
;
; INPUTS:
;  file          Original file to be locked.  The lock file is file+'.lock'.
;  =waittime     Time to wait before checking the lock file again.
;  /clear        Clear/delete a lock file.  Normally at the end of processing.
;  /unlock       If the lock file exists, then unlock/delete it.
;  /lock         Relock the file at the end of the wait.
;  =maxduration  Maximum duration of the original file being unmodified.
;                  Default is 5 hours.
;  /silent       Do not print anything to the screen.
;
; OUTPUTS:
;  Nothing is returned
;
; USAGE:
;  IDL>aplock,file,waittime=10
;
; By D. Nidever June 2023
;-

pro aplock,file,waittime=waittime,clear=clear,unlock=unlock,$
           lock=lock,maxduration=maxduration,silent=silent
  
  ;; Defaults
  if n_elements(waittime) eq 0 then waittime=10
  if n_elements(maxduration) eq 0 then maxduration=5*3600 ; default, 5 hours
  lockfile = file+'.lock'
  dir = file_expand_path(file_dirname(file))
  if file_test(dir,/directory) eq 0 then file_mkdir,dir  ;; make directory if necessary
  
  ;; Clear or unlock the lock file
  if keyword_set(clear) or keyword_set(unlock) then begin
    file_delete,lockfile,/allow
    return
  endif

  ;; Wait for the lockfile
  while file_test(lockfile) do begin
    if not keyword_set(silent) then print,'waiting for file lock: ', lockfile
    ;; How long has it been since the file has been modified
    info = file_info(file)
    curtime = systime(1)    
    if info.exists then begin  ;; make sure it exists
      if curtime gt info.mtime+maxduration then begin
        if not keyword_set(silent) then $
           print,'lock file exists but original file unchanged in over '+strtrim(string(maxduration/3600,format='(F8.2)'),2)+' hours'
        file_delete,lockfile,/allow
        if keyword_set(lock) then touchzero,lockfile  ;; Lock it again
        return
      endif
      
    ;; Original file doesn't exist, check how long we have been waiting for it      
    endif else begin
      linfo = file_info(lockfile)
      if curtime gt linfo.ctime+maxduration then begin
        if not keyword_set(silent) then $
           print,'lock file exists but waiting for over '+strtrim(string(maxduration/3600,format='(F8.2)'),2)+' hours'
        file_delete,lockfile,/allow
        if keyword_set(lock) then touchzero,lockfile  ;; Lock it again
        return
      endif
    endelse
    wait,waittime
  endwhile

  ;; Lock it
  if keyword_set(lock) then touchzero,lockfile
  
end
