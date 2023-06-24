;+
;
; APLOCK
;
; Procedure to handle using lock files.  The default behavior is to
; check for a lock file and wait until it no longer exists.
;
; INPUTS:
;  file          Original file to be locked.  The lock file is file+'.lock'.
;  =waittime     Time to wait before checking the lock file again.
;  /unlock       If the lock file exits, then unlock/delete it.
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

pro aplock,file,waittime=waittime,unlock=unlock,silent=silent,maxduration=maxduration,lock=lock


  
  ;; Defaults
  if n_elements(time) eq 0 then time=10
  if n_elements(maxduration) eq 0 then maxduration=5*3600 ; default, 5 hours
  lockfile = file+'.lock'

  ;; Clear the lock file
  if keyword_set(clear) then begin
    file_delete,lockfile,/allow
    return
  endif
  
  ;; Unlock
  if keyword_set(unlock) then begin
    if file_test(lockfile) eq 0 then filetouch,lockfile
    return
  endif
  
  while file_test(lockfile) do begin
    if not keyword_set(silent) then print,'waiting for file lock: ', lockfile
    ;; How long has it been since the file has been modified
    info = file_info(file)
    curtime = systime(1)
    if curtime gt info.mtime+maxduration then begin
      if not keyword_set(silent) then $
         print,'lock file still exists but original file has not changed in more than '+strtrim(string(maxduration/3600,format='(F8.2)'),2)+' hours')
      file_delete,lockfile,/allow
      return
    endif
    wait,time
  endwhile

end
