pro apwait,file,time,silent=silent,maxduration=maxduration
  if not keyword_set(silent) then print,'waiting for file lock: ', file
  if n_elements(maxduration) eq 0 then maxduration=5*3600  ; default, 5 hours
  if n_elements(maxduration) gt 0 then begin
     info = file_info(file)
     curtime = systime(1)
     if curtime gt info.mtime+maxduration then return
  endif
  wait,time
end
