pro apwait,file,time,silent=silent,maxduration=maxduration
  if not keyword_set(silent) then print,'waiting for file lock: ', file
  wait,time
end
