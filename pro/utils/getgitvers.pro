function getgitvers
  spawn,['apgitvers'],out,/noshell
  return,out[0]
end

