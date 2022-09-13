pro strdivide,num,divisor,ans,remainder
  ;; Divide a number as astring by a number

  ;; https://www.geeksforgeeks.org/divide-large-number-represented-string/
  ;; A function to perform division of large numbers

  ;; Divisor greater than the number
  if strlen(strtrim(divisor,2)) gt strlen(strtrim(num,2)) then begin
    ans = '0'
    remainder = num
    return
  endif
     
  number = strarr(strlen(strtrim(num,2)))
  bytes = byte(strtrim(num,2))
  for i=0,n_elements(number)-1 do number[i]=string(bytes[i])

  ;; Find prefix of number that is larger
  ;; than divisor.
  idx = 0
  temp = long(number[idx])
  while ((temp lt divisor) and (idx lt n_elements(number)-1)) do begin
    temp = temp * 10 + long(number[idx+1])
    idx += 1
  endwhile
  idx += 1
  
  ;; Repeatedly divide divisor with temp. After
  ;; every division, update temp to include one
  ;; more digit.
  ans = ''
  while (n_elements(number) gt idx) do begin
     ;; Store result in answer i.e. temp / divisor
     ans += strtrim(floor(temp / divisor),2)
     ;; Take next digit of number
     temp = ((temp mod divisor) * 10 + long(number[idx]))
     idx += 1
  endwhile
  ans += strtrim(floor(temp / divisor),2)
  remainder = temp mod divisor

  if ans eq '' then ans='0'
  
end

  
function strint2bits,num

  ;; Convert a large number (as a string) into binary bits

  ;; https://indepth.dev/posts/1019/the-simple-math-behind-decimal-binary-conversion-algorithms
  
  numleft = strtrim(num,2)
  n = 10*strlen(numleft)
  bits = bytarr(n)
  answer = 1
  i = 0
  while (answer ne '0') do begin
     strdivide,numleft,2,answer,remainder
     if remainder eq 1 then bits[i]=1
     numleft = answer
     i += 1
  endwhile
  bits = bits[0:i-1]    ;; trim
  bits = reverse(bits)  ;; reverse
  sbits = strjoin(strtrim(fix(bits),2),'')  ;; convert to string
  
  return,sbits

end
