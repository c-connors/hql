function csv_read(file, data)
    csvFile = io.open(file, 'r')  
    if csvFile == nil then
        error(csvFile .. " not found")
    end
    local i = 0  
    for line in csvFile:lines('*l') do  
      i = i + 1
      if i > data:size(1) then
          break
      end
      local l = line:split(' ')
      for key, val in ipairs(l) do
        data[i][key] = val
      end
    end
    if i < data:size(1) then
        data:resize(i, data:size(2))
    end
    csvFile:close()
end
