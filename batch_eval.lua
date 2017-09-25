path = "no_penalty"
num_iter = 20
jobs = {}

for i=1,num_iter do
    table.insert(jobs, {cmd='./eval ' .. path .. '/' .. i .. ' ta', id = path .. '/jobs/ta_' .. i})
    table.insert(jobs, {cmd='./eval ' .. path .. '/' .. i .. ' nta', id = path .. '/jobs/nta_' .. i})
    table.insert(jobs, {cmd='./eval ' .. path .. '/' .. i .. ' ta_nreg', id = path .. '/jobs/ta_nreg_' .. i})
    table.insert(jobs, {cmd='./eval ' .. path .. '/' .. i .. ' nta_nreg', id = path .. '/jobs/nta_nreg_' .. i})
    -- table.insert(jobs, {cmd='./eval ' .. path .. '/' .. i .. ' flat', id = path .. '/jobs/flat_' .. i})
end

function file_exists(name)
    local f=io.open(name,"r")
    if f~=nil then io.close(f) return true else return false end
end

for i=1,#jobs do
    os.execute(jobs[i].cmd)
end
