path = "meta"
os.execute("mkdir -p " .. path)
os.execute("mkdir -p " .. path .. "/jobs")
num_iter = 20
jobs = {}

for i=1,num_iter do
    table.insert(jobs, {cmd='./train_ta ' .. path .. '/' .. i, id = path .. '/jobs/ta_' .. i})
    table.insert(jobs, {cmd='./train_non_ta ' .. path .. '/' .. i, id = path .. '/jobs/nta_' .. i})
    table.insert(jobs, {cmd='./train_ta_non_r ' .. path .. '/' .. i, id = path .. '/jobs/ta_nreg_' .. i})
    table.insert(jobs, {cmd='./train_nta_nreg ' .. path .. '/' .. i, id = path .. '/jobs/nta_nreg_' .. i})
end

function file_exists(name)
    local f=io.open(name,"r")
    if f~=nil then io.close(f) return true else return false end
end

for i=1,#jobs do
    if not file_exists(jobs[i].id) then
        os.execute("mkdir -p " .. jobs[i].id)
        os.execute(jobs[i].cmd)
        os.execute("rmdir " .. jobs[i].id)
    end
end
