path = "final"
os.execute("mkdir -p " .. path)
os.execute("mkdir -p " .. path .. "/jobs")
jobs = {}

start_iter = 101
end_iter = 120
for i=start_iter,end_iter do
    table.insert(jobs, {cmd='./train_ta2 ' .. path .. '/' .. i, id = path .. '/jobs/ta_' .. i})
    table.insert(jobs, {cmd='./train_non_ta2 ' .. path .. '/' .. i, id = path .. '/jobs/nta_' .. i})
    table.insert(jobs, {cmd='./train_ta_non_r2 ' .. path .. '/' .. i, id = path .. '/jobs/ta_nreg_' .. i})
end

function file_exists(name)
    local f=io.open(name,"r")
    if f~=nil then io.close(f) return true else return false end
end

for i=1,#jobs do
    if not file_exists(jobs[i].id) then
        os.execute("mkdir -p " .. jobs[i].id)
        os.execute(jobs[i].cmd)
    end
end
