path = "multi_final"
os.execute("mkdir -p " .. path)
os.execute("mkdir -p " .. path .. "/jobs")
num_iter = 10
jobs = {}

for i=1,num_iter do
    table.insert(jobs, {cmd='./train_pp_am ' .. path .. '/' .. i, id = path .. '/jobs/pp_am_' .. i})
    table.insert(jobs, {cmd='./train_pp ' .. path .. '/' .. i, id = path .. '/jobs/pp' .. i})
    table.insert(jobs, {cmd='./train_cc_am ' .. path .. '/' .. i, id = path .. '/jobs/cc_am_' .. i})
    table.insert(jobs, {cmd='./train_cc ' .. path .. '/' .. i, id = path .. '/jobs/cc' .. i})
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
