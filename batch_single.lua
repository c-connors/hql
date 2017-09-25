os.execute("mkdir -p jobs")
num_iter = 10
jobs = {}

for i=1,num_iter do
    table.insert(jobs, {cmd='./visit ' .. i, id = 'jobs/visit' .. i})
    table.insert(jobs, {cmd='./pickup ' .. i, id = 'jobs/pickup' .. i})
    table.insert(jobs, {cmd='./transform ' .. i, id = 'jobs/transform' .. i})
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
