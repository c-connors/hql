os.execute("mkdir -p jobs")
num_iter = 6
jobs = {}

for i=1,num_iter do
    table.insert(jobs, {cmd='./a_train a_visit' .. i, id = 'jobs/a_visit' .. i})
    table.insert(jobs, {cmd='./a_train a_pickup' .. i, id = 'jobs/a_pickup' .. i})
    table.insert(jobs, {cmd='./a_train a_transform' .. i, id = 'jobs/a_transform' .. i})
    table.insert(jobs, {cmd='./a_train a_interact' .. i .. ' --interact', id = 'jobs/a_interact' .. i})
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
