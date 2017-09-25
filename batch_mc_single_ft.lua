os.execute("mkdir -p jobs")
num_iter = 4
jobs = {}

for i=1,num_iter do
    table.insert(jobs, {cmd='./mc_train_ft2 mc_visit' .. i, id = 'jobs/ft2_visit' .. i})
    table.insert(jobs, {cmd='./mc_train_ft2 mc_pickup' .. i, id = 'jobs/ft2_pickup' .. i})
    table.insert(jobs, {cmd='./mc_train_ft2 mc_transform' .. i, id = 'jobs/ft2_transform' .. i})
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
