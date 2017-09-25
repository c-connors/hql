require "os"
require "test_model"

function file_exists(name)
    local f=io.open(name,"r")
    if f~=nil then io.close(f) return true else return false end
end

function eval_model(game, model, n_play, csv)
    if not file_exists(csv) and not file_exists(csv .. ".tmp") then
        local f = io.open(csv .. '.tmp', "w")
        print("Evalauting " .. csv)
        test_model(game, model, nil, nil, n_play, ' --csv ' .. csv)
        f:close()
        os.execute("rm " .. csv .. '.tmp')
    end
end

local cmd = torch.CmdLine()
cmd:option('--model', '', 'model file path')
cmd:option('--csv', '', 'csv file path')
cmd:option('--n_play', 50, 'num play')
opts = cmd:parse(arg or {})

eval_model("final_nc", opts.model, opts.n_play, opts.csv .. '_train.csv')
eval_model("final_seen", opts.model, opts.n_play, opts.csv .. '_seen.csv')
eval_model("final_unseen", opts.model, opts.n_play, opts.csv .. '_unseen.csv')
eval_model("final_unseen_no_all", opts.model, opts.n_play, opts.csv .. '_unseen_no_all.csv')
eval_model("final_seen_no_all", opts.model, opts.n_play, opts.csv .. '_seen_no_all.csv')
