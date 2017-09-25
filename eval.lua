require "os"
require "test_model"

function eval_model(game, model, n_play, csv)
    --if not file_exists(csv) and not file_exists(csv .. ".tmp") then
    --    local f = io.open(csv .. '.tmp', "w")
    print("Evalauting " .. csv)
    test_model(game, model, nil, nil, n_play, ' --csv ' .. csv)
    --    f:close()
    --    os.execute("rm " .. csv .. '.tmp')
    --end
end

local cmd = torch.CmdLine()
cmd:option('--model', '', 'model file path')
cmd:option('--csv', '', 'csv file path')
cmd:option('--n_play', 4000, 'num play')
opts = cmd:parse(arg or {})

eval_model("a_count_seen_test", opts.model, opts.n_play, opts.csv .. '_seen.csv')
eval_model("a_count_unseen_test", opts.model, opts.n_play, opts.csv .. '_unseen.csv')
