require "xlua"
require "image"
require "paths"
paths.dofile('util.lua')
paths.dofile('model.lua')
paths.dofile('MazeBase/init.lua')
torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
-- game parameters
cmd:option('--game', 'goto1', 'MazeBase/config/%s.lua')
cmd:option('--load', '', 'file name to load the model')
cmd:option('--result', '', 'folder to write screen images and video')
cmd:option('--n_play', 1, 'num of play')

g_opts = cmd:parse(arg or {})
g_opts.test = true
g_opts.games_config_path = "MazeBase/config/" .. g_opts.game .. ".lua"
g_opts.game = nil
print(g_opts)
g_init_vocab()
g_init_game()
g_init_model()

local task_name = {"visit", "pick_up", "hit"}
local input
local obs = torch.Tensor(1, g_opts.max_items, g_opts.max_size, g_opts.max_size):fill(0)
local arg = torch.Tensor(1, #g_opts.num_args)
if g_opts.result ~= "" then
    os.execute("rm -rf " .. g_opts.result)
    os.execute("mkdir " .. g_opts.result)
end
for i=1,g_opts.n_play do
    local g = new_game()
    g.map:remove_item(g.agent)
    local img = g.map:to_image()
    image.save(string.format("%s/%02d.png", g_opts.result, i), img)

    xlua.progress(i, g_opts.n_play)
    for arg1=1,g_opts.num_args[1] do
        for arg2=1,g_opts.num_args[2] do
            arg:zero()
            arg[1][1] = arg1
            arg[1][2] = arg2

            local task = string.format("%s_%s", task_name[arg1], g_objects[arg2].name)
            local f = io.open(string.format("%s/%02d_%s.csv", g_opts.result, i, task), "w")
            for k = 1, g_opts.max_size * g_opts.max_size do 
                local y = math.floor((k-1) / g_opts.max_size) + 1
                local x = math.floor((k-1) % g_opts.max_size) + 1

                if g.map:is_loc_reachable(y, x) then
                    -- move the agent 
                    g.map:remove_item(g.agent)
                    g.agent.loc.y = y
                    g.agent.loc.x = x
                    g.map:add_item(g.agent)
                    g:to_tensor(obs[1])
                    
                    if g_model.args.buf == nil or g_model.args.buf == 1 then
                        input = obs
                    else
                        if not input then
                            local size = obs:size()
                            assert(size[2] == g_model.args.channels)
                            size[2] = g_model.args.buf * g_model.args.channels
                            input = torch.Tensor(size)
                        end
                        input:zero()
                        input:narrow(2, 1, g_model.args.channels):copy(obs)
                    end
                    --print(input, arg)
                    action, baseline = g_model:eval(input, arg)
                    f:write(string.format("%.2f ", baseline:squeeze()))
                else
                    f:write(string.format("%.2f ", 0))
                end
                
                if x == g_opts.max_size then
                    f:write("\n")
                end
            end
            f:close()
        end
    end
end
