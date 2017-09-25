require 'torch'
paths.dofile('util.lua')
paths.dofile('model.lua')
paths.dofile('MazeBase/init.lua')
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

function goal_text(goal)
    local str = ""
    local g = goal:clone():squeeze()
    for i=1,g:nElement() do
        if g_ivocab[g[i]] ~= "nil" then
            if i > 1 then
                str = str .. " "
            end
            assert(g_ivocab[g[i]], tostring(g[i]) .. " is not defined in vocab")
            str = str .. g_ivocab[g[i]]
        end
    end
    return str
end

g_opts = {gpu = 0}
local Demo = torch.class('Demo')
function Demo:__init(game, model, multitask, sub_agent, img_size, minecraft, counting, 
            interact, open_loop, counting2, gt_instruction)
    if multitask and multitask ~= 0 then
        g_opts.multitask = true
        g_opts.max_task = 1
        g_opts.term_action = true
    else
        g_opts.multitask = nil
        g_opts.term_action = nil
    end
    if minecraft then 
        g_opts.backend = 'minecraft'
        g_opts.task_name = 'maze'
    else
        g_opts.backend = 'maze'
    end

    g_opts.test = true
    g_opts.counting = counting
    g_opts.counting2 = counting2
    g_opts.interact = interact
    g_opts.gt_instruction = gt_instruction

    if open_loop == nil or open_loop == 0 then
        g_opts.open_loop = nil
    else
        g_opts.open_loop = true
    end
    g_opts.games_config_path = "MazeBase/config/" .. game .. ".lua"
    g_opts.game = nil
    g_init_vocab()
    g_factory = g_init_game(g_opts, g_vocab, g_opts.games_config_path)
    g_factory:hardest()
    g_factory:freeze()

    g_opts.load = model
    if sub_agent and sub_agent ~= "" then
        g_opts.sub_agent = sub_agent
    end
    g_train_factory = g_factory
    g_init_model()
    g_opts.img_size = g_model.args.input_dim
    g_opts.display_size = g_opts.img_size
    g_opts.feasible = g_model.args.feasible
    if minecraft then 
        g_opts.display_size = img_size or g_opts.img_size
        g_minecraft = Minecraft.new(g_opts.task_name, g_opts.display_size)
    end
end

function Demo:goal_text()
    local str_list = {}
    for i=1,self.game.max_task do
        local txt = goal_text(self.game.instruction[i])
        if txt == '' then
            break
        end
        table.insert(str_list, txt)
    end
    return str_list
end

function Demo:get_metadata()
    local ret = {}
    if g_minecraft then
        local topology, objects = g_minecraft:get_topology_object(self.game)
        ret.topology = topology
        ret.objects = objects
        ret.pos_y = self.game.agent.loc.y
        ret.pos_x = self.game.agent.loc.x
        ret.yaw = self.game.agent.yaw
    end
    return ret
end

function Demo:update_observation(obs)
    if obs then
        self.obs[self.t] = torch.Tensor(3, g_opts.img_size, g_opts.img_size):fill(0)
        self.obs[self.t]:copy(obs)
    else
        self.obs[self.t] = torch.Tensor(3, g_opts.img_size, g_opts.img_size):fill(0)
        self.game:to_image(self.obs[self.t])
    end
end

function Demo:new_game()
    self.game = new_game()
    self.n_task = self.game.ntasks
    self.obs = {}
    self.t = 1
    g_model:reset_init_state(1)

    self.goal = torch.Tensor(g_model.args.max_task, self.game.max_words):fill(g_vocab['nil'])
    self.game:copy_goal(self.goal)
    if self.game.copy_repeats then
        self.repeats = torch.Tensor(g_model.args.max_task):fill(1)
        self.game:copy_repeats(self.repeats)
    end

    self.args = torch.Tensor(g_model.args.max_task * g_model.args.max_args):fill(1)
    if g_opts.multitask then
        self.game:copy_args(self.args:view(g_model.args.max_task, g_model.args.max_args))
    end

    --self:update_observation()
end

function Demo:active()
    if self.game:is_active() then
        return 1
    else
        return 0
    end
end

function Demo:sub_task()
    local str = ""
    if g_model.sub_agent then
        local act = {g_model.actions[1]:clone():squeeze(), g_model.actions[2]:clone():squeeze()}
        str = string.format("%s %s", g_task_name[act[1]], g_objects[act[2]].name)
    elseif g_opts.multitask then
        if g_model.args.max_args == 2 then
            str = string.format("%s, %s", 
                    g_task_name[self.game.task[1].param.args[1]], 
                    g_objects[self.game.task[1].param.args[2]].name)
        elseif g_model.args.max_args == 3 then
            str = string.format("%s, %s, %d", 
                    g_task_name[self.game.task[1].param.args[1]], 
                    g_objects[self.game.task[1].param.args[2]].name,
                    self.game.task[1].param.args[3])
        end
    end
    return str
end

function Demo:update_decision()
    if g_model.m.t2 then
        return g_model.m.t2.data.module.output:squeeze()
    end
end

function Demo:update_value()
    if g_model.update then
        return g_model.update
    end
end

function Demo:memory_pointer()
    if g_model.m.g_prob then
        prob = g_model.m.g_prob.data.module.output:clone():squeeze()
        return prob
    end
end

function Demo:shift()
    if g_model.m.g_prob_s then
        shift = g_model.m.g_prob_s.data.module.output:clone():squeeze()
        return shift
    end
end

function Demo:get_repeats()
    return self.repeats
end

function Demo:register()
    if g_model.m.o then
        local register = g_model.m.o.data.module.output:clone():squeeze()
        return register
    end
end

function Demo:h_norm1()
    if g_model.m.next_h then
        return g_model.m.next_h.data.module.output:clone():
                narrow(2, 1, g_model.args.ldim / 2):norm()
    end
end

function Demo:h_norm2()
    if g_model.m.next_h then
        local sdim = g_model.args.ldim / 2
        return g_model.m.next_h.data.module.output:clone():
                narrow(2, sdim + 1, sdim):norm()
    end
end

function Demo:sub_giveup()
    if g_model.giveup then
        return g_model.giveup:clone():squeeze()
    end
end

function Demo:register_conv()
    if g_model.m.co_conv then
        return g_model.m.co_conv.data.module.output:clone():squeeze()
    end
end

function Demo:register_prod()
    if g_model.m.co_prod then
        return g_model.m.co_prod.data.module.output:clone():squeeze()
    end
end

function Demo:sub_term()
    if g_model.term then
        return g_model.term:clone():squeeze()
    end
end

function Demo:sub_obs(idx)
    if g_model.sub_x then
        return g_model.sub_x:clone():squeeze():narrow(1, 3*(idx-1)+1, 3)
    end
end

function Demo:term()
    if g_model.m.term then
        return torch.exp(g_model.m.term.data.module.output:clone()):squeeze()[2]
    end
end

function Demo:max_step()
    return g_opts.max_step
end

function Demo:task_idx()
    return self.game.task_idx
end

function Demo:map_image()
    local img = torch.ones(3, g_opts.max_size * 32, g_opts.max_size * 32)
    return self.game.map:to_image(img)
end

function Demo:giveup()
    if g_model.feasible and g_model:feasible() then 
        return g_model:feasible():clone():add(-1):squeeze()
    end 
end

function Demo:giveup_prob()
    if g_model.feasible and g_model:feasible() then 
        return torch.exp(g_model.m.feas.data.module.output:clone()):squeeze()[2]
    end 
end

function Demo:observation()
    local obs = torch.Tensor(3, g_opts.display_size, g_opts.display_size):fill(0)
    if g_minecraft then
        g_minecraft:get_observation(obs)
    else
        self.game:to_image(obs)
    end
    return obs
end

function Demo:close()
    if g_minecraft then
        g_minecraft.client:close()
    end
end

function Demo:update()
    --self:update_observation()
    if not self.input then
        local size = self.obs[self.t]:size()
        assert(size[1] == g_model.args.channels)
        size[1] = g_model.args.buf * g_model.args.channels
        self.input = torch.Tensor(size)
    end
    self.input:zero()
    for k=1,g_model.args.buf do
        local idx = self.t - k + 1
        if idx < 1 then
            break
        end
        self.input:narrow(1, (k-1)*g_model.args.channels+1, 
            g_model.args.channels):copy(self.obs[idx])
    end
    local prev_action = {} 
    if g_model.args.pass_act then
        for i=1,#g_model.args.num_args do
            prev_action[i] = torch.zeros(1, g_model.args.num_args[i])
            if self.t > 1 then
                prev_action[i]:scatter(2, g_model.actions[i]:long(), 1)
            end
        end
    end
    local temp_goal = torch.Tensor(g_model.args.max_task, self.game.max_words):fill(g_vocab['nil'])
    self.game:copy_goal(temp_goal)
    for i=1,self.game.max_task do
        local txt = goal_text(temp_goal[i])
        if txt == '' then
            break
        end
        print(txt)
    end

    local action, baseline, states
    --[[
    if self.repeats then
        local x = {obs = self.input:view(1, g_model.args.buf * g_model.args.channels, g_opts.max_size, g_opts.max_size),
            instructions = self.goal:view(1, g_model.args.max_task * self.game.max_words),
            prev_args = prev_action,
            }
        x.repeats = self.repeats:view(1, g_model.args.max_task)
        action, baseline, states = g_model:eval(x)
    else
        --]]
    if g_opts.multitask then
        action, baseline, states = g_model:eval(self.input:view(1, 
                g_model.args.buf * g_model.args.channels, g_opts.img_size, g_opts.img_size),
            self.args:view(1, g_model.args.max_task * self.game.max_args),
            prev_action)
    else
        action, baseline, states = g_model:eval(self.input:view(1, 
                g_model.args.buf * g_model.args.channels, g_opts.img_size, g_opts.img_size),
            temp_goal:view(1, g_model.args.max_task * self.game.max_words),
            prev_action)
    end

    if g_opts.feasible then
        local feas = g_model:feasible():clone():add(-1)
        if self.game:is_active() and feas:squeeze() == 1 then
            self.game:giveup()
        end
    end

    self.game:act(action:squeeze())
    self.game:update()
    local reward = self.game:get_reward()

    self.t = self.t + 1

    g_model:set_init_state(states)
    return {act = action:squeeze(), reward = reward}
end
