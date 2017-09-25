require('nn')
require('nngraph')
paths.dofile('LinearNB.lua')
paths.dofile('PowTable.lua')
paths.dofile('ScalarMulTable.lua')
paths.dofile('CircularConvolution.lua')
paths.dofile('Multinomial.lua')
paths.dofile('OneHot.lua')
paths.dofile('Entropy.lua')
paths.dofile('EntropyPenalty.lua')
paths.dofile('DrLimCriterion.lua')
paths.dofile('NotEqual.lua')
paths.dofile('Register.lua')
paths.dofile('ScalarAddTable.lua')
-- paths.dofile('FactoredSpatialConvolution.lua')

local Net = torch.class('Net')
function Net:__init(args)
    self.args = args
    self.m = {}
    self.entropy_list = {}
    self.share_list = {}
    self.net = self:build_model(args) 
    self.init_states = self:build_init_states(args)
    self:build_sub_agent(args)
    self.recurrent = #self.init_states > 0

    if g_opts.gpu > 0 then
       self.net:cuda()
       -- print(self.m.conv1.data.module.weight)
       -- cudnn.convert(self.net, cudnn)
    end
    -- IMPORTANT! do weight sharing after model is in cuda
    -- print("name", self.args.name, "#sub agents", #self.sub_agent)
    for k, v in pairs(self.share_list) do
        local m1 = v[1].data.module
        if #v > 2 and g_opts.verbose > 3 then
            print(string.format("[%s] %d modules are shared", k, #v))
        end
        for j = 2,#v do
            local m2 = v[j].data.module
            m2:share(m1,'weight','bias','gradWeight','gradBias')
        end
    end
    self:reset_init_state(1)
    assert(#self.n_actions > 0, "num of actions are not defined")
end

function Net:share_module(name, node) 
    if self.share_list[name] == nil then
        self.share_list[name] = {node}
    else
        table.insert(self.share_list[name], node)
    end
end

function Net:build_model(args)
    error("Should implement this")
end

function Net:build_init_states(args)
    return {}
end

function Net:init_grad_states()
    local grad_states = {}
    for j=1,#self.init_states do
        local grad = torch.Tensor(self.init_states[j]:size()):zero()
        if g_opts.gpu > 0 then
            grad = grad:cuda()
        end
        table.insert(grad_states, grad)
    end
    return grad_states
end

function Net:reset_init_state(batch_size)
    for j=1,#self.init_states do
        self.init_states[j]:fill(0)
    end
    if self.sub_agent then
        self.sub_agent:reset_init_state(batch_size)
    end
end

function Net:set_init_state(states)
    assert(#states == #self.init_states)
    for j=1,#self.init_states do
        self.init_states[j] = states[j]
    end
end

function Net:build_sub_agent(args)
    if args.sub_agent ~= "" then
        self.sub_agent = torch.load(args.sub_agent)
    end
end

function Net:eval(x, g, arg)
    local primitive_actions
    local actions, baseline, states, term = self:forward(x, g, arg)
    self.actions = {}
    self.update = 1
    if self.sub_agent then
        local batch_size = actions[1]:size(1)
        local args = torch.Tensor(batch_size, #self.args.num_args):zero()
        for i=1,#self.args.num_args do
            local prob = torch.exp(actions[i]:float())
            self.actions[i] = torch.multinomial(prob, 1)
            args:narrow(2, i, 1):copy(self.actions[i])
        end
        local buf = self.sub_agent.args.buf or 1
        local sub_x = x:narrow(2, 1, buf * self.args.channels):contiguous()
        primitive_actions = self.sub_agent:eval(sub_x, args)
    else
        local prob = torch.exp(actions[1]:float())
        self.actions[1] = torch.multinomial(prob, 1)
        primitive_actions = self.actions[1]
    end
    self:fill_internal_actions()
    return primitive_actions, baseline, states, term 
end

function Net:fill_internal_actions()
end

function Net:parse_forward(out)
    local actions = {}
    local states = {}
    for i=1,#self.init_states do
        table.insert(states, out[#out-#self.init_states+i])
    end
    local offset = 1
    if self.args.term_act then
        offset = offset + 1
    end
    for i=1,#out-#self.init_states-offset do
        table.insert(actions, out[i])
    end
    if self.args.term_act then
        assert(out[#actions+2])
        return actions, out[#actions+2], states, out[#actions+1]
    else
        return actions, out[#actions+1], states
    end
end

function Net:forward(x, g, arg, term)
    if g_opts.gpu > 0 then
        self.input = {x:cuda(), g:cuda()}
    else
        self.input = {x, g}
    end
    for i=1,#self.init_states do
        table.insert(self.input, self.init_states[i])
    end
    if self.args.pass_act then
        for i=1,#arg do
            if g_opts.gpu > 0 then
                table.insert(self.input, arg[i]:cuda())
            else
                table.insert(self.input, arg[i])
            end
        end
    end
    if self.args.pass_term then
        if g_opts.gpu > 0 then
            table.insert(self.input, term:cuda())
        else
            table.insert(self.input, term)
        end
    end 
    return self:parse_forward(self.net:forward(self.input))
end

function Net:backward(x, g, grad_action, grad_baseline, grad_state, rho, grad_term)
    if self.update == 0 then
        assert(self.args.open_loop)
        return grad_state
    end

    self:fill_grad(grad_action, grad_baseline, grad_state, grad_term)
    if rho > 0 then
        self:add_entropy_grad(rho)
    end
    self:add_l1_regularization()
    local grad_input = self.net:backward(self.input, self.grad_output)
    table.remove(grad_input, 1) -- x
    table.remove(grad_input, 1) -- g
    if self.args.pass_term then
        table.remove(grad_input, #grad_input)
    end
    if self.args.pass_act then
        for i = 1, #self.args.num_args do
            table.remove(grad_input, #grad_input)
        end
    end
    return grad_input
end

function Net:add_l1_regularization()
end

function Net:add_entropy_grad(rho)
    self.entropy = self.entropy or nn.Entropy()
    local g = self.entropy:backward(self.m.a.data.module.output, 1)
    self.grad_output[1]:add(g:mul(-rho))
    if self.entropy_list then 
        for i = 1,#self.entropy_list do
            self.entropy_list[i].w = -rho
        end
    end
end

function Net:fill_grad(grad_action, grad_baseline, grad_state, grad_term)
    local batch_size = grad_baseline:size(1)
    self.grad_output = self.grad_output or {}
    assert(#self.n_actions == #self.actions)
    for i=1,#self.actions do
        self.grad_output[i] = self.grad_output[i] or torch.Tensor()
        self.grad_output[i]:resize(batch_size, self.n_actions[i]):zero()

        if g_opts.gpu > 0 then
            self.grad_output[i] = self.grad_output[i]:cuda()
        end
        for j=1,#grad_action do 
            if grad_action[j]:nElement() == batch_size * self.n_actions[i] then
                self.grad_output[i]:add(grad_action[j])
            else
                self.g_tmp = self.g_tmp or {}
                self.g_tmp[i] = self.g_tmp[i] or torch.Tensor(batch_size, self.n_actions[i])
                if g_opts.gpu > 0 then
                    self.g_tmp[i] = self.g_tmp[i]:cuda()
                    self.actions[i] = self.actions[i]:cuda()
                end
                self.g_tmp[i]:zero()
                self.g_tmp[i]:scatter(2, self.actions[i], grad_action[j])
                self.g_tmp[i]:div(batch_size)
                self.grad_output[i]:add(self.g_tmp[i])
            end
            -- print(j, self.grad_output[i])
        end
        -- table.insert(self.grad_output, g)
        -- print(i, g, self.actions[i])
    end
    local offset = 1
    if self.args.term_act then
        assert(grad_term)
        self.grad_output[#self.actions+offset] = grad_term
        offset = offset + 1
    end
    self.grad_output[#self.actions+offset] = grad_baseline
    assert(#grad_state == #self.init_states)
    for i=1,#self.init_states do
        self.grad_output[#self.actions+offset+i] = grad_state[i]
    end
    self:clean_invalid_grad()
end

function Net:clean_invalid_grad()

end

function Net:getParameters()
    return self.net:getParameters() 
end

function Net:clone(share)
    local new_args = {}
    for k,v in pairs(self.args) do
        new_args[k] = v
    end
    if new_args.sub_agent ~= "" then
        new_args.sub_agent = ""
    end
    local clone = self.new(new_args)
    if share then 
        clone:share_weight_from(self.m)
    else
        clone:copy_weight_from(self.m)
    end
    -- TODO: make this more memory-efficient 
    -- sharing sub_agents causes NaN in spatial convolution for some reason.
    clone.sub_agent = self.sub_agent
    collectgarbage()
    return clone
end

--[[
function Net:cuda()
    self.net:cuda()
    for i=1,#self.sub_agent do
        self.sub_agent[i]:cuda()
    end
    for i=1,#self.init_states do
        self.init_states[i] = self.init_states[i]:cuda()
    end
    return self
end
--]]

function Net:float()
    self.net:float()
    for i=1,#self.sub_agent do
        self.sub_agent[i]:float()
    end
    for i=1,#self.init_states do
        self.init_states[i] = self.init_states[i]:float()
    end
    return self
end

function Net:training()
    self.net:training()
    for i=1,#self.sub_agent do
        self.sub_agent[i]:training()
    end
    return self
end

function Net:evaluate()
    self.net:evaluate()
    for i=1,#self.sub_agent do
        self.sub_agent[i]:evaluate()
    end
    return self
end

function Net:share_weight_from(m)
    for name, node in pairs(m) do
        if self.m[name] then
            local dst_module = self.m[name].data.module
            local src_module = node.data.module
            dst_module:share(src_module, 'weight','bias','gradWeight','gradBias')
        end
    end
end

function Net:copy_weight_from(m, log)
    for name, node in pairs(m) do
        if self.m[name] then
            local src_module = node.data.module
            local dst_module = self.m[name].data.module
            if src_module.weight then
                assert(dst_module.weight, name)
                if name == "g_val_h" and g_opts.counting then
                    if src_module.weight:size(1) == dst_module.weight:size(1) then
                        dst_module.weight:copy(src_module.weight)
                    else
                        assert(src_module.weight:size(1) + 5 == dst_module.weight:size(1))
                        dst_module.weight:narrow(1, 1, src_module.weight:size(1)):
                            copy(src_module.weight)
                    end
                else
                    assert(src_module.weight:nElement() == dst_module.weight:nElement(), 
                        name .. "source: " .. src_module.weight:nElement() ..
                        "dest: " .. dst_module.weight:nElement())
                    dst_module.weight:copy(src_module.weight)
                    if log then
                        print(name, "copied")
                    end
                end
            end
            if src_module.bias then
                assert(dst_module.bias)
                assert(src_module.bias:nElement() == dst_module.bias:nElement())
                dst_module.bias:copy(src_module.bias)
            end
        end
    end
end

require "queue"
local Opt = torch.class('Opt', 'Net')
function Opt:bfs(maze, obj, g)
    local agent_y = maze.agent.loc.y
    local agent_x = maze.agent.loc.x
    local q = Queue.new()
    local dist = torch.Tensor(maze.map.height, maze.map.width):fill(-1)
    local t = torch.Tensor(maze.map.height, maze.map.width):fill(0)
    q:push({agent_y, agent_x})
    dist[agent_y][agent_x] = 0
    local depth = 0
    local found = false
    local x, y 
    while not q:empty() do
        local v = q:pop()
        y = v[1]
        x = v[2]

        for i,j in pairs(maze.map.items[y][x]) do
          if j.type == 'object' and j.name == obj then
              found = true
              break
          end
        end

        if found then
            break
        end

        if maze.map:is_loc_reachable(y-1, x) and dist[y-1][x] == -1 then
          dist[y-1][x] = dist[y][x] + 1
          t[y-1][x] = 1
          q:push({y-1, x})
        end
        if maze.map:is_loc_reachable(y+1, x) and dist[y+1][x] == -1 then
          dist[y+1][x] = dist[y][x] + 1
          t[y+1][x] = 2
          q:push({y+1, x})
        end
        if maze.map:is_loc_reachable(y, x-1) and dist[y][x-1] == -1 then
          dist[y][x-1] = dist[y][x] + 1
          t[y][x-1] = 3
          q:push({y, x-1})
        end
        if maze.map:is_loc_reachable(y, x+1) and dist[y][x+1] == -1 then
          dist[y][x+1] = dist[y][x] + 1
          t[y][x+1] = 4
          q:push({y, x+1})
        end
        depth = math.max(dist[y][x] + 1, depth)
    end

    local action = nil
    if found then
        if dist[y][x] == 0 then
            if maze.map:is_loc_reachable(agent_y+1,agent_x) then
                action = 'down'
            elseif maze.map:is_loc_reachable(agent_y-1,agent_x) then
                action = 'up'
            elseif maze.map:is_loc_reachable(agent_y,agent_x+1) then
                action = 'right'
            elseif maze.map:is_loc_reachable(agent_y,agent_x-1) then
                action = 'left'
            else
                action = 'stop'
            end
        else
            if dist[y][x] == 1 and (g == 2 or g == 3) then
                action = 'hit_'
                if g == 2 then
                    action = 'pick_up_'
                end
                if y == agent_y - 1 then
                    action = action .. 'up'
                elseif y == agent_y + 1 then
                    action = action .. 'down'
                elseif x == agent_x - 1 then
                    action = action .. 'left'
                else
                    assert(x == agent_x + 1)
                    action = action .. 'right'
                end
            else
                while dist[y][x] >= 1 do
                    -- print(dist[y][x], t[y][x])
                    if t[y][x] == 1 then
                        y = y + 1
                        action = 'up'
                    elseif t[y][x] == 2 then
                        y = y - 1
                        action = 'down'
                    elseif t[y][x] == 3 then
                        x = x + 1
                        action = 'left'
                    else
                        assert(t[y][x] == 4)
                        x = x - 1
                        action = 'right'
                    end
                end
            end
        end
    else
        action = 'stop'
    end
    if action == nil then
        print(y, x, agent_y, agent_x, dist[y][x])
    end
    return maze.agent.action_ids[action], depth, found
end

function Opt:build_model(args)
    self.n_actions = {args.n_actions}
    return nil
end

function Opt:eval(batch)
    local act = torch.zeros(#batch, 1)
    local dist, found
    for i=1,#batch do
        if batch[i]:is_active() then
            local g = torch.zeros(g_opts.max_args)
            batch[i].task[batch[i].task_idx]:arguments(g)
            if batch[i].enemy_spawn and batch[i].enemy_spawn > 0 then
                local enemies = batch[i]:init_enemies()
                assert(#enemies == 1)
                act[i], dist, found = self:bfs(batch[i], enemies[1].name, 3)
                if not found or dist > 5 then
                    act[i] = self:bfs(batch[i], g_objects[g[2]].name, g[1])
                end
            else
                act[i] = self:bfs(batch[i], g_objects[g[2]].name, g[1])
            end
        end
    end
    return act
end

local Single = torch.class('Single', 'Net')
function Single:forward(x, g)
    if self.recurrent then
        self.input = {x}
        for i=1, #self.init_states do
            table.insert(self.input, self.init_states[i])
        end
    else
        self.input = x
    end
    return self:parse_forward(self.net:forward(self.input))
end

function Single:backward(x, g, grad_action, grad_baseline, grad_state, rho, grad_term)
    self:fill_grad(grad_action, grad_baseline, grad_state, grad_term)
    if rho > 0 then
        self:add_entropy_grad(rho)
    end
    local grad_input = self.net:backward(self.input, self.grad_output)
    if self.recurrent then
        table.remove(grad_input, 1)
        return grad_input
    else
        return {}
    end
end

function Single:add_entropy_grad(rho)
    self.entropy = self.entropy or {}
    for i=1,#self.n_actions do
        self.entropy[i] = self.entropy[i] or nn.Entropy()
    end
    local g = self.entropy[1]:backward(self.m.a.data.module.output, 1)
    self.grad_output[1]:add(g:mul(-rho))
end

local CNN = torch.class('CNN', 'Single')
function CNN:build_model(args)
    self.m.x = nn.Identity()()
    self.m.conv1 = args.convLayer(args.buf * args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    local last_feature = self.m.conv1_nl
    for i=1,(#args.n_units-1) do
        if args.pool[i] then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            last_feature = self.m["pool" .. i]
        end
        self.m["conv" .. i+1] = args.convLayer(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(last_feature)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        last_feature = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(g_opts.feat_size * g_opts.feat_size *  args.n_units[#args.n_units], 
            args.edim)(self.m.conv)
    self.m.fc1_nl = nn.ReLU(true)(self.m.fc1)
    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.fc1_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.fc1_nl)
    
    local input = {self.m.x}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.fc1_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

local LSTM = torch.class('LSTM', 'Single')
function LSTM:build_init_states(args)
    local states = {}
    table.insert(states, torch.Tensor(1, args.ldim))      -- c
    table.insert(states, torch.Tensor(1, args.ldim))      -- h
    return states
end

function LSTM:fill_internal_actions()
    if self.args.feasible then
        local prob = torch.exp(self.m.feas.data.module.output:float())
        local feas = torch.multinomial(prob, 1)
        table.insert(self.actions, feas)
    end
end

function LSTM:feasible()
    return self.actions[2]
end

function LSTM:build_model(args)
    self.m.x = nn.Identity()()
    self.m.c0 = nn.Identity()()
    self.m.h0 = nn.Identity()()

    local prev_c = self.m.c0
    local prev_h = self.m.h0
    local edim = args.edim
    local ldim = args.ldim

    self.m.conv1 = args.convLayer(args.buf * args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    local last_feature = self.m.conv1_nl
    for i=1,(#args.n_units-1) do
        if args.pool[i] then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            last_feature = self.m["pool" .. i]
        end
        self.m["conv" .. i+1] = args.convLayer(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(last_feature)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        last_feature = self.m["conv" .. i+1 .. "_nl"]
    end

    local nel = g_opts.feat_size * g_opts.feat_size *  args.n_units[#args.n_units]
    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    local join = nn.JoinTable(2)({self.m.conv, prev_h})
    self.m.fc1 = nn.Linear(nel + ldim, 4 * ldim)(join)

    local reshaped = nn.View(4, ldim):setNumInputDims(1)(self.m.fc1)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    local in_gate = nn.Sigmoid(true)(n1)
    local forget_gate = nn.Sigmoid(true)(n2)
    local out_gate = nn.Sigmoid(true)(n3)
    local in_transform = nn.Tanh()(n4)
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    self.m.fc = nn.Linear(ldim, args.n_actions)(next_h)
    self.m.a = nn.LogSoftMax()(self.m.fc)
    self.m.b = nn.Linear(ldim, 1)(next_h)

    local input = {self.m.x, self.m.c0, self.m.h0}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.feasible then
        table.insert(self.n_actions, 2)
        self.m.feas_fc = nn.Linear(ldim, 2)(next_h)
        self.m.feas_fc.data.module.bias[1] = 4
        self.m.feas = nn.LogSoftMax()(self.m.feas_fc)
        table.insert(output, self.m.feas)
    end

    if args.term_act then
        self.m.term_fc = nn.Linear(ldim, 2)(next_h)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end

    table.insert(output, self.m.b)
    table.insert(output, next_c)
    table.insert(output, next_h)
    return nn.gModule(input, output)

end

local MultiCNN = torch.class('MultiCNN', 'Net')
function MultiCNN:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()
    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)
    self.m.fc1_nl = nn.ReLU(true)(self.m.fc1)

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    local arg_h = {}
    for i=1,args.max_args do
        self.m["arg_h" .. i] = nn.LookupTable(args.num_args[i], args.edim)(arguments[i])
        --self.m["arg_h" .. i].data.module:reset(0.08)
        table.insert(arg_h, self.m["arg_h" .. i])
        self:share_module("arg" .. i, self.m["arg_h" .. i])
    end
    self.m.arg_h = nn.JoinTable(2)(arg_h)
    self.m.arg_fc1 = nn.Linear(args.max_args * args.edim, args.edim)(self.m.arg_h)
    self.m.arg_fc1_nl = nn.ReLU(true)(self.m.arg_fc1)
    self:share_module("arg_fc", self.m.arg_fc1)
    
    local join_h = nn.JoinTable(2)({self.m.fc1_nl, self.m.arg_fc1_nl})
    self.m.joint_h = nn.Linear(2 * args.edim, args.edim)(join_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)
    
    local input = {self.m.x, self.m.args}
    local output = {self.m.a, self.m.b}

    if g_opts.regularizer > 0 then
        self.m.args4 = nn.Identity()()
        local arg = {nn.SplitTable(1, 2)(self.m.args4):split(4)}
        local o = {}
        for i=1,4 do
            local arguments = {nn.SplitTable(1, 1)(arg[i]):split(args.max_args)}
            local arg_h = {}
            for j=1,args.max_args do
                self.m["embed" .. j] = nn.LookupTable(args.num_args[j], args.edim)(arguments[j])
                self:share_module("arg" .. j, self.m["embed" .. j])
                table.insert(arg_h, self.m["embed" .. j])
            end
            local join = nn.JoinTable(2)(arg_h)
            self.m["arg_fc_" .. i] = nn.Linear(args.max_args * args.edim, args.edim)(join)
            self.m["arg_fc_nl_" ..i] = nn.ReLU(true)(self.m["arg_fc_" .. i])
            self:share_module("arg_fc", self.m["arg_fc_" ..i])
            table.insert(o, self.m["arg_fc_nl_" .. i])
        end
        local delta1 = nn.CSubTable()({o[1], o[2]})
        local delta2 = nn.CSubTable()({o[3], o[4]})
        local err = nn.CSubTable()({delta1, delta2})

        table.insert(input, self.m.args4)
        table.insert(output, err)
    end
--[[
    self.m.joint_h2 = nn.Linear(args.edim, args.edim)(self.m.joint_h_nl)
    self.m.joint_h2_nl = nn.ReLU(true)(self.m.joint_h2)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h2_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h2_nl)
--]]

    self.n_actions = {args.n_actions}
    return nn.gModule(input, output)
end

local MultiCNN2 = torch.class('MultiCNN2', 'Net')
function MultiCNN2:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()
    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)
    self.m.fc1_nl = nn.ReLU(true)(self.m.fc1)

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    local arg_h = {}
    for i=1,args.max_args do
        self.m["arg_h" .. i] = nn.LookupTable(args.num_args[i], args.edim)(arguments[i])
        --self.m["arg_h" .. i].data.module:reset(0.08)
        table.insert(arg_h, self.m["arg_h" .. i])
    end
    self.m.arg_h = nn.CMulTable()(arg_h)
    self.m.arg_fc1 = nn.LinearNB(args.edim, args.edim)(self.m.arg_h)
    self.m.arg_fc1_nl = nn.ReLU(true)(self.m.arg_fc1)
    
    local join_h = nn.JoinTable(2)({self.m.fc1_nl, self.m.arg_fc1_nl})
    self.m.joint_h = nn.Linear(2 * args.edim, args.edim)(join_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    self.n_actions = {args.n_actions}
    return nn.gModule({self.m.x, self.m.args}, {self.m.a, self.m.b})
end

local MultiCNN3 = torch.class('MultiCNN3', 'Net')
function MultiCNN3:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()
    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    local arg_h = {}
    for i=1,args.max_args do
        self.m["arg_h" .. i] = nn.LookupTable(args.num_args[i], args.edim)(arguments[i])
        --self.m["arg_h" .. i].data.module:reset(0.08)
        table.insert(arg_h, self.m["arg_h" .. i])
    end
    self.m.arg_h = nn.CMulTable()(arg_h)
    self.m.arg_fc1 = nn.LinearNB(args.edim, args.edim)(self.m.arg_h)
    
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc1, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    self.n_actions = {args.n_actions}
    return nn.gModule({self.m.x, self.m.args}, {self.m.a, self.m.b})
end

local MultiCNN4 = torch.class('MultiCNN4', 'Net')
function MultiCNN4:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()
    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    local arg_h = {}
    for i=1,args.max_args do
        self.m["arg_h" .. i] = nn.LookupTable(args.num_args[i], args.edim)(arguments[i])
        table.insert(arg_h, self.m["arg_h" .. i])
        self:share_module("arg" .. i, self.m["arg_h" .. i])
    end
    self.m.arg_h = nn.CMulTable()(arg_h)
    self.m.arg_fc1 = nn.LinearNB(args.edim, args.edim)(self.m.arg_h)
    self:share_module("arg_fc", self.m.arg_fc1)
    
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc1, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)
    self.m.joint_h_nl_drop = nn.Dropout(0.5)(self.m.joint_h_nl)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl_drop)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl_drop)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a, self.m.b}

    if g_opts.regularizer > 0 then
        self.m.args4 = nn.Identity()()
        local arg = {nn.SplitTable(1, 2)(self.m.args4):split(4)}
        local o = {}
        for i=1,4 do
            local arguments = {nn.SplitTable(1, 1)(arg[i]):split(args.max_args)}
            local arg_h = {}
            for j=1,args.max_args do
                self.m["embed" .. j] = nn.LookupTable(args.num_args[j], args.edim)(arguments[j])
                self:share_module("arg" .. j, self.m["embed" .. j])
                table.insert(arg_h, self.m["embed" .. j])
            end
            local mult = nn.CMulTable()(arg_h)
            self.m["arg_fc_" ..i] = nn.LinearNB(args.edim, args.edim)(mult)
            self:share_module("arg_fc", self.m["arg_fc_" ..i])
            table.insert(o, self.m["arg_fc_" .. i])
        end
        local delta1 = nn.CSubTable()({o[1], o[2]})
        local delta2 = nn.CSubTable()({o[3], o[4]})
        local err = nn.CSubTable()({delta1, delta2})

        table.insert(input, self.m.args4)
        table.insert(output, err)
    end

    self.n_actions = {args.n_actions}
    return nn.gModule(input, output)
end

local MultiCNN5 = torch.class('MultiCNN5', 'Net')
function MultiCNN5:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()
    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    local arg_h = {}
    for i=1,args.max_args do
        self.m["arg_h" .. i] = nn.LookupTable(args.num_args[i], args.edim)(arguments[i])
        --self.m["arg_h" .. i].data.module:reset(0.08)
        table.insert(arg_h, self.m["arg_h" .. i])
    end
    self.m.arg_h = nn.JoinTable(2)(arg_h)
    self.m.arg_fc1 = nn.Linear(args.max_args * args.edim, args.edim)(self.m.arg_h)
    self.m.arg_fc1_nl = nn.ReLU(true)(self.m.arg_fc1)
    
    local join_h = nn.JoinTable(2)({self.m.conv, self.m.arg_fc1_nl})
    self.m.joint_h = nn.Linear(5 * 5 * args.n_units[#args.n_units] + args.edim, args.edim)(join_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

--[[
    self.m.joint_h2 = nn.Linear(args.edim, args.edim)(self.m.joint_h_nl)
    self.m.joint_h2_nl = nn.ReLU(true)(self.m.joint_h2)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h2_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h2_nl)
--]]

    self.n_actions = {args.n_actions}
    return nn.gModule({self.m.x, self.m.args}, {self.m.a, self.m.b})
end

local MultiCNN6 = torch.class('MultiCNN6', 'Net')
function MultiCNN6:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()
    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    local arg_h = {}
    for i=1,args.max_args do
        self.m["arg_h" .. i] = nn.LookupTable(args.num_args[i], args.edim)(arguments[i])
        --self.m["arg_h" .. i].data.module:reset(0.08)
        table.insert(arg_h, self.m["arg_h" .. i])
    end
    self.m.arg_h = nn.JoinTable(2)(arg_h)
    local join_h = nn.JoinTable(2)({self.m.conv, self.m.arg_h})
    self.m.joint_h = nn.Linear(5 * 5 * args.n_units[#args.n_units] + 2*args.edim, args.edim)(join_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)
    self.n_actions = {args.n_actions}
    return nn.gModule({self.m.x, self.m.args}, {self.m.a, self.m.b})
end

local MultiCNN7 = torch.class('MultiCNN7', 'Net')
function MultiCNN7:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()
    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)
    self.m.fc1_nl = nn.ReLU(true)(self.m.fc1)

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    local arg_h = {}
    for i=1,args.max_args do
        self.m["arg_h" .. i] = nn.LookupTable(args.num_args[i], args.edim)(arguments[i])
        --self.m["arg_h" .. i].data.module:reset(0.08)
        table.insert(arg_h, self.m["arg_h" .. i])
        self:share_module("arg" .. i, self.m["arg_h" .. i])
    end
    self.m.arg_h = nn.CMulTable()(arg_h)
    self.m.arg_fc1 = nn.LinearNB(args.edim, args.edim * args.n_actions)(self.m.arg_h)
    self.m.arg_fc2 = nn.LinearNB(args.edim, args.edim)(self.m.arg_h)
    self:share_module("arg_fc1", self.m.arg_fc1)
    local w = nn.View(-1, args.edim, args.n_actions)(self.m.arg_fc1)
    local w2 = nn.View(-1, args.edim, 1)(self.m.arg_fc2)
    local conv_view = nn.View(1, -1):setNumInputDims(1)(self.m.fc1_nl)

    local MMbout = nn.MM(false, false)
    local MMbout2 = nn.MM(false, false)
    self.m.fc2 = nn.Squeeze()(MMbout({conv_view, w}))
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Squeeze()(MMbout2({conv_view, w2}))
    
    local input = {self.m.x, self.m.args}
    local output = {self.m.a, self.m.b}

    if g_opts.regularizer > 0 then
        self.m.args4 = nn.Identity()()
        local arg = {nn.SplitTable(1, 2)(self.m.args4):split(4)}
        local o = {}
        for i=1,4 do
            local arguments = {nn.SplitTable(1, 1)(arg[i]):split(args.max_args)}
            local arg_h = {}
            for j=1,args.max_args do
                self.m["embed" .. j] = nn.LookupTable(args.num_args[j], args.edim)(arguments[j])
                self:share_module("arg" .. j, self.m["embed" .. j])
                table.insert(arg_h, self.m["embed" .. j])
            end
            local mult = nn.CMulTable()(arg_h)
            self.m["arg_fc1_" .. i] = nn.LinearNB(args.edim, args.edim * args.n_actions)(mult)
            self:share_module("arg_fc1", self.m["arg_fc1_" ..i])
            table.insert(o, self.m["arg_fc1_" .. i])
        end
        local delta1 = nn.CSubTable()({o[1], o[2]})
        local delta2 = nn.CSubTable()({o[3], o[4]})
        local err = nn.CSubTable()({delta1, delta2})

        table.insert(input, self.m.args4)
        table.insert(output, err)
    end

    self.n_actions = {args.n_actions}
    return nn.gModule(input, output)
end

local MultiCNN8 = torch.class('MultiCNN8', 'Net')
function MultiCNN8:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.n_units[1])(arguments[2])
    self.m.conv_att = nn.Sigmoid(true)(self.m.arg_h2)
    local att_map = nn.Replicate(10*10, 2, 1)(self.m.conv_att)
    local att_cont = nn.Contiguous()(att_map)
    local att_3d = nn.View(args.n_units[1], 10, 10):setNumInputDims(2)(att_cont)

    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.buf * args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x)
    self.m.conv1_nl_pre = nn.ReLU(true)(self.m.conv1)
    self.m.conv1_nl = nn.CMulTable()({self.m.conv1_nl_pre, att_3d})
    
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.mul_h = nn.CMulTable()({self.m.arg_h1, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

local MultiCNN8NB = torch.class('MultiCNN8NB', 'Net')
function MultiCNN8NB:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.n_units[1])(arguments[2])
    self.m.conv_att = nn.Sigmoid(true)(self.m.arg_h2)
    local att_map = nn.Replicate(10*10, 2, 1)(self.m.conv_att)
    local att_cont = nn.Contiguous()(att_map)
    local att_3d = nn.View(args.n_units[1], 10, 10):setNumInputDims(2)(att_cont)

    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.buf * args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x)
    self.m.conv1_nl_pre = nn.ReLU(true)(self.m.conv1)
    self.m.conv1_nl = nn.CMulTable()({self.m.conv1_nl_pre, att_3d})
    
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.mul_h = nn.CMulTable()({self.m.arg_h1, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.LinearNB(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

local MultiCNN9 = torch.class('MultiCNN9', 'Net')
function MultiCNN9:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h11 = nn.LookupTable(args.num_args[1], args.n_units[1])(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.n_units[1])(arguments[2])
    self.m.arg_hh = nn.CMulTable()({self.m.arg_h11, self.m.arg_h2})
    self.m.conv_att = nn.Sigmoid(true)(self.m.arg_hh)
    local att_map = nn.Replicate(10*10, 2, 1)(self.m.conv_att)
    local att_cont = nn.Contiguous()(att_map)
    local att_3d = nn.View(args.n_units[1], 10, 10):setNumInputDims(2)(att_cont)

    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.buf * args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x)
    self.m.conv1_nl_pre = nn.ReLU(true)(self.m.conv1)
    self.m.conv1_nl = nn.CMulTable()({self.m.conv1_nl_pre, att_3d})
    
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.mul_h = nn.CMulTable()({self.m.arg_h1, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

local MultiCNN10 = torch.class('MultiCNN10', 'Net')
function MultiCNN10:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h11 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    self.m.arg_hh = nn.CAddTable()({self.m.arg_h11, self.m.arg_h2})
    self.m.arg_hh_nl = nn.ReLU(true)(self.m.arg_hh)
    self.m.arg_hh2 = nn.Linear(args.edim, args.n_units[1])(self.m.arg_hh_nl)
    self.m.conv_att = nn.Sigmoid(true)(self.m.arg_hh2)
    local att_map = nn.Replicate(10*10, 2, 1)(self.m.conv_att)
    local att_cont = nn.Contiguous()(att_map)
    local att_3d = nn.View(args.n_units[1], 10, 10):setNumInputDims(2)(att_cont)

    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.buf * args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x)
    self.m.conv1_nl_pre = nn.ReLU(true)(self.m.conv1)
    self.m.conv1_nl = nn.CMulTable()({self.m.conv1_nl_pre, att_3d})
    
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.mul_h = nn.CMulTable()({self.m.arg_h1, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

local MultiCNN11 = torch.class('MultiCNN11', 'Net')
function MultiCNN11:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.buf * args.channels)(arguments[2])
    self.m.conv_att = nn.Sigmoid(true)(self.m.arg_h2)
    local att_map = nn.Replicate(10*10, 2, 1)(self.m.conv_att)
    local att_cont = nn.Contiguous()(att_map)
    local att_3d = nn.View(args.buf * args.channels, 10, 10):setNumInputDims(2)(att_cont)

    self.m.x_masked = nn.CMulTable()({self.m.x, att_3d})
    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.buf * args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_masked)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.mul_h = nn.CMulTable()({self.m.arg_h1, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

local MultiCNN100 = torch.class('MultiCNN100', 'Net')
function MultiCNN100:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local c = 3
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], c * args.n_units[1])(arguments[2])
    self.m.arg_h2_nl = nn.Sigmoid(true)(self.m.arg_h2)
    self.m.arg_h2d = nn.View(c, args.n_units[1]):setNumInputDims(1)(self.m.arg_h2_nl)

    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.buf * args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x)
    self.m.conv1_nl_pre = nn.ReLU(true)(self.m.conv1)
    local conv1_2d = nn.View(args.n_units[1], -1):setNumInputDims(3)(self.m.conv1_nl_pre)
    local MMconv = nn.MM(false, false)
    self.m.conv1_nl = nn.View(c, 10, 10):setNumInputDims(2)(MMconv({self.m.arg_h2d, conv1_2d}))

    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        local nInput = args.n_units[i]
        if i == 1 then
            nInput = c
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.mul_h = nn.CMulTable()({self.m.arg_h1, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

local MultiCNN101 = torch.class('MultiCNN101', 'Net')
function MultiCNN101:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local c = 3
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], c * args.n_units[1])(arguments[2])
    self.m.arg_h2_nl = nn.SoftMax()(nn.View(-1, args.n_units[1])(self.m.arg_h2))
    self.m.arg_h2d = nn.View(-1, c, args.n_units[1])(self.m.arg_h2_nl)

    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.buf * args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x)
    self.m.conv1_nl_pre = nn.ReLU(true)(self.m.conv1)
    local conv1_2d = nn.View(args.n_units[1], -1):setNumInputDims(3)(self.m.conv1_nl_pre)
    local MMconv = nn.MM(false, false)
    self.m.conv1_nl = nn.View(c, 10, 10):setNumInputDims(2)(MMconv({self.m.arg_h2d, conv1_2d}))

    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        local nInput = args.n_units[i]
        if i == 1 then
            nInput = c
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.mul_h = nn.CMulTable()({self.m.arg_h1, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

local MultiCNN102 = torch.class('MultiCNN102', 'Net')
function MultiCNN102:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local c = 3
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], c * args.buf * args.channels)(arguments[2])
    self.m.arg_h2_nl = nn.Sigmoid(true)(self.m.arg_h2)
    self.m.arg_h2d = nn.View(c, args.buf * args.channels):setNumInputDims(1)(self.m.arg_h2_nl)
    local conv1_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(c, 10, 10):setNumInputDims(2)(MMconv({self.m.arg_h2d, conv1_2d}))

    -- convolution
    self.m.conv1 = nn.SpatialConvolution(c, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_masked)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)

    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        local nInput = args.n_units[i]
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.mul_h = nn.CMulTable()({self.m.arg_h1, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

local MultiCNN103 = torch.class('MultiCNN103', 'Net')
function MultiCNN103:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local c = 3
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], c * args.buf * args.channels)(arguments[2])
    self.m.arg_h2_nl = nn.SoftMax()(nn.View(-1, args.buf * args.channels)(self.m.arg_h2))
    self.m.arg_h2d = nn.View(-1, c, args.buf * args.channels)(self.m.arg_h2_nl)
    local conv1_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(c, 10, 10):setNumInputDims(2)(MMconv({self.m.arg_h2d, conv1_2d}))

    -- convolution
    self.m.conv1 = nn.SpatialConvolution(c, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_masked)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)

    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        local nInput = args.n_units[i]
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.mul_h = nn.CMulTable()({self.m.arg_h1, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

local MultiCNN104 = torch.class('MultiCNN104', 'Net')
function MultiCNN104:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local c = 3
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], c * args.buf * args.channels)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], c * args.buf * args.channels)(arguments[2])
    self.m.arg_hh = nn.CMulTable()({self.m.arg_h1, self.m.arg_h2})
    self.m.arg_hh_nl = nn.Sigmoid(true)(self.m.arg_hh)
    self.m.arg_h2d = nn.View(c, args.buf * args.channels):setNumInputDims(1)(self.m.arg_hh_nl)
    local conv1_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(c, 10, 10):setNumInputDims(2)(MMconv({self.m.arg_h2d, conv1_2d}))

    -- convolution
    self.m.conv1 = nn.SpatialConvolution(c, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_masked)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)

    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        local nInput = args.n_units[i]
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.fc1)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

local MultiCNN105 = torch.class('MultiCNN105', 'Net')
function MultiCNN105:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local c = 3
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    self.m.arg_h2_nl = nn.ReLU(true)(self.m.arg_h2)
    self.m.arg_h22 = nn.Linear(args.edim, c * args.buf * args.channels)(self.m.arg_h2_nl)
    self.m.arg_h2d = nn.View(c, args.buf * args.channels):setNumInputDims(1)(self.m.arg_h22)
    local conv1_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(c, 10, 10):setNumInputDims(2)(MMconv({self.m.arg_h2d, conv1_2d}))

    -- convolution
    self.m.conv1 = nn.SpatialConvolution(c, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_masked)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)

    local prev_input = self.m.conv1_nl
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        local nInput = args.n_units[i]
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.mul_h = nn.CMulTable()({self.m.arg_h1, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

local MultiCNN106 = torch.class('MultiCNN106', 'Net')
function MultiCNN106:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local c = 3
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h1_nl = nn.ReLU(true)(self.m.arg_h1)
    self.m.arg_h11 = nn.Linear(args.edim, args.edim)(self.m.arg_h1_nl)
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    self.m.arg_h2_nl = nn.ReLU(true)(self.m.arg_h2)
    self.m.arg_h22 = nn.Linear(args.edim, c * args.buf * args.channels)(self.m.arg_h2_nl)
    self.m.arg_h2d = nn.View(c, args.buf * args.channels):setNumInputDims(1)(self.m.arg_h22)
    local conv1_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(c, 10, 10):setNumInputDims(2)(MMconv({self.m.arg_h2d, conv1_2d}))

    -- convolution
    self.m.conv1 = nn.SpatialConvolution(c, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_masked)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)

    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        local nInput = args.n_units[i]
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.mul_h = nn.CMulTable()({self.m.arg_h11, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

local MultiCNN107 = torch.class('MultiCNN107', 'Net')
function MultiCNN107:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local c = 3
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    local arg_hh = nn.CMulTable()({self.m.arg_h1, self.m.arg_h2})
    self.m.arg_h22 = nn.Linear(args.edim, c * args.buf * args.channels)(arg_hh)
    self.m.arg_h2d = nn.View(c, args.buf * args.channels):setNumInputDims(1)(self.m.arg_h22)
    local conv1_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(c, 10, 10):setNumInputDims(2)(MMconv({self.m.arg_h2d, conv1_2d}))

    -- convolution
    self.m.conv1 = nn.SpatialConvolution(c, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_masked)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)

    local prev_input = self.m.conv1_nl
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        local nInput = args.n_units[i]
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.arg_fc = nn.LinearNB(args.edim, args.edim)(arg_hh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

local MultiCNN108 = torch.class('MultiCNN108', 'Net')
function MultiCNN108:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    self.m.arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    self.m.arg_h22 = nn.Linear(args.edim, args.ch * args.buf * args.channels)(self.m.arg_hh)
    self.m.arg_h2d = nn.View(args.ch, 
            args.buf * args.channels):setNumInputDims(1)(self.m.arg_h22)
    local conv1_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(
        MMconv({self.m.arg_h2d, conv1_2d}))

    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.ch, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_masked)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)

    local prev_input = self.m.conv1_nl
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        local nInput = args.n_units[i]
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)
    
    self.m.arg_fc = nn.Linear(args.edim, args.edim)(self.m.arg_hh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)
    --self.m.joint_h_nl = nn.ReLU(true)(self.m.fc1)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

function MultiCNN108:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)

    local analogy2 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        table.insert(analogy2, module)
    end
    table.insert(self.analogy_list, analogy2)
    return self.analogy_list
end

local MultiCNN109 = torch.class('MultiCNN109', 'Net')
function MultiCNN109:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    local arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    self.m.arg_h22 = nn.Linear(args.edim, args.ch * args.buf * args.channels)(arg_hh)
    self.m.arg_h2d = nn.View(args.ch, 
            args.buf * args.channels):setNumInputDims(1)(self.m.arg_h22)
    local conv1_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(
        MMconv({self.m.arg_h2d, conv1_2d}))

    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.ch, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_masked)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)

    local prev_input = self.m.conv1_nl
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        local nInput = args.n_units[i]
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    --[[
    self.m.arg_fc = nn.Linear(args.edim, args.edim)(arg_hh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)
    --]]
    self.m.joint_h_nl = nn.ReLU(true)(self.m.fc1)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

function MultiCNN109:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h22 = nn.Linear(self.args.edim, 
                self.args.ch * self.args.buf * self.args.channels):share(
            self.m.arg_h22.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_h22)
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)
    return self.analogy_list
end

local MultiCNN110 = torch.class('MultiCNN110', 'MultiCNN108')
function MultiCNN110:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    local arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    self.m.arg_h22 = nn.Linear(args.edim, args.ch * args.n_units[1])(arg_hh)
    self.m.arg_h2d = nn.View(args.ch, args.n_units[1]):setNumInputDims(1)(self.m.arg_h22)
    
    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.buf * args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    
    local conv1_2d = nn.View(args.n_units[1], -1):setNumInputDims(3)(self.m.conv1_nl)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 15, 15):setNumInputDims(2)(
        MMconv({self.m.arg_h2d, conv1_2d}))
    
    local nInput = args.ch
    local prev_input = self.m.x_masked
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        nInput = args.n_units[i+1]
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(args.feat_size * args.feat_size * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.arg_fc = nn.Linear(args.edim, args.edim)(arg_hh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)
    -- self.m.joint_h_nl = nn.ReLU(true)(self.m.fc1)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

function MultiCNN110:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h22 = nn.Linear(self.args.edim, 
                self.args.ch * self.args.n_units[1]):share(
            self.m.arg_h22.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_h22)
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)

    local analogy2 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_fc = nn.Linear(self.args.edim, self.args.edim):share(
            self.m.arg_fc.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_fc)
        table.insert(analogy2, module)
    end
    table.insert(self.analogy_list, analogy2)
    return self.analogy_list
end

local SubControl = torch.class('SubControl', 'MultiCNN110')
function SubControl:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    local arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    self.m.arg_h22 = nn.Linear(args.edim, args.ch * args.n_units[2])(arg_hh)
    self.m.arg_h2d = nn.View(args.ch, args.n_units[2]):setNumInputDims(1)(self.m.arg_h22)
    
    -- convolution
    
    self.m.x_view = nn.View(-1, args.channels, args.input_dim, args.input_dim)(self.m.x)

    self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_view)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)

    self.m.conv1_view = nn.View(-1, args.buf * args.n_units[1],
            args.input_dim, args.input_dim)(self.m.conv1_nl)

    self.m.conv2 = nn.SpatialConvolution(args.buf * args.n_units[1], args.n_units[2],
                        args.filter_size[2], args.filter_size[2],
                        args.filter_stride[2], args.filter_stride[2],
                        args.pad[2], args.pad[2])(self.m.conv1_view)
    self.m.conv2_nl = nn.ReLU(true)(self.m.conv2)

    local conv_2d = nn.View(args.n_units[2], -1):setNumInputDims(3)(self.m.conv2_nl)
    local MMconv = nn.MM(false, false)
    self.m.conv_masked = nn.View(-1, args.ch, g_opts.max_size, g_opts.max_size)(
        MMconv({self.m.arg_h2d, conv_2d}))

    local nInput = args.ch
    local prev_input = self.m.conv_masked
    for i=2,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        nInput = args.n_units[i+1]
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(args.feat_size * args.feat_size * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.arg_fc = nn.Linear(args.edim, args.edim)(arg_hh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)
    -- self.m.joint_h_nl = nn.ReLU(true)(self.m.fc1)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

function SubControl:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h22 = nn.Linear(self.args.edim, 
                self.args.ch * self.args.n_units[2]):share(
            self.m.arg_h22.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_h22)
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)

    local analogy2 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_fc = nn.Linear(self.args.edim, self.args.edim):share(
            self.m.arg_fc.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_fc)
        table.insert(analogy2, module)
    end
    table.insert(self.analogy_list, analogy2)
    return self.analogy_list
end

local SubControl2 = torch.class('SubControl2', 'SubControl')
function SubControl2:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    local arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    self.m.arg_h22 = nn.Linear(args.edim, args.ch * args.buf * args.n_units[2])(arg_hh)
    self.m.arg_h2d = nn.View(args.ch, args.buf * args.n_units[2]):setNumInputDims(1)(self.m.arg_h22)
    
    -- convolution
    self.m.x_view = nn.View(-1, args.channels, args.input_dim, args.input_dim)(self.m.x)
    self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_view)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    
    self.m.conv2 = nn.SpatialConvolution(args.n_units[1], args.n_units[2],
                        args.filter_size[2], args.filter_size[2],
                        args.filter_stride[2], args.filter_stride[2],
                        args.pad[2], args.pad[2])(self.m.conv1_nl)
    self.m.conv2_nl = nn.ReLU(true)(self.m.conv2)

    local conv_2d = nn.View(-1, args.buf * args.n_units[2], g_opts.max_size * g_opts.max_size)(self.m.conv2_nl)
    local MMconv = nn.MM(false, false)
    self.m.conv_masked = nn.ReLU(true)(nn.View(-1, args.ch, g_opts.max_size, g_opts.max_size)(
        MMconv({self.m.arg_h2d, conv_2d})))

    local nInput = args.ch
    local prev_input = self.m.conv_masked
    for i=2,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        nInput = args.n_units[i+1]
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(args.feat_size * args.feat_size * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.arg_fc = nn.Linear(args.edim, args.edim)(arg_hh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)
    -- self.m.joint_h_nl = nn.ReLU(true)(self.m.fc1)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

function SubControl2:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h22 = nn.Linear(self.args.edim, 
                self.args.ch * self.args.buf * self.args.n_units[2]):share(
            self.m.arg_h22.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_h22)
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)

    local analogy2 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_fc = nn.Linear(self.args.edim, self.args.edim):share(
            self.m.arg_fc.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_fc)
        table.insert(analogy2, module)
    end
    table.insert(self.analogy_list, analogy2)
    return self.analogy_list
end

local SubControl3 = torch.class('SubControl3', 'SubControl2')
function SubControl3:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    local arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    self.m.arg_h22 = nn.Linear(args.edim, args.ch * args.buf * args.n_units[2])(arg_hh)
    self.m.arg_h2d = nn.View(args.ch, args.buf * args.n_units[2]):setNumInputDims(1)(self.m.arg_h22)
    
    -- convolution
    self.m.x_view = nn.View(-1, args.channels, args.input_dim, args.input_dim)(self.m.x)
    self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_view)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    
    self.m.conv2 = nn.SpatialConvolution(args.n_units[1], args.n_units[2],
                        args.filter_size[2], args.filter_size[2],
                        args.filter_stride[2], args.filter_stride[2],
                        args.pad[2], args.pad[2])(self.m.conv1_nl)
    self.m.conv2_nl = nn.ReLU(true)(self.m.conv2)


    local conv_2d = nn.View(-1, args.buf * args.n_units[2], g_opts.max_size * g_opts.max_size)(self.m.conv2_nl)
    local MMconv = nn.MM(false, false)
    self.m.conv_masked = nn.View(-1, args.ch, g_opts.max_size, g_opts.max_size)(
        MMconv({self.m.arg_h2d, conv_2d}))

    local nInput = args.ch
    local prev_input = self.m.conv_masked
    for i=2,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        nInput = args.n_units[i+1]
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(args.feat_size * args.feat_size * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.arg_fc = nn.Linear(args.edim, args.edim)(arg_hh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)
    -- self.m.joint_h_nl = nn.ReLU(true)(self.m.fc1)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

local SubControl4 = torch.class('SubControl4', 'SubControl3')
function SubControl4:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    local arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    self.m.arg_h22 = nn.Linear(args.edim, args.ch * args.buf * args.n_units[3])(arg_hh)
    self.m.arg_h2d = nn.View(args.ch, args.buf * args.n_units[3]):setNumInputDims(1)(self.m.arg_h22)
    
    -- convolution
    self.m.x_view = nn.View(-1, args.channels, args.input_dim, args.input_dim)(self.m.x)
    self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_view)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    
    self.m.conv2 = nn.SpatialConvolution(args.n_units[1], args.n_units[2],
                        args.filter_size[2], args.filter_size[2],
                        args.filter_stride[2], args.filter_stride[2],
                        args.pad[2], args.pad[2])(self.m.conv1_nl)
    self.m.conv2_nl = nn.ReLU(true)(self.m.conv2)

    self.m.conv3 = nn.SpatialConvolution(args.n_units[2], args.n_units[3],
                        args.filter_size[3], args.filter_size[3],
                        args.filter_stride[3], args.filter_stride[3],
                        args.pad[3], args.pad[3])(self.m.conv2_nl)
    self.m.conv3_nl = nn.ReLU(true)(self.m.conv3)

    local conv_2d = nn.View(-1, args.buf * args.n_units[3], g_opts.max_size * g_opts.max_size)(self.m.conv3_nl)
    local MMconv = nn.MM(false, false)
    self.m.conv_masked = nn.View(-1, args.ch, g_opts.max_size, g_opts.max_size)(
        MMconv({self.m.arg_h2d, conv_2d}))

    local nInput = args.ch
    local prev_input = self.m.conv_masked
    for i=3,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        nInput = args.n_units[i+1]
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(args.feat_size * args.feat_size * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.arg_fc = nn.Linear(args.edim, args.edim)(arg_hh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)
    -- self.m.joint_h_nl = nn.ReLU(true)(self.m.fc1)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

function SubControl4:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h22 = nn.Linear(self.args.edim, 
                self.args.ch * self.args.buf * self.args.n_units[3]):share(
            self.m.arg_h22.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_h22)
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)

    local analogy2 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_fc = nn.Linear(self.args.edim, self.args.edim):share(
            self.m.arg_fc.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_fc)
        table.insert(analogy2, module)
    end
    table.insert(self.analogy_list, analogy2)
    return self.analogy_list
end


local MultiCNN111 = torch.class('MultiCNN111', 'MultiCNN110')
function MultiCNN111:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    local arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    self.m.arg_h22 = nn.Linear(args.edim, args.ch)(arg_hh)
	
	self.m.arg_rep = nn.Contiguous()(nn.Transpose({2,3})(nn.Replicate(10*10, 2)(self.m.arg_h22)))
	self.m.arg_h3d = nn.View(args.ch, 10, 10):setNumInputDims(2)(self.m.arg_rep)
    self.m.x_concat = nn.JoinTable(2)({self.m.x, self.m.arg_h3d })

    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.ch + args.buf * args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_concat)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)

    local prev_input = self.m.conv1_nl
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        local nInput = args.n_units[i]
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.arg_fc = nn.Linear(args.edim, args.edim)(arg_hh)
	self.m.join_h = nn.JoinTable(2)({self.m.arg_fc, self.m.fc1})
    self.m.joint_h = nn.Linear(2 * args.edim, args.edim)(self.m.join_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)
    -- self.m.joint_h_nl = nn.ReLU(true)(self.m.fc1)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

local Sub = torch.class('Sub', 'SubControl3')
function Sub:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    local arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    self.m.arg_h22 = nn.Linear(args.edim, args.ch * args.buf * args.n_units[2])(arg_hh)
    self.m.arg_h2d = nn.View(args.ch, args.buf * args.n_units[2]):setNumInputDims(1)(self.m.arg_h22)
    
    -- convolution
    self.m.x_view = nn.View(-1, args.channels, args.input_dim, args.input_dim)(self.m.x)
    self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_view)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    
    self.m.conv2 = nn.SpatialConvolution(args.n_units[1], args.n_units[2],
                        args.filter_size[2], args.filter_size[2],
                        args.filter_stride[2], args.filter_stride[2],
                        args.pad[2], args.pad[2])(self.m.conv1_nl)
    self.m.conv2_nl = nn.ReLU(true)(self.m.conv2)

    local fsize = 15
    local conv_2d = nn.View(-1, args.buf * args.n_units[2], fsize * fsize)(self.m.conv2_nl)
    local MMconv = nn.MM(false, false)
    self.m.conv_masked = nn.View(-1, args.ch, fsize, fsize)(
        MMconv({self.m.arg_h2d, conv_2d}))

    local nInput = args.ch
    local prev_input = self.m.conv_masked
    for i=2,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        nInput = args.n_units[i+1]
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(args.feat_size * args.feat_size * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.arg_fc = nn.Linear(args.edim, args.edim)(arg_hh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

local SubLSTM = torch.class('SubLSTM', 'Net')
function SubLSTM:build_init_states(args)
    local states = {}
    table.insert(states, torch.Tensor(1, args.ldim))      -- c
    table.insert(states, torch.Tensor(1, args.ldim))      -- h
    return states
end

function SubLSTM:forward(x, g, arg, term)
    self.input = {x:narrow(2, 1, self.args.channels), g}
    for i=1,#self.init_states do
        table.insert(self.input, self.init_states[i])
    end
    if self.args.pass_act then
        for i=1,#arg do
            if g_opts.gpu > 0 then
                table.insert(self.input, arg[i]:cuda())
            else
                table.insert(self.input, arg[i])
            end
        end
    end
    if self.args.pass_term then
        if g_opts.gpu > 0 then
            table.insert(self.input, term:cuda())
        else
            table.insert(self.input, term)
        end
    end 
    return self:parse_forward(self.net:forward(self.input))
end

function SubLSTM:build_model(args)
    args.buf = 1
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()
    self.m.c0 = nn.Identity()()
    self.m.h0 = nn.Identity()()

    local prev_c = self.m.c0
    local prev_h = self.m.h0
    local edim = args.edim
    local ldim = args.ldim

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    local arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    self.m.arg_fc = nn.Linear(args.edim, args.edim)(arg_hh)

    self.m.arg_h22 = nn.Linear(args.edim, args.ch * args.n_units[2])(arg_hh)
    self.m.arg_h2d = nn.View(args.ch, args.n_units[2]):setNumInputDims(1)(self.m.arg_h22)
    
    -- convolution
    self.m.x_view = nn.View(-1, args.channels, args.input_dim, args.input_dim)(self.m.x)
    self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_view)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    
    self.m.conv2 = nn.SpatialConvolution(args.n_units[1], args.n_units[2],
                        args.filter_size[2], args.filter_size[2],
                        args.filter_stride[2], args.filter_stride[2],
                        args.pad[2], args.pad[2])(self.m.conv1_nl)
    self.m.conv2_nl = nn.ReLU(true)(self.m.conv2)

    local fsize = 15
    local conv_2d = nn.View(-1, args.n_units[2], fsize * fsize)(self.m.conv2_nl)
    local MMconv = nn.MM(false, false)
    self.m.conv_masked = nn.View(-1, args.ch, fsize, fsize)(
        MMconv({self.m.arg_h2d, conv_2d}))

    local nInput = args.ch
    local prev_input = self.m.conv_masked
    for i=2,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        nInput = args.n_units[i+1]
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    local nel = args.feat_size * args.feat_size * args.n_units[#args.n_units]

    -- gated lstm
    local xh = nn.JoinTable(2)({self.m.conv, prev_h})
    self.m.lstm_gate = nn.Linear(nel + ldim, edim)(xh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.lstm_gate})
    self.m.state = nn.Linear(edim, 4*ldim)(self.m.mul_h)
    self.m.lstm = nn.ReLU(true)(self.m.state)

    local reshaped = nn.View(4, ldim):setNumInputDims(1)(self.m.lstm)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    local in_gate = nn.Sigmoid(true)(n1)
    local forget_gate = nn.Sigmoid(true)(n2)
    local out_gate = nn.Sigmoid(true)(n3)
    local in_transform = nn.Tanh()(n4)
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    self.m.fc = nn.Linear(ldim, args.n_actions)(next_h)
    self.m.a = nn.LogSoftMax()(self.m.fc)
    self.m.b = nn.Linear(ldim, 1)(next_h)

    local input = {self.m.x, self.m.args, self.m.c0, self.m.h0}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.feasible then
        table.insert(self.n_actions, 2)
        self.m.feas_fc = nn.Linear(ldim, 2)(next_h)
        self.m.feas_fc.data.module.bias[1] = 3
        self.m.feas = nn.LogSoftMax()(self.m.feas_fc)
        table.insert(output, self.m.feas)
    end

    if args.term_act then
        self.m.term_fc = nn.Linear(ldim, 2)(next_h)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end

    table.insert(output, self.m.b)
    table.insert(output, next_c)
    table.insert(output, next_h)
    return nn.gModule(input, output)
end

function SubLSTM:fill_internal_actions()
    if self.args.feasible then
        local prob = torch.exp(self.m.feas.data.module.output:float())
        local feas = torch.multinomial(prob, 1)
        table.insert(self.actions, feas)
    end
end

function SubLSTM:feasible()
    return self.actions[2]
end

function SubLSTM:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h22 = nn.Linear(self.args.edim, 
                self.args.ch * self.args.n_units[2]):share(
            self.m.arg_h22.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_h22)
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)

    local analogy2 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_fc = nn.Linear(self.args.edim, self.args.edim):share(
            self.m.arg_fc.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_fc)
        table.insert(analogy2, module)
    end
    table.insert(self.analogy_list, analogy2)
    return self.analogy_list
end

local SubCount = torch.class("SubCount", "SubLSTM")
function SubCount:build_model(args)
    args.buf = 1
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()
    self.m.c0 = nn.Identity()()
    self.m.h0 = nn.Identity()()

    local prev_c = self.m.c0
    local prev_h = self.m.h0
    local edim = args.edim
    local ldim = args.ldim

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    self.m.arg_h3 = nn.LookupTable(args.num_args[3], args.edim)(arguments[3])
    local arg_hh = nn.ReLU(true)(nn.CAddTable()({self.m.arg_h1, self.m.arg_h2, self.m.arg_h3}))
    self.m.arg_fc = nn.Linear(args.edim, args.edim)(arg_hh)

    self.m.arg_h22 = nn.Linear(args.edim, args.ch * args.n_units[2])(arg_hh)
    self.m.arg_h2d = nn.View(args.ch, args.n_units[2]):setNumInputDims(1)(self.m.arg_h22)
    
    -- convolution
    self.m.x_view = nn.View(-1, args.channels, args.input_dim, args.input_dim)(self.m.x)
    self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_view)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    
    self.m.conv2 = nn.SpatialConvolution(args.n_units[1], args.n_units[2],
                        args.filter_size[2], args.filter_size[2],
                        args.filter_stride[2], args.filter_stride[2],
                        args.pad[2], args.pad[2])(self.m.conv1_nl)
    self.m.conv2_nl = nn.ReLU(true)(self.m.conv2)

    local fsize = 15
    local conv_2d = nn.View(-1, args.n_units[2], fsize * fsize)(self.m.conv2_nl)
    local MMconv = nn.MM(false, false)
    self.m.conv_masked = nn.View(-1, args.ch, fsize, fsize)(
        MMconv({self.m.arg_h2d, conv_2d}))

    local nInput = args.ch
    local prev_input = self.m.conv_masked
    for i=2,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        nInput = args.n_units[i+1]
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    local nel = args.feat_size * args.feat_size * args.n_units[#args.n_units]

    -- gated lstm
    local xh = nn.JoinTable(2)({self.m.conv, prev_h})
    self.m.lstm_gate = nn.Linear(nel + ldim, edim)(xh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.lstm_gate})
    self.m.state = nn.Linear(edim, 4*ldim)(self.m.mul_h)
    self.m.lstm = nn.ReLU(true)(self.m.state)

    local reshaped = nn.View(4, ldim):setNumInputDims(1)(self.m.lstm)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    local in_gate = nn.Sigmoid(true)(n1)
    local forget_gate = nn.Sigmoid(true)(n2)
    local out_gate = nn.Sigmoid(true)(n3)
    local in_transform = nn.Tanh()(n4)
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    self.m.fc = nn.Linear(ldim, args.n_actions)(next_h)
    self.m.a = nn.LogSoftMax()(self.m.fc)
    self.m.b = nn.Linear(ldim, 1)(next_h)

    local input = {self.m.x, self.m.args, self.m.c0, self.m.h0}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.feasible then
        table.insert(self.n_actions, 2)
        self.m.feas_fc = nn.Linear(ldim, 2)(next_h)
        self.m.feas_fc.data.module.bias[1] = 3
        self.m.feas = nn.LogSoftMax()(self.m.feas_fc)
        table.insert(output, self.m.feas)
    end

    if args.term_act then
        self.m.term_fc = nn.Linear(ldim, 2)(next_h)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end

    table.insert(output, self.m.b)
    table.insert(output, next_c)
    table.insert(output, next_h)
    return nn.gModule(input, output)
end

function SubCount:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h3 = nn.LookupTable(self.args.num_args[3], self.args.edim):share(
            self.m.arg_h3.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h22 = nn.Linear(self.args.edim, 
                self.args.ch * self.args.n_units[2]):share(
            self.m.arg_h22.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2):add(arg_h3))
        module:add(nn.CAddTable())
        module:add(nn.ReLU(true))
        module:add(arg_h22)
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)

    local analogy2 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h3 = nn.LookupTable(self.args.num_args[3], self.args.edim):share(
            self.m.arg_h3.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_fc = nn.Linear(self.args.edim, self.args.edim):share(
            self.m.arg_fc.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2):add(arg_h3))
        module:add(nn.CAddTable())
        module:add(nn.ReLU(true))
        module:add(arg_fc)
        table.insert(analogy2, module)
    end
    table.insert(self.analogy_list, analogy2)
    return self.analogy_list
end

local SubCountBase = torch.class("SubCountBase", "SubLSTM")
function SubCountBase:build_model(args)
    args.buf = 1
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()
    self.m.c0 = nn.Identity()()
    self.m.h0 = nn.Identity()()

    local prev_c = self.m.c0
    local prev_h = self.m.h0
    local edim = args.edim
    local ldim = args.ldim

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    self.m.arg_h3 = nn.Linear(1, args.edim)(arguments[3])
    local arg_hh = nn.CAddTable()({self.m.arg_h1, self.m.arg_h2, self.m.arg_h3})
    self.m.arg_fc = nn.Linear(args.edim, args.edim)(arg_hh)

    self.m.arg_h22 = nn.Linear(args.edim, args.ch * args.n_units[2])(arg_hh)
    self.m.arg_h2d = nn.View(args.ch, args.n_units[2]):setNumInputDims(1)(self.m.arg_h22)
    
    -- convolution
    self.m.x_view = nn.View(-1, args.channels, args.input_dim, args.input_dim)(self.m.x)
    self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_view)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    
    self.m.conv2 = nn.SpatialConvolution(args.n_units[1], args.n_units[2],
                        args.filter_size[2], args.filter_size[2],
                        args.filter_stride[2], args.filter_stride[2],
                        args.pad[2], args.pad[2])(self.m.conv1_nl)
    self.m.conv2_nl = nn.ReLU(true)(self.m.conv2)

    local fsize = 15
    local conv_2d = nn.View(-1, args.n_units[2], fsize * fsize)(self.m.conv2_nl)
    local MMconv = nn.MM(false, false)
    self.m.conv_masked = nn.View(-1, args.ch, fsize, fsize)(
        MMconv({self.m.arg_h2d, conv_2d}))

    local nInput = args.ch
    local prev_input = self.m.conv_masked
    for i=2,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        nInput = args.n_units[i+1]
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    local nel = args.feat_size * args.feat_size * args.n_units[#args.n_units]

    -- gated lstm
    local xh = nn.JoinTable(2)({self.m.conv, prev_h})
    self.m.lstm_gate = nn.Linear(nel + ldim, edim)(xh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.lstm_gate})
    self.m.state = nn.Linear(edim, 4*ldim)(self.m.mul_h)
    self.m.lstm = nn.ReLU(true)(self.m.state)

    local reshaped = nn.View(4, ldim):setNumInputDims(1)(self.m.lstm)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    local in_gate = nn.Sigmoid(true)(n1)
    local forget_gate = nn.Sigmoid(true)(n2)
    local out_gate = nn.Sigmoid(true)(n3)
    local in_transform = nn.Tanh()(n4)
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    self.m.fc = nn.Linear(ldim, args.n_actions)(next_h)
    self.m.a = nn.LogSoftMax()(self.m.fc)
    self.m.b = nn.Linear(ldim, 1)(next_h)

    local input = {self.m.x, self.m.args, self.m.c0, self.m.h0}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.feasible then
        table.insert(self.n_actions, 2)
        self.m.feas_fc = nn.Linear(ldim, 2)(next_h)
        self.m.feas_fc.data.module.bias[1] = 3
        self.m.feas = nn.LogSoftMax()(self.m.feas_fc)
        table.insert(output, self.m.feas)
    end

    if args.term_act then
        self.m.term_fc = nn.Linear(ldim, 2)(next_h)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end

    table.insert(output, self.m.b)
    table.insert(output, next_c)
    table.insert(output, next_h)
    return nn.gModule(input, output)
end

local SubBias = torch.class('SubBias', 'SubControl3')
function SubBias:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    local arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    self.m.arg_h22 = nn.Linear(args.edim, args.ch + args.ch * args.buf * args.n_units[2])(arg_hh)
    self.m.arg_h2d = nn.View(args.ch, args.buf * args.n_units[2]):setNumInputDims(1)(
            nn.Narrow(2, args.ch + 1, args.ch * args.buf * args.n_units[2])(self.m.arg_h22))
    self.m.arg_bias = nn.Narrow(2, 1, args.ch)(self.m.arg_h22)

    -- convolution
    self.m.x_view = nn.View(-1, args.channels, args.input_dim, args.input_dim)(self.m.x)
    self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_view)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    
    self.m.conv2 = nn.SpatialConvolution(args.n_units[1], args.n_units[2],
                        args.filter_size[2], args.filter_size[2],
                        args.filter_stride[2], args.filter_stride[2],
                        args.pad[2], args.pad[2])(self.m.conv1_nl)
    self.m.conv2_nl = nn.ReLU(true)(self.m.conv2)

    local fsize = 15
    local conv_2d = nn.View(-1, args.buf * args.n_units[2], fsize * fsize)(self.m.conv2_nl)
    local MMconv = nn.MM(false, false)
    self.m.conv_masked = nn.View(-1, args.ch, fsize * fsize)(MMconv({self.m.arg_h2d, conv_2d}))
    self.m.conv_masked_b = nn.ScalarAddTable()({self.m.conv_masked, self.m.arg_bias})
    self.m.conv_masked_nl = nn.View(-1, args.ch, fsize, fsize)(
            nn.ReLU(true)(self.m.conv_masked_b))

    local nInput = args.ch
    local prev_input = self.m.conv_masked_nl
    for i=2,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        nInput = args.n_units[i+1]
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(args.feat_size * args.feat_size * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.arg_fc = nn.Linear(args.edim, args.edim)(arg_hh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

function SubBias:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h22 = nn.Linear(self.args.edim, 
                self.args.ch + self.args.ch * self.args.buf * self.args.n_units[2]):share(
            self.m.arg_h22.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_h22)
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)

    local analogy2 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_fc = nn.Linear(self.args.edim, self.args.edim):share(
            self.m.arg_fc.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_fc)
        table.insert(analogy2, module)
    end
    table.insert(self.analogy_list, analogy2)
    return self.analogy_list
end



local Sub1 = torch.class('Sub1', 'Sub')
function Sub1:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    local arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    self.m.arg_h22 = nn.Linear(args.edim, args.ch * args.buf * args.n_units[1])(arg_hh)
    self.m.arg_h2d = nn.View(args.ch, args.buf * args.n_units[1]):setNumInputDims(1)(self.m.arg_h22)
    
    -- convolution
    self.m.x_view = nn.View(-1, args.channels, args.input_dim, args.input_dim)(self.m.x)
    self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_view)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    
    local fsize = 15
    local conv_2d = nn.View(-1, args.buf * args.n_units[1], fsize * fsize)(self.m.conv1_nl)
    local MMconv = nn.MM(false, false)
    self.m.conv_masked = nn.View(-1, args.ch, fsize, fsize)(
        MMconv({self.m.arg_h2d, conv_2d}))

    local nInput = args.ch
    local prev_input = self.m.conv_masked
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        nInput = args.n_units[i+1]
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(args.feat_size * args.feat_size * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.arg_fc = nn.Linear(args.edim, args.edim)(arg_hh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

function Sub1:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h22 = nn.Linear(self.args.edim, 
                self.args.ch * self.args.buf * self.args.n_units[1]):share(
            self.m.arg_h22.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_h22)
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)

    local analogy2 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_fc = nn.Linear(self.args.edim, self.args.edim):share(
            self.m.arg_fc.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_fc)
        table.insert(analogy2, module)
    end
    table.insert(self.analogy_list, analogy2)
    return self.analogy_list
end


function MultiCNN111:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h22 = nn.Linear(self.args.edim, self.args.ch):share(
            self.m.arg_h22.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_h22)
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)

    local analogy2 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_fc = nn.Linear(self.args.edim, self.args.edim):share(
            self.m.arg_fc.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_fc)
        table.insert(analogy2, module)
    end
    table.insert(self.analogy_list, analogy2)
    return self.analogy_list
end

local MultiCNN112 = torch.class('MultiCNN112', 'MultiCNN110')
function MultiCNN112:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    local arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.buf * args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)

    local prev_input = self.m.conv1_nl
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        local nInput = args.n_units[i]
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.arg_fc = nn.Linear(args.edim, args.edim)(arg_hh)
	self.m.join_h = nn.JoinTable(2)({self.m.arg_fc, self.m.fc1})
    self.m.joint_h = nn.Linear(2 * args.edim, args.edim)(self.m.join_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)
    -- self.m.joint_h_nl = nn.ReLU(true)(self.m.fc1)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

function MultiCNN112:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy2 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_fc = nn.Linear(self.args.edim, self.args.edim):share(
            self.m.arg_fc.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_fc)
        table.insert(analogy2, module)
    end
    table.insert(self.analogy_list, analogy2)
    return self.analogy_list
end

local MultiCNNIMG = torch.class('MultiCNNIMG', 'MultiCNN108')
function MultiCNNIMG:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    local arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    self.m.arg_h22 = nn.Linear(args.edim, args.ch * args.n_units[1])(arg_hh)
    self.m.arg_h2d = nn.View(args.ch, args.n_units[1]):setNumInputDims(1)(self.m.arg_h22)
    
    self.m.arg_h3 = nn.Linear(args.edim, args.ch * args.buf * args.channels)(arg_hh)
    local h3view = nn.View(args.ch, args.buf * args.channels):setNumInputDims(1)(self.m.arg_h3)
    local x_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, args.input_dim, args.input_dim):setNumInputDims(2)(
        MMconv({h3view, x_2d}))

    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.ch, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_masked)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    
    local conv1_2d = nn.View(args.n_units[1], -1):setNumInputDims(3)(self.m.conv1_nl)
    local MMconv2 = nn.MM(false, false)
    local conv1_masked = nn.View(args.ch, 15, 15):setNumInputDims(2)(
        MMconv2({self.m.arg_h2d, conv1_2d}))
    
    local nInput = args.ch
    local prev_input = conv1_masked
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        nInput = args.n_units[i+1]
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(args.feat_size * args.feat_size * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.arg_fc = nn.Linear(args.edim, args.edim)(arg_hh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)
    -- self.m.joint_h_nl = nn.ReLU(true)(self.m.fc1)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

function MultiCNNIMG:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h22 = nn.Linear(self.args.edim, self.args.ch * self.args.n_units[1]):share(
            self.m.arg_h22.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_h22)
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)

    local analogy2 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_fc = nn.Linear(self.args.edim, self.args.edim):share(
            self.m.arg_fc.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_fc)
        table.insert(analogy2, module)
    end
    table.insert(self.analogy_list, analogy2)
    
    local analogy3 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h3 = nn.Linear(self.args.edim, 
                self.args.ch * self.args.buf * self.args.channels):share(
            self.m.arg_h3.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_h3)
        table.insert(analogy3, module)
    end
    table.insert(self.analogy_list, analogy3)
    return self.analogy_list
end


local MultiFF = torch.class('MultiFF', 'MultiCNN110')
function MultiFF:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    local arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    self.m.arg_h22 = nn.Linear(args.edim, args.ch * args.buf * args.channels)(arg_hh)
    self.m.arg_h2d = nn.View(args.ch, 
            args.buf * args.channels):setNumInputDims(1)(self.m.arg_h22)
    local conv1_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(-1):setNumInputDims(2)(MMconv({self.m.arg_h2d, conv1_2d}))

    self.m.fc1 = nn.Linear(10 * 10 * args.ch, args.edim)(self.m.x_masked)
    self.m.fc1_nl = nn.ReLU(true)(self.m.fc1)
    self.m.fc2 = nn.Linear(args.edim, args.edim)(self.m.fc1_nl)
    self.m.arg_fc = nn.Linear(args.edim, args.edim)(arg_hh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc2})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)
    -- self.m.joint_h_nl = nn.ReLU(true)(self.m.fc1)

    self.m.fc = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

function MultiFF:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h22 = nn.Linear(self.args.edim, 
                self.args.ch * self.args.buf * self.args.channels):share(
            self.m.arg_h22.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_h22)
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)

    local analogy2 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_fc = nn.Linear(self.args.edim, self.args.edim):share(
            self.m.arg_fc.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_fc)
        table.insert(analogy2, module)
    end
    table.insert(self.analogy_list, analogy2)
    return self.analogy_list
end

local MultiTerm = torch.class('MultiTerm', 'MultiCNN110')
function MultiTerm:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    local arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    self.m.arg_h22 = nn.Linear(args.edim, args.ch * args.channels)(arg_hh)
    self.m.arg_h2d = nn.View(args.ch, args.channels):setNumInputDims(1)(self.m.arg_h22)
    local conv1_2d = nn.View(args.channels, -1):setNumInputDims(3)(
        nn.Contiguous()(nn.Narrow(2, 1, args.channels)(self.m.x)))
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(
        MMconv({self.m.arg_h2d, conv1_2d}))
        
    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.ch, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_masked)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)

    local prev_input = self.m.conv1_nl
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        local nInput = args.n_units[i]
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.arg_fc = nn.Linear(args.edim, args.edim)(arg_hh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.arg_hconv = nn.Linear(args.edim, args.ch * args.buf * args.channels)(arg_hh)
        self.m.arg_hconv2d = nn.View(args.ch, 
                args.buf * args.channels):setNumInputDims(1)(self.m.arg_hconv)
        local x_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
        local MMconv2 = nn.MM(false, false)
        self.m.x2_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(
                MMconv2({self.m.arg_hconv2d, x_2d}))
        self.m.conv1_t = nn.SpatialConvolution(args.ch, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x2_masked)
        self.m.conv1_t_nl = nn.ReLU(true)(self.m.conv1_t)
        self.m.conv_t = nn.View(-1):setNumInputDims(3)(self.m.conv1_t_nl)
        self.m.term_fc = nn.Linear(10 * 10 * args.n_units[1], 2)(self.m.conv_t)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

local MultiCNN210 = torch.class('MultiCNN210', 'MultiCNN110')
function MultiCNN210:build_model(args)
    self.m.x = nn.Identity()()
    self.m.args = nn.Identity()()

    -- argument embedding
    local arguments = {nn.SplitTable(1, 1)(self.m.args):split(args.max_args)}
    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(arguments[1])
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(arguments[2])
    local arg_hh = nn.ReLU(true)(nn.CAddTable()({self.m.arg_h1, self.m.arg_h2}))
    self.m.arg_hhh = nn.Linear(args.edim, 10)(arg_hh)
    self.m.arg_hhh2 = nn.ReLU(true)(nn.Linear(10, args.edim)(self.m.arg_hhh))
    self.m.arg_h22 = nn.Linear(args.edim, args.ch * args.buf * args.channels)(self.m.arg_hhh2)
    self.m.arg_h2d = nn.View(args.ch, 
            args.buf * args.channels):setNumInputDims(1)(self.m.arg_h22)
    local conv1_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(
        MMconv({self.m.arg_h2d, conv1_2d}))

    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.ch, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_masked)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)

    local prev_input = self.m.conv1_nl
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        local nInput = args.n_units[i]
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.arg_fc = nn.Linear(args.edim, args.edim)(self.m.arg_hhh2)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)
    -- self.m.joint_h_nl = nn.ReLU(true)(self.m.fc1)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local input = {self.m.x, self.m.args}
    local output = {self.m.a}
    self.n_actions = {args.n_actions}
    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end


function MultiTerm:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h22 = nn.Linear(self.args.edim, 
                self.args.ch * self.args.channels):share(
            self.m.arg_h22.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_h22)
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)

    local analogy2 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_fc = nn.Linear(self.args.edim, self.args.edim):share(
            self.m.arg_fc.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_fc)
        table.insert(analogy2, module)
    end
    table.insert(self.analogy_list, analogy2)

    --[[
    local analogy3 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_hconv = nn.Linear(self.args.edim, 
                self.args.ch * self.args.buf * self.args.channels):share(
            self.m.arg_hconv.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_hconv)
        table.insert(analogy3, module)
    end
    table.insert(self.analogy_list, analogy3)
    --]]
    return self.analogy_list
end

local Manager = torch.class('Manager', 'Net')

function Manager:build_conv(args, input)
    self.m.conv1 = nn.SpatialConvolution(args.buf * args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(input)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    for i=1,(#args.n_units-1) do
        if args.pool[i] then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end
    return nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
end

function Manager:build_goal_memory(args, goal)
    -- local g_flat = nn.View(-1):setNumInputDims(2)(goal)
    if args.content_att == true then
        self.m.g_key_h = nn.LookupTable(args.vocab_size, args.wdim)(goal)
        --self.m.g_key_h.data.module:reset(0.01)
        local key4d = nn.View(-1, args.max_word, args.wdim):setNumInputDims(2)(self.m.g_key_h)
        self.m.g_key = nn.Sum(3)(key4d)
    end
    self.m.g_val_h = nn.LookupTable(args.vocab_size, args.wdim)(goal)
    --self.m.g_val_h.data.module:reset(0.01)
    local val4d = nn.View(-1, args.max_word, args.wdim):setNumInputDims(2)(self.m.g_val_h)
    self.m.g_val = nn.Sum(3)(val4d)
end

function Manager:retrieve_goal(args, h, p)
    -- shift vector
    self.m.g_hs = nn.Linear(args.ldim, args.edim)(h)
    local g_hs_nl = nn.ReLU(true)(self.m.g_hs)
    self.m.g_s = nn.Linear(args.edim, 2 * args.shift + 1)(g_hs_nl)
    if args.soft_mem then 
        self.m.g_prob_s = nn.SoftMax()(self.m.g_s)
    else
        self.m.g_shift = nn.LogSoftMax()(self.m.g_s)
        self.m.shift_act = nn.Multinomial()(self.m.g_shift)
        self.m.g_prob_s = nn.OneHot(2 * args.shift + 1)(self.m.shift_act)
    end
    --self.m.g_gamma = nn.Linear(args.ldim, 1)(h)
    --self.m.g_gamma.data.module:reset(0.0001)
    --local gamma = nn.AddConstant(1)(nn.SoftPlus()(self.m.g_gamma))
    --[[
    -- gating parameter
    self.m.g_gate_lin = nn.Linear(args.ldim, 1)(h)
    self.m.g_gate = args.sigmoid()(self.m.g_gate_lin)
    -- exponential focusing parameter
    self.m.g_gamma = nn.Linear(args.ldim, 1)(h)
    local gamma = nn.AddConstant(1)(nn.SoftPlus()(self.m.g_gamma))

    local gate = nn.Reshape(-1)(nn.Replicate(args.max_task, 2, 1)(self.m.g_gate))
    local w_g = nn.CAddTable(){
        nn.CMulTable(){self.m.g_prob_c, gate},
        nn.CMulTable(){p, nn.AddConstant(1)(nn.MulConstant(-1)(gate))}
    }
    --]]
    --[[
    local w_g = nn.CAddTable(){
        nn.ScalarMulTable(){self.m["g_prob_c"], self.m["g_gate"]},
        nn.ScalarMulTable(){p, nn.AddConstant(1)(nn.MulConstant(-1)(self.m["g_gate"]))}
    }
    --]]

    local w_tilde = nn.CircularConvolution(){p, self.m.g_prob_s}
    --local w_pow = nn.PowTable(){w_tilde, gamma}
    self.m.g_prob = nn.Normalize(1)(w_tilde)
    local prob3d = nn.View(1, -1):setNumInputDims(1)(self.m.g_prob)
    local MMbout = nn.MM(false, false)
    local out3d = MMbout({prob3d, self.m.g_val})
    local out2d = nn.View(-1):setNumInputDims(1)(out3d)
    return out2d, self.m.g_prob
end

function Manager:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.c0 = nn.Identity()()
    self.m.h0 = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.p0 = nn.Identity()()
    
    local prev_c = self.m.c0
    local prev_h = self.m.h0
    local prev_r = self.m.r0
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    -- Compute CNN embeddings
    self.m.conv = self:build_conv(args, self.m.x)
    self.m.fc1  = nn.Linear(nel, edim)(self.m.conv)
    local fc1_nl = nn.ReLU(true)(self.m.fc1)
    self.m.x_s = nn.JoinTable(2)({fc1_nl, prev_r, prev_p})
    self.m.x2s = nn.Linear(edim + wdim + args.max_task, edim)(self.m.x_s)
    local xs = nn.ReLU(true)(self.m.x2s)
    local state = nn.JoinTable(2)({xs, prev_h})
    self.m.lstm = nn.Linear(edim + ldim, 4*ldim)(state)

    local reshaped = nn.View(4, ldim):setNumInputDims(1)(self.m.lstm)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    local in_gate = nn.Sigmoid(true)(n1)
    local forget_gate = nn.Sigmoid(true)(n2)
    local out_gate = nn.Sigmoid(true)(n3)
    local in_transform = nn.Tanh()(n4)
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    local next_r, next_p = self:retrieve_goal(args, next_h, prev_p)
    local hr = nn.JoinTable(2)({next_h, next_r})
    self.m.hr = nn.LinearNB(wdim + ldim, edim)(hr)
    local hr1 = nn.Narrow(2,1,args.lindim)(self.m.hr)
    local hr2 = nn.Narrow(2,1+args.lindim,edim-args.lindim)(self.m.hr)
    local hr2_nl = nn.ReLU(true)(hr2)
    self.m.s = nn.JoinTable(2)({hr1, hr2_nl})

    local input = {self.m.x, self.m.g, self.m.c0, self.m.h0, self.m.r0, self.m.p0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.term_act then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.s)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end

    self.m.b = nn.Linear(edim, 1)(self.m.s)
    table.insert(output, self.m.b)
    table.insert(output, next_c)
    table.insert(output, next_h)
    table.insert(output, next_r)
    table.insert(output, next_p)

    return nn.gModule(input, output)
end

function Manager:build_init_states(args)
    local states = {}
    local edim = args.edim
    table.insert(states, torch.Tensor(1, args.ldim))      -- c
    table.insert(states, torch.Tensor(1, args.ldim))      -- h
    table.insert(states, torch.Tensor(1, args.wdim))      -- r
    table.insert(states, torch.Tensor(1, args.max_task))  -- p
    return states
end

function Manager:reset_init_state(batch_size)
    for j=1,#self.init_states do
        local size = self.init_states[j]:size()
        size[1] = batch_size
        self.init_states[j]:resize(size)
        self.init_states[j]:fill(0)
    end
    -- Set initial memory attention to 1, 0, 0, ...
    self.init_states[4]:narrow(2, 1, 1):fill(1)
end

function Manager:fill_internal_actions()
    if not self.args.soft_mem then
        table.insert(self.actions, self.m.shift_act.data.module.output:long())
    end
end

function Manager:add_entropy_grad(rho)
    self.entropy = self.entropy or {}
    for i=1,#self.n_actions do
        self.entropy[i] = self.entropy[i] or nn.Entropy()
    end
    local idx = 0
    for i=1,#self.args.num_args do
        idx = idx + 1
        --local entropy = self.entropy[i]:forward(self.m["a" .. i].data.module.output)
        local g = self.entropy[i]:backward(self.m["a" .. i].data.module.output, 1)
        --print("Entropy", entropy, "GradNorm", g:norm(), self.grad_output[idx]:norm())
        self.grad_output[idx]:add(g:mul(-rho))
    end
    if not self.args.soft_mem then
        idx = idx + 1
        local g = self.entropy[idx]:backward(self.m.g_shift.data.module.output, 1)
        self.grad_output[idx]:add(g:mul(-rho))
    end
    if not self.args.soft_term and self.m.term then
        idx = idx + 1
        local g = self.entropy[idx]:backward(self.m.term.data.module.output, 1)
        self.grad_output[idx]:add(g:mul(-rho))
    end
    if self.entropy_list then 
        for i = 1,#self.entropy_list do
            self.entropy_list[i].data.module.w = -rho
        end
    end
end

function Manager:clean_invalid_grad()
    if not self.args.soft_term and self.m.term then
        local idx = #self.args.num_args
        if not self.args.soft_mem then
            idx = idx + 1
            if self.m.op_shift_act then
                idx = idx + 1
            end
        end
        self.grad_scale = self.grad_scale or {}
        for i = 1, idx do
            self.grad_scale[i] = self.grad_scale[i] or nn.ScalarMulTable()
            --print(self.grad_output[i]:norm(2, 2), self.m.t2.data.module.output)
            self.grad_output[i]:copy(self.grad_scale[i]:forward(
                {self.grad_output[i], self.m.t2.data.module.output}))
            --print(self.grad_output[i]:norm(2, 2))
        end
        for i = 1, #self.init_states do
            local j = #self.grad_output - i + 1
            self.grad_scale[j] = self.grad_scale[j] or nn.ScalarMulTable()
            self.grad_output[j]:copy(self.grad_scale[j]:forward(
                {self.grad_output[j], self.m.t2.data.module.output}))
            --print(self.grad_output[j]:size())
        end
    end
end

function Manager:add_l1_regularization()
    if g_opts.l1 > 0 then
        local idx = #self.args.num_args + 1
        if not self.args.soft_mem then
            idx = idx + 1
        end
        --hard termination
        if self.m.term and not self.args.soft_term then
            self.l1_penalty = self.l1_penalty or nn.Sequential():add(
                    nn.Select(2, 2)):add(
                    nn.Exp()):add(
                    nn.L1Penalty(self.args.l1))
            local out = self.l1_penalty:forward(self.m.term.data.module.output)
            self.l1_grad_output = self.l1_grad_output or 
                    torch.zeros(out:size())
            local grad = self.l1_penalty:backward(self.m.term.data.module.output, 
                    self.l1_grad_output)
            -- print(grad)
            self.grad_output[idx]:add(grad)
        end
    end
end

local FF = torch.class('FF', 'Manager')
function FF:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.p0 = nn.Identity()()
    
    local prev_r = self.m.r0
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    -- Compute CNN embeddings
    self.m.conv = self:build_conv(args, self.m.x)
    local state = nn.JoinTable(2)({self.m.conv, prev_r, prev_p})
    self.m.state = nn.Linear(nel + wdim + args.max_task, edim)(state)
    self.m.h = nn.Tanh()(self.m.state)
    local next_r, next_p = self:retrieve_goal(args, self.m.h, prev_p)
    local hr = nn.JoinTable(2)({self.m.h, next_r})
    self.m.hr = nn.LinearNB(wdim + edim, edim)(hr)
    local hr1 = nn.Narrow(2,1,args.lindim)(self.m.hr)
    local hr2 = nn.Narrow(2,1+args.lindim,edim-args.lindim)(self.m.hr)
    local hr2_nl = nn.ReLU(true)(hr2)
    self.m.s = nn.JoinTable(2)({hr1, hr2_nl})

    local input = {self.m.x, self.m.g, self.m.r0, self.m.p0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.term_act then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.s)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    table.insert(output, next_r)
    table.insert(output, next_p)

    return nn.gModule(input, output)
end

function FF:build_init_states(args)
    local states = {}
    local edim = args.edim
    table.insert(states, torch.Tensor(1, args.wdim))      -- r
    table.insert(states, torch.Tensor(1, args.max_task))  -- p
    return states
end

function FF:reset_init_state(batch_size)
    for j=1,#self.init_states do
        local size = self.init_states[j]:size()
        size[1] = batch_size
        self.init_states[j]:resize(size)
        self.init_states[j]:fill(0)
    end
    -- Set initial memory attention to 1, 0, 0, ...
    self.init_states[2]:narrow(2, 1, 1):fill(1)
end

local FFR = torch.class('FFR', 'FF')
function FFR:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.p0 = nn.Identity()()
    
    local prev_r = self.m.r0
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    -- Compute CNN embeddings
    self.m.conv = self:build_conv(args, self.m.x)
    local state = nn.JoinTable(2)({self.m.conv, prev_r, prev_p})
    self.m.state = nn.Linear(nel + wdim + args.max_task, edim)(state)
    self.m.h = nn.Tanh()(self.m.state)
    local next_r, next_p = self:retrieve_goal(args, self.m.h, prev_p)
    self.m.s = next_r
    local input = {self.m.x, self.m.g, self.m.r0, self.m.p0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.term_act then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.s)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    table.insert(output, next_r)
    table.insert(output, next_p)

    return nn.gModule(input, output)
end

local FFP = torch.class('FFP', 'FF')
function FFP:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.p0 = nn.Identity()()
    
    local prev_r = self.m.r0
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    -- Compute CNN embeddings
    self.m.conv = self:build_conv(args, self.m.x)
    local state = nn.JoinTable(2)({self.m.conv, prev_r})
    self.m.state = nn.Linear(nel + wdim, edim)(state)
    self.m.h = nn.Tanh()(self.m.state)
    local next_r, next_p = self:retrieve_goal(args, self.m.h, prev_p)
    local hr = nn.JoinTable(2)({self.m.h, next_r})
    self.m.hr = nn.LinearNB(wdim + edim, edim)(hr)
    local hr1 = nn.Narrow(2,1,args.lindim)(self.m.hr)
    local hr2 = nn.Narrow(2,1+args.lindim,edim-args.lindim)(self.m.hr)
    local hr2_nl = nn.ReLU(true)(hr2)
    self.m.s = nn.JoinTable(2)({hr1, hr2_nl})

    local input = {self.m.x, self.m.g, self.m.r0, self.m.p0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.term_act then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.s)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    table.insert(output, next_r)
    table.insert(output, next_p)

    return nn.gModule(input, output)
end

local FFPR = torch.class('FFPR', 'FF')
function FFPR:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.p0 = nn.Identity()()
    
    local prev_r = self.m.r0
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    -- Compute CNN embeddings
    self.m.conv = self:build_conv(args, self.m.x)
    local state = nn.JoinTable(2)({self.m.conv, prev_r})
    self.m.state = nn.Linear(nel + wdim, edim)(state)
    self.m.h = nn.Tanh()(self.m.state)
    local next_r, next_p = self:retrieve_goal(args, self.m.h, prev_p)
    self.m.hr = nn.LinearNB(wdim, edim)(next_r)
    local hr1 = nn.Narrow(2,1,args.lindim)(self.m.hr)
    local hr2 = nn.Narrow(2,1+args.lindim,edim-args.lindim)(self.m.hr)
    local hr2_nl = nn.ReLU(true)(hr2)
    self.m.s = nn.JoinTable(2)({hr1, hr2_nl})

    local input = {self.m.x, self.m.g, self.m.r0, self.m.p0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.term_act then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.s)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    table.insert(output, next_r)
    table.insert(output, next_p)

    return nn.gModule(input, output)
end


local FF2 = torch.class('FF2', 'FF')
function FF2:build_conv(args, input, prev_r)
    self.m.embed_r = nn.Linear(args.wdim, args.n_units[1])(prev_r)
    self.m.conv_att = nn.Sigmoid(true)(self.m.embed_r)
    local att_map = nn.Replicate(10*10, 2, 1)(self.m.conv_att)
    local att_cont = nn.Contiguous()(att_map)
    local att_3d = nn.View(args.n_units[1], 10, 10):setNumInputDims(2)(att_cont)

    self.m.conv1 = nn.SpatialConvolution(args.buf * args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(input)
    self.m.conv1_nl_pre = nn.ReLU(true)(self.m.conv1)
    self.m.conv1_nl = nn.CMulTable()({self.m.conv1_nl_pre, att_3d})
    for i=1,(#args.n_units-1) do
        if args.pool[i] then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end
    return nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
end

function FF2:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.p0 = nn.Identity()()
    
    local prev_r = self.m.r0
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    -- Compute CNN embeddings
    self.m.conv = self:build_conv(args, self.m.x, prev_r)
    self.m.state = nn.Linear(nel, edim)(self.m.conv)
    local next_r, next_p = self:retrieve_goal(args, self.m.state, prev_p)
    self.m.h = nn.Linear(wdim, edim)(next_r)
    self.m.s = nn.ReLU(true)(self.m.h)

    local input = {self.m.x, self.m.g, self.m.r0, self.m.p0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.term_act then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.s)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    table.insert(output, next_r)
    table.insert(output, next_p)

    return nn.gModule(input, output)
end

local FF3 = torch.class('FF3', 'FF2')
function FF3:retrieve_goal(args, h, p)
    local nel = 5 * 5 * args.n_units[#args.n_units]
    -- shift vector
    self.m.g_hs = nn.Linear(nel, args.edim)(h)
    local g_hs_nl = nn.ReLU(true)(self.m.g_hs)
    self.m.g_s = nn.Linear(args.edim, 2 * args.shift + 1)(g_hs_nl)
    if args.soft_mem then 
        self.m.g_prob_s = nn.SoftMax()(self.m.g_s)
    else
        self.m.g_shift = nn.LogSoftMax()(self.m.g_s)
        self.m.shift_act = nn.Multinomial()(self.m.g_shift)
        self.m.g_prob_s = nn.OneHot(2 * args.shift + 1)(self.m.shift_act)
    end
    local w_tilde = nn.CircularConvolution(){p, self.m.g_prob_s}
    --local w_pow = nn.PowTable(){w_tilde, gamma}
    self.m.g_prob = nn.Normalize(1)(w_tilde)
    local prob3d = nn.View(1, -1):setNumInputDims(1)(self.m.g_prob)
    local MMbout = nn.MM(false, false)
    local out3d = MMbout({prob3d, self.m.g_val})
    local out2d = nn.View(-1):setNumInputDims(1)(out3d)
    return out2d, self.m.g_prob
end

function FF3:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.p0 = nn.Identity()()
    
    local prev_r = self.m.r0
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    -- Compute CNN embeddings
    self.m.conv = self:build_conv(args, self.m.x, prev_r)
    local next_r, next_p = self:retrieve_goal(args, self.m.conv, prev_p)
    self.m.s = next_r

    local input = {self.m.x, self.m.g, self.m.r0, self.m.p0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.term_act then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.s)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    table.insert(output, next_r)
    table.insert(output, next_p)

    return nn.gModule(input, output)
end

local FF4 = torch.class('FF4', 'FF')
function FF4:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.p0 = nn.Identity()()
    
    local prev_r = self.m.r0
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    self.m.embed_r = nn.Linear(args.wdim, args.buf * args.channels)(prev_r)
    self.m.conv_att = nn.Sigmoid(true)(self.m.embed_r)
    local att_map = nn.Replicate(10*10, 2, 1)(self.m.conv_att)
    local att_cont = nn.Contiguous()(att_map)
    local att_3d = nn.View(args.buf * args.channels, 10, 10):setNumInputDims(2)(att_cont)
    self.m.masked_x = nn.CMulTable()({self.m.x, att_3d})

    -- Compute CNN embeddings
    self.m.conv = self:build_conv(args, self.m.masked_x)
    self.m.state = nn.Linear(nel, edim)(self.m.conv)
    self.m.h = nn.Tanh()(self.m.state)
    local next_r, next_p = self:retrieve_goal(args, self.m.h, prev_p)
    self.m.s = next_r

    local input = {self.m.x, self.m.g, self.m.r0, self.m.p0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.term_act then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.s)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    table.insert(output, next_r)
    table.insert(output, next_p)

    return nn.gModule(input, output)
end

local FF5 = torch.class('FF5', 'FF')
function FF5:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.p0 = nn.Identity()()
    
    local prev_r = self.m.r0
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    self.m.w_r = nn.Linear(args.wdim, args.buf * args.channels)(prev_r)
    self.m.w_r3d = nn.View(-1, args.buf * args.channels, 1, 1)(self.m.w_r)
    self.m.b_r = nn.Linear(args.wdim, args.buf * args.channels)(prev_r)
    -- Compute CNN embeddings
    self.m.conv1_1 = nn.SpatialConvolution(args.buf * args.channels, 
                args.buf * args.channels, 1, 1, 1, 1, 0, 0)(self.m.x)
    self.m.conv1_2 = nn.FactoredSpatialConvolution(args.buf * args.channels, 
                        1, 1, 1, 1, 0, 0)({self.m.conv1_1, self.m.w_r3d, self.m.b_r})
    self.m.conv1 = nn.SpatialConvolution(args.buf * args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.conv1_2)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    for i=1,(#args.n_units-1) do
        if args.pool[i] then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end
    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.h2 = nn.JoinTable(2)({self.m.conv, prev_r})
    self.m.state = nn.Linear(nel + wdim, edim)(self.m.h2)
    self.m.h = nn.Tanh()(self.m.state)
    local next_r, next_p = self:retrieve_goal(args, self.m.h, prev_p)
    self.m.s = next_r

    local input = {self.m.x, self.m.g, self.m.r0, self.m.p0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.term_act then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.s)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    table.insert(output, next_r)
    table.insert(output, next_p)

    return nn.gModule(input, output)
end

local FF10 = torch.class('FF10', 'FF')
function FF10:build_conv(args, input, nInput)
    self.m.conv1 = nn.SpatialConvolution(nInput, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(input)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    for i=1,(#args.n_units-1) do
        if args.pool[i] then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end
    return nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
end

function FF10:retrieve_goal(args, h, p, hdim)
    -- shift vector
    self.m.g_hs = nn.Linear(hdim, args.edim)(h)
    local g_hs_nl = nn.ReLU(true)(self.m.g_hs)
    self.m.g_s = nn.Linear(args.edim, 2 * args.shift + 1)(g_hs_nl)
    if args.soft_mem then 
        self.m.g_prob_s = nn.SoftMax()(self.m.g_s)
    else
        self.m.g_shift = nn.LogSoftMax()(self.m.g_s)
        self.m.shift_act = nn.Multinomial()(self.m.g_shift)
        self.m.g_prob_s = nn.OneHot(2 * args.shift + 1)(self.m.shift_act)
    end
    local w_tilde = nn.CircularConvolution(){p, self.m.g_prob_s}
    --local w_pow = nn.PowTable(){w_tilde, gamma}
    self.m.g_prob = nn.Normalize(1)(w_tilde)
    local prob3d = nn.View(1, -1):setNumInputDims(1)(self.m.g_prob)
    local MMbout = nn.MM(false, false)
    local out3d = MMbout({prob3d, self.m.g_val})
    local out2d = nn.View(-1):setNumInputDims(1)(out3d)
    return out2d, self.m.g_prob
end

function FF10:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.p0 = nn.Identity()()
    
    local prev_r = self.m.r0
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    -- Compute CNN embeddings
    self.m.mask_r = nn.Linear(args.wdim, args.ch * args.buf * args.channels)(prev_r)
    self.m.mask_r_nl = nn.Sigmoid(true)(self.m.mask_r)
    self.m.mask_h2d = nn.View(args.ch, args.buf * args.channels):setNumInputDims(1)(self.m.mask_r_nl)
    local x_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(MMconv({self.m.mask_h2d, x_2d}))
    
    self.m.conv = self:build_conv(args, self.m.x_masked, args.ch)
    self.m.state = nn.Linear(nel, edim)(self.m.conv)
    self.m.h = nn.Tanh()(self.m.state)
    local next_r, next_p = self:retrieve_goal(args, self.m.h, prev_p, edim)
    local hr = nn.JoinTable(2)({self.m.h, next_r})
    self.m.hr = nn.LinearNB(wdim + edim, edim)(hr)
    local hr1 = nn.Narrow(2,1,args.lindim)(self.m.hr)
    local hr2 = nn.Narrow(2,1+args.lindim,edim-args.lindim)(self.m.hr)
    local hr2_nl = nn.ReLU(true)(hr2)
    self.m.s = nn.JoinTable(2)({hr1, hr2_nl})

    local input = {self.m.x, self.m.g, self.m.r0, self.m.p0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.term_act then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.s)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    table.insert(output, next_r)
    table.insert(output, next_p)

    return nn.gModule(input, output)
end

local FF11 = torch.class('FF11', 'FF10')
function FF11:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.p0 = nn.Identity()()
    
    local prev_r = self.m.r0
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    -- Compute CNN embeddings
    self.m.mask_r = nn.Linear(args.wdim, args.ch * args.buf * args.channels)(prev_r)
    self.m.mask_r_nl = nn.Sigmoid(true)(self.m.mask_r)
    self.m.mask_h2d = nn.View(args.ch, args.buf * args.channels):setNumInputDims(1)(self.m.mask_r_nl)
    local x_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(MMconv({self.m.mask_h2d, x_2d}))
    
    self.m.conv = self:build_conv(args, self.m.x_masked, args.ch)
    local state = nn.JoinTable(2)({self.m.conv, prev_r, prev_p})
    self.m.state = nn.Linear(nel + wdim + args.max_task, edim)(state)
    self.m.h = nn.Tanh()(self.m.state)
    local next_r, next_p = self:retrieve_goal(args, self.m.h, prev_p, edim)
    local hr = nn.JoinTable(2)({self.m.h, next_r})
    self.m.hr = nn.LinearNB(wdim + edim, edim)(hr)
    local hr1 = nn.Narrow(2,1,args.lindim)(self.m.hr)
    local hr2 = nn.Narrow(2,1+args.lindim,edim-args.lindim)(self.m.hr)
    local hr2_nl = nn.ReLU(true)(hr2)
    self.m.s = nn.JoinTable(2)({hr1, hr2_nl})

    local input = {self.m.x, self.m.g, self.m.r0, self.m.p0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.term_act then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.s)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    table.insert(output, next_r)
    table.insert(output, next_p)

    return nn.gModule(input, output)
end

local FF12 = torch.class('FF12', 'FF10')
function FF12:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.p0 = nn.Identity()()
    
    local prev_r = self.m.r0
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    -- Compute CNN embeddings
    self.m.mask_r = nn.Linear(wdim, edim)(prev_r)
    self.m.mask_r_nl = nn.ReLU(true)(self.m.mask_r)
    self.m.mask_r2 = nn.Linear(edim, args.ch * args.buf * args.channels)(self.m.mask_r_nl)
    self.m.mask_h2d = nn.View(args.ch, args.buf * args.channels):setNumInputDims(1)(self.m.mask_r2)
    local x_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(MMconv({self.m.mask_h2d, x_2d}))
    
    self.m.conv = self:build_conv(args, self.m.x_masked, args.ch)
    local state = nn.JoinTable(2)({self.m.conv, prev_r})
    self.m.state = nn.Linear(nel + wdim, edim)(state)
    self.m.h = nn.Tanh()(self.m.state)
    local next_r, next_p = self:retrieve_goal(args, self.m.h, prev_p, edim)
    local hr = nn.JoinTable(2)({self.m.h, next_r})
    self.m.hr = nn.LinearNB(wdim + edim, edim)(hr)
    local hr1 = nn.Narrow(2,1,args.lindim)(self.m.hr)
    local hr2 = nn.Narrow(2,1+args.lindim,edim-args.lindim)(self.m.hr)
    local hr2_nl = nn.ReLU(true)(hr2)
    self.m.s = nn.JoinTable(2)({hr1, hr2_nl})

    local input = {self.m.x, self.m.g, self.m.r0, self.m.p0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.term_act then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.s)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    table.insert(output, next_r)
    table.insert(output, next_p)

    return nn.gModule(input, output)
end


local FF11B = torch.class('FF11B', 'FF11')

function FF11B:add_entropy_grad(rho)
    self.entropy = self.entropy or {}
    for i=1,#self.n_actions do
        self.entropy[i] = self.entropy[i] or nn.Entropy()
    end
    local idx = 1
    --local entropy = self.entropy[i]:forward(self.m["a" .. i].data.module.output)
    local g = self.entropy[1]:backward(self.m.a.data.module.output, 1)
    --print("Entropy", entropy, "GradNorm", g:norm(), self.grad_output[idx]:norm())
    self.grad_output[idx]:add(g:mul(-rho))
    if not self.args.soft_mem then
        idx = idx + 1
        local g = self.entropy[idx]:backward(self.m.g_shift.data.module.output, 1)
        self.grad_output[idx]:add(g:mul(-rho))
    end
end

function FF11B:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.p0 = nn.Identity()()
    
    local prev_r = self.m.r0
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    -- Compute CNN embeddings
    self.m.mask_r = nn.Linear(args.wdim, args.edim)(prev_r)
    self.m.mask_r_nl = nn.ReLU(true)(self.m.mask_r)
    self.m.mask_r2 = nn.Linear(args.edim, args.ch * args.buf * args.channels)(self.m.mask_r_nl)
    self.m.mask_fc = nn.Linear(args.edim, args.edim)(self.m.mask_r_nl)
    self.m.mask_h2d = nn.View(args.ch, args.buf * args.channels):setNumInputDims(1)(self.m.mask_r2)
    local x_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(MMconv({self.m.mask_h2d, x_2d}))
    
    self.m.conv = self:build_conv(args, self.m.x_masked, args.ch)
    self.m.fc1 = nn.Linear(nel, args.edim)(self.m.conv)
    self.m.mul_h = nn.CMulTable()({self.m.mask_fc, self.m.fc1})
    self.m.state = nn.Linear(edim, edim)(self.m.mul_h)
    self.m.h = nn.ReLU(true)(self.m.state)
    local next_r, next_p = self:retrieve_goal(args, self.m.h, prev_p, edim)
    local hr = nn.JoinTable(2)({self.m.h, next_r})
    self.m.hr = nn.Linear(wdim + edim, edim)(hr)
    self.m.s = nn.ReLU(true)(self.m.hr)

    self.m.fc = nn.Linear(edim, args.n_actions)(self.m.s)
    self.m.a = nn.LogSoftMax()(self.m.fc)
    
    self.n_actions = {args.n_actions}
    local input = {self.m.x, self.m.g, self.m.r0, self.m.p0}
    local output = {self.m.a}
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.term_act then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.s)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    table.insert(output, self.m.b)
    table.insert(output, next_r)
    table.insert(output, next_p)

    return nn.gModule(input, output)
end

local FF20 = torch.class('FF20', 'FF10')

function FF20:build_init_states(args)
    return {torch.Tensor(1, args.max_task)}
end

function FF20:reset_init_state(batch_size)
    for j=1,#self.init_states do
        local size = self.init_states[j]:size()
        size[1] = batch_size
        self.init_states[j]:resize(size)
        self.init_states[j]:fill(0)
    end
    -- Set initial memory attention to 1, 0, 0, ...
    --self.init_states[1]:narrow(2, 1, 1):fill(1)
    self.init_states[1]:narrow(2, self.init_states[1]:size(2), 1):fill(1)
end


function FF20:build_model(args)
    args.pass_act = true
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.p0 = nn.Identity()()
    self.m.arg1 = nn.Identity()()
    self.m.arg2 = nn.Identity()()
    
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    -- Compute CNN embeddings
    self.m.arg_h1 = nn.Linear(args.num_args[1], args.edim)(self.m.arg1)
    self.m.mask_r = nn.Linear(args.num_args[2], args.ch * args.buf * args.channels)(self.m.arg2)
    self.m.mask_r_nl = nn.Sigmoid(true)(self.m.mask_r)
    self.m.mask_h2d = nn.View(args.ch, args.buf * args.channels):setNumInputDims(1)(self.m.mask_r_nl)
    local x_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(MMconv({self.m.mask_h2d, x_2d}))
    
    self.m.conv = self:build_conv(args, self.m.x_masked, args.ch)
    self.m.fc1 = nn.Linear(nel, args.edim)(self.m.conv)
    self.m.mul_h = nn.CMulTable()({self.m.arg_h1, self.m.fc1})
    self.m.state = nn.Linear(edim, edim)(self.m.mul_h)
    self.m.h = nn.Tanh()(self.m.state)
    local next_r, next_p = self:retrieve_goal(args, self.m.h, prev_p, edim)
    local hr = nn.JoinTable(2)({self.m.h, next_r})
    self.m.hr = nn.LinearNB(wdim + edim, edim)(hr)
    local hr1 = nn.Narrow(2,1,args.lindim)(self.m.hr)
    local hr2 = nn.Narrow(2,1+args.lindim,edim-args.lindim)(self.m.hr)
    local hr2_nl = nn.ReLU(true)(hr2)
    self.m.s = nn.JoinTable(2)({hr1, hr2_nl})

    local input = {self.m.x, self.m.g, self.m.p0, self.m.arg1, self.m.arg2}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.term_act then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.s)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    table.insert(output, next_p)

    return nn.gModule(input, output)
end

local FF21 = torch.class('FF21', 'FF20')

function FF21:build_model(args)
    args.pass_act = true
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.p0 = nn.Identity()()
    self.m.arg1 = nn.Identity()()
    self.m.arg2 = nn.Identity()()
    
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    -- Compute CNN embeddings
    self.m.arg_h1 = nn.Linear(args.num_args[1], args.edim)(self.m.arg1)
    self.m.mask_r = nn.Linear(args.num_args[2], args.edim)(self.m.arg2)
    self.m.mask_r_nl = nn.ReLU(true)(self.m.mask_r)
    self.m.mask_r2 = nn.Linear(args.edim, args.ch * args.buf * args.channels)(self.m.mask_r_nl)
    self.m.mask_h2d = nn.View(args.ch, args.buf * args.channels):setNumInputDims(1)(self.m.mask_r2)
    local x_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(MMconv({self.m.mask_h2d, x_2d}))
    
    self.m.conv = self:build_conv(args, self.m.x_masked, args.ch)
    self.m.fc1 = nn.Linear(nel, args.edim)(self.m.conv)
    self.m.mul_h = nn.CMulTable()({self.m.arg_h1, self.m.fc1})
    self.m.state = nn.Linear(edim, edim)(self.m.mul_h)
    self.m.h = nn.Tanh()(self.m.state)
    local next_r, next_p = self:retrieve_goal(args, self.m.h, prev_p, edim)
    local hr = nn.JoinTable(2)({self.m.h, next_r})
    self.m.hr = nn.LinearNB(wdim + edim, edim)(hr)
    local hr1 = nn.Narrow(2,1,args.lindim)(self.m.hr)
    local hr2 = nn.Narrow(2,1+args.lindim,edim-args.lindim)(self.m.hr)
    local hr2_nl = nn.ReLU(true)(hr2)
    self.m.s = nn.JoinTable(2)({hr1, hr2_nl})

    local input = {self.m.x, self.m.g, self.m.p0, self.m.arg1, self.m.arg2}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.term_act then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.s)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    table.insert(output, next_p)

    return nn.gModule(input, output)
end


local FF30 = torch.class('FF30', 'FF20')
function FF30:conv_filter(args, z, zdim)
    self.m.mask_r = nn.Linear(zdim, args.ch * args.buf * args.channels)(z)
    return nn.Sigmoid()(self.m.mask_r)
end
function FF30:build_model(args)
    args.pass_act = true
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.p0 = nn.Identity()()
    self.m.arg1 = nn.Identity()()
    self.m.arg2 = nn.Identity()()
    
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    -- Compute CNN embeddings
    self.m.arg_h1 = nn.Linear(args.num_args[1], args.edim)(self.m.arg1)
    local conv_filter = self:conv_filter(args, self.m.arg2, args.num_args[2])
    self.m.mask_h2d = nn.View(args.ch, args.buf * args.channels):setNumInputDims(1)(conv_filter)
    local x_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(MMconv({self.m.mask_h2d, x_2d}))
    
    self.m.conv = self:build_conv(args, self.m.x_masked, args.ch)
    self.m.fc1 = nn.Linear(nel, args.edim)(self.m.conv)
    self.m.mul_h = nn.CMulTable()({self.m.arg_h1, self.m.fc1})
    self.m.state = nn.Linear(edim, edim)(self.m.mul_h)
    self.m.h = nn.ReLU(true)(self.m.state)
    
    local next_r, next_p = self:retrieve_goal(args, self.m.h, prev_p, edim)
    local hr = nn.JoinTable(2)({self.m.h, next_r})
    self.m.hr = nn.Linear(wdim + edim, edim)(hr)
    self.m.s = nn.ReLU(true)(self.m.hr)

    local input = {self.m.x, self.m.g, self.m.p0, self.m.arg1, self.m.arg2}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    table.insert(output, self.m.b)
    table.insert(output, next_p)

    return nn.gModule(input, output)
end

local FF31 = torch.class('FF31', 'FF30')
function FF31:conv_filter(args, z, zdim)
    self.m.mask_r = nn.Linear(zdim, args.edim)(z)
    self.m.mask_r_nl = nn.ReLU(true)(self.m.mask_r)
    self.m.mask_conv = nn.Linear(args.edim, args.ch * args.buf * args.channels)(self.m.mask_r_nl)
    return self.m.mask_conv
end

local FFTerm = torch.class('FFTerm', 'FF20')

function FFTerm:build_init_states(args)
    local states = {}
    table.insert(states, torch.Tensor(1, args.max_task))
    if args.pass_r then
        table.insert(states, torch.Tensor(1, args.wdim))
    end
    return states
end

function FFTerm:reset_init_state(batch_size)
    for j=1,#self.init_states do
        local size = self.init_states[j]:size()
        size[1] = batch_size
        self.init_states[j]:resize(size)
        self.init_states[j]:fill(0)
    end
    -- Set initial memory attention to 1, 0, 0, ...
    self.init_states[1]:narrow(2, self.init_states[1]:size(2), 1):fill(1)
    -- self.init_states[1]:narrow(2, 1, 1):fill(1)
end

function FFTerm:fill_internal_actions()
    if not self.args.soft_mem then
        table.insert(self.actions, self.m.shift_act.data.module.output:long())
    end
    if not self.args.soft_term then
        table.insert(self.actions, self.m.term_act.data.module.output:long())
    end
end

function FFTerm:conv_filter(args, z, zdim)
    self.m.mask_r = nn.Linear(zdim, args.ch * args.buf * args.channels)(z)
    return nn.Sigmoid()(self.m.mask_r)
end

function FFTerm:merge_pi(args, t1, t2, prev_arg, p)
    assert(args.merge_pi ~= nil)
    if args.merge_pi then
        local normalized_p = nn.SoftMax()(p)
        local prob = nn.CAddTable()({
            nn.ScalarMulTable()({prev_arg, self.m.t1}),
            nn.ScalarMulTable()({normalized_p, self.m.t2})
        })
        local eps = 1e-8
        local clipped_prob = nn.Clamp(eps, 1)(prob)
        return nn.Log()(clipped_prob)
    else
        local score = nn.CAddTable()({
            nn.ScalarMulTable()({nn.MulConstant(15)(prev_arg), self.m.t1}),
            nn.ScalarMulTable()({p, self.m.t2})
        })
        return nn.LogSoftMax()(score)
    end
end

function FFTerm:build_model(args)
    args.pass_act = true

    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.p0 = nn.Identity()()
    if args.pass_r then
        self.m.r0 = nn.Identity()()
    end
    for i=1,#args.num_args do
        self.m["arg" .. i] = nn.Identity()()
    end 
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    -- Compute CNN embeddings
    self.m.arg_h1_lin = nn.Linear(args.num_args[1], args.edim)(self.m.arg1)
    local conv_filter = self:conv_filter(args, self.m.arg2, args.num_args[2])
    self.m.mask_h2d = nn.View(args.ch, args.buf * args.channels):setNumInputDims(1)(conv_filter)
    if args.l1_arg1 > 0 then
        self.m.mask_h2d = nn.L1Penalty(args.l1_arg1)(self.m.mask_h2d)
        --print("l1_arg1: " .. args.l1_arg1)
    end
    if args.l1_arg2 > 0 then
        self.m.arg_h1 = nn.L1Penalty(args.l1_arg2)(self.m.arg_h1_lin)
        --print("l1_arg2: " .. args.l1_arg2)
    else
        self.m.arg_h1 = self.m.arg_h1_lin
    end
    local x_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(
        MMconv({self.m.mask_h2d, x_2d}))
    
    self.m.conv = self:build_conv(args, self.m.x_masked, args.ch)
    self.m.fc1 = nn.Linear(nel, args.edim)(self.m.conv)
    self.m.mul_h = nn.CMulTable()({self.m.arg_h1, self.m.fc1})
    self.m.state = nn.Linear(edim, edim)(self.m.mul_h)
    self.m.h = nn.ReLU(true)(self.m.state)

    self.m.term_fc = nn.Linear(edim, 2)(self.m.h)
    if args.soft_term then
        self.m.term_prob = nn.SoftMax()(self.m.term_fc)
        if args.soft_rho then 
            self.m.term_prob = nn.EntropyPenalty(0, false)(self.m.term_prob)
            table.insert(self.entropy_list, self.m.term_prob)
        end
    else
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        self.m.term_act = nn.Multinomial()(self.m.term)
        self.m.term_prob = nn.OneHot(2)(self.m.term_act)
    end
    self.m.t1 = nn.Narrow(2,1,1)(self.m.term_prob)
    self.m.t2 = nn.Narrow(2,2,1)(self.m.term_prob)
    if args.l1 > 0 and args.soft_term then
        self.m.t2 = nn.L1Penalty(args.l1)(self.m.t2)
    end

    local r, p
    if args.pass_r then
        r, p = self:retrieve_goal(args, self.m.r0, prev_p, wdim)
        self.m.r_embed = nn.Linear(wdim, edim)(r)
        self.m.s = nn.ReLU(true)(self.m.r_embed)
    else
        r, p = self:retrieve_goal(args, self.m.h, prev_p, edim)
        local hr = nn.JoinTable(2)({self.m.h, r})
        --[[
        self.m.hr = nn.Linear(wdim + edim, edim)(hr)
        self.m.s = nn.ReLU(true)(self.m.hr)
        --]]
        self.m.hr = nn.LinearNB(wdim + edim, edim)(hr)
        local hr1 = nn.Narrow(2,1,args.lindim)(self.m.hr)
        local hr2 = nn.Narrow(2,1+args.lindim,edim-args.lindim)(self.m.hr)
        local hr2_nl = nn.ReLU(true)(hr2)
        self.m.s = nn.JoinTable(2)({hr1, hr2_nl})
    end

    self.m.g_prob = nn.CAddTable()({
        nn.ScalarMulTable()({prev_p, self.m.t1}),
        nn.ScalarMulTable()({p,self.m.t2})
      })

    local input = {self.m.x, self.m.g, self.m.p0}
    if args.pass_r then
        table.insert(input, self.m.r0)
    end
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        table.insert(input, self.m["arg" .. i])
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = self:merge_pi(args, self.m.t1, self.m.t2, 
                self.m["arg" .. i], self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i]) 
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if not args.soft_term then
        table.insert(self.n_actions, 2)
        table.insert(output, self.m.term) 
    end
    
    table.insert(output, self.m.b)
    table.insert(output, self.m.g_prob)
    if args.pass_r then
        local next_r = nn.CAddTable()({
            nn.ScalarMulTable()({self.m.r0, self.m.t1}),
            nn.ScalarMulTable()({r,self.m.t2})
        }) 
        table.insert(output, next_r)
    end

    return nn.gModule(input, output)
end

local FFTerm2 = torch.class('FFTerm2', 'FFTerm')
function FFTerm2:__init(args)
    args.merge_pi = false
    FFTerm.__init(self, args)
end

function FFTerm2:conv_filter(args, z, zdim)
    self.m.mask_r = nn.Linear(zdim, args.edim)(z)
    self.m.mask_r_nl = nn.ReLU(true)(self.m.mask_r)
    self.m.mask_conv = nn.Linear(args.edim, args.ch * args.buf * args.channels)(self.m.mask_r_nl)
    return self.m.mask_conv
end

local FFTerm3 = torch.class('FFTerm3', 'FFTerm')
function FFTerm3:__init(args)
    args.merge_pi = true
    FFTerm.__init(self, args)
end
function FFTerm3:conv_filter(args, z, zdim)
    self.m.mask_r = nn.Linear(zdim, args.edim)(z)
    self.m.mask_r_nl = nn.ReLU(true)(self.m.mask_r)
    self.m.mask_conv = nn.Linear(args.edim, args.ch * args.buf * args.channels)(self.m.mask_r_nl)
    return self.m.mask_conv
end

local FFTerm11 = torch.class('FFTerm11', 'FFTerm')
function FFTerm11:__init(args)
    args.pass_r = true
    FFTerm.__init(self, args)
end

local FFTerm12 = torch.class('FFTerm12', 'FFTerm2')
function FFTerm12:__init(args)
    args.pass_r = true
    FFTerm2.__init(self, args)
end

local Term = torch.class('Term', 'FFTerm')

function Term:__init(args)
    args.merge_pi = false
    FFTerm.__init(self, args)
end

function Term:build_model(args)
    args.pass_act = true

    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.p0 = nn.Identity()()
    if args.pass_r then
        self.m.r0 = nn.Identity()()
    end
    for i=1,#args.num_args do
        self.m["arg" .. i] = nn.Identity()()
    end 
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    -- Compute CNN embeddings
    self.m.arg_h1 = nn.Linear(args.num_args[1], args.edim)(self.m.arg1)
    self.m.arg_h2 = nn.Linear(args.num_args[2], args.edim)(self.m.arg2)
    self.m.arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    
    self.m.arg_conv = nn.Linear(args.edim, args.ch * args.buf * args.channels)(self.m.arg_hh)
    self.m.mask_h2d = nn.View(args.ch, args.buf * args.channels):setNumInputDims(1)(self.m.arg_conv)
    if args.drop and args.drop > 0 then
        self.m.mask_h2d = nn.Dropout(args.drop)(self.m.mask_h2d)
    end
    local x_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(
        MMconv({self.m.mask_h2d, x_2d}))
    
    self.m.conv = self:build_conv(args, self.m.x_masked, args.ch)
    if args.goal_fc then 
        self.m.fc1 = nn.Linear(nel, args.edim)(self.m.conv)
        self.m.arg_fc = nn.Linear(args.edim, args.edim)(self.m.arg_hh)
        self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc1})
        if args.drop and args.drop > 0 then
            self.m.mul_h = nn.Dropout(args.drop)(self.m.mul_h)
        end
        self.m.state = nn.Linear(edim, edim)(self.m.mul_h)
    else
        self.m.state = nn.Linear(nel, edim)(self.m.conv)
    end
    self.m.h = nn.ReLU(true)(self.m.state)

    self.m.term_fc = nn.Linear(edim, 2)(self.m.h)
    if args.soft_term then
        self.m.term_prob = nn.SoftMax()(self.m.term_fc)
        if args.soft_rho then 
            self.m.term_prob = nn.EntropyPenalty(0, false)(self.m.term_prob)
            table.insert(self.entropy_list, self.m.term_prob)
        end
    else
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        self.m.term_act = nn.Multinomial()(self.m.term)
        self.m.term_prob = nn.OneHot(2)(self.m.term_act)
    end
    self.m.t1 = nn.Narrow(2,1,1)(self.m.term_prob)
    self.m.t2 = nn.Narrow(2,2,1)(self.m.term_prob)
    if args.l1 > 0 and args.soft_term then
        self.m.t2 = nn.L1Penalty(args.l1)(self.m.t2)
    end

    local r, p
    if args.pass_r then
        r, p = self:retrieve_goal(args, self.m.r0, prev_p, wdim)
        self.m.r_embed = nn.Linear(wdim, edim)(r)
        self.m.s = nn.ReLU(true)(self.m.r_embed)
    else
        r, p = self:retrieve_goal(args, self.m.h, prev_p, edim)
        local hr = nn.JoinTable(2)({self.m.h, r})
        --[[
        self.m.hr = nn.Linear(wdim + edim, edim)(hr)
        self.m.s = nn.ReLU(true)(self.m.hr)
        --]]
        self.m.hr = nn.LinearNB(wdim + edim, edim)(hr)
        local hr1 = nn.Narrow(2,1,args.lindim)(self.m.hr)
        local hr2 = nn.Narrow(2,1+args.lindim,edim-args.lindim)(self.m.hr)
        local hr2_nl = nn.ReLU(true)(hr2)
        self.m.s = nn.JoinTable(2)({hr1, hr2_nl})
    end

    self.m.g_prob = nn.CAddTable()({
        nn.ScalarMulTable()({prev_p, self.m.t1}),
        nn.ScalarMulTable()({p,self.m.t2})
      })

    local input = {self.m.x, self.m.g, self.m.p0}
    if args.pass_r then
        table.insert(input, self.m.r0)
    end
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        table.insert(input, self.m["arg" .. i])
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = self:merge_pi(args, self.m.t1, self.m.t2, 
                self.m["arg" .. i], self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i]) 
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if not args.soft_term then
        table.insert(self.n_actions, 2)
        table.insert(output, self.m.term) 
    end
    
    table.insert(output, self.m.b)
    table.insert(output, self.m.g_prob)
    if args.pass_r then
        local next_r = nn.CAddTable()({
            nn.ScalarMulTable()({self.m.r0, self.m.t1}),
            nn.ScalarMulTable()({r,self.m.t2})
        }) 
        table.insert(output, next_r)
    end

    return nn.gModule(input, output)
end

function Term:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.Linear(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.Linear(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():
            add(nn.OneHot(self.args.num_args[1])):
            add(nn.OneHot(self.args.num_args[2])))
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))

        if not self.args.analogy_on_embed then
            local arg_conv = nn.Linear(self.args.edim, 
                self.args.ch * self.args.buf * self.args.channels):share(
                self.m.arg_conv.data.module, 'weight','bias','gradWeight','gradBias')
            module:add(arg_conv)
        end

        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)

    if self.args.goal_fc then
        local analogy2 = {}
        for i=1, 4 do
            local module = nn.Sequential():add(nn.SplitTable(1, 1))
            local arg_h1 = nn.Linear(self.args.num_args[1], self.args.edim):share(
                self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
            local arg_h2 = nn.Linear(self.args.num_args[2], self.args.edim):share(
                self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
            module:add(nn.ParallelTable():
                add(nn.OneHot(self.args.num_args[1])):
                add(nn.OneHot(self.args.num_args[2])))
            module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
            module:add(nn.CMulTable())
            module:add(nn.ReLU(true))

            if not self.args.analogy_on_embed then
                local arg_fc = nn.Linear(self.args.edim, self.args.edim):share(
                    self.m.arg_fc.data.module, 'weight','bias','gradWeight','gradBias')
                module:add(arg_fc)
            end
            table.insert(analogy2, module)
        end
        table.insert(self.analogy_list, analogy2)
    end
    return self.analogy_list
end

local Term2 = torch.class('Term2', 'Term')
function Term2:__init(args)
    args.goal_fc = true
    Term.__init(self, args)
end

local Term3 = torch.class('Term3', 'Term')
function Term3:__init(args)
    args.analogy_on_embed = true
    Term.__init(self, args)
end

local Term4 = torch.class('Term4', 'Term')
function Term4:__init(args)
    args.goal_fc = true
    args.analogy_on_embed = true
    Term.__init(self, args)
end

local Term5 = torch.class('Term5', 'Term')
function Term5:build_model(args)
    args.pass_act = true
    args.pass_r = true

    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.p0 = nn.Identity()()
    self.m.r0 = nn.Identity()()
    for i=1,#args.num_args do
        self.m["arg" .. i] = nn.Identity()()
    end 
    local prev_p = self.m.p0
    local prev_r = self.m.r0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    -- Compute CNN embeddings
    self.m.arg_h1 = nn.Linear(args.num_args[1], args.edim)(self.m.arg1)
    self.m.arg_h2 = nn.Linear(args.num_args[2], args.edim)(self.m.arg2)
    self.m.arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))

    local gr = nn.JoinTable(2)({self.m.arg_hh, prev_r})
    self.m.arg_conv = nn.Linear(2 * args.edim, args.ch * args.buf * args.channels)(gr)
    self.m.mask_h2d = nn.View(args.ch, args.buf * args.channels):setNumInputDims(1)(self.m.arg_conv)
    local x_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(
        MMconv({self.m.mask_h2d, x_2d}))
    
    self.m.conv = self:build_conv(args, self.m.x_masked, args.ch)
    self.m.fc1 = nn.Linear(nel, args.edim)(self.m.conv)
    self.m.arg_fc = nn.Linear(2 * args.edim, args.edim)(gr)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc1})
    self.m.state = nn.Linear(edim, edim)(self.m.mul_h)
    self.m.h = nn.ReLU(true)(self.m.state)

    self.m.term_fc = nn.Linear(edim, 2)(self.m.h)
    if args.soft_term then
        self.m.term_prob = nn.SoftMax()(self.m.term_fc)
        if args.soft_rho then 
            self.m.term_prob = nn.EntropyPenalty(0, false)(self.m.term_prob)
            table.insert(self.entropy_list, self.m.term_prob)
        end
    else
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        self.m.term_act = nn.Multinomial()(self.m.term)
        self.m.term_prob = nn.OneHot(2)(self.m.term_act)
    end
    self.m.t1 = nn.Narrow(2,1,1)(self.m.term_prob)
    self.m.t2 = nn.Narrow(2,2,1)(self.m.term_prob)
    if args.l1 > 0 and args.soft_term then
        self.m.t2 = nn.L1Penalty(args.l1)(self.m.t2)
    end

    local r, p = self:retrieve_goal(args, self.m.h, prev_p, edim)
    local hr = nn.JoinTable(2)({self.m.h, r})
    --[[
    self.m.hr = nn.Linear(wdim + edim, edim)(hr)
    self.m.s = nn.ReLU(true)(self.m.hr)
    --]]
    self.m.hr = nn.LinearNB(wdim + edim, edim)(hr)
    local hr1 = nn.Narrow(2,1,args.lindim)(self.m.hr)
    local hr2 = nn.Narrow(2,1+args.lindim,edim-args.lindim)(self.m.hr)
    local hr2_nl = nn.ReLU(true)(hr2)
    self.m.s = nn.JoinTable(2)({hr1, hr2_nl})

    self.m.g_prob = nn.CAddTable()({
        nn.ScalarMulTable()({prev_p, self.m.t1}),
        nn.ScalarMulTable()({p,self.m.t2})
      })

    local input = {self.m.x, self.m.g, self.m.p0, self.m.r0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        table.insert(input, self.m["arg" .. i])
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = self:merge_pi(args, self.m.t1, self.m.t2, 
                self.m["arg" .. i], self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i]) 
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if not args.soft_term then
        table.insert(self.n_actions, 2)
        table.insert(output, self.m.term) 
    end
    
    table.insert(output, self.m.b)
    table.insert(output, self.m.g_prob)
    local next_r = nn.CAddTable()({
        nn.ScalarMulTable()({self.m.r0, self.m.t1}),
        nn.ScalarMulTable()({r,self.m.t2})
    }) 
    table.insert(output, next_r)

    return nn.gModule(input, output)
end


local FF100 = torch.class('FF100', 'FF20')
function FF100:build_model(args)
    args.pass_act = true
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.p0 = nn.Identity()()
    self.m.arg1 = nn.Identity()()
    self.m.arg2 = nn.Identity()()
    
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    -- Compute CNN embeddings
    self.m.arg_h1 = nn.Linear(args.num_args[1], args.edim)(self.m.arg1)
    self.m.mask_r = nn.Linear(args.num_args[2], args.edim)(self.m.arg2)
    self.m.mask_r_nl = nn.ReLU(true)(self.m.mask_r)
    self.m.mask_r2 = nn.Linear(args.edim, args.ch * args.buf * args.channels)(self.m.mask_r_nl)
    self.m.mask_h2d = nn.View(args.ch, args.buf * args.channels):setNumInputDims(1)(self.m.mask_r2)
    local x_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(MMconv({self.m.mask_h2d, x_2d}))
    
    self.m.conv = self:build_conv(args, self.m.x_masked, args.ch)
    self.m.fc1 = nn.Linear(nel, args.edim)(self.m.conv)
    self.m.mul_h = nn.CMulTable()({self.m.arg_h1, self.m.fc1})
    self.m.state = nn.Linear(edim, edim)(self.m.mul_h)
    self.m.h = nn.Tanh()(self.m.state)
    local next_r, next_p = self:retrieve_goal(args, self.m.h, prev_p, edim)
    self.m.hr = nn.LinearNB(wdim, edim)(next_r)
    local hr1 = nn.Narrow(2,1,args.lindim)(self.m.hr)
    local hr2 = nn.Narrow(2,1+args.lindim,edim-args.lindim)(self.m.hr)
    local hr2_nl = nn.ReLU(true)(hr2)
    self.m.s = nn.JoinTable(2)({hr1, hr2_nl})

    local input = {self.m.x, self.m.g, self.m.p0, self.m.arg1, self.m.arg2}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.term_act then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.s)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    table.insert(output, next_p)

    return nn.gModule(input, output)
end

local FF101 = torch.class('FF101', 'FF20')

function FF101:build_model(args)
    args.pass_act = true
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.p0 = nn.Identity()()
    self.m.arg1 = nn.Identity()()
    self.m.arg2 = nn.Identity()()
    
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    -- Compute CNN embeddings
    self.m.arg_h1 = nn.Linear(args.num_args[1], args.edim)(self.m.arg1)
    self.m.mask_r = nn.Linear(args.num_args[2], args.edim)(self.m.arg2)
    self.m.mask_r_nl = nn.ReLU(true)(self.m.mask_r)
    self.m.mask_r2 = nn.Linear(args.edim, args.ch * args.buf * args.channels)(self.m.mask_r_nl)
    self.m.mask_r2_nl = nn.Sigmoid(true)(self.m.mask_r2)
    self.m.mask_h2d = nn.View(args.ch, args.buf * args.channels):setNumInputDims(1)(self.m.mask_r2_nl)
    local x_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(MMconv({self.m.mask_h2d, x_2d}))
    
    self.m.conv = self:build_conv(args, self.m.x_masked, args.ch)
    self.m.fc1 = nn.Linear(nel, args.edim)(self.m.conv)
    self.m.mul_h = nn.CMulTable()({self.m.arg_h1, self.m.fc1})
    self.m.state = nn.Linear(edim, edim)(self.m.mul_h)
    self.m.h = nn.Tanh()(self.m.state)
    local next_r, next_p = self:retrieve_goal(args, self.m.h, prev_p, edim)
    self.m.hr = nn.LinearNB(wdim, edim)(next_r)
    local hr1 = nn.Narrow(2,1,args.lindim)(self.m.hr)
    local hr2 = nn.Narrow(2,1+args.lindim,edim-args.lindim)(self.m.hr)
    local hr2_nl = nn.ReLU(true)(hr2)
    self.m.s = nn.JoinTable(2)({hr1, hr2_nl})

    local input = {self.m.x, self.m.g, self.m.p0, self.m.arg1, self.m.arg2}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.term_act then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.s)
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term)
    end
    table.insert(output, self.m.b)
    table.insert(output, next_p)

    return nn.gModule(input, output)
end

local Planner = torch.class('Planner', 'Manager')
function Planner:__init(args)
    args.pass_act = true
    if args.pass_term == nil then
        args.pass_term = true
    end
    Manager.__init(self, args)
end

function Planner:eval(x, g, arg)
    assert(self.sub_agent)
    local batch_size = x:size(1)
    local primitive_actions = torch.Tensor(batch_size, 1):zero()
    local args = torch.Tensor(batch_size, #self.args.num_args):zero()
    local buf = self.sub_agent.args.buf or 1
    local sub_x = x:narrow(2, 1, buf * self.args.channels):contiguous()
    local sub_b, sub_s, sub_term

    self.actions = {}
    if arg[1]:sum() > 0 then -- t > 1
        for i = 1, #self.args.num_args do
            local max_val, max_idx = arg[i]:max(2)
            args:narrow(2, i, 1):copy(max_idx)
        end
        primitive_actions, sub_b, sub_s, sub_term = self.sub_agent:eval(sub_x, args)
        local max_val, max_idx = sub_term:max(2)
        self.term = max_idx:csub(1):float():view(batch_size)

    else  -- t = 0
        self.term = torch.Tensor(batch_size):fill(1)
        self.term_prob = torch.Tensor({0, 1})
    end

    if self.args.open_loop then
        assert(batch_size == 1, "batch_size is not 1")
        self.update = self.term:squeeze()
    else
        self.update = 1
    end
    
    local actions, baseline, states
    if self.update == 1 then
        -- run the meta-controller for all trajectories
        local x_view = x:narrow(2, 1, self.args.channels):contiguous()
        actions, baseline, states = self:forward(x_view, g, arg, self.term:view(-1, 1):float())

        local subtask_changed = torch.zeros(batch_size)
        for i=1,#self.args.num_args do
            local prob = torch.exp(actions[i])
            self.actions[i] = torch.multinomial(prob, 1)
            for k=1,batch_size do
                if self.actions[i][k]:squeeze() ~= args[k][i] then
                    subtask_changed[k] = 1
                end
            end
            args:narrow(2, i, 1):copy(self.actions[i])
        end
        self:fill_internal_actions()

        if batch_size == 1 and subtask_changed[1] == 0 and self.term[1] == 0 then 
            --print("skip")
        else
            -- run the sub-agent to get primitive actions
            if buf > 1 then 
                self.sub_x = self.sub_x or torch.zeros(batch_size, 
                    buf * self.args.channels, x:size(3), x:size(4)) 
                self.sub_x:copy(sub_x)
            else
                self.sub_x = sub_x
            end
            --sub_x:narrow(2, 1, self.args.channels):copy(x:narrow(2, 1, self.args.channels))
            for i = 1, batch_size do
                if self.term[i] == 1 or subtask_changed[i] == 1 then
                    if buf > 1 then 
                        self.sub_x[i]:zero()
                        self.sub_x[i]:narrow(1, 1, self.args.channels):copy(
                            x[i]:narrow(1, 1, self.args.channels))
                    else
                        self.sub_agent:reset_init_state(batch_size)
                    end
                end
            end
            primitive_actions, sub_b, sub_s = self.sub_agent:eval(self.sub_x, args)
        end
    else
        -- not updated
        assert(self.args.open_loop)
        states = self.init_states
        for i = 1, #self.args.num_args do
            self.actions[i] = args:narrow(2, i, 1):clone()
        end
        baseline = torch.Tensor({0})
    end
    if self.sub_agent.recurrent then
        self.sub_agent:set_init_state(sub_s)
    end

    return primitive_actions, baseline, states
end

function Planner:retrieve_goal(args, h, p, hdim)
    -- shift vector
    self.m.g_hs = nn.Linear(hdim, args.edim)(h)
    local g_hs_nl = nn.ReLU(true)(self.m.g_hs)
    self.m.g_s = nn.Linear(args.edim, 2 * args.shift + 1)(g_hs_nl)
    if args.soft_mem then 
        self.m.g_prob_s = nn.SoftMax()(self.m.g_s)
        if args.open_loop and args.soft_rho then
            self.m.g_prob_s = nn.EntropyPenalty(0, false)(self.m.g_prob_s)
            table.insert(self.entropy_list, self.m.g_prob_s)
        end
    else
        self.m.g_shift = nn.LogSoftMax()(self.m.g_s)
        self.m.shift_act = nn.Multinomial()(self.m.g_shift)
        self.m.g_prob_s = nn.OneHot(2 * args.shift + 1)(self.m.shift_act)
    end
    local w_tilde = nn.CircularConvolution(){p, self.m.g_prob_s}
    --local w_pow = nn.PowTable(){w_tilde, gamma}
    self.m.g_prob = nn.Normalize(1)(w_tilde)
    local prob3d = nn.View(1, -1):setNumInputDims(1)(self.m.g_prob)
    local MMbout = nn.MM(false, false)
    local out3d = MMbout({prob3d, self.m.g_val})
    local out2d = nn.View(-1):setNumInputDims(1)(out3d)
    return out2d, self.m.g_prob
end

function Planner:build_conv(args, input, nInput)
    if #args.n_units > 0 then
        self.m.conv1 = nn.SpatialConvolution(nInput, args.n_units[1],
                            args.filter_size[1], args.filter_size[1],
                            args.filter_stride[1], args.filter_stride[1],
                            args.pad[1], args.pad[1])(input)
        self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
        if #args.n_units == 1 then
            if args.pool[1] then
                self.m.pool1 = nn.SpatialMaxPooling(args.pool[1], args.pool[1], 
                        args.pool[1],args.pool[1])(self.m.conv1_nl)
            end
            return nn.View(-1):setNumInputDims(3)(self.m.pool1),
                5 * 5 * args.n_units[1]
        end
        for i=1,(#args.n_units-1) do
            if args.pool[i] then
                self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                        args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            end
            self.m["conv" .. i+1] = nn.SpatialConvolution(args.n_units[i], args.n_units[i+1],
                                args.filter_size[i+1], args.filter_size[i+1],
                                args.filter_stride[i+1], args.filter_stride[i+1],
                                args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
            self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        end
        return nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"]),
                5 * 5 * args.n_units[#args.n_units]
    else
        return input, 10 * 10 * nInput
    end
end

function Planner:merge_pi(args, t1, t2, prev_arg, p)
    assert(args.merge_pi ~= nil)
    if args.merge_pi then
        local normalized_p = nn.SoftMax()(p)
        local prob = nn.CAddTable()({
            nn.ScalarMulTable()({prev_arg, self.m.t1}),
            nn.ScalarMulTable()({normalized_p, self.m.t2})
        })
        local eps = 1e-12
        local clipped_prob = nn.Clamp(eps, 1)(prob)
        local norm_prob = nn.Normalize(1)(clipped_prob)
        return nn.Log()(norm_prob)
    else
        local score = nn.CAddTable()({
            nn.ScalarMulTable()({nn.MulConstant(15)(prev_arg), self.m.t1}),
            nn.ScalarMulTable()({p, self.m.t2})
        })
        return nn.LogSoftMax()(score)
    end
end

function Planner:fill_internal_actions()
    if not self.args.soft_mem then
        table.insert(self.actions, self.m.shift_act.data.module.output:long())
    end
    if not self.args.soft_term and self.m.term_act then
        table.insert(self.actions, self.m.term_act.data.module.output:long())
    end
end

local MetaF = torch.class('MetaF', 'Planner')
function MetaF:__init(args)
    args.merge_pi = false
    Planner.__init(self, args)
end

function MetaF:build_init_states(args)
    local states = {}
    table.insert(states, torch.Tensor(1, args.max_task))  -- p
    table.insert(states, torch.Tensor(1, args.wdim))      -- r
    return states
end

function MetaF:reset_init_state(batch_size)
    for j=1,#self.init_states do
        local size = self.init_states[j]:size()
        size[1] = batch_size
        self.init_states[j]:resize(size)
        self.init_states[j]:fill(0)
    end
    -- Set initial memory attention to 1, 0, 0, ...
    self.init_states[1]:narrow(2, 1, 1):fill(1)
    --self.init_states[1]:narrow(2, self.init_states[1]:size(2), 1):fill(1)
end

function MetaF:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.p0 = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.arg1 = nn.Identity()()
    self.m.arg2 = nn.Identity()()
    self.m.sub_term = nn.Identity()()
    
    --self.m.sub_nonterm = nn.MulConstant(-1)(nn.AddConstant(-1)(self.m.sub_term))
    local prev_r = self.m.r0
    local prev_p = self.m.p0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    self.m.arg_h1 = nn.LinearNB(args.num_args[1], edim)(self.m.arg1)
    self.m.arg_h2 = nn.LinearNB(args.num_args[2], edim)(self.m.arg2)
    self.m.arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    self.m.arg_fc = nn.Linear(args.edim, args.edim)(self.m.arg_hh)
    
    self.m.conv = self:build_conv(args, self.m.x, args.channels)
    self.m.fc = nn.Linear(nel, args.edim)(self.m.conv)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc})
    self.m.state = nn.Linear(edim, edim)(self.m.mul_h)
    self.m.state_nl = nn.ReLU(true)(self.m.state)
    local state_r = nn.JoinTable(2)({self.m.state_nl, prev_r}) 
    self.m.state_r = nn.Linear(wdim + edim, edim)(state_r)

    self.m.sub_term_fc = nn.Linear(1, edim)(self.m.sub_term)
    self.m.fc_mul = nn.CMulTable()({self.m.state_r, self.m.sub_term_fc})
    self.m.state_r_t = nn.Linear(edim, edim)(self.m.fc_mul)
    self.m.h = nn.ReLU(true)(self.m.state_r_t)

    self.m.term_fc = nn.Linear(edim, 2)(self.m.h)
    if args.soft_term then
        self.m.term_prob = nn.SoftMax()(self.m.term_fc)
        if args.soft_rho then 
            self.m.term_prob = nn.EntropyPenalty(0, false)(self.m.term_prob)
            table.insert(self.entropy_list, self.m.term_prob)
        end
    else
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        self.m.term_act = nn.Multinomial()(self.m.term)
        self.m.term_prob = nn.OneHot(2)(self.m.term_act)
    end
    self.m.t1 = nn.Narrow(2,1,1)(self.m.term_prob)
    self.m.t2 = nn.Narrow(2,2,1)(self.m.term_prob)

    if args.l1 > 0 and args.soft_term then
        self.m.t2 = nn.L1Penalty(args.l1)(self.m.t2)
    end

    --[[
    self.m.nonterm_join = nn.JoinTable(2)({self.m.meta_nonterm, self.m.sub_nonterm})
    self.m.term_join = nn.JoinTable(2)({self.m.meta_term, self.m.sub_term})

    self.m.t1 = nn.Min(2)(self.m.nonterm_join)
    self.m.t2 = nn.Max(2)(self.m.term_join)
    --]]
    --
    local r, p = self:retrieve_goal(args, self.m.h, prev_p, edim)
    local hr = nn.JoinTable(2)({self.m.h, r})
    self.m.hr = nn.Linear(wdim + edim, edim)(hr)
    self.m.s = nn.ReLU(true)(self.m.hr)
    
    self.m.g_prob = nn.CAddTable()({
        nn.ScalarMulTable()({prev_p, self.m.t1}),
        nn.ScalarMulTable()({p, self.m.t2})
    })

    local next_r = nn.CAddTable()({
        nn.ScalarMulTable()({prev_r, self.m.t1}),
        nn.ScalarMulTable()({r, self.m.t2})
    }) 

    local input = {self.m.x, self.m.g, self.m.p0, self.m.r0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        table.insert(input, self.m["arg" .. i])
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = self:merge_pi(args, self.m.t1, self.m.t2, 
                self.m["arg" .. i], self.m["fc_" .. i])

        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if not args.soft_term then
        table.insert(self.n_actions, 2)
        table.insert(output, self.m.term) 
    end

    table.insert(output, self.m.b)
    table.insert(output, self.m.g_prob)
    table.insert(output, next_r)

    table.insert(input, self.m.sub_term)
    return nn.gModule(input, output)
end

function MetaF:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LinearNB(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LinearNB(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_fc = nn.Linear(self.args.edim, self.args.edim):share(
            self.m.arg_fc.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():
                add(nn.OneHot(self.args.num_args[1])):
                add(nn.OneHot(self.args.num_args[2])))
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_fc)
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)
    return self.analogy_list
end

local MetaF2 = torch.class('MetaF2', 'MetaF')

function MetaF2:context_embedding(args, arg1, arg2, prev_r, term)
    if not self.m.context then
        local g = self:goal_embedding(args, arg1, arg2)
        local r = self:read_embedding(args, prev_r)
        local t = self:term_embedding(args, term)
        local grt = nn.JoinTable(2)({g, r, t})
        self.m.context_grt = nn.Linear(3 * args.edim, args.edim)(grt)
        self.m.context = nn.ReLU(true)(self.m.context_grt)
    end
    return self.m.context
end

function MetaF2:low_context(args, arg1, arg2, prev_r, term)
    return self:context_embedding(args, arg1, arg2, prev_r, term)
end

function MetaF2:high_context(args, arg1, arg2, prev_r, term)
    return self:context_embedding(args, arg1, arg2, prev_r, term)
end

function MetaF2:goal_embedding(args, arg1, arg2)
    if not self.m.arg_hh then
        self.m.arg_h1 = nn.LinearNB(args.num_args[1], args.edim)(arg1)
        self.m.arg_h2 = nn.LinearNB(args.num_args[2], args.edim)(arg2)
        self.m.arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    end
    return self.m.arg_hh
end

function MetaF2:read_embedding(args, prev_r)
    if not self.m.read_embed then
        self.m.read_embed = nn.Linear(args.wdim, args.edim)(prev_r)
        self.m.read_embed_nl = nn.ReLU(true)(self.m.read_embed)
    end
    return self.m.read_embed_nl
end

function MetaF2:term_embedding(args, term)
    if not self.m.sub_term_fc then
        self.m.sub_term_fc = nn.Linear(1, args.edim)(term)
        self.m.sub_term_nl = nn.ReLU(true)(self.m.sub_term_fc)
    end
    return self.m.sub_term_nl
end

function MetaF2:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.p0 = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.arg1 = nn.Identity()()
    self.m.arg2 = nn.Identity()()
    self.m.sub_term = nn.Identity()()
    
    --self.m.sub_nonterm = nn.MulConstant(-1)(nn.AddConstant(-1)(self.m.sub_term))
    local prev_r = self.m.r0
    local prev_p = self.m.p0

    local nel
    local edim = args.edim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    local context_l = self:low_context(args, self.m.arg1, self.m.arg2, prev_r, self.m.sub_term)
    local context_h = self:high_context(args, self.m.arg1, self.m.arg2, prev_r, self.m.sub_term)
    if context_h then
        self.m.context_fc = nn.Linear(edim, edim)(context_h)
    end
    if context_l then
        self.m.context_conv = nn.Linear(edim, args.ch * args.n_units[2])(context_l)
        local arg_h2d = nn.View(args.ch, args.n_units[2]):setNumInputDims(1)(self.m.context_conv)
        
        self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                            args.filter_size[1], args.filter_size[1],
                            args.filter_stride[1], args.filter_stride[1],
                            args.pad[1], args.pad[1])(self.m.x)
        self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
        
        self.m.conv2 = nn.SpatialConvolution(args.n_units[1], args.n_units[2],
                            args.filter_size[2], args.filter_size[2],
                            args.filter_stride[2], args.filter_stride[2],
                            args.pad[2], args.pad[2])(self.m.conv1_nl)
        self.m.conv2_nl = nn.ReLU(true)(self.m.conv2)
        
        local conv1_2d = nn.View(args.n_units[2], -1):setNumInputDims(3)(self.m.conv2_nl)
        local MMconv = nn.MM(false, false)
        self.m.conv3 = nn.View(args.ch, g_opts.max_size, g_opts.max_size):setNumInputDims(2)(
                MMconv({arg_h2d, conv1_2d}))
        
        self.m.pool1 = nn.SpatialMaxPooling(args.pool[1], args.pool[1], 2, 2)(self.m.conv3)
        self.m.conv =  nn.View(-1):setNumInputDims(3)(self.m.pool1)
        nel = g_opts.max_size * g_opts.max_size * args.ch / args.pool[1] / args.pool[1]

        --self.m.conv, nel = self:build_conv(args, self.m.x_masked, args.ch)
    else
        self.m.conv, nel = self:build_conv(args, self.m.x, args.channels)
    end

    self.m.fc = nn.Linear(nel, edim)(self.m.conv)
    if context_h then
        self.m.mul_h = nn.CMulTable()({self.m.context_fc, self.m.fc})
        self.m.state = nn.Linear(edim, edim)(self.m.mul_h)
        self.m.h = nn.ReLU(true)(self.m.state)
    else
        self.m.h = nn.ReLU(true)(self.m.fc)
    end

    if args.ta then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.h)
        if args.soft_term then
            self.m.term_prob = nn.SoftMax()(self.m.term_fc)
            if args.soft_rho then 
                self.m.term_prob = nn.EntropyPenalty(0, false)(self.m.term_prob)
                table.insert(self.entropy_list, self.m.term_prob)
            end
        else
            self.m.term = nn.LogSoftMax()(self.m.term_fc)
            self.m.term_act = nn.Multinomial()(self.m.term)
            self.m.term_prob = nn.OneHot(2)(self.m.term_act)
        end
        self.m.t1 = nn.Narrow(2,1,1)(self.m.term_prob)
        self.m.t2 = nn.Narrow(2,2,1)(self.m.term_prob)

        if args.l1 > 0 and args.soft_term then
            self.m.t2 = nn.L1Penalty(args.l1)(self.m.t2)
        end
    end

    local r, p = self:retrieve_goal(args, self.m.h, prev_p, edim)
    local hr = nn.JoinTable(2)({self.m.h, r})
    self.m.hr = nn.Linear(wdim + edim, edim)(hr)
    self.m.s = nn.ReLU(true)(self.m.hr)
    
    if args.ta then
        self.m.g_prob = nn.CAddTable()({
            nn.ScalarMulTable()({prev_p, self.m.t1}),
            nn.ScalarMulTable()({p, self.m.t2})
        })

        self.m.next_r = nn.CAddTable()({
            nn.ScalarMulTable()({prev_r, self.m.t1}),
            nn.ScalarMulTable()({r, self.m.t2})
        }) 
    end

    local input = {self.m.x, self.m.g, self.m.p0, self.m.r0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        table.insert(input, self.m["arg" .. i])
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        if args.ta then
            self.m["a" .. i] = self:merge_pi(args, self.m.t1, self.m.t2, 
                self.m["arg" .. i], self.m["fc_" .. i])
        else
            self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        end

        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    if args.base then
        local goal3d = nn.View(-1, args.max_task, args.max_word)(self.m.g)
        self.m.g_list = nn.NotEqual(g_vocab['nil'])(nn.Select(3, 1)(goal3d))
        local pointer = p
        if args.ta then 
            pointer = self.m.g_prob
        end
        self.m.base_concat = nn.JoinTable(2)({self.m.g_list, pointer, self.m.s})
        self.m.base_fc = nn.Linear(edim + args.max_task * 2, edim)(self.m.base_concat)
        self.m.b = nn.Linear(edim, 1)(nn.ReLU(true)(self.m.base_fc))
    else
        self.m.b = nn.Linear(edim, 1)(self.m.s)
    end
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.ta and not args.soft_term then
        table.insert(self.n_actions, 2)
        table.insert(output, self.m.term) 
    end

    table.insert(output, self.m.b)
    if args.ta then
        table.insert(output, self.m.g_prob)
        table.insert(output, self.m.next_r)
    else
        table.insert(output, p)
        table.insert(output, r)
    end

    table.insert(input, self.m.sub_term)
    return nn.gModule(input, output)
end

function MetaF2:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LinearNB(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LinearNB(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():
                add(nn.OneHot(self.args.num_args[1])):
                add(nn.OneHot(self.args.num_args[2])))
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)
    return self.analogy_list
end

local MetaGR_T = torch.class('MetaGR_T', 'MetaF2')
function MetaGR_T:low_context(args, arg1, arg2, prev_r, term)
    local g = self:goal_embedding(args, arg1, arg2)
    local r = self:read_embedding(args, prev_r)
    local gr = nn.JoinTable(2)({g, r})
    self.m.gr_fc = nn.Linear(2 * args.edim, args.edim)(gr)
    self.m.gr = nn.ReLU(true)(self.m.gr_fc)
    return self.m.gr
end
function MetaGR_T:high_context(args, arg1, arg2, prev_r, term)
    return self:term_embedding(args, term)
end

local MetaR_GT = torch.class('MetaR_GT', 'MetaF2')
function MetaR_GT:low_context(args, arg1, arg2, prev_r, term)
    return self:read_embedding(args, prev_r)
end

function MetaR_GT:high_context(args, arg1, arg2, prev_r, term)
    local g = self:goal_embedding(args, arg1, arg2)
    local t = self:term_embedding(args, term)
    local gt = nn.JoinTable(2)({g, t})
    self.m.gt_fc = nn.Linear(2 * args.edim, args.edim)(gt)
    self.m.gt = nn.ReLU(true)(self.m.gt_fc)
    return self.m.gt
end

local MetaToy = torch.class('MetaToy', 'MetaF2')
function MetaToy:low_context(args, arg1, arg2, prev_r, term)
    return nil 
end

function MetaToy:high_context(args, arg1, arg2, prev_r, term)
    return self:context_embedding(args, arg1, arg2, prev_r, term)
end

function MetaToy:build_conv(args, input, nInput)
    local x_view = nn.View(nInput, -1):setNumInputDims(3)(input)
    return nn.Max(3)(x_view), nInput
end

local MetaToy2 = torch.class('MetaToy2', 'MetaF2')
function MetaToy2:build_conv(args, input, nInput)
    self.m.conv1 = nn.SpatialConvolution(nInput, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(input)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    self.m.pool1 = nn.SpatialMaxPooling(10, 10, 10, 10)(self.m.conv1_nl)
    return nn.View(-1):setNumInputDims(3)(self.m.pool1), args.n_units[1]
end

-- 3x10x10 -> 3x5x5 -> Fully-connected
local MetaToy3 = torch.class('MetaToy3', 'MetaF2')
function MetaToy3:build_conv(args, input, nInput)
    self.m.pool1 = nn.SpatialMaxPooling(2, 2, 2, 2)(input)
    return nn.View(-1):setNumInputDims(3)(self.m.pool1), 5 * 5 * nInput 
end

-- 3x10x10 -> 3x5x5 -> Fully-connected
local MetaToy3P = torch.class('MetaToy3P', 'MetaToy3')
function MetaToy3P:__init(args)
    args.merge_pi = true
    Planner.__init(self, args)
end

-- 3x10x10 -> 3x5x5 -> 8x3x3 -> 8x1x1 -> Fully-connected
local MetaToy4 = torch.class('MetaToy4', 'MetaF2')
function MetaToy4:build_conv(args, input, nInput)
    self.m.pool1 = nn.SpatialMaxPooling(2, 2, 2, 2)(input)
    self.m.conv1 = nn.SpatialConvolution(nInput, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.pool1)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    self.m.pool2 = nn.SpatialMaxPooling(3, 3, 3, 3)(self.m.conv1_nl)

    return nn.View(-1):setNumInputDims(3)(self.m.pool2), args.n_units[1]
end

local MetaF3 = torch.class('MetaF3', 'MetaF')

function MetaF3:context_embedding(args, arg1, arg2, prev_r, term)
    if not self.m.context then
        local g = self:goal_embedding(args, arg1, arg2)
        local r = self:read_embedding(args, prev_r)
        local t = self:term_embedding(args, term)
        local grt = nn.JoinTable(2)({g, r, t})
        self.m.context_grt = nn.Linear(2 * args.edim + 1, args.edim)(grt)
        self.m.context = nn.ReLU(true)(self.m.context_grt)
    end
    return self.m.context
end

function MetaF3:low_context(args, arg1, arg2, prev_r, term)
    return self:context_embedding(args, arg1, arg2, prev_r, term)
end

function MetaF3:high_context(args, arg1, arg2, prev_r, term)
    return self:context_embedding(args, arg1, arg2, prev_r, term)
end

function MetaF3:goal_embedding(args, arg1, arg2)
    if not self.m.arg_hh then
        self.m.arg_h1 = nn.LinearNB(args.num_args[1], args.edim)(arg1)
        self.m.arg_h2 = nn.LinearNB(args.num_args[2], args.edim)(arg2)
        self.m.arg_hh = nn.CMulTable()({self.m.arg_h1, self.m.arg_h2})
    end
    return self.m.arg_hh
end

function MetaF3:read_embedding(args, prev_r)
    return prev_r
end

function MetaF3:term_embedding(args, term)
    return term 
end

function MetaF3:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.p0 = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.arg1 = nn.Identity()()
    self.m.arg2 = nn.Identity()()
    self.m.sub_term = nn.Identity()()
    
    --self.m.sub_nonterm = nn.MulConstant(-1)(nn.AddConstant(-1)(self.m.sub_term))
    local prev_r = self.m.r0
    local prev_p = self.m.p0

    local nel
    local edim = args.edim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)

    local context_l = self:low_context(args, self.m.arg1, self.m.arg2, prev_r, self.m.sub_term)
    local context_h = self:high_context(args, self.m.arg1, self.m.arg2, prev_r, self.m.sub_term)
    if context_h then
        self.m.context_fc = nn.Linear(edim, edim)(context_h)
    end
    if context_l then
        self.m.context_conv = nn.Linear(edim, args.ch * args.channels)(context_l)
        self.m.mask_h2d = nn.View(args.ch, args.channels):setNumInputDims(1)(self.m.context_conv)
        local x_2d = nn.View(args.channels, -1):setNumInputDims(3)(self.m.x)
        local MMconv = nn.MM(false, false)
        self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(
                MMconv({self.m.mask_h2d, x_2d}))
        self.m.conv, nel = self:build_conv(args, self.m.x_masked, args.ch)
    else
        self.m.conv, nel = self:build_conv(args, self.m.x, args.channels)
    end

    self.m.fc = nn.Linear(nel, edim)(self.m.conv)
    if context_h then
        self.m.mul_h = nn.CMulTable()({self.m.context_fc, self.m.fc})
        self.m.state = nn.Linear(edim, edim)(self.m.mul_h)
        self.m.h = nn.ReLU(true)(self.m.state)
    else
        self.m.h = nn.ReLU(true)(self.m.fc)
    end

    if args.ta then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.h)
        if args.soft_term then
            self.m.term_prob = nn.SoftMax()(self.m.term_fc)
            if args.soft_rho then 
                self.m.term_prob = nn.EntropyPenalty(0, false)(self.m.term_prob)
                table.insert(self.entropy_list, self.m.term_prob)
            end
        else
            self.m.term = nn.LogSoftMax()(self.m.term_fc)
            self.m.term_act = nn.Multinomial()(self.m.term)
            self.m.term_prob = nn.OneHot(2)(self.m.term_act)
        end
        self.m.t1 = nn.Narrow(2,1,1)(self.m.term_prob)
        self.m.t2 = nn.Narrow(2,2,1)(self.m.term_prob)

        if args.l1 > 0 and args.soft_term then
            self.m.t2 = nn.L1Penalty(args.l1)(self.m.t2)
        end
    end

    local r, p = self:retrieve_goal(args, self.m.h, prev_p, edim)
    local hr = nn.JoinTable(2)({self.m.h, r})
    self.m.hr = nn.Linear(wdim + edim, edim)(hr)
    self.m.s = nn.ReLU(true)(self.m.hr)
    
    if args.ta then
        self.m.g_prob = nn.CAddTable()({
            nn.ScalarMulTable()({prev_p, self.m.t1}),
            nn.ScalarMulTable()({p, self.m.t2})
        })

        self.m.next_r = nn.CAddTable()({
            nn.ScalarMulTable()({prev_r, self.m.t1}),
            nn.ScalarMulTable()({r, self.m.t2})
        }) 
    end

    local input = {self.m.x, self.m.g, self.m.p0, self.m.r0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        table.insert(input, self.m["arg" .. i])
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        if args.ta then
            self.m["a" .. i] = self:merge_pi(args, self.m.t1, self.m.t2, 
                self.m["arg" .. i], self.m["fc_" .. i])
        else
            self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        end

        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    if args.base then
        local goal3d = nn.View(-1, args.max_task, args.max_word)(self.m.g)
        self.m.g_list = nn.NotEqual(g_vocab['nil'])(nn.Select(3, 1)(goal3d))
        local pointer = p
        if args.ta then 
            pointer = self.m.g_prob
        end
        self.m.base_concat = nn.JoinTable(2)({self.m.g_list, pointer, self.m.s})
        self.m.base_fc = nn.Linear(edim + args.max_task * 2, edim)(self.m.base_concat)
        self.m.b = nn.Linear(edim, 1)(nn.ReLU(true)(self.m.base_fc))
    else
        self.m.b = nn.Linear(edim, 1)(self.m.s)
    end
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.ta and not args.soft_term then
        table.insert(self.n_actions, 2)
        table.insert(output, self.m.term) 
    end

    table.insert(output, self.m.b)
    if args.ta then
        table.insert(output, self.m.g_prob)
        table.insert(output, self.m.next_r)
    else
        table.insert(output, p)
        table.insert(output, r)
    end

    table.insert(input, self.m.sub_term)
    return nn.gModule(input, output)
end

function MetaF3:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LinearNB(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LinearNB(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():
                add(nn.OneHot(self.args.num_args[1])):
                add(nn.OneHot(self.args.num_args[2])))
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)
    return self.analogy_list
end

-- 3x10x10 -> 3x5x5 -> Fully-connected
local MetaF3P = torch.class('MetaF3P', 'MetaF3')
function MetaF3P:build_conv(args, input, nInput)
    self.m.pool1 = nn.SpatialMaxPooling(2, 2, 2, 2)(input)
    return nn.View(-1):setNumInputDims(3)(self.m.pool1), 5 * 5 * nInput 
end

function MetaF3P:__init(args)
    args.merge_pi = true
    Planner.__init(self, args)
end


local MetaRNN = torch.class('MetaRNN', 'Planner')
function MetaRNN:__init(args)
    args.merge_pi = true
    Planner.__init(self, args)
end

function MetaRNN:build_init_states(args)
    local states = {}
    table.insert(states, torch.Tensor(1, args.max_task))  -- p
    table.insert(states, torch.Tensor(1, args.wdim))      -- r
    table.insert(states, torch.Tensor(1, args.ldim))      -- c
    table.insert(states, torch.Tensor(1, args.ldim))      -- h
    return states
end

function MetaRNN:reset_init_state(batch_size)
    for j=1,#self.init_states do
        local size = self.init_states[j]:size()
        size[1] = batch_size
        self.init_states[j]:resize(size)
        self.init_states[j]:fill(0)
    end
    -- Set initial memory attention to 1, 0, 0, ...
    self.init_states[1]:narrow(2, 1, 1):fill(1)
end

function MetaRNN:build_conv(args, input, nInput)
    self.m.conv1 = nn.SpatialConvolution(nInput, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(input)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    
    self.m.conv2 = nn.SpatialConvolution(args.n_units[1], args.n_units[2],
                        args.filter_size[2], args.filter_size[2],
                        args.filter_stride[2], args.filter_stride[2],
                        args.pad[2], args.pad[2])(self.m.conv1_nl)
    self.m.conv2_nl = nn.ReLU(true)(self.m.conv2)

    self.m.pool1 = nn.SpatialMaxPooling(args.pool[1], args.pool[1], 
            args.pool[1], args.pool[1])(self.m.conv2_nl)
    return nn.View(args.n_units[2], -1):setNumInputDims(3)(self.m.pool1), 15 / args.pool[1]
end

function MetaRNN:context_embedding(args, arg1, arg2, prev_r, term)
    if not self.m.context then
        local g = self:goal_embedding(args, arg1, arg2)
        local r = self:read_embedding(args, prev_r)
        if term then
            local t = self:term_embedding(args, term)
            local grt = nn.JoinTable(2)({g, r, t})
            self.m.context_grt = nn.Linear(3 * args.edim, args.edim)(grt)
            self.m.context = nn.ReLU(true)(self.m.context_grt)
        else 
            assert(args.pass_term ~= true)
            local gr = nn.JoinTable(2)({g, r})
            self.m.context_gr = nn.Linear(2 * args.edim, args.edim)(gr)
            self.m.context = nn.ReLU(true)(self.m.context_gr)
        end
    end
    return self.m.context
end

function MetaRNN:low_context(...)
    return self:context_embedding(...)
end

function MetaRNN:high_context(...)
    return self:context_embedding(...)
end

function MetaRNN:goal_embedding(args, arg1, arg2)
    if not self.m.arg_hh then
        self.m.arg_h1 = nn.LinearNB(args.num_args[1], args.edim)(arg1)
        self.m.arg_h2 = nn.LinearNB(args.num_args[2], args.edim)(arg2)
        self.m.arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    end
    return self.m.arg_hh
end

function MetaRNN:read_embedding(args, prev_r)
    if not self.m.read_embed then
        self.m.read_embed = nn.Linear(args.wdim, args.edim)(prev_r)
        self.m.read_embed_nl = nn.ReLU(true)(self.m.read_embed)
    end
    return self.m.read_embed_nl
end

function MetaRNN:term_embedding(args, term)
    if not self.m.sub_term_fc then
        self.m.sub_term_fc = nn.Linear(1, args.edim)(term)
        self.m.sub_term_nl = nn.ReLU(true)(self.m.sub_term_fc)
    end
    return self.m.sub_term_nl
end

function MetaRNN:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.p0 = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.c0 = nn.Identity()()
    self.m.h0 = nn.Identity()()
    self.m.arg1 = nn.Identity()()
    self.m.arg2 = nn.Identity()()
    self.m.sub_term = nn.Identity()()
    
    local prev_r = self.m.r0
    local prev_p = self.m.p0
    local prev_c = self.m.c0
    local prev_h = self.m.h0

    local feat_size
    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local wdim = args.wdim
    local ldim = args.ldim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)
    
    local context_l = self:low_context(args, self.m.arg1, self.m.arg2, prev_r, self.m.sub_term)
    local context_h = self:high_context(args, self.m.arg1, self.m.arg2, prev_r, self.m.sub_term)
    if context_h then
        self.m.context_fc = nn.Linear(edim, edim)(context_h)
    end
    if context_l then
        self.m.context_conv = nn.Linear(edim, args.ch * args.n_units[2])(context_l)
        self.m.conv_low, feat_size = self:build_conv(args, self.m.x, args.channels)

        self.m.mask_h2d = nn.View(args.ch, args.n_units[2]):setNumInputDims(1)(
                self.m.context_conv)
        local MMconv = nn.MM(false, false)
        self.m.conv_masked = MMconv({self.m.mask_h2d, self.m.conv_low})
        self.m.conv = nn.View(-1):setNumInputDims(2)(self.m.conv_masked)

        nel = feat_size * feat_size * args.ch
    else
        self.m.conv, nel = self:build_conv(args, self.m.x, args.channels)
    end

    self.m.fc = nn.Linear(nel, edim)(self.m.conv)
    if context_h then
        self.m.mul_h = nn.CMulTable()({self.m.context_fc, self.m.fc})
        self.m.state = nn.Linear(edim, edim)(self.m.mul_h)
        self.m.h = nn.ReLU(true)(self.m.state)
    else
        self.m.h = nn.ReLU(true)(self.m.fc)
    end

    if args.ta then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.h)
        if args.soft_term then
            self.m.term_prob = nn.SoftMax()(self.m.term_fc)
            if args.soft_rho then 
                self.m.term_prob = nn.EntropyPenalty(0, false)(self.m.term_prob)
                table.insert(self.entropy_list, self.m.term_prob)
            end
        else
            self.m.term = nn.LogSoftMax()(self.m.term_fc)
            self.m.term_act = nn.Multinomial()(self.m.term)
            self.m.term_prob = nn.OneHot(2)(self.m.term_act)
        end
        self.m.t1 = nn.Narrow(2,1,1)(self.m.term_prob)
        self.m.t2 = nn.Narrow(2,2,1)(self.m.term_prob)

        if args.l1 > 0 and args.soft_term then
            self.m.t2 = nn.L1Penalty(args.l1)(self.m.t2)
        end
    end

    -- LSTM
    self.m.lstm = nn.Linear(edim, 4*ldim)(self.m.h)

    local reshaped = nn.View(4, ldim):setNumInputDims(1)(self.m.lstm)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    local in_gate = nn.Sigmoid(true)(n1)
    local forget_gate = nn.Sigmoid(true)(n2)
    local out_gate = nn.Sigmoid(true)(n3)
    local in_transform = nn.Tanh()(n4)
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    local r, p = self:retrieve_goal(args, next_h, prev_p, ldim)
    local hr = nn.JoinTable(2)({next_h, r})
    self.m.hr = nn.Linear(wdim + ldim, edim)(hr)
    self.m.s = nn.ReLU(true)(self.m.hr)
    
    if args.ta then
        self.m.g_prob = nn.CAddTable()({
            nn.ScalarMulTable()({prev_p, self.m.t1}),
            nn.ScalarMulTable()({p, self.m.t2})
        })

        self.m.next_r = nn.CAddTable()({
            nn.ScalarMulTable()({prev_r, self.m.t1}),
            nn.ScalarMulTable()({r, self.m.t2})
        }) 

        self.m.next_c = nn.CAddTable()({
            nn.ScalarMulTable()({prev_c, self.m.t1}),
            nn.ScalarMulTable()({next_c, self.m.t2})
        }) 

        self.m.next_h = nn.CAddTable()({
            nn.ScalarMulTable()({prev_h, self.m.t1}),
            nn.ScalarMulTable()({next_h, self.m.t2})
        }) 
    end

    local input = {self.m.x, self.m.g, self.m.p0, self.m.r0, self.m.c0, self.m.h0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        table.insert(input, self.m["arg" .. i])
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        if args.ta then
            self.m["a" .. i] = self:merge_pi(args, self.m.t1, self.m.t2, 
                self.m["arg" .. i], self.m["fc_" .. i])
        else
            self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        end

        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    if args.base then
        local goal3d = nn.View(-1, args.max_task, args.max_word)(self.m.g)
        self.m.g_list = nn.NotEqual(g_vocab['nil'])(nn.Select(3, 1)(goal3d))
        local pointer = p
        if args.ta then 
            pointer = self.m.g_prob
        end
        self.m.base_concat = nn.JoinTable(2)({self.m.g_list, pointer, self.m.s})
        self.m.base_fc = nn.Linear(edim + args.max_task * 2, edim)(self.m.base_concat)
        self.m.b = nn.Linear(edim, 1)(nn.ReLU(true)(self.m.base_fc))
    else
        self.m.b = nn.Linear(edim, 1)(self.m.s)
    end
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.ta and not args.soft_term then
        table.insert(self.n_actions, 2)
        table.insert(output, self.m.term) 
    end

    table.insert(output, self.m.b)
    if args.ta then
        table.insert(output, self.m.g_prob)
        table.insert(output, self.m.next_r)
        table.insert(output, self.m.next_c)
        table.insert(output, self.m.next_h)
    else
        table.insert(output, p)
        table.insert(output, r)
        table.insert(output, next_c)
        table.insert(output, next_h)
    end

    table.insert(input, self.m.sub_term)
    return nn.gModule(input, output)
end

function MetaRNN:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LinearNB(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LinearNB(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():
                add(nn.OneHot(self.args.num_args[1])):
                add(nn.OneHot(self.args.num_args[2])))
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)
    return self.analogy_list
end

local MetaLSTM = torch.class("MetaLSTM", "MetaRNN")
function MetaLSTM:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.p0 = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.c0 = nn.Identity()()
    self.m.h0 = nn.Identity()()
    self.m.arg1 = nn.Identity()()
    self.m.arg2 = nn.Identity()()
    self.m.sub_term = nn.Identity()()
    
    local prev_r = self.m.r0
    local prev_p = self.m.p0
    local prev_c = self.m.c0
    local prev_h = self.m.h0

    local feat_size
    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local wdim = args.wdim
    local ldim = args.ldim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)
    
    local context_l = self:low_context(args, self.m.arg1, self.m.arg2, prev_r, self.m.sub_term)
    local context_h = self:high_context(args, self.m.arg1, self.m.arg2, prev_r, self.m.sub_term)
    if context_h then
        self.m.context_fc = nn.Linear(edim, edim)(context_h)
    end
    if context_l then
        self.m.context_conv = nn.Linear(edim, args.ch * args.n_units[2])(context_l)
        self.m.conv_low, feat_size = self:build_conv(args, self.m.x, args.channels)

        self.m.mask_h2d = nn.View(args.ch, args.n_units[2]):setNumInputDims(1)(
                self.m.context_conv)
        local MMconv = nn.MM(false, false)
        self.m.conv_masked = MMconv({self.m.mask_h2d, self.m.conv_low})
        self.m.conv = nn.View(-1):setNumInputDims(2)(self.m.conv_masked)

        nel = feat_size * feat_size * args.ch
    else
        self.m.conv, nel = self:build_conv(args, self.m.x, args.channels)
    end

    self.m.fc = nn.Linear(nel, edim)(self.m.conv)
    if context_h then
        self.m.mul_h = nn.CMulTable()({self.m.context_fc, self.m.fc})
        self.m.state = nn.Linear(edim, edim)(self.m.mul_h)
        self.m.h = nn.ReLU(true)(self.m.state)
    else
        self.m.h = nn.ReLU(true)(self.m.fc)
    end

    -- LSTM
    local xh = nn.JoinTable(2)({self.m.h, prev_h})
    self.m.lstm = nn.Linear(edim + ldim, 4*ldim)(xh)

    local reshaped = nn.View(4, ldim):setNumInputDims(1)(self.m.lstm)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    local in_gate = nn.Sigmoid(true)(n1)
    local forget_gate = nn.Sigmoid(true)(n2)
    local out_gate = nn.Sigmoid(true)(n3)
    local in_transform = nn.Tanh()(n4)
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    if args.ta then
        self.m.term_fc = nn.Linear(ldim, 2)(next_h)
        if args.soft_term then
            self.m.term_prob = nn.SoftMax()(self.m.term_fc)
            if args.soft_rho then 
                self.m.term_prob = nn.EntropyPenalty(0, false)(self.m.term_prob)
                table.insert(self.entropy_list, self.m.term_prob)
            end
        else
            self.m.term = nn.LogSoftMax()(self.m.term_fc)
            self.m.term_act = nn.Multinomial()(self.m.term)
            self.m.term_prob = nn.OneHot(2)(self.m.term_act)
        end
        self.m.t1 = nn.Narrow(2,1,1)(self.m.term_prob)
        self.m.t2 = nn.Narrow(2,2,1)(self.m.term_prob)

        if args.l1 > 0 and args.soft_term then
            self.m.t2 = nn.L1Penalty(args.l1)(self.m.t2)
        end
    end

    local r, p = self:retrieve_goal(args, next_h, prev_p, ldim)
    local hr = nn.JoinTable(2)({next_h, r})
    self.m.hr = nn.Linear(wdim + ldim, edim)(hr)
    self.m.s = nn.ReLU(true)(self.m.hr)
    
    if args.ta then
        self.m.g_prob = nn.CAddTable()({
            nn.ScalarMulTable()({prev_p, self.m.t1}),
            nn.ScalarMulTable()({p, self.m.t2})
        })

        self.m.next_r = nn.CAddTable()({
            nn.ScalarMulTable()({prev_r, self.m.t1}),
            nn.ScalarMulTable()({r, self.m.t2})
        }) 

        self.m.next_c = next_c 
        self.m.next_h = next_h 
    end

    local input = {self.m.x, self.m.g, self.m.p0, self.m.r0, self.m.c0, self.m.h0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        table.insert(input, self.m["arg" .. i])
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        if args.ta then
            self.m["a" .. i] = self:merge_pi(args, self.m.t1, self.m.t2, 
                self.m["arg" .. i], self.m["fc_" .. i])
        else
            self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        end

        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    if args.base then
        local goal3d = nn.View(-1, args.max_task, args.max_word)(self.m.g)
        self.m.g_list = nn.NotEqual(g_vocab['nil'])(nn.Select(3, 1)(goal3d))
        local pointer = p
        if args.ta then 
            pointer = self.m.g_prob
        end
        self.m.base_concat = nn.JoinTable(2)({self.m.g_list, pointer, self.m.s})
        self.m.base_fc = nn.Linear(edim + args.max_task * 2, edim)(self.m.base_concat)
        self.m.b = nn.Linear(edim, 1)(nn.ReLU(true)(self.m.base_fc))
    else
        self.m.b = nn.Linear(edim, 1)(self.m.s)
    end
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.ta and not args.soft_term then
        table.insert(self.n_actions, 2)
        table.insert(output, self.m.term) 
    end

    table.insert(output, self.m.b)
    if args.ta then
        table.insert(output, self.m.g_prob)
        table.insert(output, self.m.next_r)
        table.insert(output, self.m.next_c)
        table.insert(output, self.m.next_h)
    else
        table.insert(output, p)
        table.insert(output, r)
        table.insert(output, next_c)
        table.insert(output, next_h)
    end

    table.insert(input, self.m.sub_term)
    return nn.gModule(input, output)
end

local HLSTM = torch.class("HLSTM", "MetaLSTM")
function HLSTM:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.p0 = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.c0 = nn.Identity()()
    self.m.h0 = nn.Identity()()
    self.m.arg1 = nn.Identity()()
    self.m.arg2 = nn.Identity()()

    if args.pass_term then
        self.m.sub_term = nn.Identity()()
    end
    
    local prev_r = self.m.r0
    local prev_p = self.m.p0
    local prev_c = self.m.c0
    local prev_h = self.m.h0

    local feat_size
    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local wdim = args.wdim
    local ldim = args.ldim
    local sdim = ldim/2

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)
    
    local context_l = self:low_context(args, self.m.arg1, self.m.arg2, prev_r, self.m.sub_term)
    local context_h = self:high_context(args, self.m.arg1, self.m.arg2, prev_r, self.m.sub_term)
    if context_h then
        self.m.context_fc = nn.Linear(edim, edim)(context_h)
    end
    if context_l then
        self.m.context_conv = nn.Linear(edim, args.ch * args.n_units[2])(context_l)
        self.m.conv_low, feat_size = self:build_conv(args, self.m.x, args.channels)

        self.m.mask_h2d = nn.View(args.ch, args.n_units[2]):setNumInputDims(1)(
                self.m.context_conv)
        local MMconv = nn.MM(false, false)
        self.m.conv_masked = MMconv({self.m.mask_h2d, self.m.conv_low})
        self.m.conv = nn.View(-1):setNumInputDims(2)(self.m.conv_masked)

        nel = feat_size * feat_size * args.ch
    else
        self.m.conv, nel = self:build_conv(args, self.m.x, args.channels)
    end

    self.m.fc = nn.Linear(nel, edim)(self.m.conv)
    self.m.fc_nl = nn.ReLU(true)(self.m.fc)

    -- LSTM
    local xh = nn.JoinTable(2)({self.m.fc_nl, prev_h})
    self.m.lstm_gate = nn.Linear(edim + ldim, edim)(xh)
    self.m.mul_h = nn.CMulTable()({self.m.context_fc, self.m.lstm_gate})
    self.m.lstm = nn.Linear(edim, 4*ldim)(self.m.mul_h)

    local reshaped = nn.View(4, ldim):setNumInputDims(1)(self.m.lstm)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    local in_gate = nn.Sigmoid(true)(n1)
    local forget_gate = nn.Sigmoid(true)(n2)
    local out_gate = nn.Sigmoid(true)(n3)
    local in_transform = nn.Tanh()(n4)
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    if args.ta then
        self.m.term_fc = nn.Linear(ldim, 2)(next_h)
        if args.soft_term then
            self.m.term_prob = nn.SoftMax()(self.m.term_fc)
            if args.soft_rho then 
                self.m.term_prob = nn.EntropyPenalty(0, false)(self.m.term_prob)
                table.insert(self.entropy_list, self.m.term_prob)
            end
        else
            self.m.term = nn.LogSoftMax()(self.m.term_fc)
            self.m.term_act = nn.Multinomial()(self.m.term)
            self.m.term_prob = nn.OneHot(2)(self.m.term_act)
        end
        self.m.t1 = nn.Narrow(2,1,1)(self.m.term_prob)
        self.m.t2 = nn.Narrow(2,2,1)(self.m.term_prob)

        if args.l1 > 0 and args.soft_term then
            self.m.t2 = nn.L1Penalty(args.l1)(self.m.t2)
        end
    end

    local r, p = self:retrieve_goal(args, next_h, prev_p, ldim)
    local hr = nn.JoinTable(2)({next_h, r})
    self.m.hr = nn.Linear(wdim + ldim, edim)(hr)
    self.m.s = nn.ReLU(true)(self.m.hr)
    
    if args.ta then
        self.m.g_prob = nn.CAddTable()({
            nn.ScalarMulTable()({prev_p, self.m.t1}),
            nn.ScalarMulTable()({p, self.m.t2})
        })

        self.m.next_r = nn.CAddTable()({
            nn.ScalarMulTable()({prev_r, self.m.t1}),
            nn.ScalarMulTable()({r, self.m.t2})
        }) 

        local prev_c2 = nn.Narrow(2, 1, sdim)(prev_c)
        local prev_h2 = nn.Narrow(2, 1, sdim)(prev_h)

        local next_c2 = nn.Narrow(2, 1, sdim)(next_c)
        local next_h2 = nn.Narrow(2, 1, sdim)(next_h)

        local merge_c = nn.CAddTable()({
            nn.ScalarMulTable()({prev_c2, self.m.t1}),
            nn.ScalarMulTable()({next_c2, self.m.t2})
        }) 

        local merge_h = nn.CAddTable()({
            nn.ScalarMulTable()({prev_h2, self.m.t1}),
            nn.ScalarMulTable()({next_h2, self.m.t2})
        }) 

        self.m.next_c = nn.JoinTable(2)({merge_c, nn.Narrow(2, sdim+1, sdim)(next_c)})
        self.m.next_h = nn.JoinTable(2)({merge_h, nn.Narrow(2, sdim+1, sdim)(next_h)})
    end

    local input = {self.m.x, self.m.g, self.m.p0, self.m.r0, self.m.c0, self.m.h0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        table.insert(input, self.m["arg" .. i])
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        if args.ta then
            self.m["a" .. i] = self:merge_pi(args, self.m.t1, self.m.t2, 
                self.m["arg" .. i], self.m["fc_" .. i])
        else
            self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        end

        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    if args.base then
        local goal3d = nn.View(-1, args.max_task, args.max_word)(self.m.g)
        self.m.g_list = nn.NotEqual(g_vocab['nil'])(nn.Select(3, 1)(goal3d))
        local pointer = p
        if args.ta then 
            pointer = self.m.g_prob
        end
        self.m.base_concat = nn.JoinTable(2)({self.m.g_list, pointer, self.m.s})
        self.m.base_fc = nn.Linear(edim + args.max_task * 2, edim)(self.m.base_concat)
        self.m.b = nn.Linear(edim, 1)(nn.ReLU(true)(self.m.base_fc))
    else
        self.m.b = nn.Linear(edim, 1)(self.m.s)
    end
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.ta and not args.soft_term then
        table.insert(self.n_actions, 2)
        table.insert(output, self.m.term) 
    end

    table.insert(output, self.m.b)
    if args.ta then
        table.insert(output, self.m.g_prob)
        table.insert(output, self.m.next_r)
        table.insert(output, self.m.next_c)
        table.insert(output, self.m.next_h)
    else
        table.insert(output, p)
        table.insert(output, r)
        table.insert(output, next_c)
        table.insert(output, next_h)
    end

    table.insert(input, self.m.sub_term)
    return nn.gModule(input, output)
end

local HRNN = torch.class("HRNN", "MetaLSTM")
function HRNN:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.p0 = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.c0 = nn.Identity()()
    self.m.h0 = nn.Identity()()
    self.m.arg1 = nn.Identity()()
    self.m.arg2 = nn.Identity()()
    self.m.sub_term = nn.Identity()()
    
    local prev_r = self.m.r0
    local prev_p = self.m.p0
    local prev_c = self.m.c0
    local prev_h = self.m.h0

    local feat_size
    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local wdim = args.wdim
    local ldim = args.ldim
    local sdim = ldim/2

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)
    
    local context_l = self:low_context(args, self.m.arg1, self.m.arg2, prev_r, self.m.sub_term)
    local context_h = self:high_context(args, self.m.arg1, self.m.arg2, prev_r, self.m.sub_term)
    if context_h then
        self.m.context_fc = nn.Linear(edim, edim)(context_h)
    end
    if context_l then
        self.m.context_conv = nn.Linear(edim, args.ch * args.n_units[2])(context_l)
        self.m.conv_low, feat_size = self:build_conv(args, self.m.x, args.channels)

        self.m.mask_h2d = nn.View(args.ch, args.n_units[2]):setNumInputDims(1)(
                self.m.context_conv)
        local MMconv = nn.MM(false, false)
        self.m.conv_masked = MMconv({self.m.mask_h2d, self.m.conv_low})
        self.m.conv = nn.View(-1):setNumInputDims(2)(self.m.conv_masked)

        nel = feat_size * feat_size * args.ch
    else
        self.m.conv, nel = self:build_conv(args, self.m.x, args.channels)
    end

    self.m.fc = nn.Linear(nel, edim)(self.m.conv)
    self.m.fc_nl = nn.ReLU(true)(self.m.fc)

    -- LSTM
    local xh = nn.JoinTable(2)({self.m.fc_nl, prev_h})
    self.m.lstm_gate = nn.Linear(edim + ldim, edim)(xh)
    self.m.mul_h = nn.CMulTable()({self.m.context_fc, self.m.lstm_gate})
    self.m.lstm = nn.Linear(edim, 4*ldim)(self.m.mul_h)

    local reshaped = nn.View(4, ldim):setNumInputDims(1)(self.m.lstm)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    local in_gate = nn.Sigmoid(true)(n1)
    local forget_gate = nn.Sigmoid(true)(n2)
    local out_gate = nn.Sigmoid(true)(n3)
    local in_transform = nn.Tanh()(n4)
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    local prev_c2 = nn.Narrow(2, 1, sdim)(prev_c)
    local prev_h2 = nn.Narrow(2, 1, sdim)(prev_h)

    local next_c2 = nn.Narrow(2, 1, sdim)(next_c)
    local next_h2 = nn.Narrow(2, 1, sdim)(next_h)
    
    local next_h_low = nn.Narrow(2, sdim+1, sdim)(next_h)

    local h_hl = nn.JoinTable(2)({prev_h2, next_h_low})
    self.m.term_fc = nn.Linear(ldim, 2)(h_hl)
    if args.soft_term then
        self.m.term_prob = nn.SoftMax()(self.m.term_fc)
        if args.soft_rho then 
            self.m.term_prob = nn.EntropyPenalty(0, false)(self.m.term_prob)
            table.insert(self.entropy_list, self.m.term_prob)
        end
    else
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        self.m.term_act = nn.Multinomial()(self.m.term)
        self.m.term_prob = nn.OneHot(2)(self.m.term_act)
    end
    self.m.t1 = nn.Narrow(2,1,1)(self.m.term_prob)
    self.m.t2 = nn.Narrow(2,2,1)(self.m.term_prob)

    if args.l1 > 0 and args.soft_term then
        self.m.t2 = nn.L1Penalty(args.l1)(self.m.t2)
    end

    local r, p = self:retrieve_goal(args, next_h, prev_p, ldim)
    local hr = nn.JoinTable(2)({next_h, r})
    self.m.hr = nn.Linear(wdim + ldim, edim)(hr)
    self.m.s = nn.ReLU(true)(self.m.hr)
    
    self.m.g_prob = nn.CAddTable()({
        nn.ScalarMulTable()({prev_p, self.m.t1}),
        nn.ScalarMulTable()({p, self.m.t2})
    })

    self.m.next_r = nn.CAddTable()({
        nn.ScalarMulTable()({prev_r, self.m.t1}),
        nn.ScalarMulTable()({r, self.m.t2})
    }) 

    local merge_c = nn.CAddTable()({
        nn.ScalarMulTable()({prev_c2, self.m.t1}),
        nn.ScalarMulTable()({next_c2, self.m.t2})
    }) 

    local merge_h = nn.CAddTable()({
        nn.ScalarMulTable()({prev_h2, self.m.t1}),
        nn.ScalarMulTable()({next_h2, self.m.t2})
    }) 

    self.m.next_c = nn.JoinTable(2)({merge_c, nn.Narrow(2, sdim+1, sdim)(next_c)})
    self.m.next_h = nn.JoinTable(2)({merge_h, nn.Narrow(2, sdim+1, sdim)(next_h)})

    local input = {self.m.x, self.m.g, self.m.p0, self.m.r0, self.m.c0, self.m.h0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        table.insert(input, self.m["arg" .. i])
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = self:merge_pi(args, self.m.t1, self.m.t2, 
            self.m["arg" .. i], self.m["fc_" .. i])

        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    if args.base then
        local goal3d = nn.View(-1, args.max_task, args.max_word)(self.m.g)
        self.m.g_list = nn.NotEqual(g_vocab['nil'])(nn.Select(3, 1)(goal3d))
        local pointer = p
        if args.ta then 
            pointer = self.m.g_prob
        end
        self.m.base_concat = nn.JoinTable(2)({self.m.g_list, pointer, self.m.s})
        self.m.base_fc = nn.Linear(edim + args.max_task * 2, edim)(self.m.base_concat)
        self.m.b = nn.Linear(edim, 1)(nn.ReLU(true)(self.m.base_fc))
    else
        self.m.b = nn.Linear(edim, 1)(self.m.s)
    end
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end
    if args.ta and not args.soft_term then
        table.insert(self.n_actions, 2)
        table.insert(output, self.m.term) 
    end

    table.insert(output, self.m.b)
    table.insert(output, self.m.g_prob)
    table.insert(output, self.m.next_r)
    table.insert(output, self.m.next_c)
    table.insert(output, self.m.next_h)

    table.insert(input, self.m.sub_term)
    return nn.gModule(input, output)
end

local HRNNOne = torch.class("HRNNOne", "MetaLSTM")
function HRNNOne:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.c0 = nn.Identity()()
    self.m.h0 = nn.Identity()()
    self.m.arg1 = nn.Identity()()
    self.m.arg2 = nn.Identity()()
    self.m.sub_term = nn.Identity()()
    
    local prev_r = self.m.r0
    local prev_c = self.m.c0
    local prev_h = self.m.h0

    local feat_size
    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local wdim = args.wdim
    local ldim = args.ldim
    local sdim = ldim/2

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)
    
    local context_l = self:low_context(args, self.m.arg1, self.m.arg2, prev_r, self.m.sub_term)
    local context_h = self:high_context(args, self.m.arg1, self.m.arg2, prev_r, self.m.sub_term)
    if context_h then
        self.m.context_fc = nn.Linear(edim, edim)(context_h)
    end
    if context_l then
        self.m.context_conv = nn.Linear(edim, args.ch * args.n_units[2])(context_l)
        self.m.conv_low, feat_size = self:build_conv(args, self.m.x, args.channels)

        self.m.mask_h2d = nn.View(args.ch, args.n_units[2]):setNumInputDims(1)(
                self.m.context_conv)
        local MMconv = nn.MM(false, false)
        self.m.conv_masked = MMconv({self.m.mask_h2d, self.m.conv_low})
        self.m.conv = nn.View(-1):setNumInputDims(2)(self.m.conv_masked)

        nel = feat_size * feat_size * args.ch
    else
        self.m.conv, nel = self:build_conv(args, self.m.x, args.channels)
    end

    self.m.fc = nn.Linear(nel, edim)(self.m.conv)
    self.m.fc_nl = nn.ReLU(true)(self.m.fc)

    -- LSTM
    local xh = nn.JoinTable(2)({self.m.fc_nl, prev_h})
    self.m.lstm_gate = nn.Linear(edim + ldim, edim)(xh)
    self.m.mul_h = nn.CMulTable()({self.m.context_fc, self.m.lstm_gate})
    self.m.lstm = nn.Linear(edim, 4*ldim)(self.m.mul_h)

    local reshaped = nn.View(4, ldim):setNumInputDims(1)(self.m.lstm)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    local in_gate = nn.Sigmoid(true)(n1)
    local forget_gate = nn.Sigmoid(true)(n2)
    local out_gate = nn.Sigmoid(true)(n3)
    local in_transform = nn.Tanh()(n4)
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    local prev_c2 = nn.Narrow(2, 1, sdim)(prev_c)
    local prev_h2 = nn.Narrow(2, 1, sdim)(prev_h)

    local next_c2 = nn.Narrow(2, 1, sdim)(next_c)
    local next_h2 = nn.Narrow(2, 1, sdim)(next_h)
    
    local next_h_low = nn.Narrow(2, sdim+1, sdim)(next_h)

    local h_hl = nn.JoinTable(2)({prev_h2, next_h_low})
    self.m.term_fc = nn.Linear(ldim, 2)(h_hl)
    if args.soft_term then
        self.m.term_prob = nn.SoftMax()(self.m.term_fc)
        if args.soft_rho then 
            self.m.term_prob = nn.EntropyPenalty(0, false)(self.m.term_prob)
            table.insert(self.entropy_list, self.m.term_prob)
        end
    else
        self.m.term = nn.LogSoftMax()(self.m.term_fc)
        self.m.term_act = nn.Multinomial()(self.m.term)
        self.m.term_prob = nn.OneHot(2)(self.m.term_act)
    end
    self.m.t1 = nn.Narrow(2,1,1)(self.m.term_prob)
    self.m.t2 = nn.Narrow(2,2,1)(self.m.term_prob)

    if args.l1 > 0 and args.soft_term then
        self.m.t2 = nn.L1Penalty(args.l1)(self.m.t2)
    end

    local r = nn.View(-1):setNumInputDims(2)(self.m.g_val)
    local hr = nn.JoinTable(2)({next_h, r})
    self.m.hr = nn.Linear(wdim + ldim, edim)(hr)
    self.m.s = nn.ReLU(true)(self.m.hr)
    
    self.m.next_r = nn.CAddTable()({
        nn.ScalarMulTable()({prev_r, self.m.t1}),
        nn.ScalarMulTable()({r, self.m.t2})
    }) 

    local merge_c = nn.CAddTable()({
        nn.ScalarMulTable()({prev_c2, self.m.t1}),
        nn.ScalarMulTable()({next_c2, self.m.t2})
    }) 

    local merge_h = nn.CAddTable()({
        nn.ScalarMulTable()({prev_h2, self.m.t1}),
        nn.ScalarMulTable()({next_h2, self.m.t2})
    }) 

    self.m.next_c = nn.JoinTable(2)({merge_c, nn.Narrow(2, sdim+1, sdim)(next_c)})
    self.m.next_h = nn.JoinTable(2)({merge_h, nn.Narrow(2, sdim+1, sdim)(next_h)})

    local input = {self.m.x, self.m.g, self.m.r0, self.m.c0, self.m.h0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        table.insert(input, self.m["arg" .. i])
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        self.m["a" .. i] = self:merge_pi(args, self.m.t1, self.m.t2, 
            self.m["arg" .. i], self.m["fc_" .. i])

        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    if args.ta and not args.soft_term then
        table.insert(self.n_actions, 2)
        table.insert(output, self.m.term) 
    end

    table.insert(output, self.m.b)
    table.insert(output, self.m.next_r)
    table.insert(output, self.m.next_c)
    table.insert(output, self.m.next_h)

    table.insert(input, self.m.sub_term)
    return nn.gModule(input, output)
end

function HRNNOne:build_init_states(args)
    local states = {}
    table.insert(states, torch.Tensor(1, args.wdim))      -- r
    table.insert(states, torch.Tensor(1, args.ldim))      -- c
    table.insert(states, torch.Tensor(1, args.ldim))      -- h
    return states
end


local MetaRegister = torch.class('MetaRegister', 'MetaRNN')
function MetaRegister:build_init_states(args)
    local states = {}
    table.insert(states, torch.Tensor(1, args.max_task))  -- p
    table.insert(states, torch.Tensor(1, args.odim))      -- o 
    table.insert(states, torch.Tensor(1, args.wdim))      -- r
    return states
end

function MetaRegister:reset_init_state(batch_size)
    for j=1,#self.init_states do
        local size = self.init_states[j]:size()
        size[1] = batch_size
        self.init_states[j]:resize(size)
        self.init_states[j]:fill(0)
    end
    -- Set initial memory attention to 1, 0, 0, ...
    self.init_states[1]:narrow(2, 1, 1):fill(1)
    self.init_states[2]:narrow(2, 1, 1):fill(1)
end

function MetaRegister:register_embedding(args, prev_o)
    if not self.m.o_embed then
        self.m.o_embed = nn.Linear(args.odim, args.edim)(prev_o)
        self.m.o_embed_nl = nn.ReLU(true)(self.m.o_embed)
    end
    return self.m.o_embed_nl
end

function MetaRegister:context_embedding(args, arg1, arg2, prev_r, term, prev_o)
    if not self.m.context then
        local g = self:goal_embedding(args, arg1, arg2)
        local r = self:read_embedding(args, prev_r)
        local t = self:term_embedding(args, term)
        local o = self:register_embedding(args, prev_o)
        local grto = nn.JoinTable(2)({g, r, t, o})
        self.m.context_grto = nn.Linear(4 * args.edim, args.edim)(grto)
        self.m.context = nn.ReLU(true)(self.m.context_grto)
    end
    return self.m.context
end

function MetaRegister:low_context(args, arg1, arg2, prev_r, term, prev_o)
    return self:context_embedding(args, arg1, arg2, prev_r, term, prev_o)
end

function MetaRegister:high_context(args, arg1, arg2, prev_r, term, prev_o)
    return self:context_embedding(args, arg1, arg2, prev_r, term, prev_o)
end


function MetaRegister:modify_register(args, prev_o, h, hdim)
    self.m.op_fc = nn.Linear(hdim, 2 * args.shift + 2)(h)
    if args.soft_mem then 
        self.m.op_prob = nn.SoftMax()(self.m.op_fc)
    else
        self.m.op_shift = nn.LogSoftMax()(self.m.op_fc)
        self.m.op_shift_act = nn.Multinomial()(self.m.op_shift)
        self.m.op_prob = nn.OneHot(2 * args.shift + 2)(self.m.op_shift_act)
    end

    self.m.o = nn.Register()({prev_o, self.m.op_prob})
    return self.m.o 
end

function MetaRegister:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.p0 = nn.Identity()()
    self.m.o0 = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.arg1 = nn.Identity()()
    self.m.arg2 = nn.Identity()()
    self.m.sub_term = nn.Identity()()
    
    local prev_p = self.m.p0
    local prev_o = self.m.o0
    local prev_r = self.m.r0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local wdim = args.wdim
    local odim = args.odim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)
    
    local context_l = self:low_context(args, self.m.arg1, self.m.arg2, prev_r, self.m.sub_term, prev_o)
    local context_h = self:high_context(args, self.m.arg1, self.m.arg2, prev_r, self.m.sub_term, prev_o)
    if context_h then
        self.m.context_fc = nn.Linear(edim, edim)(context_h)
    end
    if context_l then
        self.m.context_conv = nn.Linear(edim, args.ch * args.n_units[2])(context_l)
        self.m.conv_low, feat_size = self:build_conv(args, self.m.x, args.channels)

        self.m.mask_h2d = nn.View(args.ch, args.n_units[2]):setNumInputDims(1)(
                self.m.context_conv)
        local MMconv = nn.MM(false, false)
        self.m.conv_masked = MMconv({self.m.mask_h2d, self.m.conv_low})
        self.m.conv = nn.View(-1):setNumInputDims(2)(self.m.conv_masked)

        nel = feat_size * feat_size * args.ch
    else
        self.m.conv, nel = self:build_conv(args, self.m.x, args.channels)
    end

    self.m.fc = nn.Linear(nel, edim)(self.m.conv)
    if context_h then
        self.m.mul_h = nn.CMulTable()({self.m.context_fc, self.m.fc})
        self.m.state = nn.Linear(edim, edim)(self.m.mul_h)
        self.m.h = nn.ReLU(true)(self.m.state)
    else
        self.m.h = nn.ReLU(true)(self.m.fc)
    end

    if args.ta then
        self.m.term_fc = nn.Linear(edim, 2)(self.m.h)
        if args.soft_term then
            self.m.term_prob = nn.SoftMax()(self.m.term_fc)
            if args.soft_rho then 
                self.m.term_prob = nn.EntropyPenalty(0, false)(self.m.term_prob)
                table.insert(self.entropy_list, self.m.term_prob)
            end
        else
            self.m.term = nn.LogSoftMax()(self.m.term_fc)
            self.m.term_act = nn.Multinomial()(self.m.term)
            self.m.term_prob = nn.OneHot(2)(self.m.term_act)
        end
        self.m.t1 = nn.Narrow(2,1,1)(self.m.term_prob)
        self.m.t2 = nn.Narrow(2,2,1)(self.m.term_prob)

        if args.l1 > 0 and args.soft_term then
            self.m.t2 = nn.L1Penalty(args.l1)(self.m.t2)
        end
    end

    local o = self:modify_register(args, prev_o, self.m.h, edim)
    local r, p = self:retrieve_goal(args, self.m.h, prev_p, edim)
    local hro = nn.JoinTable(2)({self.m.h, r, o})
    self.m.hro = nn.Linear(wdim + edim + odim, edim)(hro)
    self.m.s = nn.ReLU(true)(self.m.hro)
    
    if args.ta then
        self.m.g_prob = nn.CAddTable()({
            nn.ScalarMulTable()({prev_p, self.m.t1}),
            nn.ScalarMulTable()({p, self.m.t2})
        })

        self.m.next_r = nn.CAddTable()({
            nn.ScalarMulTable()({prev_r, self.m.t1}),
            nn.ScalarMulTable()({r, self.m.t2})
        }) 

        self.m.next_o = nn.CAddTable()({
            nn.ScalarMulTable()({prev_o, self.m.t1}),
            nn.ScalarMulTable()({o, self.m.t2})
        }) 
    end

    local input = {self.m.x, self.m.g, self.m.p0, self.m.o0, self.m.r0}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        table.insert(input, self.m["arg" .. i])
        self.m["fc_" .. i] = nn.Linear(edim, args.num_args[i])(self.m.s)
        if args.ta then
            self.m["a" .. i] = self:merge_pi(args, self.m.t1, self.m.t2, 
                self.m["arg" .. i], self.m["fc_" .. i])
        else
            self.m["a" .. i] = nn.LogSoftMax()(self.m["fc_" .. i])
        end

        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end
    if args.base then
        local goal3d = nn.View(-1, args.max_task, args.max_word)(self.m.g)
        self.m.g_list = nn.NotEqual(g_vocab['nil'])(nn.Select(3, 1)(goal3d))
        local pointer = p
        if args.ta then 
            pointer = self.m.g_prob
        end
        self.m.base_concat = nn.JoinTable(2)({self.m.g_list, pointer, self.m.s})
        self.m.base_fc = nn.Linear(edim + args.max_task * 2, edim)(self.m.base_concat)
        self.m.b = nn.Linear(edim, 1)(nn.ReLU(true)(self.m.base_fc))
    else
        self.m.b = nn.Linear(edim, 1)(self.m.s)
    end
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)

        table.insert(self.n_actions, 2*args.shift+2)
        table.insert(output, self.m.op_shift)
    end
    if args.ta and not args.soft_term then
        table.insert(self.n_actions, 2)
        table.insert(output, self.m.term) 
    end

    table.insert(output, self.m.b)
    if args.ta then
        table.insert(output, self.m.g_prob)
        table.insert(output, self.m.next_o)
        table.insert(output, self.m.next_r)
    else
        table.insert(output, p)
        table.insert(output, o)
        table.insert(output, r)
    end

    table.insert(input, self.m.sub_term)
    return nn.gModule(input, output)
end

function MetaRegister:fill_internal_actions()
    if not self.args.soft_mem then
        table.insert(self.actions, self.m.shift_act.data.module.output:long())
        table.insert(self.actions, self.m.op_shift_act.data.module.output:long())
    end
    if not self.args.soft_term and self.m.term_act then
        table.insert(self.actions, self.m.term_act.data.module.output:long())
    end
end

function MetaRegister:add_l1_regularization()
    if g_opts.l1 > 0 then
        local idx = #self.args.num_args + 1
        if not self.args.soft_mem then
            idx = idx + 2
        end
        --hard termination
        if self.m.term and not self.args.soft_term then
            self.l1_penalty = self.l1_penalty or nn.Sequential():add(
                    nn.Select(2, 2)):add(
                    nn.Exp()):add(
                    nn.L1Penalty(self.args.l1))
            local out = self.l1_penalty:forward(self.m.term.data.module.output)
            self.l1_grad_output = self.l1_grad_output or 
                    torch.zeros(out:size())
            local grad = self.l1_penalty:backward(self.m.term.data.module.output, 
                    self.l1_grad_output)
            -- print(grad)
            self.grad_output[idx]:add(grad)
        end
    end
end


local FlatRegister = torch.class('FlatRegister', 'FF11')
function FlatRegister:build_init_states(args)
    local states = {}
    table.insert(states, torch.Tensor(1, args.max_task))  -- p
    table.insert(states, torch.Tensor(1, args.odim))      -- o 
    table.insert(states, torch.Tensor(1, args.wdim))      -- r
    return states
end

function FlatRegister:reset_init_state(batch_size)
    for j=1,#self.init_states do
        local size = self.init_states[j]:size()
        size[1] = batch_size
        self.init_states[j]:resize(size)
        self.init_states[j]:fill(0)
    end
    -- Set initial memory attention to 1, 0, 0, ...
    self.init_states[1]:narrow(2, 1, 1):fill(1)
    self.init_states[2]:narrow(2, 1, 1):fill(1)
end

function FlatRegister:read_embedding(args, prev_r)
    if not self.m.mask_r_nl then
        self.m.mask_r = nn.Linear(args.wdim, args.edim)(prev_r)
        self.m.mask_r_nl = nn.ReLU(true)(self.m.mask_r)
    end
    return self.m.mask_r_nl
end

function FlatRegister:register_embedding(args, prev_o)
    if not self.m.o_embed then
        self.m.o_embed = nn.Linear(args.odim, args.edim)(prev_o)
        self.m.o_embed_nl = nn.ReLU(true)(self.m.o_embed)
    end
    return self.m.o_embed_nl
end

function FlatRegister:context_embedding(args, prev_r, prev_o)
    if not self.m.context then
        local r = self:read_embedding(args, prev_r)
        local o = self:register_embedding(args, prev_o)
        local ro = nn.JoinTable(2)({r, o})
        self.m.context_ro = nn.Linear(2 * args.edim, args.edim)(ro)
        self.m.context = nn.ReLU(true)(self.m.context_ro)
    end
    return self.m.context
end

function FlatRegister:low_context(args, prev_r, prev_o)
    return self:context_embedding(args, prev_r, prev_o)
end

function FlatRegister:high_context(args, prev_r, prev_o)
    return self:context_embedding(args, prev_r, prev_o)
end

function FlatRegister:modify_register(args, prev_o, h, hdim)
    self.m.op_fc = nn.Linear(hdim, 2 * args.shift + 2)(h)
    if args.soft_mem then 
        self.m.op_prob = nn.SoftMax()(self.m.op_fc)
    else
        self.m.op_shift = nn.LogSoftMax()(self.m.op_fc)
        self.m.op_shift_act = nn.Multinomial()(self.m.op_shift)
        self.m.op_prob = nn.OneHot(2 * args.shift + 2)(self.m.op_shift_act)
    end

    self.m.o = nn.Register()({prev_o, self.m.op_prob})
    return self.m.o 
end

function FlatRegister:add_entropy_grad(rho)
    self.entropy = self.entropy or {}
    for i=1,#self.n_actions do
        self.entropy[i] = self.entropy[i] or nn.Entropy()
    end
    local idx = 1
    --local entropy = self.entropy[i]:forward(self.m["a" .. i].data.module.output)
    local g = self.entropy[1]:backward(self.m.a.data.module.output, 1)
    --print("Entropy", entropy, "GradNorm", g:norm(), self.grad_output[idx]:norm())
    self.grad_output[idx]:add(g:mul(-rho))
    if not self.args.soft_mem then
        idx = idx + 1
        local g = self.entropy[idx]:backward(self.m.g_shift.data.module.output, 1)
        self.grad_output[idx]:add(g:mul(-rho))
    end
end


function FlatRegister:build_conv(args, input, nInput)
    if #args.n_units > 0 then
        self.m.conv1 = nn.SpatialConvolution(nInput, args.n_units[1],
                            args.filter_size[1], args.filter_size[1],
                            args.filter_stride[1], args.filter_stride[1],
                            args.pad[1], args.pad[1])(input)
        self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
        if #args.n_units == 1 then
            if args.pool[1] then
                self.m.pool1 = nn.SpatialMaxPooling(args.pool[1], args.pool[1], 
                        args.pool[1],args.pool[1])(self.m.conv1_nl)
            end
            return nn.View(-1):setNumInputDims(3)(self.m.pool1),
                5 * 5 * args.n_units[1]
        end
        for i=1,(#args.n_units-1) do
            if args.pool[i] then
                self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                        args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            end
            self.m["conv" .. i+1] = nn.SpatialConvolution(args.n_units[i], args.n_units[i+1],
                                args.filter_size[i+1], args.filter_size[i+1],
                                args.filter_stride[i+1], args.filter_stride[i+1],
                                args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
            self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        end
        return nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"]),
                5 * 5 * args.n_units[#args.n_units]
    else
        return input, 10 * 10 * nInput
    end
end

function FlatRegister:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.p0 = nn.Identity()()
    self.m.o0 = nn.Identity()()
    self.m.r0 = nn.Identity()()
    
    local prev_p = self.m.p0
    local prev_o = self.m.o0
    local prev_r = self.m.r0

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local wdim = args.wdim
    local odim = args.odim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)
    
    local context_l = self:low_context(args, prev_r, prev_o)
    local context_h = self:high_context(args, prev_r, prev_o)
    if context_h then
        self.m.context_fc = nn.Linear(edim, edim)(context_h)
    end
    if context_l then
        self.m.context_conv = nn.Linear(edim, args.ch * args.buf * args.channels)(context_l)
        self.m.mask_h2d = nn.View(args.ch, args.buf * args.channels):setNumInputDims(1)(self.m.context_conv)
        local x_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
        local MMconv = nn.MM(false, false)
        self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(
                MMconv({self.m.mask_h2d, x_2d}))
        self.m.conv, nel = self:build_conv(args, self.m.x_masked, args.ch)
    else
        self.m.conv, nel = self:build_conv(args, self.m.x, args.buf * args.channels)
    end

    self.m.fc1 = nn.Linear(nel, edim)(self.m.conv)
    if context_h then
        self.m.mul_h = nn.CMulTable()({self.m.context_fc, self.m.fc1})
        self.m.state = nn.Linear(edim, edim)(self.m.mul_h)
        self.m.h = nn.ReLU(true)(self.m.state)
    else
        self.m.h = nn.ReLU(true)(self.m.fc1)
    end

    local o = self:modify_register(args, prev_o, self.m.h, edim)
    local r, p = self:retrieve_goal(args, self.m.h, prev_p, edim)
    local hro = nn.JoinTable(2)({self.m.h, r, o})
    self.m.hro = nn.Linear(wdim + edim + odim, edim)(hro)
    self.m.s = nn.ReLU(true)(self.m.hro)
    
    self.m.fc = nn.Linear(edim, args.n_actions)(self.m.s)
    self.m.a = nn.LogSoftMax()(self.m.fc)
    local output = {self.m.a}
    self.n_actions = {args.n_actions}

    if args.base then
        local goal3d = nn.View(-1, args.max_task, args.max_word)(self.m.g)
        self.m.g_list = nn.NotEqual(g_vocab['nil'])(nn.Select(3, 1)(goal3d))
        local pointer = p
        if args.ta then 
            pointer = self.m.g_prob
        end
        self.m.base_concat = nn.JoinTable(2)({self.m.g_list, pointer, self.m.s})
        self.m.base_fc = nn.Linear(edim + args.max_task * 2, edim)(self.m.base_concat)
        self.m.b = nn.Linear(edim, 1)(nn.ReLU(true)(self.m.base_fc))
    else
        self.m.b = nn.Linear(edim, 1)(self.m.s)
    end
    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)

        table.insert(self.n_actions, 2*args.shift+2)
        table.insert(output, self.m.op_shift)
    end

    local input = {self.m.x, self.m.g, self.m.p0, self.m.o0, self.m.r0}
    table.insert(output, self.m.b)
    table.insert(output, p)
    table.insert(output, o)
    table.insert(output, r)

    return nn.gModule(input, output)
end

function FlatRegister:fill_internal_actions()
    if not self.args.soft_mem then
        table.insert(self.actions, self.m.shift_act.data.module.output:long())
        table.insert(self.actions, self.m.op_shift_act.data.module.output:long())
    end
end

function g_create_network(args)

    args.edim           = args.edim or 256
    args.lindim         = args.lindim or args.edim / 2
    args.wdim           = args.wdim or 40
    args.ldim           = args.ldim or 40
    args.shift          = args.shift or 1

    local new_args = {}
    for k,v in pairs(args) do
        new_args[k] = v
    end

    return _G[args.name](new_args)
end

function g_init_model()
    -- model initialize
    if g_opts.load ~= "" then
        local msg, f = pcall(torch.load, g_opts.load)
        if not msg then
            error("Could not find network file " .. g_opts.load)
        else
            if f.model then
                g_model = f.model
                g_log = f.log
                g_plot_stat = {}
                for i = 1, #g_log do
                    g_plot_stat[i] = {g_log[i].epoch, g_log[i].reward, 
                        g_log[i].success, g_log[i].bl_cost}
                end
                if f['rmsprop_state'] then g_rmsprop_state = f['rmsprop_state'] end
            else
                g_model = f
                if g_opts.sub_agent and g_opts.sub_agent ~= "" then
                    g_model.sub_agent = torch.load(g_opts.sub_agent)
                    print(g_opts.sub_agent .. " loaded")
                end
            end
            if g_opts.open_loop then
                g_model.args.open_loop = g_opts.open_loop
            end
        end
    else
        local args = {}
        args.name         = g_opts.model
        args.edim         = g_opts.edim
        args.ldim         = g_opts.ldim
        args.wdim         = g_opts.wdim
        args.odim         = g_opts.odim or 10
        args.sub_agent    = g_opts.sub_agent -- parse_to_table(g_opts.sub_agents)
        args.n_actions    = g_opts.n_actions
        args.channels     = g_opts.channels
        args.max_task     = g_opts.max_task
        args.max_word     = g_opts.max_word
        args.vocab_size   = g_opts.nwords
        args.shift        = g_opts.shift or 1
        args.n_units      = parse_to_num_table(g_opts.n_filters)
        args.pool         = parse_to_num_table(g_opts.pool)
        args.filter_size  = parse_to_num_table(g_opts.filter_size)
        args.filter_stride= parse_to_num_table(g_opts.stride)
        args.pad          = parse_to_num_table(g_opts.pad)
        args.feat_size    = g_opts.feat_size 
        args.content_att  = g_opts.content_att or false
        args.soft_mem     = g_opts.soft_mem
        args.soft_term    = g_opts.soft_mem
        args.max_args     = g_opts.max_args
        args.num_args     = g_opts.num_args
        args.term_act     = g_opts.term_action
        args.feasible     = g_opts.feasible
        args.regularizer  = g_opts.regularizer
        args.buf          = g_opts.buf
        args.ch           = g_opts.ch or 3
        args.l1           = g_opts.l1 or 0
        args.drop         = g_opts.drop or 0
        args.l1_arg1      = g_opts.l1_arg1 or 0
        args.l1_arg2      = g_opts.l1_arg2 or 0
        args.soft_rho     = g_opts.soft_rho
        args.ta           = g_opts.ta or false
        args.base         = g_opts.base or false
        args.pass_term    = g_opts.pass_term
        args.open_loop    = g_opts.open_loop or false
        args.input_dim    = g_opts.img_size
        args.convLayer    = nn.SpatialConvolution 

        if g_opts.gpu > 0 then
            args.convLayer = cudnn.SpatialConvolution
        end
        g_model = g_create_network(args)

        if g_model.sub_agent and g_opts.initialize then
            -- g_model:copy_weight_from(g_model.sub_agent.m, true)
            --[[
            if g_model.sub_agent.m.arg_h1 and g_model.m.arg_h1 then
                g_model.m.arg_h1.data.module.weight:copy(
                    g_model.sub_agent.m.arg_h1.data.module.weight:transpose(1, 2))
                g_model.m.arg_h2.data.module.weight:copy(
                    g_model.sub_agent.m.arg_h2.data.module.weight:transpose(1, 2))
            end
            --]] 
            --[[
            local input_channel = g_model.m.conv1.data.module.nInputPlane
            g_model.m.conv1.data.module.weight:copy(
                g_model.sub_agent.m.conv1.data.module.weight:narrow(
                2, 1, input_channel))
            g_model.m.conv1.data.module.bias:copy(
                g_model.sub_agent.m.conv1.data.module.bias)
            print("Conv1 filters are initialized by sub-agent")
            --]]
            --
            g_model.m.conv1.data.module.weight:copy(
                g_model.sub_agent.m.conv1.data.module.weight)
            g_model.m.conv1.data.module.bias:copy(
                g_model.sub_agent.m.conv1.data.module.bias)
            g_model.m.conv2.data.module.weight:copy(
                g_model.sub_agent.m.conv2.data.module.weight)
            g_model.m.conv2.data.module.bias:copy(
                g_model.sub_agent.m.conv2.data.module.bias)
            print("Conv1/2 filters are initialized by sub-agent")
        end
    end
    if g_opts.param and g_opts.param ~= "" then
        local msg, f = pcall(torch.load, g_opts.param)
        if not msg then
            error("Could not find network file " .. g_opts.param)
        else
            if f.m then 
                g_model:copy_weight_from(f.m)
                print("Weights are initiailized from " .. g_opts.param)
            else
                g_paramx, g_paramdx = g_model:getParameters()
                g_paramx:copy(f)
            end
        end
    end
    if g_opts.teachers and g_opts.teachers ~= "" and g_opts.test ~= true then
        local teacher_path = parse_to_table(g_opts.teachers)
        if #teacher_path > 1 then
            local num_teachers = torch.Tensor(unpack(g_opts.num_args)):nElement()
            assert(#teacher_path == num_teachers, 
                "num teachers should be " .. num_teachers)
        end
        g_teacher = {}
        for i=1,#teacher_path do
            if teacher_path[i] ~= "" and teacher_path[i] ~= " " then
                local msg, m = pcall(torch.load, teacher_path[i])
                if not msg then
                    error("Could not find network file " .. teacher_path[i])
                end
                g_teacher[i] = m
            end
        end
        g_dist_loss = nn.DistKLDivCriterion()
    end
    if g_opts.test ~= true then
        g_paramx, g_paramdx = g_model:getParameters()
        -- make clones for training
        g_model_t = {}
        for t=1, g_opts.max_step do
            table.insert(g_model_t, g_model:clone(true))
        end
        g_bl_loss = nn.MSECriterion()

        if g_opts.gpu > 0 then
            g_bl_loss:cuda()
        end
    end
    if g_opts.term_action then
        g_term_loss = nn.ClassNLLCriterion()

        if g_opts.gpu > 0 then
            g_term_loss:cuda()
        end
    end
    if g_opts.regularizer and g_opts.regularizer > 0 then
        g_arg_loss = nn.MSECriterion()
        if g_opts.gpu > 0 then
            g_arg_loss:cuda()
        end
    end
    collectgarbage()
end

function g_save_model()
    if g_opts.save ~= '' then
        if g_log[#g_log].test[1].term_acc then
            if g_opts.best_success_acc == nil or 
                g_log[#g_log].test[1].success + g_log[#g_log].test[1].term_acc > g_opts.best_success_acc then
                torch.save(g_opts.save .. '.t7', g_model)
                g_opts.best_success_acc = g_log[#g_log].test[1].success + g_log[#g_log].test[1].term_acc
            end
        else
            if g_opts.best_reward == nil or g_log[#g_log].test[1].reward > g_opts.best_reward then
                torch.save(g_opts.save .. '.t7', g_model)
                g_opts.best_reward = g_log[#g_log].test[1].reward
            end
        end
        if g_opts.save_param then
            torch.save(string.format("%s.%03d.param.t7", g_opts.save, #g_log), g_paramx)
        end
        torch.save(g_opts.save .. '.latest.t7', g_model)
        torch.save(g_opts.save .. '.stat.t7', {opts=g_opts, log=g_log})
        local log_filename = g_opts.save .. '.log'
        local log_file = io.open(log_filename, 'a')
        log_file:write(#g_log)
        for i = 1, #g_log[#g_log].test do
            log_file:write(string.format(' %.3f', g_log[#g_log].test[i].reward))
        end
        if g_log[#g_log].test[1].success then
            for i = 1, #g_log[#g_log].test do
                log_file:write(string.format(' %.3f', g_log[#g_log].test[i].success))
            end
        end
        if g_log[#g_log].test[1].term_acc then
            for i = 1, #g_log[#g_log].test do
                log_file:write(string.format(' %.3f', g_log[#g_log].test[i].term_acc))
            end
        end
        log_file:write('\n')
        log_file:flush()
        log_file:close()
        print('model saved to ' .. g_opts.save)
    end
end

local Pretrain = torch.class("Pretrain", "Manager")
function Pretrain:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    
    local edim = args.edim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)
    self.m.g_embed = nn.View(-1):setNumInputDims(2)(self.m.g_val)
    self.m.g_fc = nn.Linear(wdim, edim)(self.m.g_embed)
    self.m.h = nn.ReLU(true)(self.m.g_fc)

    local input = {self.m.x, self.m.g}
    local output = {}
    self.n_actions = {}
    for i=1,#args.num_args do
        self.m["arg_" .. i .. "_fc"] = nn.Linear(edim, args.num_args[i])(self.m.h)
        self.m["a" .. i] = nn.LogSoftMax()(self.m["arg_" .. i .. "_fc"])
        self.m["arg_" .. i .. "_sample"] = nn.Multinomial()(self.m["a" .. i])
        self.m["arg_" .. i] = nn.View(-1)(self.m["arg_" .. i .. "_sample"])
    end

    self.m.arg_h1 = nn.LookupTable(args.num_args[1], args.edim)(self.m.arg_1)
    self.m.arg_h2 = nn.LookupTable(args.num_args[2], args.edim)(self.m.arg_2)
    local arg_hh = nn.ReLU(true)(nn.CMulTable()({self.m.arg_h1, self.m.arg_h2}))
    self.m.arg_h22 = nn.Linear(args.edim, args.ch * args.buf * args.channels)(arg_hh)
    self.m.arg_h2d = nn.View(args.ch, 
            args.buf * args.channels):setNumInputDims(1)(self.m.arg_h22)
    local conv1_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(
        MMconv({self.m.arg_h2d, conv1_2d}))

    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.ch, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_masked)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)

    local prev_input = self.m.conv1_nl
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        local nInput = args.n_units[i]
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.arg_fc = nn.Linear(args.edim, args.edim)(arg_hh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)
    -- self.m.joint_h_nl = nn.ReLU(true)(self.m.fc1)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local output = {self.m.a}
    self.n_actions = {args.n_actions}

    for i=1,#args.num_args do
        table.insert(self.n_actions, args.num_args[i])
        table.insert(output, self.m["a" .. i])
    end

    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term_softmax = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term_softmax)
    end
	
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

function Pretrain:build_init_states(args)
 	return {}
end

function Pretrain:reset_init_state(batch_size)
end

function Pretrain:fill_internal_actions()
    for i=1,#self.args.num_args do
        table.insert(self.actions, 
            self.m["arg_" .. i .. "_sample"].data.module.output:long())
    end
end

function Pretrain:add_entropy_grad(rho)
    self.entropy = self.entropy or {}
    for i=1,#self.n_actions do
        self.entropy[i] = self.entropy[i] or nn.Entropy()
    end
    local g = self.entropy[1]:backward(self.m.a.data.module.output, 1)
    self.grad_output[1]:add(g:mul(-rho))
    local idx = 1
    for i=1,#self.args.num_args do
        idx = idx + 1
        --local entropy = self.entropy[i]:forward(self.m["a" .. i].data.module.output)
        local g = self.entropy[i]:backward(self.m["a" .. i].data.module.output, 1)
        --print("Entropy", entropy, "GradNorm", g:norm(), self.grad_output[idx]:norm())
        self.grad_output[idx]:add(g:mul(-rho))
    end
    if self.entropy_list then 
        for i = 1,#self.entropy_list do
            self.entropy_list[i].data.module.w = -rho
        end
    end
end

function Pretrain:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h22 = nn.Linear(self.args.edim, 
                self.args.ch * self.args.buf * self.args.channels):share(
            self.m.arg_h22.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_h22)
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)

    local analogy2 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_fc = nn.Linear(self.args.edim, self.args.edim):share(
            self.m.arg_fc.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_fc)
        table.insert(analogy2, module)
    end
    table.insert(self.analogy_list, analogy2)
    return self.analogy_list
end


local PretrainB = torch.class("PretrainB", "Manager")
function PretrainB:build_conv(args, input, nInput)
    self.m.conv1 = nn.SpatialConvolution(nInput, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(input)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    for i=1,(#args.n_units-1) do
        if args.pool[i] then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(self.m["pool" .. i])
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
    end
    return nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
end

function PretrainB:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    
    local nel = 5 * 5 * args.n_units[#args.n_units]
    local edim = args.edim
    local ldim = args.ldim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)
    self.m.g_embed = nn.View(-1):setNumInputDims(2)(self.m.g_val)

    -- Compute CNN embeddings
    self.m.mask_r = nn.Linear(args.wdim, args.edim)(self.m.g_embed)
    self.m.mask_r_nl = nn.ReLU(true)(self.m.mask_r)
    self.m.mask_r2 = nn.Linear(args.edim, args.ch * args.buf * args.channels)(self.m.mask_r_nl)
    self.m.mask_fc = nn.Linear(args.edim, args.edim)(self.m.mask_r_nl)
    self.m.mask_h2d = nn.View(args.ch, args.buf * args.channels):setNumInputDims(1)(self.m.mask_r2)
    local x_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(MMconv({self.m.mask_h2d, x_2d}))
    
    self.m.conv = self:build_conv(args, self.m.x_masked, args.ch)
    self.m.fc1 = nn.Linear(nel, args.edim)(self.m.conv)
    self.m.mul_h = nn.CMulTable()({self.m.mask_fc, self.m.fc1})
    self.m.state = nn.Linear(edim, edim)(self.m.mul_h)
    self.m.h = nn.ReLU(true)(self.m.state)
    local hr = nn.JoinTable(2)({self.m.h, self.m.g_embed})
    self.m.hr = nn.Linear(wdim + edim, edim)(hr)
    self.m.s = nn.ReLU(true)(self.m.hr)

    self.m.fc = nn.Linear(edim, args.n_actions)(self.m.s)
    self.m.a = nn.LogSoftMax()(self.m.fc)
    
    self.n_actions = {args.n_actions}
    local input = {self.m.x, self.m.g}
    local output = {self.m.a}
    self.m.b = nn.Linear(edim, 1)(self.m.s)
    table.insert(output, self.m.b)

    return nn.gModule(input, output)
end

function PretrainB:build_init_states(args)
 	return {}
end

function PretrainB:reset_init_state(batch_size)
end

function PretrainB:fill_internal_actions()
end

function PretrainB:add_entropy_grad(rho)
    self.entropy = self.entropy or {}
    for i=1,#self.n_actions do
        self.entropy[i] = self.entropy[i] or nn.Entropy()
    end
    local g = self.entropy[1]:backward(self.m.a.data.module.output, 1)
    self.grad_output[1]:add(g:mul(-rho))
    if self.entropy_list then 
        for i = 1,#self.entropy_list do
            self.entropy_list[i].data.module.w = -rho
        end
    end
end

local PretrainC = torch.class("PretrainC", "Manager")
function PretrainC:build_model(args)
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    
    local argdim = 10
    local edim = args.edim
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)
    self.m.g_embed = nn.View(-1):setNumInputDims(2)(self.m.g_val)
    self.m.g_fc = nn.Linear(wdim, edim)(self.m.g_embed)
    self.m.h = nn.ReLU(true)(self.m.g_fc)

    local input = {self.m.x, self.m.g}
    local output = {}
    self.n_actions = {}
    self.m.arg_lin = nn.Linear(edim, argdim)(self.m.h)
    self.m.arg_embed = nn.ReLU(true)(self.m.arg_lin)

    self.m.arg_h22 = nn.Linear(argdim, args.ch * args.buf * args.channels)(self.m.arg_embed)
    self.m.arg_h2d = nn.View(args.ch, 
            args.buf * args.channels):setNumInputDims(1)(self.m.arg_h22)
    local conv1_2d = nn.View(args.buf * args.channels, -1):setNumInputDims(3)(self.m.x)
    local MMconv = nn.MM(false, false)
    self.m.x_masked = nn.View(args.ch, 10, 10):setNumInputDims(2)(
        MMconv({self.m.arg_h2d, conv1_2d}))

    -- convolution
    self.m.conv1 = nn.SpatialConvolution(args.ch, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x_masked)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)

    local prev_input = self.m.conv1_nl
    for i=1,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        local nInput = args.n_units[i]
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    self.m.fc1 = nn.Linear(5 * 5 * args.n_units[#args.n_units], args.edim)(self.m.conv)

    self.m.arg_fc = nn.Linear(argdim, args.edim)(self.m.arg_embed)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.fc1})
    self.m.joint_h = nn.Linear(args.edim, args.edim)(self.m.mul_h)
    self.m.joint_h_nl = nn.ReLU(true)(self.m.joint_h)
    -- self.m.joint_h_nl = nn.ReLU(true)(self.m.fc1)

    self.m.fc2 = nn.Linear(args.edim, args.n_actions)(self.m.joint_h_nl)
    self.m.a = nn.LogSoftMax()(self.m.fc2)
    self.m.b = nn.Linear(args.edim, 1)(self.m.joint_h_nl)

    local output = {self.m.a}
    self.n_actions = {args.n_actions}

    if args.term_act then
        self.m.term_fc = nn.Linear(args.edim, 2)(self.m.joint_h_nl)
        self.m.term_softmax = nn.LogSoftMax()(self.m.term_fc)
        table.insert(output, self.m.term_softmax)
    end
	
    table.insert(output, self.m.b)
    return nn.gModule(input, output)
end

function PretrainC:build_init_states(args)
 	return {}
end

function PretrainC:reset_init_state(batch_size)
end

function PretrainC:fill_internal_actions()
end

function PretrainC:add_entropy_grad(rho)
    self.entropy = self.entropy or {}
    for i=1,#self.n_actions do
        self.entropy[i] = self.entropy[i] or nn.Entropy()
    end
    local g = self.entropy[1]:backward(self.m.a.data.module.output, 1)
    self.grad_output[1]:add(g:mul(-rho))
    if self.entropy_list then 
        for i = 1,#self.entropy_list do
            self.entropy_list[i].data.module.w = -rho
        end
    end
end

function PretrainC:get_args()
    if self.analogy_list and #self.analogy_list > 0 then
        return self.analogy_list
    end
    self.analogy_list = {}
    local analogy1 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h22 = nn.Linear(self.args.edim, 
                self.args.ch * self.args.buf * self.args.channels):share(
            self.m.arg_h22.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_h22)
        table.insert(analogy1, module)
    end
    table.insert(self.analogy_list, analogy1)

    local analogy2 = {}
    for i=1, 4 do
        local module = nn.Sequential():add(nn.SplitTable(1, 1))
        local arg_h1 = nn.LookupTable(self.args.num_args[1], self.args.edim):share(
            self.m.arg_h1.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_h2 = nn.LookupTable(self.args.num_args[2], self.args.edim):share(
            self.m.arg_h2.data.module, 'weight','bias','gradWeight','gradBias')
        local arg_fc = nn.Linear(self.args.edim, self.args.edim):share(
            self.m.arg_fc.data.module, 'weight','bias','gradWeight','gradBias')
        module:add(nn.ParallelTable():add(arg_h1):add(arg_h2))
        module:add(nn.CMulTable())
        module:add(nn.ReLU(true))
        module:add(arg_fc)
        table.insert(analogy2, module)
    end
    table.insert(self.analogy_list, analogy2)
    return self.analogy_list
end

local FlatPretrain = torch.class("FlatPretrain", "Manager")
function FlatPretrain:build_model(args)
    args.buf = 1
    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.c0 = nn.Identity()()
    self.m.h0 = nn.Identity()()

    local prev_c = self.m.c0
    local prev_h = self.m.h0
    local edim = args.edim
    local ldim = args.ldim

    local nel = 5 * 5 * args.n_units[#args.n_units]
    local wdim = args.wdim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)
    self.m.g_embed = nn.View(-1):setNumInputDims(2)(self.m.g_val)

    -- Compute CNN embeddings
    self.m.mask_r = nn.Linear(args.wdim, args.edim)(self.m.g_embed)
    self.m.mask_r_nl = nn.ReLU(true)(self.m.mask_r)
    self.m.mask_r2 = nn.Linear(args.edim, args.ch * args.n_units[2])(self.m.mask_r_nl)
    self.m.arg_fc = nn.Linear(args.edim, args.edim)(self.m.mask_r_nl)
    self.m.arg_h2d = nn.View(args.ch, args.n_units[2]):setNumInputDims(1)(self.m.mask_r2)
    
    self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    
    self.m.conv2 = nn.SpatialConvolution(args.n_units[1], args.n_units[2],
                        args.filter_size[2], args.filter_size[2],
                        args.filter_stride[2], args.filter_stride[2],
                        args.pad[2], args.pad[2])(self.m.conv1_nl)
    self.m.conv2_nl = nn.ReLU(true)(self.m.conv2)

    local fsize = 15
    local conv_2d = nn.View(-1, args.n_units[2], fsize * fsize)(self.m.conv2_nl)
    local MMconv = nn.MM(false, false)
    self.m.conv_masked = nn.View(-1, args.ch, fsize, fsize)(
        MMconv({self.m.arg_h2d, conv_2d}))

    local nInput = args.ch
    local prev_input = self.m.conv_masked
    for i=2,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        nInput = args.n_units[i+1]
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    local nel = args.feat_size * args.feat_size * args.n_units[#args.n_units]

    -- gated lstm
    local xh = nn.JoinTable(2)({self.m.conv, prev_h})
    self.m.lstm_gate = nn.Linear(nel + ldim, edim)(xh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.lstm_gate})
    self.m.state = nn.Linear(edim, 4*ldim)(self.m.mul_h)
    self.m.lstm = nn.ReLU(true)(self.m.state)

    local reshaped = nn.View(4, ldim):setNumInputDims(1)(self.m.lstm)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    local in_gate = nn.Sigmoid(true)(n1)
    local forget_gate = nn.Sigmoid(true)(n2)
    local out_gate = nn.Sigmoid(true)(n3)
    local in_transform = nn.Tanh()(n4)
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    self.m.fc = nn.Linear(ldim, args.n_actions)(next_h)
    self.m.a = nn.LogSoftMax()(self.m.fc)
    self.m.b = nn.Linear(ldim, 1)(next_h)

    self.n_actions = {args.n_actions}
    local input = {self.m.x, self.m.g, self.m.c0, self.m.h0}
    local output = {self.m.a, self.m.b, next_c, next_h}
    return nn.gModule(input, output)
end

function FlatPretrain:build_init_states(args)
 	local states = {}
    table.insert(states, torch.Tensor(1, args.ldim))      -- c
    table.insert(states, torch.Tensor(1, args.ldim))      -- h
    return states
end

function FlatPretrain:fill_internal_actions()
end

function FlatPretrain:reset_init_state(batch_size)
    for j=1,#self.init_states do
        local size = self.init_states[j]:size()
        size[1] = batch_size
        self.init_states[j]:resize(size)
        self.init_states[j]:fill(0)
    end
end

function FlatPretrain:add_entropy_grad(rho)
    self.entropy = self.entropy or {}
    for i=1,#self.n_actions do
        self.entropy[i] = self.entropy[i] or nn.Entropy()
    end
    local g = self.entropy[1]:backward(self.m.a.data.module.output, 1)
    self.grad_output[1]:add(g:mul(-rho))
    if self.entropy_list then 
        for i = 1,#self.entropy_list do
            self.entropy_list[i].data.module.w = -rho
        end
    end
end

function FlatPretrain:forward(x, g, arg, term)
    self.input = {x:narrow(2, 1, self.args.channels), g}
    for i=1,#self.init_states do
        table.insert(self.input, self.init_states[i])
    end
    if self.args.pass_act then
        for i=1,#arg do
            if g_opts.gpu > 0 then
                table.insert(self.input, arg[i]:cuda())
            else
                table.insert(self.input, arg[i])
            end
        end
    end
    if self.args.pass_term then
        if g_opts.gpu > 0 then
            table.insert(self.input, term:cuda())
        else
            table.insert(self.input, term)
        end
    end 
    return self:parse_forward(self.net:forward(self.input))
end

local FlatLSTM = torch.class("FlatLSTM", "Manager")
function FlatLSTM:build_init_states(args)
 	local states = {}
    table.insert(states, torch.Tensor(1, args.max_task))  -- p
    table.insert(states, torch.Tensor(1, args.wdim))      -- r
    table.insert(states, torch.Tensor(1, args.ldim))      -- c
    table.insert(states, torch.Tensor(1, args.ldim))      -- h
    return states
end

function FlatLSTM:reset_init_state(batch_size)
    for j=1,#self.init_states do
        local size = self.init_states[j]:size()
        size[1] = batch_size
        self.init_states[j]:resize(size)
        self.init_states[j]:fill(0)
    end
    -- Set initial memory attention to 1, 0, 0, ...
    self.init_states[1]:narrow(2, 1, 1):fill(1)
end

function FlatLSTM:retrieve_goal(args, h, p, hdim)
    -- shift vector
    self.m.g_hs = nn.Linear(hdim, args.edim)(h)
    local g_hs_nl = nn.ReLU(true)(self.m.g_hs)
    self.m.g_s = nn.Linear(args.edim, 2 * args.shift + 1)(g_hs_nl)
    if args.soft_mem then 
        self.m.g_prob_s = nn.SoftMax()(self.m.g_s)
    else
        self.m.g_shift = nn.LogSoftMax()(self.m.g_s)
        self.m.shift_act = nn.Multinomial()(self.m.g_shift)
        self.m.g_prob_s = nn.OneHot(2 * args.shift + 1)(self.m.shift_act)
    end
    local w_tilde = nn.CircularConvolution(){p, self.m.g_prob_s}
    --local w_pow = nn.PowTable(){w_tilde, gamma}
    self.m.g_prob = nn.Normalize(1)(w_tilde)
    local prob3d = nn.View(1, -1):setNumInputDims(1)(self.m.g_prob)
    local MMbout = nn.MM(false, false)
    local out3d = MMbout({prob3d, self.m.g_val})
    local out2d = nn.View(-1):setNumInputDims(1)(out3d)
    return out2d, self.m.g_prob
end

function FlatLSTM:fill_internal_actions()
    if not self.args.soft_mem then
        table.insert(self.actions, self.m.shift_act.data.module.output:long())
    end
end

function FlatLSTM:build_model(args)
    args.pass_act = false
    args.pass_term = false

    self.m.x = nn.Identity()()
    self.m.g = nn.Identity()()
    self.m.p0 = nn.Identity()()
    self.m.r0 = nn.Identity()()
    self.m.c0 = nn.Identity()()
    self.m.h0 = nn.Identity()()
    
    local prev_r = self.m.r0
    local prev_p = self.m.p0
    local prev_c = self.m.c0
    local prev_h = self.m.h0

    local edim = args.edim
    local wdim = args.wdim
    local ldim = args.ldim

    -- Construct a goal memory embedding
    self:build_goal_memory(args, self.m.g)
    
    self.m.mask_r = nn.Linear(args.wdim, args.edim)(prev_r)
    self.m.mask_r_nl = nn.ReLU(true)(self.m.mask_r)
    self.m.mask_r2 = nn.Linear(args.edim, args.ch * args.n_units[2])(self.m.mask_r_nl)
    self.m.arg_fc = nn.Linear(args.edim, args.edim)(self.m.mask_r_nl)
    self.m.arg_h2d = nn.View(args.ch, args.n_units[2]):setNumInputDims(1)(self.m.mask_r2)
    
    self.m.conv1 = nn.SpatialConvolution(args.channels, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],
                        args.pad[1], args.pad[1])(self.m.x)
    self.m.conv1_nl = nn.ReLU(true)(self.m.conv1)
    
    self.m.conv2 = nn.SpatialConvolution(args.n_units[1], args.n_units[2],
                        args.filter_size[2], args.filter_size[2],
                        args.filter_stride[2], args.filter_stride[2],
                        args.pad[2], args.pad[2])(self.m.conv1_nl)
    self.m.conv2_nl = nn.ReLU(true)(self.m.conv2)

    local fsize = 15
    local conv_2d = nn.View(-1, args.n_units[2], fsize * fsize)(self.m.conv2_nl)
    local MMconv = nn.MM(false, false)
    self.m.conv_masked = nn.View(-1, args.ch, fsize, fsize)(
        MMconv({self.m.arg_h2d, conv_2d}))

    local nInput = args.ch
    local prev_input = self.m.conv_masked
    for i=2,(#args.n_units-1) do
        if args.pool[i] and args.pool[i] > 1 then
            self.m["pool" .. i] = nn.SpatialMaxPooling(args.pool[i], args.pool[i], 
                    args.pool[i],args.pool[i])(self.m["conv" .. i .."_nl"])
            prev_input = self.m["pool" .. i]
        end
        self.m["conv" .. i+1] = nn.SpatialConvolution(nInput, args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1],
                            args.pad[i+1], args.pad[i+1])(prev_input)
        self.m["conv" .. i+1 .. "_nl"] = nn.ReLU(true)(self.m["conv" .. i+1])
        nInput = args.n_units[i+1]
        prev_input = self.m["conv" .. i+1 .. "_nl"]
    end

    self.m.conv = nn.View(-1):setNumInputDims(3)(self.m["conv" .. #args.n_units .. "_nl"])
    local nel = args.feat_size * args.feat_size * args.n_units[#args.n_units]

    -- LSTM
    local xh = nn.JoinTable(2)({self.m.conv, prev_h})
    self.m.lstm_gate = nn.Linear(nel + ldim, edim)(xh)
    self.m.mul_h = nn.CMulTable()({self.m.arg_fc, self.m.lstm_gate})
    self.m.state = nn.Linear(edim, 4*ldim)(self.m.mul_h)
    self.m.lstm = nn.ReLU(true)(self.m.state)

    local reshaped = nn.View(4, ldim):setNumInputDims(1)(self.m.lstm)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    local in_gate = nn.Sigmoid(true)(n1)
    local forget_gate = nn.Sigmoid(true)(n2)
    local out_gate = nn.Sigmoid(true)(n3)
    local in_transform = nn.Tanh()(n4)
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    local r, p = self:retrieve_goal(args, next_h, prev_p, ldim)
    local hr = nn.JoinTable(2)({next_h, r})
    self.m.hr = nn.Linear(wdim + ldim, edim)(hr)
    self.m.s = nn.ReLU(true)(self.m.hr)

    self.n_actions = {args.n_actions}
    self.m.fc_ = nn.Linear(edim, args.n_actions)(self.m.s)
    self.m.a = nn.LogSoftMax()(self.m.fc_)
    self.m.b_ = nn.Linear(edim, 1)(self.m.s)
    
    local input = {self.m.x, self.m.g, self.m.p0, self.m.r0, self.m.c0, self.m.h0}
    local output = {self.m.a}

    if not args.soft_mem then
        table.insert(self.n_actions, 2*args.shift+1)
        table.insert(output, self.m.g_shift)
    end

    table.insert(output, self.m.b_)
    table.insert(output, p)
    table.insert(output, r)
    table.insert(output, next_c)
    table.insert(output, next_h)
    
    return nn.gModule(input, output)
end

function FlatLSTM:add_entropy_grad(rho)
    self.entropy = self.entropy or {}
    for i=1,#self.n_actions do
        self.entropy[i] = self.entropy[i] or nn.Entropy()
    end
    local g = self.entropy[1]:backward(self.m.a.data.module.output, 1)
    self.grad_output[1]:add(g:mul(-rho))
    if self.entropy_list then 
        for i = 1,#self.entropy_list do
            self.entropy_list[i].data.module.w = -rho
        end
    end
end
