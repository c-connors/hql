-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

function teacher_idx(args)
    local multiplier = 1
    local idx = 1 
    for k=#g_opts.num_args, 1, -1 do
        idx = idx + (args[k]-1)*multiplier
        multiplier = multiplier * g_opts.num_args[k]
    end
    return idx
end

function sample_args(batch_size)
    assert(#g_opts.num_args == 2)
    assert(g_opts.max_args == 2)
    local args1 = torch.zeros(batch_size, 2)
    local args2 = torch.zeros(batch_size, 2)
    local args3 = torch.zeros(batch_size, 2)
    local args4 = torch.zeros(batch_size, 2)
    
    for i = 1,batch_size do
        local task = torch.randperm(g_opts.num_args[1])
        local obj = torch.randperm(g_opts.num_args[2])
        args1[i][1] = task[1]
        args1[i][2] = obj[1]
        args2[i][1] = task[1]
        args2[i][2] = obj[2]
        args3[i][1] = task[2]
        args3[i][2] = obj[1]
        args4[i][1] = task[2]
        args4[i][2] = obj[2]
    end
    return {args1, args2, args3, args4}
end

function apply_analogy()
    local alpha = g_opts.regularizer * 5
    local beta = g_opts.regularizer
    local margin = 3

    local batch_size = g_opts.analogy_bs 
    if batch_size == 0 then
        batch_size = g_opts.batch_size * 10
    end

    local arg_x = sample_args(batch_size)
    local dis_y = torch.ones(arg_x[1]:size(1)):mul(-1)
    local sim_y = torch.ones(arg_x[1]:size(1))
    local grad_w = torch.Tensor({1})
    local analogy_list = g_model:get_args()

    for i=1,#analogy_list do
        arg_h = analogy_list[i]
        local h11 = arg_h[1]:forward(arg_x[1])
        local h12 = arg_h[2]:forward(arg_x[2])
        local h21 = arg_h[3]:forward(arg_x[3])
        local h22 = arg_h[4]:forward(arg_x[4])
        local zeros = torch.zeros(h11:size())
        
        if not g_sim_loss then
            local x11 = nn.Identity()()
            local x12 = nn.Identity()()
            local x21 = nn.Identity()()
            local x22 = nn.Identity()()
            local y = nn.Identity()()
            local delta = nn.CSubTable()({
                nn.CSubTable()({x11, x12}),
                nn.CSubTable()({x21, x22})})
            local dist = nn.Sqrt()(nn.Sum(1, 1, false)(nn.Power(2)(delta)))
            local loss = nn.DrLimCriterion()({dist, y})
            g_sim_loss = nn.gModule({x11, x12, x21, x22, y}, {loss})
        end

        if not g_dis_loss then
            local x1 = nn.Identity()()
            local x2 = nn.Identity()()
            local y = nn.Identity()()
            local delta = nn.CSubTable()({x1, x2})
            local dist = nn.Sqrt()(nn.Sum(1, 1, false)(nn.Power(2)(delta)))
            local loss = nn.DrLimCriterion(margin)({dist, y})
            g_dis_loss = nn.gModule({x1, x2, y}, {loss})
        end

        if not g_cos_loss then
            g_cos_loss = nn.CosineEmbeddingCriterion(0)
            g_cos_loss2 = nn.CosineEmbeddingCriterion(0.5)
        end

        if not g_mse_loss then
            g_mse_loss = nn.MSECriterion()
        end

        local loss = g_sim_loss:forward({h11, h12, h21, h22, sim_y})
        local dh = g_sim_loss:backward({h11, h12, h21, h22, sim_y}, grad_w)
        for j = 1, 4 do
            arg_h[i]:backward(arg_x[i], dh[i]:mul(alpha))
        end
        
        local d1, d2
        loss = g_dis_loss:forward({h11, h12, dis_y})
        d1, d2 = unpack(g_dis_loss:backward({h11, h12, dis_y}, grad_w))
        arg_h[1]:backward(arg_x[1], d1:mul(beta))
        arg_h[2]:backward(arg_x[2], d2:mul(beta))

        loss = g_dis_loss:forward({h12, h22, dis_y})
        d1, d2 = unpack(g_dis_loss:backward({h12, h22, dis_y}, grad_w))
        arg_h[2]:backward(arg_x[2], d1:mul(beta))
        arg_h[4]:backward(arg_x[4], d2:mul(beta))
        
        loss = g_dis_loss:forward({h21, h22, dis_y})
        d1, d2 = unpack(g_dis_loss:backward({h21, h22, dis_y}, grad_w))
        arg_h[3]:backward(arg_x[3], d1:mul(beta))
        arg_h[4]:backward(arg_x[4], d2:mul(beta))

        loss = g_dis_loss:forward({h11, h21, dis_y})
        d1, d2 = unpack(g_dis_loss:backward({h11, h21, dis_y}, grad_w))
        arg_h[1]:backward(arg_x[1], d1:mul(beta))
        arg_h[3]:backward(arg_x[3], d2:mul(beta))

        loss = g_dis_loss:forward({h11, h22, dis_y})
        d1, d2 = unpack(g_dis_loss:backward({h11, h22, dis_y}, grad_w))
        arg_h[1]:backward(arg_x[1], d1:mul(beta))
        arg_h[4]:backward(arg_x[4], d2:mul(beta))

        loss = g_dis_loss:forward({h12, h21, dis_y})
        d1, d2 = unpack(g_dis_loss:backward({h12, h21, dis_y}, grad_w))
        arg_h[2]:backward(arg_x[2], d1:mul(beta))
        arg_h[3]:backward(arg_x[3], d2:mul(beta))

        if g_opts.cos > 0 then
            local d1, d2
            loss = g_cos_loss2:forward({h11, h12}, dis_y)
            d1, d2 = unpack(g_cos_loss2:backward({h11, h12}, dis_y))
            arg_h[1]:backward(arg_x[1], d1:mul(g_opts.cos))
            arg_h[2]:backward(arg_x[2], d2:mul(g_opts.cos))

            loss = g_cos_loss2:forward({h12, h22}, dis_y)
            d1, d2 = unpack(g_cos_loss2:backward({h12, h22}, dis_y))
            arg_h[2]:backward(arg_x[2], d1:mul(g_opts.cos))
            arg_h[4]:backward(arg_x[4], d2:mul(g_opts.cos))
            
            loss = g_cos_loss2:forward({h21, h22}, dis_y)
            d1, d2 = unpack(g_cos_loss2:backward({h21, h22}, dis_y))
            arg_h[3]:backward(arg_x[3], d1:mul(g_opts.cos))
            arg_h[4]:backward(arg_x[4], d2:mul(g_opts.cos))

            loss = g_cos_loss2:forward({h11, h21}, dis_y)
            d1, d2 = unpack(g_cos_loss2:backward({h11, h21}, dis_y))
            arg_h[1]:backward(arg_x[1], d1:mul(g_opts.cos))
            arg_h[3]:backward(arg_x[3], d2:mul(g_opts.cos))

            loss = g_cos_loss:forward({h11, h22}, dis_y)
            d1, d2 = unpack(g_cos_loss:backward({h11, h22}, dis_y))
            arg_h[1]:backward(arg_x[1], d1:mul(g_opts.cos))
            arg_h[4]:backward(arg_x[4], d2:mul(g_opts.cos))

            loss = g_cos_loss:forward({h12, h21}, dis_y)
            d1, d2 = unpack(g_cos_loss:backward({h12, h21}, dis_y))
            arg_h[2]:backward(arg_x[2], d1:mul(g_opts.cos))
            arg_h[3]:backward(arg_x[3], d2:mul(g_opts.cos))
        end

        if g_opts.l2 > 0 then
            arg_h[1]:backward(arg_x[1], g_mse_loss:backward(h11, zeros):mul(g_opts.l2))
            arg_h[2]:backward(arg_x[2], g_mse_loss:backward(h12, zeros):mul(g_opts.l2))
            arg_h[3]:backward(arg_x[3], g_mse_loss:backward(h21, zeros):mul(g_opts.l2))
            arg_h[4]:backward(arg_x[4], g_mse_loss:backward(h22, zeros):mul(g_opts.l2))
        end
    end
end

function sample_args_v2(batch_size)
    if g_opts.max_args == 2 then
        assert(#g_opts.num_args == 2)
        local args1 = torch.zeros(batch_size, 2)
        local args2 = torch.zeros(batch_size, 2)
        local args3 = torch.zeros(batch_size, 2)
        local args4 = torch.zeros(batch_size, 2)
        local sim = torch.zeros(batch_size)
        local weight = torch.ones(batch_size)
        
        for i = 1,batch_size do
            local idx = torch.random(#g_opts.analogy)
            local task1 = g_opts.analogy[idx][1] 
            local obj1 = g_opts.analogy[idx][2]
            local task2 = g_opts.analogy[idx][3] 
            local obj2 = g_opts.analogy[idx][4]

            args1[i][1] = task1
            args1[i][2] = obj1
            args2[i][1] = task1
            args2[i][2] = obj2
            args3[i][1] = task2
            args3[i][2] = obj1
            args4[i][1] = task2
            args4[i][2] = obj2

            sim[i] = g_opts.analogy[idx][5]
            if g_opts.analogy[idx][6] then
                weight[i] = g_opts.analogy[idx][6]
            end
        end
        return {args1, args2, args3, args4, sim, weight}
    elseif g_opts.max_args == 3 then
        assert(#g_opts.num_args == 3)
        local args1 = torch.zeros(batch_size, 3)
        local args2 = torch.zeros(batch_size, 3)
        local args3 = torch.zeros(batch_size, 3)
        local args4 = torch.zeros(batch_size, 3)
        local sim = torch.zeros(batch_size)
        local weight = torch.ones(batch_size)
        
        for i = 1,batch_size do
            local idx = torch.random(#g_opts.analogy)

            if #g_opts.analogy[idx] == 8 then
                local task1 = g_opts.analogy[idx][1] 
                local obj1 = g_opts.analogy[idx][2]
                local count1 = g_opts.analogy[idx][3]
                local task2 = g_opts.analogy[idx][4] 
                local obj2 = g_opts.analogy[idx][5]
                local count2 = g_opts.analogy[idx][6]

                args1[i][1] = task1
                args1[i][2] = obj1
                args1[i][3] = count1
                
                args2[i][1] = task1
                args2[i][2] = obj1
                args2[i][3] = count2
                
                args3[i][1] = task2
                args3[i][2] = obj2
                args3[i][3] = count1
                
                args4[i][1] = task2
                args4[i][2] = obj2
                args4[i][3] = count2

                sim[i] = g_opts.analogy[idx][7]
                if g_opts.analogy[idx][8] then
                    weight[i] = g_opts.analogy[idx][8]
                end
            elseif #g_opts.analogy[idx] == 9 then
                local task1 = g_opts.analogy[idx][1] 
                local obj1 = g_opts.analogy[idx][2]
                local count1 = g_opts.analogy[idx][3]
                local task2 = g_opts.analogy[idx][4] 
                local obj2 = g_opts.analogy[idx][5]
                local count2 = g_opts.analogy[idx][6]
                local diff = g_opts.analogy[idx][7]

                args1[i][1] = task1
                args1[i][2] = obj1
                args1[i][3] = count1
                
                args2[i][1] = task1
                args2[i][2] = obj1
                args2[i][3] = count1 + diff
                
                args3[i][1] = task2
                args3[i][2] = obj2
                args3[i][3] = count2
                
                args4[i][1] = task2
                args4[i][2] = obj2
                args4[i][3] = count2 + diff

                sim[i] = g_opts.analogy[idx][8]
                if g_opts.analogy[idx][9] then
                    weight[i] = g_opts.analogy[idx][9]
                end
            elseif #g_opts.analogy[idx] == 10 then
                local task1 = g_opts.analogy[idx][1] 
                local obj1 = g_opts.analogy[idx][2]
                local count1 = g_opts.analogy[idx][3]
                local task2 = g_opts.analogy[idx][4] 
                local obj2 = g_opts.analogy[idx][5]
                local count2 = g_opts.analogy[idx][6]
                local diff = g_opts.analogy[idx][7]
                local diff2 = g_opts.analogy[idx][8]

                args1[i][1] = task1
                args1[i][2] = obj1
                args1[i][3] = count1
                
                args2[i][1] = task1
                args2[i][2] = obj1
                args2[i][3] = count1 + diff
                
                args3[i][1] = task2
                args3[i][2] = obj2
                args3[i][3] = count2
                
                args4[i][1] = task2
                args4[i][2] = obj2
                args4[i][3] = count2 + diff2

                sim[i] = g_opts.analogy[idx][9]
                
                assert(diff ~= diff2)
                assert(sim[i] == -1)
                if g_opts.analogy[idx][10] then
                    weight[i] = g_opts.analogy[idx][10]
                end
            else
                error("invalid analogy")
            end
        end
        return {args1, args2, args3, args4, sim, weight}
    else
        error("num of args should be 2 or 3!")
    end
end

function apply_analogy_v2()
    local alpha = g_opts.regularizer * 5
    local beta = g_opts.regularizer
    local margin = 3
    local batch_size = g_opts.analogy_bs 
    if batch_size == 0 then
        batch_size = g_opts.batch_size * 10
    end

    local arg_x = sample_args_v2(batch_size)
    local dis_y = torch.ones(arg_x[1]:size(1)):mul(-1)
    local sim_y = arg_x[5] -- torch.ones(arg_x[1]:size(1))
    local weight = arg_x[6] -- torch.ones(arg_x[1]:size(1))
    local grad_w = torch.Tensor({1})
    local analogy_list = g_model:get_args()

    for i=1,#analogy_list do
        arg_h = analogy_list[i]
        local h11 = arg_h[1]:forward(arg_x[1])
        local h12 = arg_h[2]:forward(arg_x[2])
        local h21 = arg_h[3]:forward(arg_x[3])
        local h22 = arg_h[4]:forward(arg_x[4])
        local zeros = torch.zeros(h11:size())
        
        if not g_sim_loss then
            local x11 = nn.Identity()()
            local x12 = nn.Identity()()
            local x21 = nn.Identity()()
            local x22 = nn.Identity()()
            local y = nn.Identity()()
            local delta = nn.CSubTable()({
                nn.CSubTable()({x11, x12}),
                nn.CSubTable()({x21, x22})})
            local dist = nn.Sqrt()(nn.Sum(1, 1, false)(nn.Power(2)(delta)))
            local loss = nn.DrLimCriterion(margin)({dist, y})
            g_sim_loss = nn.gModule({x11, x12, x21, x22, y}, {loss})
        end

        if not g_dis_loss then
            local x1 = nn.Identity()()
            local x2 = nn.Identity()()
            local y = nn.Identity()()
            local delta = nn.CSubTable()({x1, x2})
            local dist = nn.Sqrt()(nn.Sum(1, 1, false)(nn.Power(2)(delta)))
            local loss = nn.DrLimCriterion(margin)({dist, y})
            g_dis_loss = nn.gModule({x1, x2, y}, {loss})
        end

        if not g_mse_loss then
            g_mse_loss = nn.MSECriterion()
        end

        local loss = g_sim_loss:forward({h11, h12, h21, h22, sim_y})
        local dh = g_sim_loss:backward({h11, h12, h21, h22, sim_y}, grad_w)
        for i = 1, 4 do
            for l=1,dh[i]:size(1) do
                dh[i][l]:mul(alpha * weight[l])
            end
            arg_h[i]:backward(arg_x[i], dh[i])
        end
        
        local d1, d2
        loss = g_dis_loss:forward({h11, h12, dis_y})
        d1, d2 = unpack(g_dis_loss:backward({h11, h12, dis_y}, grad_w))
        arg_h[1]:backward(arg_x[1], d1:mul(beta))
        arg_h[2]:backward(arg_x[2], d2:mul(beta))

        loss = g_dis_loss:forward({h12, h22, dis_y})
        d1, d2 = unpack(g_dis_loss:backward({h12, h22, dis_y}, grad_w))
        arg_h[2]:backward(arg_x[2], d1:mul(beta))
        arg_h[4]:backward(arg_x[4], d2:mul(beta))
        
        loss = g_dis_loss:forward({h21, h22, dis_y})
        d1, d2 = unpack(g_dis_loss:backward({h21, h22, dis_y}, grad_w))
        arg_h[3]:backward(arg_x[3], d1:mul(beta))
        arg_h[4]:backward(arg_x[4], d2:mul(beta))

        loss = g_dis_loss:forward({h11, h21, dis_y})
        d1, d2 = unpack(g_dis_loss:backward({h11, h21, dis_y}, grad_w))
        arg_h[1]:backward(arg_x[1], d1:mul(beta))
        arg_h[3]:backward(arg_x[3], d2:mul(beta))

        loss = g_dis_loss:forward({h11, h22, dis_y})
        d1, d2 = unpack(g_dis_loss:backward({h11, h22, dis_y}, grad_w))
        arg_h[1]:backward(arg_x[1], d1:mul(beta))
        arg_h[4]:backward(arg_x[4], d2:mul(beta))

        loss = g_dis_loss:forward({h12, h21, dis_y})
        d1, d2 = unpack(g_dis_loss:backward({h12, h21, dis_y}, grad_w))
        arg_h[2]:backward(arg_x[2], d1:mul(beta))
        arg_h[3]:backward(arg_x[3], d2:mul(beta))
    end
end

function train_batch(rho, xi)
    -- start a new episode
    g_paramdx:zero()
    local stat = {}
    for iter=1,g_opts.iter_size do
        collectgarbage()
        collectgarbage()
        local batch = batch_init(g_opts.batch_size)
        local reward = {}
        local obs = {}
        local input = {}
        local args = {}
        local action = {}
        local term = {}
        local sub_term = {}
        local term_prob = {}
        local act_prob = {}
        local active = {}
        local baseline = {}
        local states = {}
        local prev_action = {}
        local t_act_prob = {}
        local feas = {}
        local win = nil
        
        local success
        local giveup = torch.Tensor(g_opts.batch_size):zero()
        local total_reward = torch.Tensor(g_opts.batch_size):zero() 
        local R = torch.Tensor(g_opts.batch_size):zero()
        local A = torch.Tensor(g_opts.batch_size, 1):zero()
        local A_grad = torch.Tensor(g_opts.batch_size, 1):zero()
        local delta = torch.Tensor(g_opts.batch_size):zero()
        local V_next = torch.Tensor(g_opts.batch_size):zero()

        if g_opts.gpu > 0 then
            R = R:cuda()
            A = A:cuda()
            A_grad = A_grad:cuda()
            delta = delta:cuda()
            V_next = V_next:cuda()
            total_reward = total_reward:cuda()
        end

        if g_opts.verbose > 3 then
            local active_tmp = batch_active(batch)
            local input_tmp = batch_input(batch, active_tmp, t)
            for i=1,g_opts.max_task do
                local txt = goal_text(input_tmp[2][1]:narrow(1, 
                        g_opts.max_word * (i-1) + 1, g_opts.max_word))
                if txt ~= "" then
                    print(txt)
                end
            end
            if g_opts.multitask then
                print(input_tmp[3])
            end
        end

        -- play the games
        g_model_t[1]:reset_init_state(g_opts.batch_size)
        if g_teacher and g_teacher[1].recurrent then
            for k,v in pairs(g_teacher) do
                v:reset_init_state(g_opts.batch_size)
            end
        end
        for t = 1, g_opts.max_step do
            --print("forward", i, t)
            if g_opts.display then 
                win = image.display({image=batch[1].map:to_image(), win=win, saturate=false})
                os.execute('sleep 0.2')
            end
            active[t] = batch_active(batch)
            term[t] = torch.zeros(g_opts.batch_size)
            if t > 1 then
                for i=1, g_opts.batch_size do
                    if active[t][i] == 0 and active[t-1][i] == 1 then
                        term[t][i] = 1
                    end
                end
            end
            input[t] = input[t] or {} 
            obs[t], input[t][2], input[t][3] = unpack(batch_input(batch, active[t], t, term[t]))
            if g_opts.buf == 1 then
                input[t][1] = obs[t]
            else
                if not input[t][1] then
                    local size = obs[t]:size()
                    assert(size[2] == g_opts.channels)
                    size[2] = g_opts.buf * g_opts.channels
                    input[t][1] = torch.Tensor(size)
                end
                input[t][1]:zero()
                for k=1,g_opts.buf do
                    local idx = t - k + 1
                    if idx < 1 then
                        break
                    end
                    input[t][1]:narrow(2, (k-1)*g_opts.channels+1, g_opts.channels):copy(obs[idx])
                end
                -- print(input[t][1])
            end
            local x = {input[t][1]}
            if g_opts.multitask then
                table.insert(x, input[t][3])
            else
                table.insert(x, input[t][2])
            end
            if g_model.args.pass_act then
                prev_action[t] = prev_action[t] or {}
                for i=1,#g_model.args.num_args do
                    prev_action[t][i] = torch.zeros(g_opts.batch_size, g_model.args.num_args[i])
                    if t > 1 then
                        prev_action[t][i]:scatter(2, g_model_t[t-1].actions[i]:long(), 1)
                    end
                end
                table.insert(x, prev_action[t])
            end 

            action[t], baseline[t], states, term_prob[t] = g_model_t[t]:eval(unpack(x))
            if g_teacher then
                act_prob[t] = g_model_t[t].m.a.data.module.output
                assert(g_opts.batch_size == 1)
                for k = 1, g_opts.batch_size do
                    if active[t][k] > 0 then
                        local t_idx = teacher_idx(input[t][3][k])
                        local x_t = input[t][1]:narrow(1, k, 1):narrow(2, 1, 
                                g_teacher[t_idx].args.buf * g_opts.channels)
                        assert(g_teacher[t_idx], "teacher " .. t_idx .. " does not exist")
                        local target, t_b, t_s = g_teacher[t_idx]:forward(x_t)
                        target = target[1]
                        t_act_prob[t] = torch.exp(target)
                        
                        if t < g_opts.max_step and g_teacher[1].recurrent then
                            g_teacher[t_idx]:set_init_state(t_s)
                        end
                    end
                end
            end

            if g_opts.feasible then
                feas[t] = g_model_t[t]:feasible():clone():add(-1)
                batch_giveup(batch, feas[t], active[t])
                for k=1,g_opts.batch_size do
                    if giveup[k] == 0 and feas[t][k][1] == 1 then
                        giveup[k] = 1
                    end
                end
            end

            batch_act(batch, action[t]:view(-1), active[t]) 
            batch_update(batch, active[t])
            reward[t] = batch_reward(batch, active[t], t == g_opts.max_step)
            if g_opts.gpu > 0 then
                reward[t] = reward[t]:cuda()
                active[t] = active[t]:cuda()
            end
            -- copy hidden states
            if t < g_opts.max_step and g_model.recurrent then
                assert(#states == #g_model.init_states)
                g_model_t[t+1]:set_init_state(states)
                if g_opts.verbose > 1 and __threadid == 1 then
                    print(t, g_model_t[t].update, states[1]:norm())
                end
            end
            if g_opts.verbose > 3 then
                if g_opts.nworker == 1 or __threadid == 1 then 
                    print(giveup[1], reward[t][1], batch[1].finished, batch[1].finish_by_giveup)
                    print("Step", t, "Action", action[t][1]:squeeze(), 
                        "Reward", reward[t][1], "Term", term[t][1])
                end
            end
            if active[t]:sum() == 0 then 
                break 
            end
        end
        success = batch_success(batch)

        -- increase difficulty if necessary
        if g_opts.curriculum == 1 then
            apply_curriculum(batch, success)
        end

        -- do back-propagation
        local prev_grad_states
        local grad_states = g_model_t[1]:init_grad_states()
        local act_grad = torch.Tensor(g_opts.batch_size, g_opts.n_actions):zero()
        local term_grad = torch.Tensor(g_opts.batch_size, 2):zero()
        for t = g_opts.max_step, 1, -1 do
            --print("backward", t)
            if (active[t] ~= nil and active[t]:sum() > 0) or 
                    (g_opts.term_action and term[t] ~= nil and term[t]:sum() > 0) then
                local V = baseline[t] 
                local bl_grad = nil
                if reward[t] then
                    A:mul(g_opts.gamma * g_opts.lambda)
                    R:mul(g_opts.gamma) -- discount factor
                    R:add(reward[t]) -- cumulative reward
                    total_reward:add(reward[t])
                    V:cmul(active[t])
                    R:cmul(active[t])
                    A:cmul(active[t])
                    total_reward:cmul(active[t])
                    
                    -- Baseline loss
                    if g_model_t[t].update == 1 then
                        stat.bl_cost = (stat.bl_cost or 0) + g_bl_loss:forward(V, R)
                        stat.bl_count = (stat.bl_count or 0) + active[t]:sum()
                        bl_grad = g_bl_loss:backward(V, R)
                        --print(V:mean(), R:mean())
                        if g_opts.clip_bl then
                            bl_grad[bl_grad:ge(1/g_opts.batch_size)] = 1/g_opts.batch_size
                            bl_grad[bl_grad:le(-1/g_opts.batch_size)] = -1/g_opts.batch_size
                        end
                        bl_grad:mul(g_opts.alpha)
                    else
                        bl_grad = torch.zeros(g_opts.batch_size)
                    end
                    
                    -- Generalized advantage
                    V_next:mul(g_opts.gamma)
                    delta:add(V, -1, torch.cmul(reward[t], active[t]):add(V_next))
                    A:add(delta)
                end
                    
                local grad = {}
                if g_teacher then
                    -- Policy Distillation
                    act_grad:zero()
                    local dist_loss = 0
                    local dist_count = active[t]:sum()
                    assert(g_opts.batch_size == 1)
                    for k = 1, g_opts.batch_size do
                        if active[t][k] > 0 then
                            local target_prob = t_act_prob[t]
                            dist_loss = dist_loss + 
                                g_dist_loss:forward(act_prob[t]:narrow(1, k, 1), target_prob)
                            local dist_grad = 
                                g_dist_loss:backward(act_prob[t]:narrow(1, k, 1), target_prob)
                            act_grad[k]:copy(dist_grad)
                        end
                    end
                    stat.dist_cost = (stat.dist_cost or 0) + dist_loss
                    stat.dist_count = (stat.dist_count or 0) + dist_count
                    if g_opts.xi > 0 then
                        table.insert(grad, act_grad:mul(xi))
                        table.insert(grad, A)
                    else
                        table.insert(grad, act_grad)
                        bl_grad:zero()
                    end
                else
                    -- Actor-Critic
                    if g_opts.clip_adv then
                        local A_grad = A:clone()
                        A_grad[A_grad:ge(2)] = 2
                        A_grad[A_grad:le(-2)] = -2
                        table.insert(grad, A_grad)
                    else
                        table.insert(grad, A)
                    end
                end

                if g_opts.term_action then
                    term_grad:zero()
                    local term_loss = 0
                    local term_count = 0
                    for k = 1, g_opts.batch_size do
                        if active[t][k] > 0 or (term[t][k] == 1 and giveup[k] == 0) then
                            local target = torch.Tensor({term[t][k] + 1})
                            term_count = term_count + 1
                            term_loss = term_loss + 
                               g_term_loss:forward(term_prob[t]:narrow(1, k, 1), target)
                            local term_error = 
                                g_term_loss:backward(term_prob[t]:narrow(1, k, 1), target)
                            term_grad[k]:copy(term_error)
                        end
                    end
                    term_grad:mul(g_opts.zeta)
                    prev_grad_states = g_model_t[t]:backward(input[t][1], 
                            input[t][2], grad, bl_grad, grad_states, rho, term_grad)
                    stat.term_cost = (stat.term_cost or 0) + term_loss
                    stat.term_count = (stat.term_count or 0) + term_count
                else
                    prev_grad_states = g_model_t[t]:backward(input[t][1], 
                            input[t][2], grad, bl_grad, grad_states, rho)
                end
                
                -- backpropagation through time 
                if g_model.recurrent then
                    assert(#prev_grad_states == #grad_states,
                        #prev_grad_states .. " " .. #grad_states)
                    grad_states = prev_grad_states
                    if g_opts.verbose > 1 and __threadid == 1 then
                        print(t, g_model_t[t].update, grad_states[1]:norm())
                    end
                end
                V_next:copy(V)
            end
        end

        if g_opts.regularizer and g_opts.regularizer > 0 then
            if g_opts.analogy then
                apply_analogy_v2()
            else
                apply_analogy()
            end
        end
        -- stat by game type
        for i, g in pairs(batch) do
            stat.reward = (stat.reward or 0) + total_reward:mean()
            stat.success = (stat.success or 0) + success[i]
            stat.count = (stat.count or 0) + 1

            local t = torch.type(batch[i])
            stat['reward_' .. t] = (stat['reward_' .. t] or 0) + total_reward:mean()
            stat['success_' .. t] = (stat['success_' .. t] or 0) + success[i]
            stat['count_' .. t] = (stat['count_' .. t] or 0) + 1
        end
    end
    return stat
end

function apply_curriculum(batch,success)
    for i = 1, #batch do
        local gname = batch[i].__typename
        g_factory:collect_result(gname,success[i])
        local count = g_factory:count(gname)
        local total_count = g_factory:total_count(gname)
        local pct = g_factory:success_percent(gname)
        if not g_factory.helpers[gname].frozen then
            if total_count > g_opts.curriculum_total_count * g_opts.iter_size then
                print('freezing ' .. gname)
                g_factory:hardest(gname)
                g_factory:freeze(gname)
            else
                if count > g_opts.curriculum_min_count * g_opts.iter_size then
                    if pct > g_opts.curriculum_pct_high then
                        g_factory:harder(gname)
                        print('making ' .. gname .. ' harder')
                        print(format_helpers())
                    end
                    if pct < g_opts.curriculum_pct_low then
                        g_factory:easier(gname)
                        print('making ' .. gname .. ' easier')
                        print(format_helpers())
                    end
                    g_factory:reset_counters(gname)
                end
            end
        end
    end
end

function train_batch_thread(opts_orig, paramx_orig, rho, xi)
    g_opts = opts_orig
    g_paramx:copy(paramx_orig)
    local stat = train_batch(rho, xi)
    return g_paramdx, stat
end

function train(N)
    g_opts.test = true
    local log = {train = {}, test = {}}
    for i=1,#g_opts.test_game do
        print("Test #" .. i .. ": " .. g_opts.test_game[i])
        log.test[i] = test(g_opts.test_iter, i)
    end
    g_opts.test = false
    table.insert(g_log, log)
    g_save_model()
    for n = 1, N do
        local rho = g_opts.rho
        if rho > 0 and g_opts.rho_epoch > 0 then
            rho = math.max(g_opts.rho*(1 - n/g_opts.rho_epoch), 0)
        end
        local xi = g_opts.xi
        if xi > 0 and g_opts.xi_epoch > 0 then
            xi = math.max(g_opts.xi*(1 - n/g_opts.xi_epoch), 0)
        end

        --print(g_opts)
        local stat = {}
        for k = 1, g_opts.nbatches do
            xlua.progress(k, g_opts.nbatches)
            if g_opts.nworker > 1 then
                g_paramdx:zero()
                for w = 1, g_opts.nworker do
                    g_workers:addjob(w, train_batch_thread,
                        function(paramdx_thread, s)
                            g_paramdx:add(paramdx_thread)
                            for k, v in pairs(s) do
                                stat[k] = (stat[k] or 0) + v
                            end
                        end,
                        g_opts, g_paramx, rho, g_opts.xi
                    )
                end
                g_workers:synchronize()
            else
                local s = train_batch(rho, xi)
                for k, v in pairs(s) do
                    stat[k] = (stat[k] or 0) + v
                end
            end
            g_update_param()
        end
        g_grad_report()
        for k, v in pairs(stat) do
            if string.sub(k, 1, 5) == 'count' then
                local s = string.sub(k, 6)
                stat['reward' .. s] = stat['reward' .. s] / v
                stat['success' .. s] = stat['success' .. s] / v
            end
        end
        if stat.bl_count ~= nil and stat.bl_count > 0 then
            stat.bl_cost = stat.bl_cost / stat.bl_count
        else
            stat.bl_cost = 0
        end
        if stat.dist_count ~= nil and stat.dist_count > 0 then
            stat.dist_cost = stat.dist_cost / stat.dist_count
        end
        if stat.term_count ~= nil and stat.term_count > 0 then
            stat.term_cost = stat.term_cost / stat.term_count
        end
        if stat.arg_count  and stat.arg_count > 0 then
            stat.arg_loss = stat.arg_loss / stat.arg_count
        end
        stat.epoch = #g_log + 1
        --print(string.format("%s rho: %g xi: %g", format_stat(stat), rho, xi))
        print(string.format("%s rho: %g", format_stat(stat), rho))
        g_opts.test = true
        local log = {train = stat, test = {}}
        for i=1,#g_opts.test_game do
            print("Test #" .. i .. ": " .. g_opts.test_game[i])
            log.test[i] = test(g_opts.test_iter, i)
        end
        g_opts.test = false
        table.insert(g_log, log)
        g_save_model()
        g_opts.lr = g_opts.lr * g_opts.lr_mult
        collectgarbage()
    end
end

function g_update_param()
    g_paramdx:div(g_opts.nworker * g_opts.iter_size)
    if g_opts.max_grad_norm > 0 then
        if g_paramdx:norm() > g_opts.max_grad_norm then
            g_paramdx:div(g_paramdx:norm() / g_opts.max_grad_norm)
        end
    end
    local f = function(x) return g_paramx, g_paramdx end
    local config = {
       learningRate = g_opts.lr,
       alpha = g_opts.beta,
       epsilon = 1e-6
    }
    rmsprop(f, g_paramx, config, g_rmsprop_state)
end

function g_grad_report()
    print(get_weight_norms(g_model.net))
    print(get_grad_norms(g_model.net))
    print("Grad Norm: " .. tostring(g_paramdx:norm()))
end
