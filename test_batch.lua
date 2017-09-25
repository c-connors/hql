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

function test_batch()
    local batch = batch_init(g_opts.batch_size)
    local reward, input, action, active, out, states, term, prev_act
    local prev_action = {}
    local prev_action2 = {}
    local prev_active = {}
    local obs = {}
    local giveup = torch.Tensor(g_opts.batch_size):zero()
    local term_acc = torch.ones(g_opts.batch_size)
    local steps = torch.zeros(g_opts.batch_size)
    local R = torch.Tensor(g_opts.batch_size):zero()
    local giveup_prec, giveup_recall
    if g_opts.feasible then
        giveup_prec = torch.ones(g_opts.batch_size)
        giveup_recall = torch.ones(g_opts.batch_size)
    end
    local step = 0
    if g_opts.video and g_opts.video ~= "" or g_opts.display then
        img = batch[1].map:to_image()
        local img_size = img:size()
        img_h = img_size[2]
        img_w = img_size[3]
        canvas = torch.FloatTensor(3, img_h, img_w + 180):fill(1)
    end
    local prev_act, prev_reward, inter_prob, inter_act, term_prob
    local prev_task_idx = 1
    local term_log
    if g_opts.video and g_opts.save_meta then
        term_log = io.open(string.format("%s/term.csv", video_dir), "w")
        memory_log = io.open(string.format("%s/memory.csv", video_dir), "w")
        shift_log = io.open(string.format("%s/shift.csv", video_dir), "w")
        goal_log = io.open(string.format("%s/goal.csv", video_dir), "w")
    end
    if g_opts.save_lstm and g_opts.save_lstm > 0 then
        if lstm_cnt == nil then
            lstm_x = torch.zeros(g_opts.save_lstm, g_model.args.ldim)
            lstm_y = torch.zeros(g_opts.save_lstm):byte()
            lstm_y2 = torch.zeros(g_opts.save_lstm):byte()
            lstm_cnt = 0
        end
    end
    local prev_task_idx = 0
    local time = sys.clock()
    g_model:reset_init_state(g_opts.batch_size)
    for t = 1, g_opts.max_step do 
        active = batch_active(batch)
        input = input or {} 
        obs[t], input[2], input[3] = unpack(batch_input(batch, active, t))
        if g_model.args.buf == nil or g_model.args.buf == 1 then
            input[1] = obs[t]
        else
            if not input[1] then
                local size = obs[t]:size()
                assert(size[2] == g_model.args.channels)
                size[2] = g_model.args.buf * g_model.args.channels
                input[1] = torch.Tensor(size)
            end
            input[1]:zero()
            for k=1,g_model.args.buf do
                local idx = t - k + 1
                if idx < 1 then
                    break
                end
                input[1]:narrow(2, (k-1)*g_model.args.channels+1, 
                    g_model.args.channels):copy(obs[idx])
            end
        end
        
        for i=1, g_opts.batch_size do
            if active[i] == 1 then
                steps[i] = t
            end
        end

        if g_opts.verbose > 2 then
            print(string.format("input fetch: %.3f", sys.clock() - time))
            time = sys.clock()
        end

        if g_model.args.pass_act then
            for i=1,#g_model.args.num_args do
                prev_action[i] = torch.zeros(g_opts.batch_size, g_model.args.num_args[i])
                if t > 1 then
                    prev_action[i]:scatter(2, g_model.actions[i]:long(), 1)
                end
            end
        end
        if g_opts.multitask then
            action, baseline, states, term = g_model:eval(input[1], input[3], prev_action)
        elseif g_model.args.name == "Opt" then
            action, baseline, states, term = g_model:eval(batch)
        else
            if g_opts.gt_instruction and batch[1].task_idx > prev_task_idx then
                prev_task_idx = batch[1].task_idx
                g_model:reset_init_state(g_opts.batch_size)
                for i=1,#g_model.args.num_args do
                    prev_action[i] = torch.zeros(g_opts.batch_size, g_model.args.num_args[i])
                end
            end
            action, baseline, states, term = g_model:eval(input[1], input[2], prev_action)
        end
        if g_opts.term_action then 
            for k=1,g_opts.batch_size do
                if prev_active[k] == nil or prev_active[k] == 1 then
                    local max_val, max_idx = term[k]:squeeze():max(1)
                    if 2-active[k] ~= max_idx:squeeze() then
                        if active[k] > 0 or giveup[k] == 0 then 
                            term_acc[k] = 0
                        end
                    end
                end
            end
        end

        if g_opts.verbose > 2 then
            print(string.format("eval: %.3f", sys.clock() - time))
            time = sys.clock()
        end

        if g_opts.feasible then
            local feas = g_model:feasible():clone():add(-1)
            batch_giveup(batch, feas, active)
            for k=1,g_opts.batch_size do
                if giveup[k] == 0 and feas[k][1] == 1 then
                    giveup[k] = 1
                end
            end
        end

        if g_opts.save_lstm and g_opts.save_lstm > 0 then
            assert(g_model.m.next_h)
            local tmp, max_idx = g_model.m.g_prob.data.module.output:clone():squeeze():max(1)
            max_idx = max_idx:squeeze()
            local game = batch[1].task[batch[1].task_idx]
            if batch[1]:is_active() and game.k and-- game.k == 3 and 
               max_idx == batch[1].task_idx and
               torch.rand(1):squeeze() < 0.4 and 
               lstm_cnt < g_opts.save_lstm then
                lstm_cnt = lstm_cnt + 1
                lstm_x[lstm_cnt]:copy(g_model.m.next_h.data.module.output)
                local count = game.num_pick or game.num_hit
                lstm_y[lstm_cnt] = game.k - count
                lstm_y2[lstm_cnt] = count + 1
                print(lstm_cnt, game.k - count, count + 1)
                if lstm_cnt == g_opts.save_lstm then
                    torch.save("lstm.t7", {lstm_x, lstm_y, lstm_y2})
                    print("Saved to lstm.t7")
                elseif lstm_cnt % 1000 == 0 then
                    torch.save("lstm.t7", {lstm_x:narrow(1, 1, lstm_cnt), 
                                        lstm_y:narrow(1, 1, lstm_cnt),
                                        lstm_y2:narrow(1, 1, lstm_cnt)})
                    print("Saved to lstm.t7")
                end
            end
        end

        batch_act(batch, action:view(-1), active)
        batch_update(batch, active)
        reward = batch_reward(batch, active, t == g_opts.max_step)
        R:add(torch.cmul(reward, active))

        if g_opts.verbose > 2 then
            print(string.format("update: %.3f", sys.clock() - time))
            time = sys.clock()
        end
        if g_opts.verbose > 3 then
            print("Step", t, "Action", action:squeeze(), "Reward", 
                reward[1], "Term", term:squeeze())
        end
        -- copy hidden states
        if t < g_opts.max_step and g_model.recurrent then
            g_model:set_init_state(states)
        end
        prev_active = active:clone()
        if active:sum() == 0 then break end
    end
    local success = batch_success(batch)

    if g_opts.save_meta then
        local obs_file = video_dir .. "/obs.t7"
        local goal_path = video_dir .. "/goal.txt"
        --torch.save(obs_file, obs)
        local goal_file = io.open(goal_path, "w")
        for k=1,g_opts.max_task do
            local goal_text = goal_text(batch[1].instruction[k])
            if goal_text ~= "" then
                goal_file:write(goal_text)
                goal_file:write("\n")
            else
                break
            end
        end
        goal_file:close()
    end
    R:resize(g_opts.batch_size)
    -- stat by game type
    local stat = {}
    for i, g in pairs(batch) do
        stat[g.ntasks] = stat[g.ntasks] or {}
        stat[g.ntasks].reward = (stat[g.ntasks].reward or 0) + R[i]
        stat[g.ntasks].success = (stat[g.ntasks].success or 0) + success[i]
        stat[g.ntasks].n_task = (stat[g.ntasks].n_task or 0) + g.task_idx - 1
        stat[g.ntasks].step = (stat[g.ntasks].step or 0) + steps[i]
        stat[g.ntasks].count = (stat[g.ntasks].count or 0) + 1
        if g_opts.term_action then
            stat[g.ntasks].term_acc = (stat[g.ntasks].term_acc or 0) + term_acc[i]
        end
        --if g_opts.feasible then
        --    stat[g.ntasks].term_acc = (stat[g.ntasks].term_acc or 0) + term_acc[i]
        --end
    end
    return stat
end

function test_batch_thread(opts_orig, game_idx)
    if game_idx == nil or game_idx == 0 then
        g_opts = opts_orig
        g_factory = g_train_factory
    else
        g_opts = g_test_opts[game_idx]
        g_factory = g_test_factory[game_idx]
    end
    local stat = test_batch()
    g_opts = opts_orig
    g_factory = g_train_factory
    -- print(stat[1].success)
    collectgarbage()
    collectgarbage()
    return stat
end

function test(N, game_idx)
    local stat = {}
    if g_opts.video and g_opts.video ~= '' then
        os.execute("rm -rf " .. g_opts.video)
        os.execute("mkdir " .. g_opts.video)
        os.execute("rm -rf " .. g_opts.video .. "/video")
        os.execute("mkdir " .. g_opts.video .. "/video")
        win = nil
    end
    for k = 1, N do
        collectgarbage()
        if g_opts.progress then
            xlua.progress(k, N)
        end
        if g_opts.nworker > 1 then
            for w = 1, g_opts.nworker do
                g_workers:addjob(w, test_batch_thread,
                    function(s)
                        for k, v in pairs(s) do
                            stat[k] = stat[k] or {}
                            for k2, v2 in pairs(v) do
                                stat[k][k2] = (stat[k][k2] or 0) + v2
                            end
                        end
                    end,
                    g_opts, game_idx
                )
            end
            g_workers:synchronize()
        else
            video_dir = string.format("%s/video/%d", g_opts.video, k)
            if g_opts.video and g_opts.video ~= '' then
                os.execute("mkdir " .. video_dir)
            end

            local s = test_batch()
            for k, v in pairs(s) do
                stat[k] = stat[k] or {}
                for k2, v2 in pairs(v) do
                    stat[k][k2] = (stat[k][k2] or 0) + v2
                end
            end
        end
    end
    --print(stat)
    for k = 1, 100 do
        if type(stat[k]) == "table" then
            v = stat[k]
            stat.reward = (stat.reward or 0) + stat[k].reward
            stat.success = (stat.success or 0) + stat[k].success
            stat.n_task = (stat.n_task or 0) + stat[k].n_task
            if g_opts.term_action then
                stat.term_acc = (stat.term_acc or 0) + stat[k].term_acc
            end
            stat.count = (stat.count or 0) + stat[k].count
            for k2, v2 in pairs(v) do 
                if string.sub(k2, 1, 5) == 'count' then
                    local s = string.sub(k2, 6)
                    stat[k]['reward' .. s] = stat[k]['reward' .. s] / v2
                    stat[k]['n_task' .. s] = stat[k]['n_task' .. s] / v2
                    --if stat[k]['success' .. s] > 0 then
                    --    stat[k]['step' .. s] = stat[k]['step' .. s] / stat[k]['success' .. s]
                    --end
                    stat[k]['step' .. s] = stat[k]['step' .. s] / v2 
                    stat[k]['success' .. s] = stat[k]['success' .. s] / v2
                    if g_opts.term_action then
                        stat[k]['term_acc' .. s] = stat[k]['term_acc' .. s] / v2
                    end
                end
            end
        end
    end
    --print(stat)
    stat.reward = stat.reward / stat.count
    stat.success = stat.success / stat.count
    stat.n_task = stat.n_task / stat.count
    if g_opts.term_action then
        stat.term_acc = stat.term_acc / stat.count
    end
    if g_opts.video and g_opts.video ~= '' then
        os.execute('../video/merge_video ' .. g_opts.video .. '/video')
        --[[
        os.execute('../video/make_gif ' .. g_opts.video .. '/video.mp4 ' .. 
                g_opts.video .. '/video.gif')
        --]]
    end
    print(format_stat_test(stat))
    return stat
end
