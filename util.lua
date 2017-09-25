-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

function tensor_to_words(input, show_prob)
    for i = 1, input:size(1) do
        local line = i .. ':'
        for j = 1, input:size(2) do
            line = line .. '\t'  .. g_ivocab[input[i][j]]
        end
        if show_prob then
            for h = 1, g_opts.nhop do
                line = line .. '\t' .. string.format('%.2f', g_modules[h]['prob'].output[1][i])
            end
        end
        print(line)
    end
end

function rmsprop(opfunc, x, config, state)
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-2
    local alpha = config.alpha or 0.99
    local epsilon = config.epsilon or 1e-8

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)

    -- (2) initialize mean square values and square gradient storage
    if not state.m then
      state.m = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
      state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
    end

    -- (3) calculate new (leaky) mean squared values
    state.m:mul(alpha)
    state.m:addcmul(1.0-alpha, dfdx, dfdx)

    -- (4) perform update
    state.tmp:sqrt(state.m):add(epsilon)
    x:addcdiv(-lr, dfdx, state.tmp)

    -- return x*, f(x) before optimization
    return x, {fx}
end

function format_stat(stat)
    local a = {}
    for n in pairs(stat) do table.insert(a, n) end
    table.sort(a)
    local str = ''
    if stat.dist_cost then
        str = str .. 'dist_cost: ' .. string.format("%2.4g",stat['dist_cost']) .. ' '
    end
    if stat.term_cost then
        str = str .. 'term_cost: ' .. string.format("%2.4g",stat['term_cost']) .. ' '
    end
    if stat.arg_loss then
        str = str .. 'arg_loss: ' .. string.format("%2.4g",stat['arg_loss']) .. ' '
    end
    if stat.bl_cost then
        str = str .. 'bl_cost: ' .. string.format("%2.4g",stat['bl_cost']) .. ' '
    end
    str = str .. 'reward: ' .. string.format("%2.4g",stat['reward']) .. ' '
    str = str .. 'success: ' .. string.format("%2.4g",stat['success']) .. ' '
    --str = str .. 'step: ' .. string.format("%2.2g",stat['step']) .. ' '
    if stat.term_acc then
        str = str .. 'term_acc: ' .. string.format("%2.4g",stat['term_acc']) .. ' '
    end
    if stat.epoch then
        str = str .. 'epoch: ' .. stat['epoch']
    end
    return str
end

function format_stat_test(stat)
    local f = nil
    if g_opts.csv and g_opts.csv ~= '' then
        f = io.open(g_opts.csv, "w")
    end
    local str2 = ''
    local str = ''
    for s=1,100 do
        if stat[s] then
            str = str .. '#task: ' .. s .. ' '
            str = str .. 'reward: ' .. string.format("%2.4f",stat[s]['reward']) .. ' '
            str = str .. 'success: ' .. string.format("%2.4f",stat[s]['success']) .. ' '
            str = str .. 'n_task: ' .. string.format("%2.4f",stat[s]['n_task']) .. ' '
            str = str .. 'step: ' .. string.format("%2.1f",stat[s]['step']) .. ' '
            if stat.term_acc then
                str = str .. 'term_acc: ' .. string.format("%2.4f",stat[s]['term_acc']) .. ' '
            end
            str = str .. '\n'

            str2 = str2 .. s .. ' '
            str2 = str2 .. string.format("%2.4f",stat[s]['reward']) .. ' '
            str2 = str2 .. string.format("%2.4f",stat[s]['success']) .. ' '
            str2 = str2 .. string.format("%2.4f",stat[s]['n_task']) .. ' '
            str2 = str2 .. string.format("%2.1f",stat[s]['step']) .. ' '
            str2 = str2 .. '\n'
        end
    end
    str = str .. 'reward: ' .. string.format("%2.4f",stat.reward) .. ' '
    str = str .. 'success: ' .. string.format("%2.4f",stat.success) .. ' '
    str = str .. 'n_task: ' .. string.format("%2.4f",stat.n_task) .. ' '
    if stat.term_acc then
        str = str .. 'term_acc: ' .. string.format("%2.4f",stat.term_acc) .. ' '
    end
    if f then
        f:write(str2)
        f:flush()
    end
    return str
end

function print_tensor(a)
    local str = ''
    for s = 1, a:size(1) do str = str .. string.format("%2.4g",a[s]) .. ' '  end
    return str
end
function format_helpers(gname)
    local str = ''
    if not gname then
        for i,j in pairs(g_factory.helpers) do
            str = str .. i .. ' :: '
            str = str .. 'task: ' .. print_tensor(j.ntasks) .. ' ||| '
            str = str .. 'mapW: ' .. print_tensor(j.mapW) .. ' ||| '
            str = str .. 'mapH: ' .. print_tensor(j.mapH) .. ' ||| '
            str = str .. 'wpct: ' .. print_tensor(j.waterpct) .. ' ||| '
            str = str .. 'bpct: ' .. print_tensor(j.blockspct) .. ' ||| '
            str = str .. '\n'
        end
    else
        local j = g_factory.helpers[gname]
        str = str .. gname .. ' :: '
        str = str .. 'task: ' .. print_tensor(j.ntasks) .. ' ||| '
        str = str .. 'mapW: ' .. print_tensor(j.mapW) .. ' ||| '
        str = str .. 'mapH: ' .. print_tensor(j.mapH) .. ' ||| '
        str = str .. 'wpct: ' .. print_tensor(j.waterpct) .. ' ||| '
        str = str .. 'bpct: ' .. print_tensor(j.blockspct) .. ' ||| '
        str = str .. '\n'
    end
    return str
end

--[[
function g_load_model()
    if g_opts.load ~= '' then
        if paths.filep(g_opts.load) == false then
            print('WARNING: Failed to load from ' .. g_opts.load)
            return
        end
        local f = torch.load(g_opts.load)
        g_paramx:copy(f.paramx)
        g_log = f.log
        g_plot_stat = {}
        for i = 1, #g_log do
            g_plot_stat[i] = {g_log[i].epoch, g_log[i].reward, g_log[i].success, g_log[i].bl_cost}
        end
        if f['rmsprop_state'] then g_rmsprop_state = f['rmsprop_state'] end
        print('model loaded from ', g_opts.load)
    end
end

function g_save_model()
    if g_opts.save ~= '' then
        f = {opts=g_opts, paramx=g_paramx, log=g_log}
        if g_rmsprop_state then f['rmsprop_state'] = g_rmsprop_state end
        torch.save(g_opts.save, f)
        print('model saved to ', g_opts.save)
    end
end
--]]

function parse_to_table(str)
    local t = {}
    if str then
        for s in string.gmatch(str, '([^,]+)') do
            if s ~= "" then
                table.insert(t, s)
            end
        end
    end
    return t
end

function parse_to_num_table(str)
    local t = {}
    if str then
        for s in string.gmatch(str, '([^,]+)') do
            if s ~= "" then
                table.insert(t, tonumber(s))
            end
        end
    end
    return t
end

function merge_table(dst, src)
    for k,v in pairs(src) do
        dst[k] = v
    end
end
