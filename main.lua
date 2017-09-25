-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

local function init()
    paths.dofile('util.lua')
    paths.dofile('nnutils.lua')
    paths.dofile('model.lua')
    paths.dofile('train.lua')
    paths.dofile('test_batch.lua')
    paths.dofile('MazeBase/init.lua')
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.setnumthreads(1)
end

require('xlua')
local function init_threads()
    print('starting ' .. g_opts.nworker .. ' workers')
    local threads = require('threads')
    threads.Threads.serialization('threads.sharedserialize')
    g_mutex = threads.Mutex()
    local mutex_id = g_mutex:id()
    local workers = threads.Threads(g_opts.nworker, init)
    workers:specific(true)
    for w = 1, g_opts.nworker do
        workers:addjob(w,
            function(opts_orig, vocab_orig, ivocab_orig, mutex_id)
                g_opts = opts_orig
                g_vocab = vocab_orig
                g_ivocab = ivocab_orig
                local threads = require "threads"
                g_mutex = threads.Mutex(mutex_id)
                old_print = print
                print = function(...)
                    io.write("[" .. __threadid .. "] ")
                    old_print(...)
                    io.flush()
                end 
                if g_opts.backend == 'minecraft' then
                    g_init_minecraft() 
                end
                g_factory = g_init_game(g_opts, g_vocab, g_opts.games_config_path)
                g_train_opts = g_opts
                g_train_factory = g_factory
                g_test_opts = {}
                g_test_factory = {}
                for i=1,#g_opts.test_game do
                    g_test_opts[i] = {}
                    merge_table(g_test_opts[i], g_opts)
                    g_test_opts[i].test = true
                    g_test_opts[i].games_config_path = "MazeBase/config/" .. g_opts.test_game[i] .. ".lua"
                    g_test_factory[i] = g_init_game(g_test_opts[i], g_vocab, g_test_opts[i].games_config_path)
                    if g_opts.curriculum == 1 then
                        g_test_factory[i]:hardest()
                        g_test_factory[i]:freeze()
                    end
                end
                g_init_model() 
            end,
            function() end,
            g_opts, g_vocab, g_ivocab, mutex_id
        )
    end
    workers:synchronize()
    return workers
end

init()

local cmd = torch.CmdLine()
--filter_size 8,5 --stride 4,2 --pad 0,0 --n_filters 32,64 --feat_size 6
-- model parameters
cmd:option('--ch', 3, 'the number of channels for conditional weight')
cmd:option('--edim', 256, 'the size of the hidden vector')
cmd:option('--ldim', 64, 'the size of the lstm vector')
cmd:option('--wdim', 256, 'the size of the embedding vector')
cmd:option('--model', 'CNN', 'model type: conv | manager')
cmd:option('--init_std', 0, 'STD of initial weights')
cmd:option('--n_filters', '32,64', 'num of convolution filters')
cmd:option('--pool', '', 'size of pooling')
cmd:option('--stride', '4,2', 'size of pooling')
cmd:option('--pad', '0,0', 'size of pooling')
cmd:option('--filter_size', '8,5', 'size of pooling')
cmd:option('--feat_size', 6, 'size of pooling')
cmd:option('--soft_mem', false, 'soft memory attention if true')
cmd:option('--buf', 1, 'the number of recent observations as input at each time')
-- game parameters
cmd:option('--backend', 'maze', 'maze or minecraft')
cmd:option('--sub_agent', '', 'file paths for sub-agents (comma-separated)')
cmd:option('--game', 'visit1', 'MazeBase/config/%s.lua')
cmd:option('--task_name', 'maze', 'save model name')
cmd:option('--channels', 3, 'the number of input channels')
cmd:option('--n_actions', 13, 'the number of actions')
cmd:option('--max_task', 20, 'the number of instructions')
cmd:option('--max_word', 6, 'max number of words for each instruction')
cmd:option('--max_args', 2, 'max number of argumnets for subagents')
cmd:option('--img_size', 64, 'image size')
-- training parameters
cmd:option('--multitask', false, 'multi-task sub-agent pre-training if true')
cmd:option('--xi', 0, 'weight for distillation')
cmd:option('--xi_epoch', 150, 'max epoch for distillation')
cmd:option('--teachers', '', 'pretrained teachers for multi-task sub-agent training')
cmd:option('--lr', 1e-4, 'learning rate')
cmd:option('--max_grad_norm', 40, 'gradient clip value')
cmd:option('--alpha', 0.03, 'coefficient of baseline term in the cost function')
cmd:option('--epochs', 1000, 'the number of training epochs')
cmd:option('--nbatches', 50, 'the number of mini-batches in one epoch')
cmd:option('--batch_size', 32, 'size of mini-batch (the number of parallel games) in each thread')
cmd:option('--nworker', 16, 'the number of threads used for training')
cmd:option('--gamma', 0.99, 'discount factor')
cmd:option('--lambda', 0.96, 'generalized advantage multiplier')
cmd:option('--lr_mult', 1, 'learning rate multiplier after each epoch')
cmd:option('--test_iter', 10, 'the number of test iterations during training')
cmd:option('--test_game', 'visit1', 'test environments during training')
cmd:option('--rho', 0.01, 'weight for entropy regularization')
cmd:option('--rho_epoch', 75, 'max epoch that applies entropy regularization')
cmd:option('--initialize', false, 'init manager from sub-agent if true')
cmd:option('--regularizer', 0, 'weight for argument regularzer')
cmd:option('--clip_bl', true, 'clip baseline grad to [-1, 1] if true')
cmd:option('--clip_adv', false, 'clip advantage grad to [-1, 1] if true')
cmd:option('--term_action', false, 'the agent should terminate the episode if true')
cmd:option('--feasible', false, 'the agent should prediction feasibility if true')
cmd:option('--zeta', 0.1, 'weight for termination classification')
cmd:option('--l1', 0, 'l1 regularization')
cmd:option('--l2', 0, 'l1 regularization')
cmd:option('--drop', 0, 'l1 regularization')
cmd:option('--cos', 0, 'l1 regularization')
cmd:option('--eps', 0, 'randomness for multi-task')
cmd:option('--l1_arg1', 0, 'l1 regularization for argument')
cmd:option('--l1_arg2', 0, 'l1 regularization for argument')
cmd:option('--soft_rho', false, 'entroy regularization for soft models')
cmd:option('--hrl', false, 'hierarchical rl')
cmd:option('--ta', false, 'temporal abstraction')
cmd:option('--base', false, 'richer baseline')
cmd:option('--save_param', false, 'save param')
cmd:option('--iter_size', 1, 'number of iterations for efficient memory')
cmd:option('--analogy_bs', 0, 'size of mini-batch (the number of parallel games) in each thread')
cmd:option('--counting', false, 'counting task if true')
cmd:option('--counting2', false, 'counting task if true')
cmd:option('--interact', false, 'interact task if true')
cmd:option('--pass_term', true, 'pass termination information if true')
cmd:option('--open_loop', false, 'use termination signal as time-scale if true')
cmd:option('--strict_eval', false, 'strict evaluation if true')
cmd:option('--gt_instruction', false, 'ground truth instruction')

cmd:option('--comp', 'conv', 'compare function between register and repeat number')
-- for rmsprop
cmd:option('--beta', 0.97, 'parameter of RMSProp')
--other
cmd:option('--save_folder', 'model', 'folder name to save the model')
cmd:option('--save', '', 'save model name')
cmd:option('--postfix', '', 'file name to save the model')
cmd:option('--load', '', 'file name to load the model (replace)')
cmd:option('--param', '', 'file name to pre-trained the model (copy only params)')
cmd:option('--display', false, 'display the screen')
cmd:option('--verbose', 1, 'debug message level')
cmd:option('--gpu', 0, 'gpu id')

g_opts = cmd:parse(arg or {})
if g_opts.multitask then
    g_opts.max_task = 1
end

if g_opts.gpu > 0 then
    g_opts.nworker = 1
    print("Multi-thread is disabled due to GPU mode.")

    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    cutorch.setDevice(g_opts.gpu)
    gpu = cutorch.getDevice()
    print('Using GPU device id:', gpu-1)
    cudnn.benchmark = true
end

if g_opts.save_folder ~= '' then
    os.execute("mkdir -p " .. g_opts.save_folder)
end
if g_opts.save_folder ~= '' and g_opts.save == '' then
    local model_name = g_opts.model
    if g_opts.postfix ~= '' then
        model_name = model_name .. "_" .. g_opts.postfix
    end
    if g_opts.ta then
        model_name = model_name .. "_t"
    end
    if g_opts.soft_mem then
        model_name = model_name .. "_soft"
    end
    if g_opts.soft_rho then
        model_name = model_name .. '_e'
    end
    if g_opts.multitask and g_opts.teachers ~= "" then
        model_name = model_name .. "_dist"
    end
    if g_opts.multitask and g_opts.xi > 0 then
        model_name = model_name .. "_xi" .. g_opts.xi
    end
    if g_opts.rho > 0 then
        model_name = model_name .. "_r" .. g_opts.rho
    end
    if g_opts.term_action and g_opts.zeta > 0 then
        model_name = model_name .. "_z" .. g_opts.zeta
    end
    if g_opts.multitask and g_opts.buf > 1 then
        model_name = model_name .. "_b" .. g_opts.buf
    end
    if g_opts.l1 > 0 then
        model_name = model_name .. "_l" .. g_opts.l1
    end
    if g_opts.l2 > 0 then
        model_name = model_name .. "_z" .. g_opts.l2
    end
    if g_opts.base then
        model_name = model_name .. "_b"
    end
    if g_opts.cos > 0 then
        model_name = model_name .. "_c" .. g_opts.cos
    end
    if g_opts.drop > 0 then
        model_name = model_name .. "_d" .. g_opts.drop
    end
    if g_opts.eps > 0 then
        model_name = model_name .. "_e" .. g_opts.eps
    end
    if g_opts.l1_arg1 > 0 then
        model_name = model_name .. "_a1" .. g_opts.l1_arg1
    end
    if g_opts.l1_arg2 > 0 then
        model_name = model_name .. "_a2" .. g_opts.l1_arg2
    end
    if g_opts.regularizer > 0 then
        -- assert(g_opts.multitask)
        model_name = model_name .. "_g" .. g_opts.regularizer
    end
    g_opts.save = string.format('%s/%s_%s_%g', g_opts.save_folder,
            g_opts.game, model_name, g_opts.lr)
else
    g_opts.save = string.format('%s/%s_%s', g_opts.save_folder,
        g_opts.game, g_opts.save)
end
local temp_file = io.open(g_opts.save .. ".tmp", "w")
g_opts.games_config_path = "MazeBase/config/" .. g_opts.game .. ".lua"
g_test_opts = {}
if g_opts.test_game == '' then
    --g_opts.test_game = {g_opts.game}
else
    g_opts.test_game = parse_to_table(g_opts.test_game)
    --table.insert(g_opts.test_game, 1, g_opts.game)
    for i=1,#g_opts.test_game do
        g_test_opts[i] = {}
        merge_table(g_test_opts[i], g_opts)
    end
end
print(g_opts)
if g_opts.backend == 'minecraft' then
    g_minecraft = true
end
g_init_vocab()
g_opts.game = nil
g_factory = g_init_game(g_opts, g_vocab, g_opts.games_config_path)
g_train_opts = g_opts
g_train_factory = g_factory
g_test_factory = {}
for i=1,#g_opts.test_game do
    g_test_opts[i].test = true
    g_test_opts[i].games_config_path = "MazeBase/config/" .. g_opts.test_game[i] .. ".lua"
    g_test_factory[i] = g_init_game(g_test_opts[i], g_vocab, g_test_opts[i].games_config_path)
    if g_opts.curriculum == 1 then
        g_test_factory[i]:hardest()
        g_test_factory[i]:freeze()
    end
end

if g_opts.nworker > 1 then
    g_workers = init_threads()
end
g_init_model()

g_log = {}
g_rmsprop_state = {}
--[[
for k,v in pairs(g_model.m) do
    if v.data.module.weight then
        print(k, v.data.module.weight:size())
    end
end
--]]
train(g_opts.epochs)
g_save_model()
os.execute("rm -rf " .. g_opts.save .. ".tmp")
