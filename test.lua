-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

local function init()
    paths.dofile('util.lua')
    paths.dofile('model.lua')
    paths.dofile('test_batch.lua')
    paths.dofile('MazeBase/init.lua')
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.setnumthreads(1)
end

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
            function(opts_orig, vocab_orig, mutex_id)
                old_print = print
                print = function(...)
                    io.write("[" .. __threadid .. "] ")
                    old_print(...)
                    io.flush()
                end 
                g_opts = opts_orig
                g_vocab = vocab_orig
                local threads = require "threads"
                g_mutex = threads.Mutex(mutex_id)
                g_factory = g_init_game(g_opts, g_vocab, g_opts.games_config_path)
                if g_opts.curriculum == 1 then
                    g_factory:hardest()
                    g_factory:freeze()
                end
                g_train_factory = g_factory
                --g_init_game()
                g_init_model()
                g_opts.img_size = g_model.args.img_size or 64
                if g_opts.backend == 'minecraft' then
                    g_init_minecraft() 
                end
            end,
            function() end,
            g_opts, g_vocab, mutex_id
        )
    end
    workers:synchronize()
    return workers
end

require('xlua')
init()

local cmd = torch.CmdLine()
-- model parameters
cmd:option('--ch', 3, 'the number of channels for conditional weight')
cmd:option('--edim', 256, 'the size of the hidden vector')
cmd:option('--ldim', 256, 'the size of the lstm vector')
cmd:option('--wdim', 256, 'the size of the embedding vector')
cmd:option('--model', 'conv', 'model type: conv | manager')
cmd:option('--n_filters', '32,64', 'num of convolution filters')
cmd:option('--pool', '2', 'size of pooling')
cmd:option('--stride', '1,1', 'size of pooling')
cmd:option('--pad', '1,1', 'size of pooling')
cmd:option('--filter_size', '3,3', 'size of pooling')
cmd:option('--soft_mem', false, 'soft memory attention if true')

cmd:option('--param', '', 'file name to pre-trained the model (copy only params)')
cmd:option('--task_name', 'maze', 'save model name')
-- game parameters
cmd:option('--sub_agent', '', 'file paths for sub-agents (comma-separated)')
cmd:option('--backend', 'minecraft', 'maze or minecraft')
cmd:option('--game', 'goto1', 'MazeBase/config/%s.lua')
cmd:option('--channels', 18, 'the number of input channels')
cmd:option('--max_task', 20, 'the number of instructions')
cmd:option('--max_word', 6, 'max number of words for each instruction')
cmd:option('--max_args', 2, 'max number of argumnets for subagents')
cmd:option('--term_action', false, 'the agent should terminate the episode if true')
cmd:option('--n_actions', 13, 'the number of actions')
cmd:option('--counting', true, 'counting task if true')
cmd:option('--counting2', false, 'counting task if true')
cmd:option('--interact', false, 'interact task if true')
cmd:option('--open_loop', false, 'use termination signal as time-scale if true')
cmd:option('--strict_eval', true, 'strict evaluation if true')

--other
cmd:option('--batch_size', 1, 'size of mini-batch (the number of parallel games) in each thread')
cmd:option('--nworker', 1, 'the number of threads used for training')
cmd:option('--load', '', 'file name to load the model')
cmd:option('--video', '', 'folder to write screen images and video')
cmd:option('--display', false, 'display the screen')
cmd:option('--verbose', 1, 'debug message level')
cmd:option('--n_play', 30, 'num of play')
cmd:option('--multitask', false, 'multi-task sub-agent pre-training if true')
cmd:option('--progress', true, 'show progress bar if true')
cmd:option('--regularizer', 0, '')
cmd:option('--csv', '', 'csv output')
cmd:option('--save_meta', false, 'save meta data output')
cmd:option('--save_raw', false, 'save raw images')
cmd:option('--save_meta', false, 'save meta data')
cmd:option('--save_lstm', 0, 'save meta data')
cmd:option('--gt_instruction', false, 'ground truth instruction')
cmd:option('--gpu', 0, 'gpu id')

g_opts = cmd:parse(arg or {})
g_opts.test = true
if g_opts.video ~= '' or g_opts.display then
    g_opts.batch_size = 1
    g_opts.nworker = 1
end
g_opts.games_config_path = "MazeBase/config/" .. g_opts.game .. ".lua"
g_opts.game = nil
-- print(g_opts)
g_init_vocab()
if g_opts.nworker > 1 then
    g_workers = init_threads()
end
g_factory = g_init_game(g_opts, g_vocab, g_opts.games_config_path)
if g_opts.curriculum == 1 then
    g_factory:hardest()
    g_factory:freeze()
end
g_train_factory = g_factory
g_init_model()
g_opts.img_size = g_model.args.img_size or 64
if g_opts.backend == 'minecraft' then
    if g_opts.nworker == 1 then 
        g_init_minecraft() 
    else
        g_minecraft = true
    end
end
test(g_opts.n_play)
