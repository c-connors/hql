-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

paths.dofile('MazeBase.lua')
paths.dofile('OptsHelper.lua')
paths.dofile('MultiGoals.lua')
paths.dofile('MultiGoalsAbsolute.lua')
paths.dofile('MultiGoalsHelper.lua')
paths.dofile('CondGoals.lua')
paths.dofile('CondGoalsHelper.lua')
paths.dofile('Exclusion.lua')
paths.dofile('ExclusionHelper.lua')
paths.dofile('Switches.lua')
paths.dofile('SwitchesHelper.lua')
paths.dofile('LightKey.lua')
paths.dofile('Goto.lua')
paths.dofile('GotoHelper.lua')
paths.dofile('GotoHidden.lua')
paths.dofile('GotoHiddenHelper.lua')
paths.dofile('GameFactory.lua')
paths.dofile('StarEnemy.lua')
paths.dofile('Bullet.lua')
paths.dofile('PushableBlock.lua')
paths.dofile('PushBlock.lua')
paths.dofile('PushBlockCardinal.lua')
paths.dofile('BlockedDoor.lua')
paths.dofile('MultiAgentsStar.lua')
paths.dofile('MultiAgentsStarHelper.lua')
paths.dofile('GotoCardinal.lua')
paths.dofile('GotoSwitch.lua')
paths.dofile('SingleGoal.lua')
paths.dofile('batch.lua')

paths.dofile('SequentialBase.lua')
paths.dofile('SubTask.lua')
paths.dofile('SubSingleGoal.lua')
paths.dofile('SubPickUp.lua')
paths.dofile('SubHit.lua')
paths.dofile('SubCollectAll.lua')
paths.dofile('SubHitAll.lua')
paths.dofile('SubCollectK.lua')
paths.dofile('SubHitK.lua')
paths.dofile('SubInteract.lua')

paths.dofile('Minecraft.lua')

local function init_game_opts(opts, vocab)
    local games = {}
    local helpers = {}
    local tasks = {}
    games.MultiGoals = MultiGoals
    helpers.MultiGoals = MultiGoalsHelper
    games.MultiGoalsAbsolute = MultiGoalsAbsolute
    helpers.MultiGoalsAbsolute = MultiGoalsHelper
    games.CondGoals = CondGoals
    helpers.CondGoals = CondGoalsHelper
    games.Exclusion = Exclusion
    helpers.Exclusion = ExclusionHelper
    games.Switches = Switches
    helpers.Switches = SwitchesHelper
    games.GotoSwitch = GotoSwitch
    helpers.GotoSwitch = SwitchesHelper
    games.LightKey = LightKey
    helpers.LightKey = OptsHelper
    games.Goto = Goto
    helpers.Goto = GotoHelper
    games.GotoCardinal = GotoCardinal
    helpers.GotoCardinal = OptsHelper
    games.GotoHidden = GotoHidden
    helpers.GotoHidden = GotoHiddenHelper
    games.PushBlock = PushBlock
    helpers.PushBlock = OptsHelper
    games.PushBlockCardinal = PushBlockCardinal
    helpers.PushBlockCardinal = OptsHelper
    games.BlockedDoor = BlockedDoor
    helpers.BlockedDoor = OptsHelper
    games.MultiAgentsStar = MultiAgentsStar
    helpers.MultiAgentsStar = MultiAgentsStarHelper
    games.SingleGoal = SingleGoal
    helpers.SingleGoal = OptsHelper
    games.SequentialBase = SequentialBase
    helpers.SequentialBase = OptsHelper
    tasks.SubSingleGoal = SubSingleGoal
    tasks.SubPickUp = SubPickUp
    tasks.SubHit = SubHit
    tasks.SubCollectAll = SubCollectAll
    tasks.SubHitAll = SubHitAll
    tasks.SubCollectK = SubCollectK
    tasks.SubHitK = SubHitK
    tasks.SubInteract = SubInteract
    return GameFactory(opts,vocab,games,helpers,tasks)
end

function g_init_vocab()
    local function vocab_add(word)
        if g_vocab[word] == nil then
            local ind = g_opts.nwords + 1
            g_opts.nwords = g_opts.nwords + 1
            g_vocab[word] = ind
            g_ivocab[ind] = word
        end
    end
    g_vocab = {}
    g_ivocab = {}
    g_ivocabx = {}
    g_ivocaby = {}
    g_opts.nwords = 0

    -- general
    vocab_add('nil')
    vocab_add('agent')
    vocab_add('block')
    vocab_add('water')

    vocab_add('pick')
    vocab_add('up')
    vocab_add('transform')

    if g_opts.backend ~= 'minecraft' then
        -- objects
        vocab_add('candy')
        vocab_add('wood')
        vocab_add('pig')
        vocab_add('tree')
        vocab_add('rock')
    end

    -- for LightKey
    -- for Exclusion
    vocab_add('visit')
    vocab_add('all')

    if g_opts.backend == "minecraft" then
        vocab_add('cow')
        vocab_add('pig')
        vocab_add('chicken')
        vocab_add('box')
        vocab_add('horse')
        vocab_add('ice')
        vocab_add('enemy')
        vocab_add('sheep')
        vocab_add('cat')
    else
        vocab_add('cow')
        vocab_add('milk')
        vocab_add('box')
        vocab_add('diamond')
        vocab_add('meat')
        vocab_add('duck')
        vocab_add('egg')
        vocab_add('heart')
        vocab_add('enemy')
        vocab_add('brick')
    end
    if g_opts.counting then
        for i=1,5 do
            vocab_add(tostring(i))
        end
    end
    if g_opts.counting2 then
        for i=1,7 do
            vocab_add(tostring(i))
        end
    end

    if g_opts.interact then
        vocab_add('interact')
        vocab_add('with')
    end
end

function g_init_minecraft(opts)
    g_minecraft = Minecraft.new(g_opts.task_name, g_opts.img_size)
end

function g_init_game(opts, vocab, config_path)
    local load_game = dofile(config_path)
    opts = load_game(opts)
    return init_game_opts(opts, vocab)
end

function new_game()
    local game
    if g_opts.game == nil or g_opts.game == '' then
        game = g_factory:init_random_game()
    else
        game = g_factory:init_game(g_opts.game)
    end
    if g_minecraft then
        g_minecraft:reset(game)
    end
    return game
end

function g_new_task(name, param, maze)
    return g_factory:new_task(name, param, maze)
end
