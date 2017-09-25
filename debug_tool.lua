function print_goal()
    for i=1,g.instruction:size(1) do
        print(g_ivocab[g.instruction[i][1]], 
            g_ivocab[g.instruction[i][2]], 
            g_ivocab[g.instruction[i][3]], 
            g_ivocab[g.instruction[i][4]])
    end
end

function start() 
    display=require('display') 
    dofile('MazeBase/init.lua')
    dofile('util.lua') 
    g_opts = {games_config_path = 'MazeBase/config/subtask_giveup.lua'} 
    g_init_vocab() 
    g_factory = g_init_game(g_opts, g_vocab, g_opts.games_config_path) 
    g = new_game() 
    win = display.image(g.map:to_image(), {win=win})  
    print_goal()
end

function act(idx)
    g:act(idx) g:update() win = display.image(g.map:to_image(), {win=win}) print(g:get_reward(), g:is_active()) 
end

function new() 
    g = new_game() 
    win = display.image(g.map:to_image(), {win=win})  
    print_goal()
end

function call()
    print("th -ldisplay.start 8000 0.0.0.0")
end
