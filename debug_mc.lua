function print_goal()
    for i=1,g.instruction:size(1) do
        if g.instruction[i][1] == g_vocab['nil'] then
            break
        end
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
    img_size = 64
    g_opts = {games_config_path = 'MazeBase/config/mc_nc.lua', 
            img_size = img_size, backend = 'minecraft'} 
    g_init_vocab() 
    g_factory = g_init_game(g_opts, g_vocab, g_opts.games_config_path) 
    g_init_minecraft()
    g = new_game()
    print_goal()
   
    g_img = torch.FloatTensor(3, img_size, img_size)
    g:to_image(g_img)
    win = display.image({g_img, g.map:to_image()}, {win=win, saturate=false})  
    image.save("temp.png", g_img)
end

function act(idx)
    g:act(idx) 
    g:update() 
    g:to_image(g_img)
    win = display.image({g_img, g.map:to_image()}, {win=win, saturate=false})  
    print(g:get_reward(), g:is_active()) 
end

function new() 
    g = new_game() 
    g:to_image(g_img)
    win = display.image({g_img, g.map:to_image()}, {win=win, saturate=false})  
    print_goal()
end

function call()
    print("th -ldisplay.start 8000 0.0.0.0")
end
