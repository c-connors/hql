require "os"
function test_model(game, model, video, multitask, num_play, extra)
    local etc = extra or ""
    if multitask then
        etc = etc .. " --multitask --max_task 1"
    end
    if num_play then
        etc = etc .. " --n_play " .. num_play
    end
    if video then
        os.execute("qlua test.lua --game " .. game .. " --load " .. model .. " --video " .. video  .. " --progress --n_play 10" .. etc)
        itorch.video(video .. "/video.mp4")  
    else
        os.execute("OMP_NUM_THREADS=1 th test.lua --game " .. game .. " --load " .. model .. " --batch_size 1 --nworker 16 --n_play 10" .. etc)
    end
end
