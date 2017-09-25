require "nn"
require "optim"
require "xlua"

local cmd = torch.CmdLine()
cmd:option('--input', 'all', 'all | high | low')
opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')

lr = 0.001
batch_size = 100
hdim = 128
data_x, data_y2, data_y = unpack(torch.load("lstm.t7"))

num_data = data_x:size(1)
num_train = num_data * 0.9
num_test = num_data * 0.1

train_x = data_x:narrow(1, 1, num_train)
train_y = data_y:narrow(1, 1, num_train)
test_x = data_x:narrow(1, num_train + 1, num_test)
test_y = data_y:narrow(1, num_train + 1, num_test)

local ldim = data_x:size(2)
local sdim = ldim / 2

net = nn.Sequential()
if opt.input == 'all' then
    x_dim = ldim
else
    x_dim = sdim
end
--net:add(nn.Linear(x_dim, hdim))
--net:add(nn.ReLU(true))
net:add(nn.Linear(x_dim, 3))
net:add(nn.LogSoftMax())
w, dw = net:getParameters()
criterion = nn.ClassNLLCriterion()

print(string.format("#train: %d, #test: %d", num_train, num_test))
-- Optim
train_confusion = optim.ConfusionMatrix(3)
test_confusion = optim.ConfusionMatrix(3)
local config = {learningRate = lr}
local adam_state = {}

for epoch=1,10 do
    idx = torch.randperm(train_x:size(1))
    idx = idx:view(-1, batch_size):long()
    collectgarbage()

    for iter=1,idx:size(1) do
        x = train_x:index(1, idx[iter])
        y = train_y:index(1, idx[iter])

        if opt.input == 'high' then
            x = x:narrow(2, 1, sdim)
        elseif opt.input == 'low' then
            x = x:narrow(2, sdim+1, sdim)
        end
        local feval = function(param)
            if param ~= w then
                w:copy(param)
            end
            dw:zero()
            local out = net:forward(x)
            local loss = criterion:forward(out, y)
            local dy = criterion:backward(out, y)
            net:backward(x, dy)

            for i = 1, batch_size do
                train_confusion:add(out[i], y[i])
            end
            return loss, dw
        end
        optim.adam(feval, w, config, adam_state)
    end

    for iter=1,num_test/batch_size do
        x = test_x:narrow(1, (iter-1)*batch_size+1, batch_size)
        y = test_y:narrow(1, (iter-1)*batch_size+1, batch_size)
        if opt.input == 'high' then
            x = x:narrow(2, 1, sdim)
        elseif opt.input == 'low' then
            x = x:narrow(2, sdim+1, sdim)
        end
        local out = net:forward(x)
        for i = 1, batch_size do
            test_confusion:add(out[i], y[i])
        end
    end
    print(train_confusion)
    print(test_confusion)
    print(string.format("Epoch %d, Train Accuracy: %.2f, Test Accuracy: %.2f", 
            epoch, train_confusion.totalValid * 100, test_confusion.totalValid * 100))
    train_confusion:zero()
    test_confusion:zero()
end
