--[[

 Input: A table {x, k} of a vector x and a convolution kernel k.

 Output: Circular convolution of x with k.

--]]

if nn.CircularConvolution == nil then
    paths.dofile("CircularConvolution.lua")
end

if nn.ScalarMulTable == nil then
    paths.dofile("ScalarMulTable.lua")
end
require "nngraph"

local Register, parent = torch.class('nn.Register', 'nn.Module')

function Register:__init()
    parent.__init(self)
    self.gradInput = {}
    
    local o = nn.Identity()()
    local op = nn.Identity()()
    local init = nn.Identity()()
    local shift_vec = nn.Narrow(2, 1, 3)(op)
    local reset = nn.Narrow(2, 4, 1)(op)
    local o_tilde = nn.CircularConvolution(){o, shift_vec}
    local out = nn.CAddTable()({
        nn.ScalarMulTable()({init, reset}),
        o_tilde})

    self.net = nn.gModule({o, op, init}, {out})
end

function Register:updateOutput(input)
    local o, op = unpack(input)
    
    self.init = self.init or o.new()
    self.init:resizeAs(o):zero()

    if o:dim() == 1 then
        self.init[1] = 1
    elseif o:dim() == 2 then
        self.init:narrow(2, 1, 1):fill(1)
    else
        error("Dimension mismatch")
    end

    self.output = self.net:forward({o, op, self.init})
    return self.output
end

function Register:updateGradInput(input, gradOutput)
    local o, op = unpack(input)
    --[[
    self.gradInput[1] = self.gradInput[1] or o.new()
    self.gradInput[2] = self.gradInput[2] or op.new()
    self.gradInput[1]:resizeAs(o)
    self.gradInput[2]:resizeAs(op)
    --]]
    
    self.gradInput[1], self.gradInput[2], self.tmp = unpack(self.net:backward({o, op, self.init}, gradOutput))
    return self.gradInput
end
