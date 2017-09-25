--[[

 Input: A table {x, y} of a Tensor x and a scalar y.

 Output: x^y (elementwise)

--]]

local PowTable, parent = torch.class('nn.PowTable', 'nn.Module')

function PowTable:__init()
  parent.__init(self)
  self.gradInput = {}
end

function PowTable:updateOutput(input)
    local v, p = unpack(input)
    if v:dim() == 1 then
        return self.output:set(torch.pow(v, p[1]))
    elseif v:dim() == 2 then
        self.output:resizeAs(v)
        self.output:copy(v)
        for i=1, v:size(2) do
            self.output:narrow(2, i, 1):cpow(p)
        end
        return self.output
    else 
        error("input must be a vector or matrix")
    end
end

function PowTable:updateGradInput(input, gradOutput)
    local v, p = unpack(input)
    self.gradInput[1] = self.gradInput[1] or input[1].new()
    self.gradInput[2] = self.gradInput[2] or input[2].new()
    self.gradInput[1]:resizeAs(input[1])
    self.gradInput[2]:resizeAs(input[2])

    if v:dim() == 1 then
        p = p[1]
        self.gradInput[1]:set(torch.cmul(gradOutput, torch.pow(v, p - 1)) * p)
        local pgrad = 0
        for i = 1, v:size(1) do
            if v[i] > 0 then
                pgrad = pgrad + math.log(v[i]) * self.output[i] * gradOutput[i]
            end
        end
        self.gradInput[2][1] = pgrad
    elseif v:dim() == 2 then
        self.gradInput[1]:copy(v)
        self.gradInput[2]:zero()
        for i=1, v:size(2) do
            self.gradInput[1]:narrow(2, i, 1):cpow(torch.add(p, -1)):cmul(p)
            self.gradInput[2]:add(torch.log(v:narrow(2, i, 1)):cmul(
                    self.output:narrow(2, i, 1)):cmul(gradOutput:narrow(2, i, 1)))
        end
    else
        error("input must be a vector or matrix")
    end
    return self.gradInput
end
