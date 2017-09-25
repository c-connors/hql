--[[

 Input: A table {x, y} of a Tensor x and a scalar y.

 Output: x * y

--]]

local ScalarMulTable, parent = torch.class('nn.ScalarMulTable', 'nn.Module')

function ScalarMulTable:__init()
    parent.__init(self)
    self.gradInput = {}
end

function ScalarMulTable:updateOutput(input)
    local v, scale = unpack(input)
    if v:dim() == 1 then
        self.output:set(v * scale[1])
    elseif v:dim() == 2 then
        self.output:resizeAs(v)
        self.output:copy(v)
        for i=1, v:size(2) do
            self.output:narrow(2, i, 1):cmul(scale)
        end
    else
        error("Dimension mismatch")
    end
    return self.output
end

function ScalarMulTable:updateGradInput(input, gradOutput)
    local v, scale = unpack(input)
    self.gradInput[1] = self.gradInput[1] or input[1].new()
    self.gradInput[2] = self.gradInput[2] or input[2].new()

    if v:dim() == 1 then
        self.gradInput[2]:resizeAs(input[2])
        self.gradInput[1]:set(gradOutput * scale[1])
        self.gradInput[2][1] = gradOutput:dot(v)
    elseif v:dim() == 2 then
        self.gradInput[1]:resizeAs(v):copy(gradOutput)
        for i=1, v:size(2) do
            self.gradInput[1]:narrow(2, i, 1):cmul(scale)
        end
        self.tmp = self.tmp or input[1].new()
        self.tmp:resizeAs(v)
        self.tmp:copy(v):cmul(gradOutput)

        self.gradInput[2]:resizeAs(scale)
        self.gradInput[2]:copy(self.tmp:sum(2))
    else
        error("Dimension mismatch")
    end
    return self.gradInput
end
