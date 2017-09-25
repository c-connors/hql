--[[

 Input: A table {x, y} of a Tensor x and a scalar y.

 Output: x * y

--]]

local ScalarAddTable, parent = torch.class('nn.ScalarAddTable', 'nn.Module')

function ScalarAddTable:__init()
    parent.__init(self)
    self.gradInput = {}
end

function ScalarAddTable:updateOutput(input)
    local v, constant = unpack(input)
    if v:dim() == 1 then
        error("Dimension mismatch")
    elseif v:dim() == 2 then
        error("Dimension mismatch")
    elseif v:dim() == 3 then
        assert(constant:dim() == 2)
        self.output:resizeAs(v)
        self.output:copy(v)
        for i=1, v:size(1) do
            for j=1, v:size(2) do
                self.output[i][j]:add(constant[i][j])
            end
        end
    else
        error("Dimension mismatch")
    end
    return self.output
end

function ScalarAddTable:updateGradInput(input, gradOutput)
    local v, constant = unpack(input)
    self.gradInput[1] = self.gradInput[1] or input[1].new()
    self.gradInput[2] = self.gradInput[2] or input[2].new()

    if v:dim() == 1 then
        error("Dimension mismatch")
    elseif v:dim() == 2 then
        error("Dimension mismatch")
    elseif v:dim() == 3 then
        self.gradInput[1]:resizeAs(v):copy(gradOutput)
        self.gradInput[2]:resizeAs(constant):copy(gradOutput:sum(3):view(v:size(1), v:size(2)))
    else
        error("Dimension mismatch")
    end
    return self.gradInput
end
