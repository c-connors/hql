--[[

 Input: Log probability distribution

 Output: Entropy 

--]]

local Entropy, parent = torch.class('nn.Entropy', 'nn.Module')

function Entropy:__init()
    parent.__init(self)
end

function Entropy:updateOutput(input)
    self.tmp = self.tmp or input.new()
    self.tmp:resizeAs(input)
    self.tmp:copy(input):exp():cmul(input)
    
    if input:dim() == 1 then
        self.output:resize(1)
        self.output[1] = -self.tmp:sum()
    elseif input:dim() == 2 then
        self.output = self.tmp:sum(2):mul(-1.0):mean()
    else
        error("input must be a vector or matrix")
    end
    return self.output
end

function Entropy:updateGradInput(input, gradOutput)
    if torch.type(self.gradInput) ~= torch.type(input) then
        self.gradInput = input.new()
    end
    self.gradInput:resizeAs(input)
    self.gradInput:copy(input):add(1.0)
    self.gradInput:cmul(torch.exp(input)):mul(-1.0)
    if input:dim() == 1 then
        self.gradInput:mul(gradOutput)
    elseif input:dim() == 2 then
        self.gradInput:mul(gradOutput / input:size(1))
    else
        error("input must be a vector or matrix")
    end

    return self.gradInput
end
