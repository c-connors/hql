--[[

 Input: log probability distirbution

 Output: samples from multinomial distribution

--]]

local Multinomial, parent = torch.class('nn.Multinomial', 'nn.Module')

function Multinomial:__init()
  parent.__init(self)
end

function Multinomial:updateOutput(input)
    local prob = torch.exp(input)
    if input:dim() == 1 then
        self.output:resize(1):zero()
        self.output[1] = torch.multinomial(prob, 1)
        return self.output
    elseif input:dim() == 2 then
        local idx = torch.multinomial(prob, 1)
        return self.output:set(idx:typeAs(self.output))
    else 
        error("input must be a vector or matrix")
    end
end

function Multinomial:updateGradInput(input, gradOutput)
   -- the input can be of any type (as in the forward it's 
   -- converted anyway to LongTensor) thus, need to allocate
   -- new memory each time the user changes the input type
   if torch.type(self.gradInput) ~= torch.type(input) then
      self.gradInput = input.new()
   end
   if not self.gradInput:isSameSizeAs(input) then
      self.gradInput:resizeAs(input):zero()
   end
   return self.gradInput
end
