--[[

 Input: indices 

 Output: one hot vectors 

--]]

local OneHot, parent = torch.class('nn.OneHot', 'nn.Module')

function OneHot:__init(outputSize)
    parent.__init(self)
    self.outputSize = outputSize
    -- We'll construct one-hot encodings by using the index method to
    -- reshuffle the rows of an identity matrix. To avoid recreating
    -- it every iteration we'll cache it.
    self._eye = torch.eye(outputSize)
end

function OneHot:updateOutput(input)
    self.output:resize(input:size(1), self.outputSize):zero()
    if self._eye == nil then self._eye = torch.eye(self.outputSize) end
    self._eye = self._eye:float()
    local longInput = input:long():squeeze()
    if type(longInput) == 'number' then
        longInput = torch.Tensor({longInput}):long()
    end
    self.output:copy(self._eye:index(1, longInput))
    return self.output
end

function OneHot:updateGradInput(input, gradOutput)
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
