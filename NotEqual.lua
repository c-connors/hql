local NotEqual, parent = torch.class('nn.NotEqual', 'nn.Module')

function NotEqual:__init(value)
   parent.__init(self)
   self.value = value
end

function NotEqual:updateOutput(input)
   self.output = input:ne(self.value):typeAs(input)
   return self.output
end

function NotEqual:updateGradInput(input, gradOutput)
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
