local DrLimCriterion, parent = torch.class('nn.DrLimCriterion', 'nn.Criterion')

function DrLimCriterion:__init(margin)
   parent.__init(self)
   self.margin = margin or 1
   self.sizeAverage = true
end 
 
function DrLimCriterion:updateOutput(input, y)
   self.buffer = self.buffer or input.new()
   if not torch.isTensor(y) then 
      self.ty = self.ty or input.new():resize(1)
      self.ty[1]=y
      y=self.ty
   end

   self.buffer:resizeAs(input):copy(input)
   self.buffer[torch.eq(y, -1)] = 0
   self.output = self.buffer:pow(2):sum()

   self.buffer:fill(self.margin):add(-1, input)
   self.buffer:cmax(0)
   self.buffer[torch.eq(y, 1)] = 0
   self.output = self.output + self.buffer:pow(2):sum()
   
   if (self.sizeAverage == nil or self.sizeAverage == true) then 
      self.output = self.output / input:nElement()
   end
   return self.output
end

function DrLimCriterion:updateGradInput(input, y)
   if not torch.isTensor(y) then self.ty[1]=y; y=self.ty end
   self.gradInput:resizeAs(input):copy(input)
   self.gradInput[torch.eq(y, -1)] = 0
   self.buffer:copy(input):add(-self.margin)
   self.buffer:cmin(0)
   self.gradInput:add(torch.cmul(torch.eq(y, -1):typeAs(self.buffer), self.buffer))
   self.gradInput:mul(2)
   if (self.sizeAverage == nil or self.sizeAverage == true) then
      self.gradInput:mul(1 / input:nElement())
   end
      
   return self.gradInput 
end
