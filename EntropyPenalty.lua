local EntropyPenalty, parent = torch.class('nn.EntropyPenalty', 'nn.Module')

function EntropyPenalty:__init(w, log)
    parent.__init(self)
    assert(w, "w is not specified")
    self.w = w
    self.log = log
    self.eps = 1e-12
    if self.log == nil then
        self.log = true
    end
end

function EntropyPenalty:updateOutput(input)
    if self.w ~= 0 then
        self.tmp = self.tmp or input.new()
        self.tmp:resizeAs(input)

        if self.log then
            self.tmp:copy(input):exp():cmul(input)
        else
            self.tmp:copy(input):add(self.eps):log():cmul(input)
        end
        
        if input:dim() == 1 then
            self.loss = -self.tmp:sum() * self.w
        elseif input:dim() == 2 then
            self.loss = self.tmp:sum(2):mul(-1.0):mean() * self.w
        else
            error("input must be a vector or matrix")
        end
    end
    self.output = input
    return self.output
end

function EntropyPenalty:updateGradInput(input, gradOutput)
    if torch.type(self.gradInput) ~= torch.type(input) then
        self.gradInput = input.new()
    end
    self.gradInput:resizeAs(input)
    if self.w ~= 0 then
        self.tmp2 = self.tmp2 or input.new()
        self.tmp2:resizeAs(input)
        if self.log then
            self.tmp2:copy(input):add(1.0)
            self.tmp2:cmul(torch.exp(input)):mul(-1.0)
        else
            self.tmp2:copy(input):add(self.eps):log():add(1.0):mul(-1.0)
        end
        if input:dim() == 1 then
            self.tmp2:mul(self.w)
        elseif input:dim() == 2 then
            self.tmp2:mul(self.w / input:size(1))
        else
            error("input must be a vector or matrix")
        end
        self.gradInput:copy(gradOutput):add(self.tmp2)
    else
        self.gradInput:copy(gradOutput)
    end
    return self.gradInput
end
