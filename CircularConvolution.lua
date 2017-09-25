--[[

 Input: A table {x, k} of a vector x and a convolution kernel k.

 Output: Circular convolution of x with k.

 TODO: This module can probably be implemented more efficiently.

--]]

local CircularConvolution, parent = torch.class('nn.CircularConvolution', 'nn.Module')

function CircularConvolution:__init()
    parent.__init(self)
    self.gradInput = {}
end

function rotate_left(input, step)
    local output = input.new():resizeAs(input)
    if input:dim() == 1 then
        local size = input:size(1)
        output[{{1, size - step}}] = input[{{step + 1, size}}]
        output[{{size - step + 1, size}}] = input[{{1, step}}]
    elseif input:dim() == 2 then
        local size = input:size(2)
        output[{{},{1, size - step}}] = input[{{},{step + 1, size}}]
        output[{{},{size - step + 1, size}}] = input[{{},{1, step}}]
    else
        error("Dimension mismatch")
    end
  return output
end

function rotate_right(input, step)
    local output = input.new():resizeAs(input)
    if input:dim() == 1 then
        local size = input:size(1)
        output[{{step + 1, size}}] = input[{{1, size - step}}]
        output[{{1, step}}] = input[{{size - step + 1, size}}]
    elseif input:dim() == 2 then
        local size = input:size(2)
        output[{{},{step + 1, size}}] = input[{{},{1, size - step}}]
        output[{{},{1, step}}] = input[{{},{size - step + 1, size}}]
    else
        error("Dimension mismatch")
    end
    return output
end

function CircularConvolution:updateOutput(input)
    local v, k = unpack(input)
    if v:dim() == 1 then
        self.size = v:size(1)
        self.kernel_size = k:size(1)
        self.kernel_shift = math.floor(self.kernel_size / 2)
        self.output = v.new():resize(self.size):zero()
        for i = 1, self.size do
            for j = 1, self.kernel_size do
                local idx = i + self.kernel_shift - j + 1
                if idx < 1 then idx = idx + self.size end
                if idx > self.size then idx = idx - self.size end
                self.output[{{i}}]:add(k[j] * v[idx])
            end
        end
    elseif v:dim() == 2 then
        self.size = v:size(2)
        self.kernel_size = k:size(2)
        self.kernel_shift = math.floor(self.kernel_size / 2)
        self.output = v.new():resizeAs(v):zero()
        for i = 1, self.size do
            for j = 1, self.kernel_size do
                local idx = i + self.kernel_shift - j + 1
                if idx < 1 then idx = idx + self.size end
                if idx > self.size then idx = idx - self.size end
                self.output:narrow(2, i, 1):add(
                    torch.cmul(k:narrow(2, j, 1), v:narrow(2, idx, 1)))
            end
        end
    else
        error("Dimension mismatch")
    end
    return self.output
end

function CircularConvolution:updateGradInput(input, gradOutput)
    local v, k = unpack(input)
    self.gradInput[1] = self.gradInput[1] or v.new()
    self.gradInput[2] = self.gradInput[2] or k.new()
    self.gradInput[1]:resizeAs(v)
    self.gradInput[2]:resizeAs(k)

    if v:dim() == 1 then
        local gradOutput2 = rotate_right(gradOutput:repeatTensor(1, 2):view(2 * self.size), 
                self.kernel_shift)
        for i = 1, self.size do
            self.gradInput[1][i] = k:dot(gradOutput2:narrow(1, i, self.kernel_size))
        end
        local v2 = rotate_left(v:repeatTensor(1, 2):view(2 * self.size), 
                self.kernel_shift + 1)
        for i = 1, self.kernel_size do
            self.gradInput[2][i] = gradOutput:dot(v2:narrow(1, self.size - i + 1, self.size))
        end
    elseif v:dim() == 2 then
        local gradOutput2 = rotate_right(gradOutput:repeatTensor(1, 2), self.kernel_shift)
        for i = 1, self.size do
            self.gradInput[1]:narrow(2, i, 1):copy(torch.cmul(k, 
                gradOutput2:narrow(2, i, self.kernel_size)):sum(2))
        end

        local v2 = rotate_left(v:repeatTensor(1, 2), self.kernel_shift + 1)
        for i = 1, self.kernel_size do
            self.gradInput[2]:narrow(2, i, 1):copy(torch.cmul(gradOutput, 
                v2:narrow(2, self.size - i + 1, self.size)):sum(2))
        end
    else
        error("Dimension mismatch")
    end
    return self.gradInput
end
