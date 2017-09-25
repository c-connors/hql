require "model"
require "os"

local cmd = torch.CmdLine()
cmd:option('--model', '')
cmd:option('--output', 'temp')
opt = cmd:parse(arg or {})

m = torch.load(opt.model)
graph.dot(m.net.fg, m.args.name, opt.output)
