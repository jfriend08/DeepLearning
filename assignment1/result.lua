require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

print '==> processing options'
cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('-size', 'full', 'how many samples do we load: small | full')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-output', '', 'All: output predictions and target, else use empty')
cmd:text()
opt = cmd:parse(arg or {})

dofile '1_data.lua'
dofile 'testResult.lua'
dofile 'Csv.lua'

print '==> load test data'
print(testData)

print '==> load model'
local filename = paths.concat(opt.save, 'model.net')
model = torch.load(filename)
print(model)

print '==> testing'
classes = {'1','2','3','4','5','6','7','8','9','0'}
confusion = optim.ConfusionMatrix(classes)
predicts = torch.FloatStorage(testData.size())
targets = torch.FloatStorage(testData.size())
test()

print '==> output predition to csv'
local separator = ','
local csv = Csv("predictions.csv", "w", separator)

if opt.output == 'All' then
      csv:write({"Id","Prediction", "Targets"}) -- write header
      for i = 1,predicts:size() do
           csv:write({i, predicts[i], targets[i]}) -- write each data row
      end
else 
      csv:write({"Id","Prediction"}) -- write header
      for i = 1,predicts:size() do
           csv:write({i, predicts[i]}) -- write each data row
      end
end
csv:close()



