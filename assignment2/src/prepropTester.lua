require 'svm'
require("train-svm")
dofile './dataKmeanPreProp.lua'
dofile './provider.lua'

require 'xlua'
require 'optim'
require 'cunn'
local c = require 'trepl.colorize'

dataKmeanPreProp = dataKmeanPreProp('./patchProvider_22_20000_400.t7')

-- local raw_train = torch.load('stl-10/extra.t7b')

provider = torch.load 'provider.t7'
provider.trainData.data = provider.trainData.data:float()
provider.valData.data = provider.valData.data:float()

opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 20)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default kmeansModel)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
]]

print(opt)

function parseData(d, numSamples, numChannels, height, width)
  idxs = torch.randperm(100000):long():sub(1,numSamples)
  local t = torch.Tensor(numSamples, numChannels, height, width)
  local idx = 1
  for i = 1, numSamples do
    local this_d = d[idxs[i]]

    t[idx]:copy(this_d)
    idx = idx + 1
  end
  assert(idx == numSamples+1)
  return t
end


-- local FIG_dim = {3, 96, 96}
-- local trsize = 10
-- local data = parseData(raw_train.data[1], trsize, FIG_dim[1], FIG_dim[2], FIG_dim[3])
-- local normFeatures = dataKmeanPreProp:prePropHandler(data, 16, 22, 2, false)

local trsize = 4000
local vasize = 5
local numPatch = 16
local w = 22
local gap = 2
local doNorm = true
local doWhiten = true
local max_epoch = 200
local modelName = ''

idxs = torch.randperm(4000):long():sub(1,trsize)
local train = provider.trainData.data:index(1,idxs)
local trainlabels = provider.trainData.labels:index(1,idxs)

-- idxs = torch.randperm(1000):long():sub(1,vasize)
-- local test = provider.valData.data[{{1,vasize},{}}]
-- local testlabels = provider.valData.labels:index(1,idxs)
-- -- local train = provider.valData.data:index(1,idxs)
-- -- local testlabels = provider.valData.labels[{{1,vasize}}]

local trainFeatures = dataKmeanPreProp:prePropHandler(train, numPatch, w, gap, doNorm, doWhiten)
-- local testFeatures = dataKmeanPreProp:prePropHandler(test, numPatch, w, gap, doNorm, doWhiten)
trainFeatures = trainFeatures:float()
-- testFeatures = testFeatures:float()

print ("hi!", trainFeatures:size())

-- -- svm method
-- t = {}
-- for l=1,trainlabels:size()[1] do
--   table.insert(t,{trainlabels[l],{torch.range(1,4*1600):int(), trainFeatures[l]}})
-- end
-- print(t)

-- t2 = {}
-- for l=1,testlabels:size()[1] do
--   table.insert(t2,{testlabels[l],{torch.range(1,4*1600):int(), trainFeatures[l]}})
-- end

trainFeatures = torch.cat(trainFeatures, torch.ones(trainFeatures:size(1)), 2)
local theta = train_svm(trainFeatures, trainlabels, 100);
print(theta:size())
local val,idx = torch.max(trainFeatures * theta, 2)
local match = torch.eq(trainlabels, idx:float():squeeze()):sum()
local accuracy = match/trainlabels:size(1)*100
print('==> train accuracy is '..accuracy..'%')

-- model = liblinear.train(t)
-- labels,accuracy,dec = liblinear.predict(t,model)




-- print(c.blue '==>' ..' configuring model')
-- local model = nn.Sequential()
-- -- model:add(nn.BatchFlip():float())
-- -- model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
-- model:add(dofile('models/'..opt.model..'.lua'))
-- model:get(1).updateGradInput = function(input) return end

-- confusion = optim.ConfusionMatrix(10)
-- print('Will save at '..opt.save)
-- paths.mkdir(opt.save)
-- valLogger = optim.Logger(paths.concat(opt.save, 'val.log'))
-- valLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (val set)'}
-- valLogger.showPlot = false

-- parameters,gradParameters = model:getParameters()


-- print(c.blue'==>' ..' setting criterion')
-- criterion = nn.CrossEntropyCriterion():cuda()


-- print(c.blue'==>' ..' configuring optimizer')
-- optimState = {
--   learningRate = opt.learningRate,
--   weightDecay = opt.weightDecay,
--   momentum = opt.momentum,
--   learningRateDecay = opt.learningRateDecay,
-- }

-- function train()
--   model:training()
--   epoch = epoch or 1

--   -- drop learning rate every "epoch_step" epochs
--   if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
--   print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

--   trainFeatures = trainFeatures:float()
--   local targets = torch.CudaTensor(opt.batchSize)
--   local indices = torch.randperm(trainFeatures:size(1)):long():split(opt.batchSize)
--   -- remove last element so that all the batches have equal size
--   indices[#indices] = nil

--   local tic = torch.tic()
--   for t,v in ipairs(indices) do
--     xlua.progress(t, #indices)
--     local inputs = trainFeatures:index(1,v)
--     targets:copy(trainlabels:index(1,v))

--     local feval = function(x)
--       if x ~= parameters then parameters:copy(x) end
--       gradParameters:zero()
--       local outputs = model:forward(inputs)
--       print("outputs",outputs)
--       print(targets)
--       local f = criterion:forward(outputs, targets)
--       print("f",f)
--       local df_do = criterion:backward(outputs, targets)
--       print("df_do", df_do)
--       model:backward(inputs, df_do)

--       confusion:batchAdd(outputs, targets)

--       return f,gradParameters
--     end
--     optim.sgd(feval, parameters, optimState)
--   end

--   confusion:updateValids()
--   print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
--         confusion.totalValid * 100, torch.toc(tic)))

--   train_acc = confusion.totalValid * 100

--   confusion:zero()
--   epoch = epoch + 1
-- end

-- function val()
--   -- disable flips, dropouts and batch normalization
--   model:evaluate()
--   print(c.blue '==>'.." valing")
--   local bs = 25
--   for i=1,testFeatures:size(1),bs do
--     local outputs = model:forward(testFeatures:narrow(1,i,bs))
--     confusion:batchAdd(outputs, testlabels:narrow(1,i,bs))
--   end

--   confusion:updateValids()
--   print('val accuracy:', confusion.totalValid * 100)
--   if valLogger then
--     paths.mkdir(opt.save)
--     valLogger:add{train_acc, confusion.totalValid * 100}
--     valLogger:style{'-','-'}
--     valLogger:plot()

--     local base64im
--     do
--       os.execute(('convert -density 200 %s/val.log.eps %s/val.png'):format(opt.save,opt.save))
--       os.execute(('openssl base64 -in %s/val.png -out %s/val.base64'):format(opt.save,opt.save))
--       local f = io.open(opt.save..'/val.base64')
--       if f then base64im = f:read'*all' end
--     end

--     local file = io.open(opt.save..'/report.html','w')
--     file:write(([[
--     <!DOCTYPE html>
--     <html>
--     <body>
--     <title>%s - %s</title>
--     <img src="data:image/png;base64,%s">
--     <h4>optimState:</h4>
--     <table>
--     ]]):format(opt.save,epoch,base64im))
--     for k,v in pairs(optimState) do
--       if torch.type(v) == 'number' then
--         file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
--       end
--     end
--     file:write'</table><pre>\n'
--     file:write(tostring(confusion)..'\n')
--     file:write(tostring(model)..'\n')
--     file:write'</pre></body></html>'
--     file:close()
--   end

--   -- save model every 50 epochs
--   if epoch % 5 == 0 then
--     local filename = paths.concat(opt.save, 'model.net')
--     print('==> saving model to '..filename)
--     torch.save(filename, model:get(3))
--   end

--   confusion:zero()
-- end


-- for i=1,max_epoch do
--   train()
--   -- val()
-- end