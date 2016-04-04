require 'torch'   -- torch
require 'image'   -- image
require 'nn'   	  -- nn
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
---------------------------------------------------------------------------
print '==> processing options'
cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('-size', 'full', 'how many samples do we load: small | full')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:option('-output', 'notAll', 'output: all | notAll')
cmd:text()
opt = cmd:parse(arg or {})

----------------------------------------------------------------------
print '==> downloading dataset' 

tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'
data_path = 'mnist.t7'
train_file = paths.concat(data_path, 'train_32x32.t7')
test_file = paths.concat(data_path, 'test_32x32.t7')

if not paths.filep(train_file) or not paths.filep(test_file) then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

----------------------------------------------------------------------
-- training/test size

if opt.size == 'full' then
   print '==> using regular, full training data'
   trsize = 60000
   tesize = 10000
elseif opt.size == 'small' then
   print '==> using reduced training data, for fast experiments'
   trsize = 6000
   tesize = 1000
end

----------------------------------------------------------------------
print '==> loading dataset'

loaded = torch.load(train_file, 'ascii')
trainData = {
   data = loaded.data,
   labels = loaded.labels,
   size = function() return trsize end
}

loaded = torch.load(test_file, 'ascii')
testData = {
   data = loaded.data,
   labels = loaded.labels,
   size = function() return tesize end
}

----------------------------------------------------------------------
print '==> preprocessing data'

trainData.data = trainData.data:float()
testData.data = testData.data:float()

print '==> preprocessing data: normalize globally'
mean = trainData.data[{ {},1,{},{} }]:mean()
std = trainData.data[{ {},1,{},{} }]:std()

-- Normalize test data, using the training means/stds
testData.data[{ {},1,{},{} }]:add(-mean)
testData.data[{ {},1,{},{} }]:div(std)

-- Local normalization
print '==> preprocessing data: normalize locally'

-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(7)
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- Normalize all channels locally:
for i = 1,testData:size() do
   testData.data[{ i,{1},{},{} }] = normalization:forward(testData.data[{ i,{1},{},{} }])
end

--------------------------------------------------------------------------
print '==> load model'
model = torch.load('model.net')

--------------------------------------------------------------------------
print '==> testing'
classes = {'1','2','3','4','5','6','7','8','9','0'}
confusion = optim.ConfusionMatrix(classes)
predicts = torch.FloatStorage(testData.size())
targets = torch.FloatStorage(testData.size())

-- local vars
local time = sys.clock()

-- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
model:evaluate()

-- test over test data
print('==> testing on test set:')
for t = 1,testData:size() do
   -- disp progress
   xlua.progress(t, testData:size())

   -- get new sample
   local input = testData.data[t]
   if opt.type == 'double' then input = input:double()
   elseif opt.type == 'cuda' then input = input:cuda() end
   local target = testData.labels[t]

   -- test sample
   local pred = model:forward(input)
   confusion:add(pred, target)
   local maxIdx = 0
   local maxValue = -9999999
   for idx = 1,10 do 
      if maxValue < pred[idx] then
         maxIdx = idx
	 maxValue = pred[idx]
      end
   end
   predicts[t] = maxIdx
   targets[t] = target
end

-- timing
time = sys.clock() - time
time = time / testData:size()
print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

-- print confusion matrix
print(confusion)

------------------------------------------------------------------------
print '==> output predition to csv'
csv = io.open("predictions.csv", "w")

if opt.output == 'all' then
   csv:write('Id,Prediction,Target'..'\n')
   for i = 1,predicts:size() do
      csv:write(i..','..predicts[i]..','..targets[i]..'\n')
   end
else 
   csv:write('Id,Prediction'..'\n')
   for i = 1,predicts:size() do
      csv:write(i..','..predicts[i]..'\n')
   end
end
csv:close()
