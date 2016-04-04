require 'nn'
require 'cunn'
require 'xlua'
require 'optim'
dofile './provider.lua'

----------------------------------------
modelPath = 'vgg_kmeans_surro/model.net'
testDataPath = 'stl-10/test.t7b'
providerPath = 'provider.t7'

print('==> loading model')
model = torch.load(modelPath)

print('==> loading testData')
rawTestData = torch.load(testDataPath)

print('==> loading provider')
provider = torch.load(providerPath)

----------------------------------------
print('==> re-formatting testData')
numSamples = 8000
channel = 3
height = 96
width = 96
testData = {
	data = torch.Tensor(),
	labels = torch.Tensor(),
	size = function() return numSamples end
}
testData.data = torch.ByteTensor(numSamples, channel, height, width)
testData.labels = torch.ByteTensor(numSamples)
local idx = 1
for i = 1, #rawTestData.data do
	local this_d = rawTestData.data[i]
	for j = 1, #this_d do
		--print(#this_d)
		testData.data[idx]:copy(this_d[j])
		testData.labels[idx] = i
		idx = idx + 1
      	end
end
assert(idx == numSamples+1)
testData.data = testData.data:float()
testData.labels = testData.labels:float()

-------------------------------------------
print('==> normalize testData')
local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
for i = 1,testData:size() do
	xlua.progress(i, testData:size())
     	-- rgb -> yuv
     	local rgb = testData.data[i]
     	local yuv = image.rgb2yuv(rgb)
     	-- normalize y locally:
     	yuv[1] = normalization(yuv[{{1}}])
     	testData.data[i] = yuv
end
-- normalize u globally:
mean_u = provider.trainData.mean_u
std_u = provider.trainData.std_u
mean_v = provider.trainData.mean_v
std_v = provider.trainData.std_v

testData.data:select(2,2):add(-mean_u)
testData.data:select(2,2):div(std_u)
-- normalize v globally:
testData.data:select(2,3):add(-mean_v)
testData.data:select(2,3):div(std_v)

-------------------------------------------
print('==> start testing')
model:cuda()
--model:float()
confusion = optim.ConfusionMatrix(10)
predicts = torch.FloatStorage(testData.size())
targets = torch.FloatStorage(testData.size())

-- disable flips, dropouts and batch normalization
model:evaluate()
local bs = 25
local currIdx = 1
for i=1,testData.data:size(1),bs do
	print('testing', i)
	local outputs = model:forward(testData.data:narrow(1,i,bs):cuda())
	--if i == 1 then
	--	print(outputs)
	--	print(testData.labels:narrow(1,i,bs))
	--end
   	confusion:batchAdd(outputs, testData.labels:narrow(1,i,bs))
	--mapping output to predicted class
	for k = 1,bs do
		local maxIdx = 0
   		local maxValue = -9999999
		local pred = outputs[k]
   		for idx = 1,10 do 
      			if maxValue < pred[idx] then
        			maxIdx = idx
	 			maxValue = pred[idx]
      			end
   		end
   		predicts[currIdx] = maxIdx
		targets[currIdx] = testData.labels[currIdx]
		currIdx = currIdx + 1
	end
end

confusion:updateValids()
print('val accuracy:', confusion.totalValid * 100)

print(confusion)

------------------------------------------------------------------------
print '==> output predition to csv'
csv = io.open("predictions.csv", "w")
--csv:write('Id,Prediction'..'\n')
--for i = 1,predicts:size() do
--	csv:write(i..','..predicts[i]..'\n')
--end

   csv:write('Id,Prediction,Target'..'\n')
   for i = 1,predicts:size() do
      csv:write(i..','..predicts[i]..','..targets[i]..'\n')
   end
csv:close()