require 'svm'

dofile './dataKmeanPreProp.lua'
dofile './provider.lua'

dataKmeanPreProp = dataKmeanPreProp('./patchProvider_22_20000_400.t7')

-- local raw_train = torch.load('stl-10/extra.t7b')

provider = torch.load 'provider.t7'
provider.trainData.data = provider.trainData.data:float()
provider.valData.data = provider.valData.data:float()


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

local trsize = 1000
local numPatch = 16
local w = 22
local gap = 2
local doWhiten = true

idxs = torch.randperm(4000):long():sub(1,trsize)
local train = provider.trainData.data:index(1,idxs)
local trainlabels = provider.trainData.labels:index(1,idxs)

-- local test = provider.valData.data[{{1,trsize},{}}]
-- local testlabels = provider.valData.labels[{{1,trsize}}]

local trainFeatures = dataKmeanPreProp:prePropHandler(train, numPatch, w, gap, doWhiten)
-- local testFeatures = dataKmeanPreProp:prePropHandler(test, numPatch, w, gap, doWhiten)

print ("hi!", trainFeatures:size())

t = {}
for l=1,trainlabels:size()[1] do
  table.insert(t,{trainlabels[l],{torch.range(1,4*1600):int(), trainFeatures[l]}})
end
print(t)

-- t2 = {}
-- for l=1,testlabels:size()[1] do
--   table.insert(t2,{testlabels[l],{torch.range(1,4*1600):int(), trainFeatures[l]}})
-- end

model = liblinear.train(t)
labels,accuracy,dec = liblinear.predict(t,model)
-- labels,accuracy,dec = liblinear.predict(t2,model)






