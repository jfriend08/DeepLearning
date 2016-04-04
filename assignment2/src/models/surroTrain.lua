require 'cunn'
nn = require 'nn'
dofile './surrogate.lua'

model = nn.Sequential()
model:add(nn.SpatialConvolution(3, 64, 5,5, 1,1, 1,1))
model:add(nn.SpatialBatchNormalization(64,1e-3))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
model:add(nn.Dropout(0.3))

model:add(nn.SpatialConvolution(64, 128, 5,5, 1,1, 1,1))
model:add(nn.SpatialBatchNormalization(128,1e-3))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
model:add(nn.Dropout(0.4))

model:add(nn.SpatialConvolution(128, 256, 5,5, 1,1, 1,1))
model:add(nn.SpatialBatchNormalization(256,1e-3))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
model:add(nn.View(256*3*3))

classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256*3*3,512))
classifier:add(nn.BatchNormalization(512))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512,4000))

model:add(classifier)
r = model:forward(inputs)

provider = torch.load 'Surrogate.t7'
provider.trainData.data = provider.surrogateData.data:float()

batchSize = 120
indices = torch.randperm(provider.trainData.data:size(1)):long():split(batchSize)
targets = torch.CudaTensor(batchSize)

indices[#indices] = nil
inputs = provider.trainData.data:index(1,indices[1])
targets:copy(provider.surrogateData.labels:index(1,indices[1]))