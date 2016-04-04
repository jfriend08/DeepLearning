require 'nn'

local vgg = nn.Sequential()

vgg:add(nn.SpatialConvolution(3, 64, 5,5, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(64,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
vgg:add(nn.Dropout(0.1))

vgg:add(nn.SpatialConvolution(64, 128, 5,5, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(128,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
vgg:add(nn.Dropout(0.1))

vgg:add(nn.SpatialConvolution(128, 256, 5,5, 1,1, 1,1))
vgg:add(nn.SpatialBatchNormalization(256,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
vgg:add(nn.View(256*3*3))

classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256*3*3,512))
classifier:add(nn.BatchNormalization(512))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.1))
classifier:add(nn.Linear(512,4000))

vgg:add(classifier)


-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'nn.SpatialConvolution'
end

MSRinit(vgg)

-- check that we can propagate forward without errors
-- should get 16x10 tensor
--print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))

return vgg
