require 'nn'

local vgg = nn.Sequential()
classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(6400,1600))
classifier:add(nn.BatchNormalization(1600))
-- classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(1600,10))
classifier:add(nn.BatchNormalization(10))

-- classifier:add(nn.Dropout(0.5))
-- classifier:add(nn.Linear(1600,800))
-- classifier:add(nn.BatchNormalization(800))

-- classifier:add(nn.Dropout(0.5))
-- classifier:add(nn.Linear(800,400))
-- classifier:add(nn.BatchNormalization(400))

-- classifier:add(nn.Dropout(0.5))
-- classifier:add(nn.Linear(400,10))
-- classifier:add(nn.BatchNormalization(10))
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
