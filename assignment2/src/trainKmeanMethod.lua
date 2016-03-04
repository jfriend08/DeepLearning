require 'xlua'
require 'optim'
require 'cunn'
-- dofile './provider.lua'
dofile './patchRunII.lua'

local c = require 'trepl.colorize'

--this method includes:
--1. patch parsing
--2. feature mapping by centroids
--3. 4 quadrants convolution
--4. layers of training

patchRun = torch.load 'patchProvider_4000.t7'
patchRun.patches.data = patchRun.patches.data:float()
patchRun.patches.centroids = patchRun.patches.centroids:float()
print(patchRun.patches.data:size())
print(patchRun.patches.centroids:size())

provider = torch.load 'provider.t7'
provider.trainData.data = provider.trainData.data:float()
provider.valData.data = provider.valData.data:float()
print(provider.trainData.data:size())

local FIG_dim = {3, 96, 96}

trainData, TrainLabels = parsePatch(provider.trainData.data, 16, FIG_dim[1], FIG_dim[2], FIG_dim[3], 22, 2)







function parsePatch(d, numPatch, numChannels, height, width, kSize, gap)
  local t = torch.Tensor(numSamples*numPatch, numChannels*kSize*kSize)
  local idx = 1
  for i = 1, numSamples do
    local this_d = d[i]

    for row = 0, 3 do
      for col = 0, 3 do
        if row*(kSize+gap)+kSize < height and col*(kSize+gap)+kSize < width then
          c1 = image.crop(this_d, row*(kSize+gap),col*(kSize+gap), row*(kSize+gap)+kSize,col*(kSize+gap)+kSize)

          -- local filename = paths.concat("../fig/patchII", i.."_"..idx.."_After"..".png")
          -- image.save(filename, c1)

          t[idx]:copy(c1:resize(numChannels*kSize*kSize))
          idx = idx + 1
        end
      end
    end
  end
  assert(idx == numSamples*numPatch+1)
  return t
end