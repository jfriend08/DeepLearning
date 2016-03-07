#!/usr/bin/env th
-- An Analysis of Single-Layer Networks in Unsupervised Feature Learning
-- by Adam Coates et al. 2011
--
-- The original MatLab code can be found in http://www.cs.stanford.edu/~acoates/
-- Tranlated to Lua/Torch7
--
require 'xlua'
require 'image'
require 'unsup'
require 'kmeans'
-- require("extract")
-- require("train-svm")
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

dofile './provider.lua'

opt = {
   whiten = true,
}
print(opt)

local patchRun = torch.class 'patchRun'

function patchRun:__init(full)
  local FIG_dim = {3, 96, 96}
  local trsize = 4000
  local tesize = 1000
  local kSize = 28
  local gap = 2
  local nPatch = 9
  -- local nkernel1 = 32
  -- local nkernel2 = 32
  -- local fanIn1 = 1
  -- local fanIn2 = 4

  print("==> loading dataset")

  local raw_train = torch.load('stl-10/extra.t7b')
  -- local raw_val = torch.load('stl-10/val.t7b')
  -- parse train data
  self.trainData = {
     data = torch.Tensor(trsize, FIG_dim[1]*FIG_dim[2]*FIG_dim[3]),
     labels = torch.Tensor(trsize),
     size = function() return trsize end
  }
  self.trainData.data = parsePatch(raw_train.data[1], trsize, nPatch, FIG_dim[1], FIG_dim[2], FIG_dim[3], kSize, gap)

  if opt.whiten then
    print("==> whiten patches")
    local function zca_whiten(x)
      local dims = x:size()
      local nsamples = dims[1]
      local ndims    = dims[2]
      local M = torch.mean(x, 1)
      local D, V = unsup.pcacov(x)
      x:add(torch.ger(torch.ones(nsamples), M:squeeze()):mul(-1))
      local diag = torch.diag(D:add(0.1):sqrt():pow(-1))
      local P = V * diag * V:t()
      x = x * P
      return x, M, P
    end
  self.trainData.data, M, P = zca_whiten(self.trainData.data)
  end

  print("==> find clusters")
  local ncentroids = 1600
  kernels, counts = unsup.kmeans_modified(self.trainData.data, ncentroids, nil, 0.1, 300, 1000, nil, true)
  print(self.trainData.data:size())
  print(kernels:size())
  print(counts:size())

end

-- create Surrogate set
function parsePatch(d, numSamples, numPatch, numChannels, height, width, kSize, gap)
  local t = torch.Tensor(numSamples*numPatch, numChannels*kSize*kSize)
  local idx = 1
  for i = 1, numSamples do
    local this_d = d[i]

    local mean_u = this_d:float():mean()
    local std_u = this_d:float():std()

    for row = 0, 3 do
      for col = 0, 3 do
        if row*(kSize+gap)+kSize < height and col*(kSize+gap)+kSize < width then
          c1 = image.crop(this_d, row*(kSize+gap),col*(kSize+gap), row*(kSize+gap)+kSize,col*(kSize+gap)+kSize)
          c1 = c1:add(-mean_u)
          c1 = c1:div(std_u)
          -- local filename = paths.concat("../fig/patch", i.."_"..idx.."_After"..".png")
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

-- function parseDataLabel(d, numSamples, numChannels, height, width)
--    local t = torch.ByteTensor(numSamples, numChannels*height*width)
--    local l = torch.ByteTensor(numSamples)
--    local idx = 1
--    for i = 1, #d do
--       local this_d = d[i]
--       for j = 1, #this_d do
--         t[idx]:copy(this_d[j]:resize(numChannels*height*width))
--         l[idx] = i
--         idx = idx + 1
--       end
--    end
--    assert(idx == numSamples+1)
--    return t, l
-- end