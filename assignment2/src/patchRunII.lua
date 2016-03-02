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
  local kSize = 28
  local gap = 2
  local nPatch = 9

  print("==> loading dataset")

  local raw_train = torch.load('stl-10/extra.t7b')
  -- local raw_val = torch.load('stl-10/val.t7b')
  -- parse train data
  self.trainData = {
     data = torch.Tensor(trsize, FIG_dim[1]*FIG_dim[2]*FIG_dim[3]),
     labels = torch.Tensor(trsize),
     size = function() return trsize end
  }
  self.trainData.data = parseData(raw_train.data[1], trsize, FIG_dim[1], FIG_dim[2], FIG_dim[3])
  local trainData = self.trainData
end

function patchRun:normalize()
  local trainData = self.trainData
  print '<trainer> preprocessing data (color space + normalization)'
  collectgarbage()

  -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,trainData:size() do
     xlua.progress(i, trainData:size())
     -- rgb -> yuv
     local rgb = trainData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     trainData.data[i] = yuv
  end
  -- normalize u globally:
  local mean_u = trainData.data:select(2,2):mean()
  local std_u = trainData.data:select(2,2):std()
  trainData.data:select(2,2):add(-mean_u)
  trainData.data:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = trainData.data:select(2,3):mean()
  local std_v = trainData.data:select(2,3):std()
  trainData.data:select(2,3):add(-mean_v)
  trainData.data:select(2,3):div(std_v)

  trainData.mean_u = mean_u
  trainData.std_u = std_u
  trainData.mean_v = mean_v
  trainData.std_v = std_v

end

function patchRun:whiten()
  print 'start whiten patches'
  collectgarbage()

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
  self.patches.data, M, P = zca_whiten(self.patches.data)
end

function patchRun:runKmean(ncentroids, niter)
  print 'start whiten centroids'
  -- local ncentroids = 1600
  self.patches.centroids, counts = unsup.kmeans_modified(self.patches.data, ncentroids, nil, 0.1, niter, 1000, nil, true)
  print(self.patches.data:size())
  print(self.patches.centroids:size())
end

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


function patchRun:getPatch()
  print 'start getting patches'
  collectgarbage()

  local kSize = 28
  local gap = 2
  local nPatch = 9
  local FIG_dim = {3, 96, 96}

  numFig = self.trainData:size()

  self.patches = {
     data = torch.Tensor(),
     centroids = torch.Tensor(),
     size = function() return numFig*nPatch end
  }
  self.patches.data = parsePatch(self.trainData.data, numFig, nPatch, FIG_dim[1], FIG_dim[2], FIG_dim[2], kSize, gap)
end

function parsePatch(d, numSamples, numPatch, numChannels, height, width, kSize, gap)
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
