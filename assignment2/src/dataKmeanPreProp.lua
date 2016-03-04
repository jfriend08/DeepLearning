require 'nn'
require 'image'

local dataKmeanPreProp = torch.class 'dataKmeanPreProp'
dofile './patchRunII.lua'

function dataKmeanPreProp:__init(path)
  self.kmeanProvider = torch.load(path)
end


function dataKmeanPreProp:prePropHandler(d, numPatch, kSize, gap, runWhiten)
  local numSamples = d:size()[1]
  local FIG_dim = {3,96,96}
  data_norm = normalize(d)
  patches = parsePatch(data_norm, numSamples, numPatch, FIG_dim[1], FIG_dim[2], FIG_dim[3], kSize, gap)
  print(patches:size())
  print(self.kmeanProvider.patches.centroids:size())
  if runWhiten then
    patches = whiten(patches)
    print(patches:size())
  end

end


function normalize(d)
  local data = d
  print 'preprocessing data'
  collectgarbage()
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1, data:size()[1] do
     xlua.progress(i, data:size()[1])
     -- rgb -> yuv
     local rgb = d[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     d[i] = yuv
  end
  -- normalize u globally:
  local mean_u = data:select(2,2):mean()
  local std_u = data:select(2,2):std()
  data:select(2,2):add(-mean_u)
  data:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = data:select(2,3):mean()
  local std_v = data:select(2,3):std()
  data:select(2,3):add(-mean_v)
  data:select(2,3):div(std_v)
  return data
end

function parsePatch(d, numSamples, numPatch, numChannels, height, width, kSize, gap)
  local t = torch.Tensor(numSamples, numPatch, numChannels*kSize*kSize)
  local idx = 1
  for i = 1, numSamples do
    local this_d = d[i]
    local p = torch.Tensor(numPatch, numChannels*kSize*kSize)
    for row = 0, 3 do
      for col = 0, 3 do
        if row*(kSize+gap)+kSize < height and col*(kSize+gap)+kSize < width then
          c1 = image.crop(this_d, row*(kSize+gap),col*(kSize+gap), row*(kSize+gap)+kSize,col*(kSize+gap)+kSize)

          -- local filename = paths.concat("../fig/patchII", i.."_"..idx.."_After"..".png")
          -- image.save(filename, c1)

          p[idx]:copy(c1:resize(numChannels*kSize*kSize))

        end
      end
    end
    idx = idx + 1
  end
  assert(idx == numSamples+1)
  return t
end

function whiten(d)
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
    return x
    -- return x, M, P
  end
  for i = 1, d:size()[1] do
    d[1] = zca_whiten(d[1])
  end
  return d
end