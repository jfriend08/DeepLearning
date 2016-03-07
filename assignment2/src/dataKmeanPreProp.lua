require 'nn'
require 'image'

local dataKmeanPreProp = torch.class 'dataKmeanPreProp'
dofile './patchRunII.lua'

function dataKmeanPreProp:__init(path)
  self.kmeanProvider = torch.load(path)
  self.kmeanProvider.patches.centroids = self.kmeanProvider.patches.centroids:float()
end


function dataKmeanPreProp:prePropHandler(d, numPatch, kSize, gap, runNorm, runWhiten)
  local numSamples = d:size()[1]
  local FIG_dim = {3,96,96}

  data_norm = d:float()
  if runNorm then
    data_norm = normalize(d)
  end
  data_norm = data_norm:float()

  patches = parsePatch(data_norm, numSamples, numPatch, FIG_dim[1], FIG_dim[2], FIG_dim[3], kSize, gap)
  patches = patches:float()
  print(patches:size())
  print(self.kmeanProvider.patches.centroids:size())

  if runWhiten then
    patches = whiten(patches)
    patches = patches:float()
    print(patches:size())
  end

  return getFeatureFromCentroids(patches, self.kmeanProvider.patches.centroids)

end

function getFeatureFromCentroids(p, centers)
  print 'start feature extraction'
  collectgarbage()
  local numSamples = p:size()[1]
  local numPatch = p:size()[2]
  local numK = centers:size()[1]
  local centerDim = centers:size()[2]
  local norm_res = torch.Tensor(numSamples, 4*numK)
  -- local norm_res = torch.Tensor()
  local mlp_l2 = nn.PairwiseDistance(2)
  for i = 1, numSamples do
    xlua.progress(i, numSamples)
    local fk = torch.Tensor(centerDim)
    local zk = torch.Tensor(numPatch,numK)
    --Calculate distance of each centroid to each patch
    for k = 1, numK do
      for pidx = 1, numPatch do
        zk[pidx][k] = mlp_l2:forward({p[i][pidx], centers[k]})
        local mean = zk[pidx]:mean()
        zk[pidx]:neg():add(mean)
        zk[pidx]:apply(stepFunction)
      end
    end
    r1 = zk[1]:add(zk[2]):add(zk[5]):add(zk[6])
    r2 = zk[2]:add(zk[3]):add(zk[7]):add(zk[8])
    r3 = zk[9]:add(zk[10]):add(zk[13]):add(zk[14])
    r4 = zk[11]:add(zk[12]):add(zk[15]):add(zk[16])
    norm_res[i] = torch.cat(r1,r2,1):cat(r3,1):cat(r4,1)
  end
  return norm_res
end

function stepFunction(x)
  if x > 0 then
    return x
  else
    return 0
  end
end

function normalize(d)
  local data = d
  print 'normalize data'
  collectgarbage()
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1, data:size()[1] do
    xlua.progress(i, data:size()[1])
    -- rgb -> yuv
    local rgb = data[i]
    local yuv = image.rgb2yuv(rgb)
    -- normalize y locally:
    yuv[1] = normalization(yuv[{{1}}])
    data[i] = yuv
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
    local p_idx = 1
    for row = 0, 3 do
      for col = 0, 3 do
        if row*(kSize+gap)+kSize < height and col*(kSize+gap)+kSize < width then
          c1 = image.crop(this_d, row*(kSize+gap),col*(kSize+gap), row*(kSize+gap)+kSize,col*(kSize+gap)+kSize)

          -- local filename = paths.concat("../fig/patchII", i.."_"..idx.."_After"..".png")
          -- image.save(filename, c1)

          p[p_idx]:copy(c1:resize(numChannels*kSize*kSize))
          p_idx = p_idx + 1
        end
      end
    end
    assert(p_idx == numPatch+1)
    t[idx]:copy(p)
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
    xlua.progress(i, d:size()[1])
    d[i] = zca_whiten(d[i])
  end
  return d
end