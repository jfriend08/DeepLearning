dofile './dataKmeanPreProp.lua'

local raw_train = torch.load('stl-10/extra.t7b')
dataKmeanPreProp = dataKmeanPreProp('./patchProvider_22_10000_2.t7')


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


local FIG_dim = {3, 96, 96}
local trsize = 10
local data = parseData(raw_train.data[1], trsize, FIG_dim[1], FIG_dim[2], FIG_dim[3])
dataKmeanPreProp:prePropHandler(data, 16, 22, 2)








