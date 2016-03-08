require 'nn'
require 'image'
require 'xlua'
gfx = require 'gfx.js'
m = require ' '

function parseDataLabel(d, numSamples, numChannels, height, width, oneDim)
  local t = torch.ByteTensor()
  if oneDim then
    t = torch.ByteTensor(numSamples, height, width)
  else
    t = torch.ByteTensor(numSamples, numChannels, height, width)
  end
  local l = torch.ByteTensor(numSamples)
  local idx = 1
  for i = 1, #d do
    local this_d = d[i]
    for j = 1, #this_d do
      if oneDim then
        t[idx]:copy(this_d[j][1])
      else
        t[idx]:copy(this_d[j])
      end
      l[idx] = i
      idx = idx + 1
    end
  end
  assert(idx == numSamples+1)
  return t, l
end


local valsize = 1000  -- Use the validation here as the valing set
local channel = 3
local height = 96
local width = 96

raw_val = torch.load('./val.t7b')
valData = {
   data = torch.Tensor(),
   labels = torch.Tensor(),
   size = valsize
}
valData.data, valData.labels = parseDataLabel(raw_val.data,
                                               valsize, channel, height, width, true)
mydata, mylabels = parseDataLabel(raw_val.data,
                                               valsize, channel, height, width, false)
print(valData.data:size())
print(mydata:size())

x = torch.DoubleTensor(valData.data:size()):copy(valData.data)
x:resize(x:size(1), x:size(2) * x:size(3))
labels = valData.label
x:size()

opts = {ndims = 2, perplexity = 30, pca = 50, use_bh = true, theta=0.5}
mapped_x1 = d(x, opts)

mapped_x1:size()
im_size = 4096
-- map_im = m.draw_image_map(mapped_x1, x:resize(x:size(1), 1, 96, 96), im_size, 0, true)
map_im = m.draw_image_map(mapped_x1, mydata, im_size, 0, true)
mapped_x1:size()
gfx.image(map_im)
