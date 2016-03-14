require 'cunn'
require 'nn'
require 'image'
require 'xlua'
m = require 'manifold';
-- gfx = require 'gfx.js'

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

raw_val = torch.load('./stl-10/val.t7b')
mydata, mylabels = parseDataLabel(raw_val.data,valsize, channel, height, width, false)
mydata_norm = torch.Tensor(mydata:size()):copy(mydata)
print(mydata_norm:size())
local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
for i = 1,mydata_norm:size(1) do
   xlua.progress(i,mydata_norm:size(1))
   -- rgb -> yuv
   local rgb = mydata_norm[i]
   local yuv = image.rgb2yuv(rgb)
   -- normalize y locally:
   yuv[1] = normalization(yuv[{{1}}])
   mydata_norm[i] = yuv
end
-- normalize u globally:
local mean_u = mydata_norm:select(2,2):mean()
local std_u = mydata_norm:select(2,2):std()
mydata_norm:select(2,2):add(-mean_u)
mydata_norm:select(2,2):div(std_u)
-- normalize v globally:
local mean_v = mydata_norm:select(2,3):mean()
local std_v = mydata_norm:select(2,3):std()
mydata_norm:select(2,3):add(-mean_v)
mydata_norm:select(2,3):div(std_v)


print('==> loading model')
model = torch.load('./vgg_kmeans_surro/model_758.net')
model:remove(54)

print('==> input to model')
x = torch.DoubleTensor(valsize,4608)
local bs = 25
local currIdx = 1
for i=1,mydata:size(1),bs do
  outputs = model:forward(mydata_norm:narrow(1,i,25):cuda())
  for k=1,bs do
    x[i+k-1]:copy(outputs[k])
  end
end
-- x = torch.DoubleTensor(valData.data:size()):copy(valData.data)
-- x:resize(x:size(1), x:size(2) * x:size(3))
-- labels = valData.label
-- x:size()
print(x:size())
print(mydata:size())
print('==> start mapping')
opts = {ndims = 2, perplexity = 30, pca = 50, use_bh = true, theta=0.5}
mapped_x1 = m.embedding.tsne(x, opts)

im_size = 2048
map_im = m.draw_image_map(mapped_x1, mydata, im_size, 0, true)
print(map_im:size())
image.save('tsneTest.jpeg', map_im)
torch.save('testobj', map_im)



