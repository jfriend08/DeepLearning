require 'nn'
require 'image'
require 'xlua'

torch.setdefaulttensortype('torch.FloatTensor')

-- create a new class called Surrogate
local Surrogate = torch.class 'Surrogate'

torch.manualSeed(123)

function Surrogate:__init(full)
  local trsize = 4000
  local valsize = 1000  -- Use the validation here as the valing set
  local channel = 3
  local height = 96
  local width = 96

  local raw_train = torch.load('stl-10/train.t7b')
  local raw_val = torch.load('stl-10/val.t7b')

  -- load and parse dataset
  self.trainData = {
     data = torch.Tensor(),
     labels = torch.Tensor(),
     size = function() return trsize end
  }
  self.trainData.data, self.trainData.labels = parseDataLabel(raw_train.data, trsize, channel, height, width)
  local trainData = self.trainData

  self.valData = {
     data = torch.Tensor(),
     labels = torch.Tensor(),
     size = function() return valsize end
  }
  self.valData.data, self.valData.labels = parseDataLabel(raw_val.data, valsize, channel, height, width)
  local valData = self.valData

  -- convert from ByteTensor to Float
  self.trainData.data = self.trainData.data:float()
  self.trainData.labels = self.trainData.labels:float()
  self.valData.data = self.valData.data:float()
  self.valData.labels = self.valData.labels:float()
  collectgarbage()
end

function Surrogate:getSurrogate(numFig)
  local channel = 3
  local height = 32
  local width = 32

  local numN = 100

  -- start to create surrogateData
  self.surrogateData = {
     data = torch.Tensor(),
     labels = torch.Tensor(),
     size = function() return numFig*numN end
  }
  self.surrogateData.data, self.surrogateData.labels = getSurrogate(self.trainData.data, numFig*numN, channel, height, width, numFig, numN)
  local surrogateData = self.surrogateData

  -- convert from ByteTensor to Float
  self.surrogateData.data = self.surrogateData.data:float()
  self.surrogateData.labels = self.surrogateData.labels:float()
  collectgarbage()

end




-- parse STL-10 data from table into Tensor
function parseDataLabel(d, numSamples, numChannels, height, width)
   local t = torch.ByteTensor(numSamples, numChannels, height, width)
   local l = torch.ByteTensor(numSamples)
   local idx = 1
   for i = 1, #d do
      local this_d = d[i]
      for j = 1, #this_d do
        t[idx]:copy(this_d[j])
        l[idx] = i
        idx = idx + 1
      end
   end
   assert(idx == numSamples+1)
   return t, l
end

-- create Surrogate set
function getSurrogate(d, numSamples, numChannels, height, width, numFig, numN)
   local t = torch.ByteTensor(numSamples, numChannels, height, width)
   local l = torch.ByteTensor(numSamples)
   local idx = 1

   for i = 1, numFig do
      local this_d = d[i]
      for n = 1, numN do
        degree = torch.random(-20,20)
        t1 = torch.uniform(0,0.1)
        t2 = torch.uniform(0,0.1)
        scale = torch.uniform(0.7,1.4)
        r = image.rotate(this_d, degree)
        r = image.translate(r, t1, t2)
        r = image.scale(r, scale*96, scale*96)

        r = image.crop(r, 32,32 , 64,64)
        t[idx]:copy(r)
        l[idx] = idx

        -- local filename = paths.concat("./img/", i.."_"..n..".png")
        -- image.save(filename, r)
      end
      idx = idx + 1
   end
   return t, l
end

function Surrogate:saveFigure(path)
  for i = 1, 50 do
    local filename = paths.concat("./", path, i..".png")
    image.save(filename, self.trainData.data[i])
  end
end

function Surrogate:normalize()
  local trainData = self.trainData
  local valData = self.valData

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

    -- preprocess valSet
  for i = 1,valData:size() do
    xlua.progress(i, valData:size())
     -- rgb -> yuv
     local rgb = valData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     valData.data[i] = yuv
  end
  -- normalize u globally:
  valData.data:select(2,2):add(-mean_u)
  valData.data:select(2,2):div(std_u)
  -- normalize v globally:
  valData.data:select(2,3):add(-mean_v)
  valData.data:select(2,3):div(std_v)

  print (self.trainData.data:size())
  print (self.valData.data:size())
end



