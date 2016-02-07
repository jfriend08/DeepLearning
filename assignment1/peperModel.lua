----------------------------------------------------------------------
-- Need comment
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('MNIST Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> define parameters'

-- 10-class problem
noutputs = 10

-- input dimensions
nfeats = 1
width = 32
height = 32
ninputs = nfeats*width*height

-- number of hidden units (for MLP only):
nhiddens = ninputs / 2

-- hidden units, filter sizes (for ConvNet only):
nstates = {64,64,128}
filtsize = 5
poolsize = 2
normkernel = image.gaussian1D(7)

----------------------------------------------------------------------
print '==> construct model'

if opt.model == 'linear' then

   -- Simple linear model
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,noutputs))

elseif opt.model == 'mlp' then

   -- Simple 2-layer neural network, with tanh hidden units
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,nhiddens))
   model:add(nn.Tanh())
   model:add(nn.Linear(nhiddens,noutputs))

elseif opt.model == 'convnet' then

   if opt.type == 'cuda' then
      -- a typical modern convolution network (conv+relu+pool)
      model = nn.Sequential()

      -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

      -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

      -- stage 3 : standard 2-layer neural network
      model:add(nn.View(nstates[2]*filtsize*filtsize))
      model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
      model:add(nn.Tanh())
      model:add(nn.Linear(nstates[3], noutputs))

   else
      -- regarding to fig3 in http://arxiv.org/abs/1204.3968
      -- construct stage1-->stage2 plus stage1 that fed to classifier together

      model1 = nn.Sequential() 
      model2 = nn.Sequential()
      c = nn.Parallel(1,2)
      model = nn.Sequential();

      -- model1 stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      model1:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      model1:add(nn.Tanh())
      model1:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
      model1:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

      -- model1 stage 2 : filter bank -> squashing -> L2 pooling -> normalization
      model1:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      model1:add(nn.Tanh())
      model1:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
      model1:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

      -- model2 stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      model2:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      model2:add(nn.Tanh())
      model2:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
      model2:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

      -- add construct model1 model2 to model
      c:add(model1)
      c:add(model2)
      model:add(c)

      -- stage 3 : standard 2-layer neural network
      model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))
      model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
      model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
      model:add(nn.Tanh())
      model:add(nn.Linear(nstates[3], noutputs))
   end
else

   error('unknown -model')

end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------
-- Visualization is quite easy, using itorch.image().

if opt.visualize then
   if opt.model == 'convnet' then
      if itorch then
   print '==> visualizing ConvNet filters'
   print('Layer 1 filters:')
   itorch.image(model:get(1).weight)
   print('Layer 2 filters:')
   itorch.image(model:get(5).weight)
      else
   print '==> To visualize filters, start the script in itorch notebook'
      end
   end
end
