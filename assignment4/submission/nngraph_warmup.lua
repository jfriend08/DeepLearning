nn = require 'nn'
require 'nngraph'

--First part for tanh
h1     = nn.Linear(4,2)()
h1_add = nn.Add(1)(h1)
hh1    = nn.Tanh()(h1_add)
hhh1   = nn.Square()(hh1)

--Second part for sigmold
h2     = nn.Linear(5,2)()
h2_add = nn.Add(1)(h2)
hh2    = nn.Sigmoid()(h2_add)
hhh2   = nn.Square()(hh2)

--Third part for adding input z
h3     = nn.Identity()()

--Combinding graph
mmul   = nn.CMulTable()({hhh1, hhh2})
madd   = nn.CAddTable()({mmul, h3})

model = nn.gModule({h1,h2,h3},{madd})

x = torch.rand(4)
y = torch.rand(5)
z = torch.rand(2)
gradient = torch.ones(2)

-- Forward propagate, output in size 2x2
print (model:forward({x,y,z}))

-- Backward propagate
res = model:backward({x,y,z},gradient)
print(res)
print(res[1])
print(res[2])
print(res[3])

