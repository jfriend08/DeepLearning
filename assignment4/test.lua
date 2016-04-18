nn = require 'nn'
require 'nngraph'

-- l1 = nn.Linear(4,2)()
-- a = nn.Tanh()(l1)
-- -- b = nn.Identity()()
-- l2 = nn.Linear(4,2)()
-- b = nn.Tanh()(l2)

-- x = nn.CAddTable()({a,b})
-- m = nn.gModule({a,b},{x})


-- t1 = torch.Tensor{1,1,1,1}
-- t2 = torch.Tensor{2,2,2,2}

-- print (m:forward({t1,t2}))


-- h1 = nn.Linear(20, 10)()
-- h2 = nn.Linear(10, 1)(  nn.Tanh()(  nn.Linear(10, 10)( nn.Tanh()(h1) )  )   )
-- mlp = nn.gModule({h1}, {h2})

-- x = torch.rand(20)
-- -- print(x)
-- print (mlp:forward(x))


h1 = nn.Linear(20, 20)()
h2 = nn.Linear(20, 1)(h1)
model = nn.gModule({h1},{h2})
x = torch.rand(20)
print (model:forward(x))



h1 = nn.Linear(4,2)()
hh1 = nn.Tanh()(h1)

h2 = nn.Linear(5,2)()
hh2 = nn.Sigmoid()(h2)

mmul = nn.CMulTable()({hh1, hh2})

model = nn.gModule({h1,h2},{mmul})

x = torch.rand(4)
y = torch.rand(5)
print (model:forward({x,y}))