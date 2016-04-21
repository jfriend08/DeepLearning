require('base')
require 'io'
require('nngraph')

ptb = require('data')
stringx = require('pl.stringx')

local params = {
                batch_size=20, -- minibatch
                seq_length=20, -- unroll length
                layers=2,
                decay=2,
                rnn_size=200, -- hidden unit size
                dropout=0, 
                init_weight=0.1, -- random weight initialization limits
                lr=1, --learning rate
                vocab_size=10000, -- limit on the vocabulary size
                max_epoch=4,  -- when to start decaying learning rate
                max_max_epoch=13, -- final epoch
                max_grad_norm=5 -- clip when gradients exceed this norm value
               }

function transfer_data(x)
    if gpu then
        return x:cuda()
    else
        return x
    end
end

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if tonumber(line[1]) == nil then error({code="init"}) end
  -- for i = 2,#line do
  --   if line[i] ~= 'foo' then error({code="vocab", word = line[i]}) end
  -- end
  return line
end

function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2 * params.layers do
            model.start_s[d]:zero()
        end
    end
end

function runPredict(line)
    reset_state(state_test)
    g_disable_dropout(model.rnns)

    local len = #line-1
    local perp = 0
    local pre_word

    print('Input length is: ' .. len)

    g_replace_table(model.s[0], model.start_s)
    for i = 2, #line-1 do
      local x = torch.Tensor{ptb.vocab_map[line[i]]}
      local y = torch.Tensor{ptb.vocab_map[line[i+1]]}
      x = x:expand(params.batch_size)
      y = y:expand(params.batch_size)

      perp_tmp, model.s[1], pred = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
      y, i = torch.max(pred[1], 1)
      pre_word = ptb.vocabRev_map[i[1]]

      perp = perp + perp_tmp[1]
      g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    print('Prediction: ' .. pre_word)

    g_enable_dropout(model.rnns)
end

-- get data in batches
state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}

model = torch.load('./model/model_34845.net')

while true do
  print("Query: len word1 word2 etc")
  local ok, line = pcall(readline)
  if not ok then
    if line.code == "EOF" then
      break -- end loop
    -- elseif line.code == "vocab" then
    --   print("Word not in vocabulary, only 'foo' is in vocabulary: ", line.word)
    elseif line.code == "init" then
      print("Start with a number")
    else
      print(line)
      print("Failed, try again")
    end
  else
    runPredict(line)
    -- local prev_s = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    -- local prev_word
    -- for i = 2, #line-1 do
    --   print(line[i])
    --   perp_tmp, model.s[1], pred = unpack(model.rnns[i]:forward({ptb.vocab_map[line[i]], ptb.vocab_map[line[i+1]], model.s[0]}))

    --   -- local embedding = protos.embed:forward(prev_char)
    --   -- local next_c, next_h = unpack(protos.lstm:forward{embedding, prev_c, prev_h})

    --   -- prev_c:copy(next_c) -- TODO: this shouldn't be needed... check if we can just use an assignment?
    --   -- prev_h:copy(next_h)
    -- end
    -- io.write('\n')
  end
end
