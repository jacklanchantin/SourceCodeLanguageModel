require 'torch'
require 'nn'
require 'optim'

require 'LanguageModel'
require 'util.DataLoader'

-- /usr/cs/bin/th train.lua -input_h5 data/catjava.h5 -input_json data/catjava.json -resume 169000

torch.setdefaulttensortype('torch.FloatTensor')

local utils = require 'util.utils'
local unpack = unpack or table.unpack

local cmd = torch.CmdLine()

cmd:option('-rnn_model', 'global')

-- Dataset options
-- cmd:option('-input_h5', 'data/global_lm.h5')
-- cmd:option('-input_json', 'data/global_lm.json')
cmd:option('-batch_size', 1)
cmd:option('-seq_length', 50)

-- Model options
cmd:option('-model_type', 'lstm')
cmd:option('-wordvec_size', 64)
cmd:option('-rnn_size', 128)
cmd:option('-num_layers', 2)
cmd:option('-dropout', 0)
cmd:option('-batchnorm', 0)


-- Output options
cmd:option('-checkpoint_name', 'cv/global/checkpoint')

-- Backend options
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')

local opt = cmd:parse(arg)

if opt.rnn_model == 'global' then
  checkpoint = torch.load('./cv/global/checkpoint_251000.t7')
  opt.input_h5 = 'data/global_lm.h5'
  opt.input_json = 'data/global_lm.json'
elseif opt.rnn_model == 'buggy' then
  checkpoint = torch.load('./cv/buggy/checkpoint_320000.t7')
  opt.input_h5 = 'data/buggy_lm.h5'
  opt.input_json = 'data/buggy_lm.json'
elseif opt.rnn_model == 'local' then
  checkpoint = torch.load('./cv/local/checkpoint_256000.t7')
  opt.input_h5 = 'data/locallm.h5'
  opt.input_json = 'data/locallm.json'
else
  print('invalid model type')
  os.exit()
end
model = checkpoint.model


-- Initialize the DataLoader and vocabulary
-- local loader = DataLoader(opt)
local vocab = utils.read_json(opt.input_json)
local idx_to_token = {}
local token_to_idx = {}
for k, v in pairs(vocab.idx_to_token) do
  idx_to_token[tonumber(k)] = v
  token_to_idx[v] = tonumber(k)
end

-- Set up GPU stuff
local dtype = 'torch.FloatTensor'
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  dtype = 'torch.CudaTensor'
  print(string.format('Running with CUDA on GPU %d', opt.gpu))
else
  -- Memory benchmarking is only supported in CUDA mode
  opt.memory_benchmark = 0
  print 'Running in CPU mode'
end


-- Initialize the model and criterion
-- local opt_clone = torch.deserialize(torch.serialize(opt))
-- opt_clone.idx_to_token = idx_to_token
-- local model = nn.LanguageModel(opt_clone):type(dtype)


-- Set up some variables we will use below
local N, T = opt.batch_size, opt.seq_length


model:evaluate()
model:resetStates()


function log2(log_in)
  return torch.log(log_in)/torch.log(2)
end


results = {}
results[1] = {}
results[1]['rnn'] = {}
results[1]['ngram'] = {}
results[0] = {}
results[0]['rnn'] = {}
results[0]['ngram'] = {}

-- line_idx = 0
-- count = 0
-- for line in io.lines("data/testdata.txt") do
--       if (line_idx % 3) == 0 then
--         flag = 0
--         if #line > 1 then
--           flag = 1
--         end
--       elseif (line_idx % 3) == 2 then
--           if flag == 1 and tonumber(line) == 0 then
--             count = count + 1
--           end
--       end
--       line_idx = line_idx + 1
-- end
-- print(count)
-- os.exit()



line_idx = 0
for line in io.lines("data/testdata.txt") do
  -- if (line_idx % 100) <= 2 then
      if (line_idx % 3) == 0 then
        flag = 0
        if #line > 1 then
          xv = torch.Tensor(#line)
          for i = 1,#line do
            char = string.sub(line,i,i)
            if token_to_idx[char] then
              xv[i] = token_to_idx[char]
            else
              xv[i] = 1
            end
          end
          seq_len = xv:size(1)
          xv = xv:reshape(1,seq_len)
          local scores = model:forward(xv):view(N * seq_len, -1)
          H = 0
          for i = 1,scores:size(1) do
            probs = nn.SoftMax():forward(scores[i])
            for j = 1,probs:size(1) do
              if probs[j] > 0 then
                H = H + probs[j]*log2(probs[j])
              end
            end
          end
          H = (H*-1)/scores:size(1)
        else
          flag = 1
        end
      elseif (line_idx % 3) == 1 then
        if flag == 0 then
          ngram_H = tonumber(line)
        end
      elseif (line_idx % 3) == 2 then
        if flag == 0 then
          table.insert(results[tonumber(line)]['rnn'], H)
          table.insert(results[tonumber(line)]['ngram'], ngram_H)
        end
      end
  -- end
  if (line_idx % 5000) == 0 then
    print(line_idx)
  end
  line_idx = line_idx + 1
end


if opt.rnn_model == 'global' then
  torch.save('./global_results.t7', results)
elseif opt.rnn_model == 'buggy' then
  torch.save('./buggy_results.t7', results)
elseif opt.rnn_model == 'local' then
  torch.save('./local_results.t7', results)
end
