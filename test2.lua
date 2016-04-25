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

opt.input_h5 = 'data/global_lm.h5'
opt.input_json = 'data/global_lm.json'

global_checkpoint = torch.load('./cv/global/checkpoint_251000.t7')
buggy_checkpoint = torch.load('./cv/buggy/checkpoint_320000.t7')

global_model = global_checkpoint.model
buggy_model = buggy_checkpoint.model

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


global_model:evaluate()
global_model:resetStates()

buggy_model:evaluate()
buggy_model:resetStates()

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
--         if #line <= 1 then
--           count = count + 1
--         end
--       end
--       line_idx = line_idx + 1
-- end
-- print(count)
-- os.exit()


lambda = 0.5

line_idx = 0
for line in io.lines("data/testdata.txt") do
  if (line_idx % 100) <= 2 then
      if (line_idx % 3) == 0 then
        if #line > 0 then
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
          local global_scores = global_model:forward(xv):view(N * seq_len, -1)
          local buggy_scores = buggy_model:forward(xv):view(N * seq_len, -1)
          H = 0
          for i = 1,global_scores:size(1) do
            global_probs = nn.SoftMax():forward(global_scores[i])
            buggy_probs = nn.SoftMax():forward(buggy_scores[i])
            t1 = global_probs*lambda
            t2 = buggy_probs*(1-lambda)
            for j = 1,probs:size(1) do
              if probs[j] > 0 then
                H = H + probs[j]*log2(probs[j])
              end
            end
          end
          H = (H*-1)/global_scores:size(1)
        else
          H = 0
        end
      elseif (line_idx % 3) == 1 then
          ngram_H = tonumber(line)
      elseif (line_idx % 3) == 2 then
        table.insert(results[tonumber(line)]['rnn'], H)
        table.insert(results[tonumber(line)]['ngram'], ngram_H)
      end
  end

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
