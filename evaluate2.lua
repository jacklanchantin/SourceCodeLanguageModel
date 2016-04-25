require 'torch'
require 'nn'
require 'optim'
include('auRoc.lua')


torch.setdefaulttensortype('torch.FloatTensor')

local utils = require 'util.utils'
local unpack = unpack or table.unpack

local cmd = torch.CmdLine()

-- cmd:option('-file', './global_results.t7')
--
--
-- local opt = cmd:parse(arg)


results = torch.load('./global_results.t7')
global_rnn_0 = torch.Tensor(results[0]['rnn'])
global_rnn_1 = torch.Tensor(results[1]['rnn'])


results = torch.load('./local_results.t7')
local_rnn_0 = torch.Tensor(results[0]['rnn'])
local_rnn_1 = torch.Tensor(results[1]['rnn'])



rnn_0 = (global_rnn_0 + local_rnn_0)/2
rnn_1 = (global_rnn_1 + local_rnn_1)/2

ngram_0 = torch.Tensor(results[0]['ngram'])
ngram_1 = torch.Tensor(results[1]['ngram'])


-- normalize
-- rnn_0 = (rnn_0-rnn_0:min())/rnn_0:max()
-- rnn_1 = (rnn_1-rnn_1:min())/rnn_1:max()
-- ngram_0 = (ngram_0-ngram_0:min())/ngram_0:max()
-- ngram_1 = (ngram_1-ngram_1:min())/ngram_1:max()



rnn_0_avg = rnn_0:sum()/rnn_0:size(1)
rnn_1_avg = rnn_1:sum()/rnn_1:size(1)


ngram_0_avg = ngram_0:sum()/ngram_0:size(1)
ngram_1_avg = ngram_1:sum()/ngram_1:size(1)


print('Non-buggy rnn: '..tostring(rnn_0_avg))
print('Buggy rnn: '..tostring(rnn_1_avg))
print('Difference: '..tostring(rnn_1_avg-rnn_0_avg))
print('')

print('Non-buggy ngram: '..tostring(ngram_0_avg))
print('Buggy ngram: '..tostring(ngram_1_avg))
print('Difference: '..tostring(ngram_1_avg-ngram_0_avg))
print('')




AUC = auRoc:new()




-- file = io.open("rnn_nonbuggy_entropy.txt", "w")
-- for i = 1,rnn_0:size(1) do
--   file:write(tostring(rnn_0[i])..'\n')
-- end
-- file:close()
-- file = io.open("rnn_buggy_entropy.txt", "w")
-- for i = 1,rnn_1:size(1) do
--   file:write(tostring(rnn_1[i])..'\n')
-- end
-- file:close()
-- os.exit()


for i = 1,rnn_0:size(1) do
  AUC:add(rnn_0[i], -1)
end
for i = 1,rnn_1:size(1) do
  AUC:add(rnn_1[i], 1)
end



AUROC = AUC:calculateAuc()
print('RNN AUROC: '..AUROC)
AUC:zero()



for i = 1,ngram_0:size(1) do
  AUC:add(ngram_0[i], -1)
end
for i = 1,ngram_1:size(1) do
  AUC:add(ngram_1[i], 1)
end
AUROC = AUC:calculateAuc()
print('ngram AUROC: '..AUROC)
AUC:zero()
