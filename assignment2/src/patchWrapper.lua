dofile './patchRunII.lua'

patchRun = patchRun(40000)
patchRun:normalize()1
patchRun:getPatch(3, 2, 10, true) -- kSize, gap, nPatch, randomPatch
patchRun:whiten()
patchRun:runKmean(64, 1000) --ncentroids, niter
torch.save('patchProvider_3_40000_1000_64c.t7', patchRun)

