dofile './patchRunII.lua'

patchRun = patchRun(100)
patchRun:normalize()
patchRun:getPatch(22, 2, 16) -- kSize, gap, nPatch
patchRun:whiten()
patchRun:runKmean(1600, 50) --ncentroids, niter
torch.save('patchProvider_22_100_50.t7', patchRun)

