dofile './patchRunII.lua'

patchRun = patchRun(10000)
patchRun:normalize()
patchRun:getPatch(22, 2, 16) -- kSize, gap, nPatch
patchRun:whiten()
patchRun:runKmean(1600, 2) --ncentroids, niter
torch.save('patchProvider_22_10000_2.t7', patchRun)

