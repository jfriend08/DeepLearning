dofile './patchRunII.lua'

patchRun = patchRun(20000)
patchRun:normalize()
patchRun:getPatch(22, 2, 16) -- kSize, gap, nPatch
patchRun:whiten()
patchRun:runKmean(1600, 400) --ncentroids, niter
torch.save('patchProvider_22_20000_400.t7', patchRun)

