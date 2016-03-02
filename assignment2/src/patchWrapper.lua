dofile './patchRunII.lua'

patchRun = patchRun(100)
patchRun:normalize()
patchRun:getPatch()
patchRun:whiten()
patchRun:runKmean(1600, 300) --ncentroids, niter
torch.save('patchProvider_10000.t7',patchRun)

