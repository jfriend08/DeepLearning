dofile './patchRunII.lua'

patchRun = patchRun()
patchRun:normalize()
patchRun:getPatch()
patchRun:whiten()
patchRun:runKmean(1600, 300) --ncentroids, niter
torch.save('patchProvider.t7',patchRun)

