import subprocess
import numpy as np


job = 's48'
time = "20:00:00"
ram = "8GB"
l_cmd = "walltime=%s,mem=%s"%(time, ram)
for idx in xrange(2,5):
  for layer in [2,3,4]:
    for isSoft in ['true', 'false']:
      for maxNorm in xrange(5,21,5):
        d = idx*0.05
  # layer = 2*i
        jobName = map(str,[d,layer,isSoft,maxNorm])
        jobName = '_'.join(jobName)
        subprocess.call('qsub run.pbs -N {0} -q {1} -l {2} -v drop={3},name={4},layer={5},isSoft={6},maxNorm={7}'.format(
                jobName, job, l_cmd, d, jobName+"_", layer, isSoft, maxNorm), shell=True)
  # subprocess.call('qsub run.pbs -N {0} -q {1} -l {2} -v n={3},k={4},b={5},C={6}'.format(
  #         jobName, job, l_cmd, d, jobName, batch / 10.0, C), shell=True)