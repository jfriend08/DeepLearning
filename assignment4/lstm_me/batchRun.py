import subprocess
import numpy as np


job = 's48'
time = "24:00:00"
ram = "8GB"
l_cmd = "walltime=%s,mem=%s"%(time, ram)
for i in xrange(3):
  for layer in [1,2]:
    for isSoft in ['true', 'false']:
      for maxNorm in xrange(5):
        for rnn in xrange(200, 300, 25):
          drop = 0.4 + i * 0.05
          jobName = map(str,[drop,layer,isSoft,maxNorm,rnn])
          jobName = '_'.join(jobName)
          subprocess.call('qsub run.pbs -N {0} -q {1} -l {2} -v drop={3},name={4},layer={5},isSoft={6},maxNorm={7},rnn={8}'.format(
                  jobName, job, l_cmd, drop, jobName+"_", layer, isSoft, maxNorm, rnn), shell=True)


  # subprocess.call('qsub run.pbs -N {0} -q {1} -l {2} -v n={3},k={4},b={5},C={6}'.format(
  #         jobName, job, l_cmd, d, jobName, batch / 10.0, C), shell=True)