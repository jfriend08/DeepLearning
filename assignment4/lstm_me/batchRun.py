import subprocess
import numpy as np


job = 's48'
time = "15:00:00"
ram = "8GB"
l_cmd = "walltime=%s,mem=%s"%(time, ram)
for i in xrange(1,6):
  d = 0
  layer = 2*i
  jobName = 'modelLayer' + str(layer)
  subprocess.call('qsub run.pbs -N {0} -q {1} -l {2} -v drop={3},name={4},layer={5}'.format(
          jobName, job, l_cmd, d, jobName+"_", layer), shell=True)
  # subprocess.call('qsub run.pbs -N {0} -q {1} -l {2} -v n={3},k={4},b={5},C={6}'.format(
  #         jobName, job, l_cmd, d, jobName, batch / 10.0, C), shell=True)