import sys, re
from os import listdir
import numpy as np
from os.path import isfile, join
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("inputDir", help="input results directory")
parser.add_argument("namePrefix", help="file name prefix")
# parser.add_argument("selected_nSplit", help="select n_split so can generate 3dim figure")

args = parser.parse_args()
mypath = args.inputDir
namePrefix = args.namePrefix
# selected_nSplit = args.selected_nSplit

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and re.match(r'(.*)\.o(.*)', f, re.M) ]
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.find(".o")!= -1 ]
print onlyfiles

allTable = {}
for eachfile in onlyfiles:
  nameSplit = re.split('\.o', eachfile)
  allTable[nameSplit[0]] = {'train':[], 'val':[]}
  fo = open(mypath+"/"+eachfile, "r+")
  for line in fo:
    line = line.rstrip()
    if line.find("epoch") != -1 and line.find("norm()") != -1:
      elms = re.split(',|\s+|=', line)
      allTable[nameSplit[0]]['train'] += [elms[9]]

    elif line.find("Validation") != -1:
      line = re.sub('\x1b[^m]*m', '', line)
      val = re.split('\s+:\s+', line)[1]
      allTable[nameSplit[0]]['val'] += [val]

style = ['-.','--', '+', '.', ':']
plt.figure(figsize=(12, 12))
legend = []
for idx, k in enumerate(sorted(allTable.keys())):
  arr = map(lambda x:float(x), allTable[k]["train"])
  print arr, range(len(arr))
  plt.plot(range(len(arr)), arr, style[idx%len(style)])
  ylim((0,1000))
  legend += [k]
plt.legend(legend, loc=1, fontsize=10)
plt.title('Perplexity per Steps -- Training')
plt.xlabel("Steps")
plt.ylabel("Perplexity")
plt.savefig('./'+namePrefix+'_train'+'.png')

plt.figure(figsize=(12,12))
legend = []
for idx, k in enumerate(sorted(allTable.keys())):
  arr = map(lambda x: float(x), allTable[k]["val"])
  plt.plot(range(len(arr)), arr, style[idx%len(style)])
  ylim((0,1000))
  legend += [k]
plt.legend(legend, loc=1, fontsize=10)
plt.title('Perplexity per Steps -- Validation')
plt.xlabel("Steps")
plt.ylabel("Perplexity")
plt.savefig('./'+namePrefix+'_val'+'.png')