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
parser.add_argument("prefix", help="output filename prefix")
# parser.add_argument("selected_nSplit", help="select n_split so can generate 3dim figure")

args = parser.parse_args()
mypath = args.inputDir
prefix = args.prefix
# selected_nSplit = args.selected_nSplit

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.find(".o")!= -1 ]
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

leng = (len(allTable) if len(allTable)%2==0 else len(allTable)+1)
print leng, leng/2
plt.figure(figsize=(20, 30))
for idx, k in enumerate(sorted(allTable.keys())):
  plt.subplot(((leng)/2)-1, 3, idx+1)
  arr = map(lambda x: 1/float(x), allTable[k]["train"])
  print arr, range(len(arr))
  plt.plot(range(len(arr)), arr)
  ylim((0,0.025))
  plt.title(k)
plt.savefig('./'+prefix+'_train.png')

plt.figure(figsize=(20, 30))
for idx, k in enumerate(sorted(allTable.keys())):
  plt.subplot(((leng)/2)-1, 3, idx+1)
  arr = map(lambda x: 1/float(x), allTable[k]["val"])
  plt.plot(range(len(arr)), arr)
  ylim((0,0.01))
  plt.title(k)
plt.savefig('./'+prefix+'_val.png')