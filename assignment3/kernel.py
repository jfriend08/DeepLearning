

def doKernel(m, k):
  r, c = len(k), len(k[0])
  res = []
  for i in xrange(len(m)-r+1):
    each = []
    for j in xrange(len(m[0])-c+1):
      print (i,j)
      val = 0
      for di in xrange(r):
        for dj in xrange(c):
          val += m[i+di][j+dj]*k[di][dj]
      each += [val]
    res +=[each]
  return res






m = [[4,5,2,2,1],[3,3,2,2,4],[4,3,4,1,1],[5,1,4,1,2],[5,1,3,1,4]]
k = [[4,3,3],[5,5,5],[2,4,3]]
print doKernel(m,k)