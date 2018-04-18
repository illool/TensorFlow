#coding=utf-8
from SVM import *

data=[
        [[1,1],1],
        [[2,1],1],
        [[1,0],1],
        [[3,7],-1],
        [[4,8],-1],
        [[4,10],-1],
      ]
#如果为gauss核的话  ['Gauss',标准差]
svm=SVM(data,'Line',1000,0.02,0.001)
svm.train()
print svm.predict([4,0])
print svm.a
print svm.w
print svm.b
