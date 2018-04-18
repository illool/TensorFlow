#coding=utf-8
"""
      主要思想及算法流程来自李航的《统计学习方法》
      《理解SVM的三重境界》
      yi={1,-1}
      难点：
      KKT条件
      SMO算法
"""
import time
import random
import numpy as np
import math
import copy
a=np.matrix([[1.2,3.1,3.1]])
#print a.astype(int)
#print a.A

class SVM:
      def __init__(self,data,kernel,maxIter,C,epsilon):
            self.trainData=data
            self.C=C  #惩罚因子
            self.kernel=kernel
            self.maxIter=maxIter
            self.epsilon=epsilon
            self.a=[0 for i in range(len(self.trainData))]
            self.w=[0 for i in range(len(self.trainData[0][0]))]
            self.eCache=[[0,0] for i in range(len(self.trainData))]
            self.b=0
            self.xL=[self.trainData[i][0] for i in range(len(self.trainData))]
            self.yL=[self.trainData[i][1] for i in range(len(self.trainData))]

      def train(self):
            #support_Vector=self.__SMO()
            self.__SMO()
            self.__update()

      def __kernel(self,A,B):
            #核函数 是对输入的向量进行变形 从低维映射到高维度
            res=0
            if self.kernel=='Line':
                  res=self.__Tdot(A,B)
            elif self.kernel[0]=='Gauss':
                  K=0
                  for m in range(len(A)):
                       K+=(A[m]-B[m])**2 
                  res=math.exp(-0.5*K/(self.kernel[1]**2))
            return res


      def __Tdot(self,A,B):
            res=0
            for k in range(len(A)):
                  res+=A[k]*B[k]
            return res


      def __SMO(self):
            #SMO是基于 KKT 条件的迭代求解最优化问题算法
            #SMO是SVM的核心算法
            support_Vector=[]
            self.a=[0 for i in range(len(self.trainData))]
            pre_a=copy.deepcopy(self.a)
            for it in range(self.maxIter):
                  flag=1
                  for i in range(len(self.xL)):
                        #print self.a
                        #更新 self.a  使用 机器学习实战的求解思路
                        #计算 j更新
                        diff=0
                        self.__update()
                        #选择有最大误差的j 丹麦理工大学的算法是 对j在数据集上循环, 随机选取i 显然效率不是很高
                        #机器学习实战 硬币书表述正常 代码混乱且有错误 启发式搜索
                        Ei=self.__calE(self.xL[i],self.yL[i])
                        j,Ej=self.__chooseJ(i,Ei)
                        #计算 L H
                        (L,H)=self.__calLH(pre_a,j,i)
                        #思路是先表示为self.a[j] 的唯一变量的函数 再进行求导（一阶导数=0 更新）
                        kij=self.__kernel(self.xL[i],self.xL[i])+self.__kernel(self.xL[j],self.xL[j])-2*self.__kernel(self.xL[i],self.xL[j])
                        #print kij,"aa"
                        if(kij==0):
                              continue
                        self.a[j] = pre_a[j] + float(1.0*self.yL[j]*(Ei-Ej))/kij
                        #下届是L 也就是截距,小于0时为0
                        #上届是H 也就是最大值,大于H时为H
                        self.a[j] = min(self.a[j], H)
                        self.a[j] = max(self.a[j], L)
                        #self.a[j] = min(self.a[j], H)
                        #print L,H
                        self.eCache[j]=[1,self.__calE(self.xL[j],self.yL[j])]
                        self.a[i] = pre_a[i]+self.yL[i]*self.yL[j]*(pre_a[j]-self.a[j])
                        self.eCache[i]=[1,self.__calE(self.xL[i],self.yL[i])]
                        diff=sum([abs(pre_a[m]-self.a[m]) for m in range(len(self.a))])
                        #print diff,pre_a,self.a
                        if diff < self.epsilon:
                              flag=0
                        pre_a=copy.deepcopy(self.a)
                  if flag==0:
                        print it,"break"
                        break

            #return support_Vector

      def __chooseJ(self,i,Ei):
            self.eCache[i]=[1,Ei]
            chooseList=[]
            #print self.eCache
            #从误差缓存中得到备选的j的列表 chooseList  误差缓存的作用：解决初始选择问题
            for p in range(len(self.eCache)):
                  if self.eCache[p][0]!=0 and p!=i:
                        chooseList.append(p)
            if len(chooseList)>1:
                  delta_E=0
                  maxE=0
                  j=0
                  Ej=0
                  for k in chooseList:
                        Ek=self.__calE(self.xL[k],self.yL[k])
                        delta_E=abs(Ek-Ei)
                        if delta_E>maxE:
                              maxE=delta_E
                              j=k
                              Ej=Ek
                  return j,Ej
            else:
                  #最初始状态
                  j=self.__randJ(i)
                  Ej=self.__calE(self.xL[j],self.yL[j])
                  return j,Ej

      def __randJ(self,i):
            j=i
            while(j==i):
                  j=random.randint(0,len(self.xL)-1)
            return j

      def __calLH(self,pre_a,j,i):
            if(self.yL[j]!= self.yL[i]):
                  return (max(0,pre_a[j]-pre_a[i]),min(self.C,self.C-pre_a[i]+pre_a[j]))
            else:
                  return (max(0,-self.C+pre_a[i]+pre_a[j]),min(self.C,pre_a[i]+pre_a[j]))

      def __calE(self,x,y):
            #print x,y
            y_,q=self.predict(x)
            return y_-y

      def __calW(self):
            self.w=[0 for i in range(len(self.trainData[0][0]))]
            for i in range(len(self.trainData)):
                  for j in range(len(self.w)):
                        self.w[j]+=self.a[i]*self.yL[i]*self.xL[i][j]

      def __update(self):
            #更新 self.b 和 self.w
            self.__calW()
            #得到了self.w 下面求b
            #print self.a
            maxf1=-99999
            min1=99999
            for k in range(len(self.trainData)):
                  y_v=self.__Tdot(self.w,self.xL[k])
                  #print y_v
                  if self.yL[k]==-1:
                        if y_v>maxf1:
                              maxf1=y_v
                  else:
                        if y_v<min1:
                              min1=y_v
            self.b=-0.5*(maxf1+min1)

      def predict(self,testData):
            pre_value=0
            #从trainData 改成 suport_Vector
            for i in range(len(self.trainData)):
                  pre_value+=self.a[i]*self.yL[i]*self.__kernel(self.xL[i],testData)
            pre_value+=self.b
            #print pre_value,"pre_value"
            if pre_value<0:
                  y=-1
            else:
                  y=1
            return y,abs(pre_value-0)

      def save(self):
            pass




def LoadSVM():
      pass
