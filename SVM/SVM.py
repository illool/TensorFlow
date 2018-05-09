#coding=utf-8
"""
      主要思想及算法流程来自李航的《统计学习方法》
      《理解SVM的三重境界》
      yi={1,-1}
      难点：
      KKT条件
      SMO算法
      比较详细的算法流程
      https://blog.csdn.net/zhongshijunacm/article/details/52006552
"""
import time
import random
import numpy as np
import math
import copy
#a=np.matrix([[1.2,3.1,3.1]])
#print(a.astype(int))
#print(a.A)

class SVM:
      def __init__(self,data,kernel,maxIter,C,epsilon):
            self.trainData=data
            self.C=C  #惩罚因子
            self.kernel=kernel
            self.maxIter=maxIter #最大迭代次数 
            self.epsilon=epsilon #结果的精确性 
            #a中部等于零，对应到train_data就是支持向量
            self.a=[0 for i in range(len(self.trainData))]#用来存放a_i
            self.w=[0 for i in range(len(self.trainData[0][0]))]#参数的权重
            self.eCache=[[0,0] for i in range(len(self.trainData))]#放着Ei，表示预测值与真实值之差为
            self.b=0
            #样本数据xL
            self.xL=[self.trainData[i][0] for i in range(len(self.trainData))]
            #样本结果yL
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
                       #print(A[m],B[m],self.kernel[1])
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
            #初始化的时候已经设置过了可一干掉
            #self.a=[0 for i in range(len(self.trainData))]
            #将a_older拷贝出来，后面更新要用
            pre_a=copy.deepcopy(self.a)
            #迭代计算
            for it in range(self.maxIter):
                #flag用于判断时候达到计算精度的要求
                  flag=1
                  for i in range(len(self.xL)):#样本数据xL
                        #print self.a
                        #更新 self.a  使用 机器学习实战的求解思路
                        #计算 j更新
                        diff=0
                        self.__update()
                        #选择有最大误差的j 丹麦理工大学的算法是 对j在数据集上循环, 随机选取i 显然效率不是很高
                        #机器学习实战 硬币书表述正常 代码混乱且有错误 启发式搜索
                        Ei=self.__calE(self.xL[i],self.yL[i])
                        j,Ej=self.__chooseJ(i,Ei)
                        #计算 L H,aj的可行域
                        (L,H)=self.__calLH(pre_a,j,i)
                        #思路是先表示为self.a[j] 的唯一变量的函数 再进行求导（一阶导数=0 更新）
                        #kij = k(ii)+k(jj)-2*k(ij),其中k是核函数
                        #kij
                        kij=self.__kernel(self.xL[i],self.xL[i]) + self.__kernel(self.xL[j],self.xL[j]) - 2*self.__kernel(self.xL[i],self.xL[j])
                        #print kij,"aa"
                        if(kij==0): #分母不能为0，为0放弃计算
                              continue
                        #计算aj_new，pre_a[j]是a_old
                        #aj_new = a_old+y(j)*(Ei-Ej)/k(ii)+k(jj)-2*k(ij)
                        self.a[j] = pre_a[j] + float(1.0*self.yL[j]*(Ei-Ej))/kij
                        #下届是L 也就是截距,小于0时为0
                        #上届是H 也就是最大值,大于H时为H
                        #更新算出来的aj_new
                        self.a[j] = min(self.a[j], H)
                        self.a[j] = max(self.a[j], L)
                        #self.a[j] = min(self.a[j], H)
                        #print L,H
                        self.eCache[j]=[1,self.__calE(self.xL[j],self.yL[j])]
                        #计算ai_new = ai_older + yL[i]*yL[j]*((aj_older-aj_new)
                        self.a[i] = pre_a[i]+self.yL[i]*self.yL[j]*(pre_a[j]-self.a[j])
                        self.eCache[i]=[1,self.__calE(self.xL[i],self.yL[i])]
                        #计算误差
                        diff=sum([abs(pre_a[m]-self.a[m]) for m in range(len(self.a))])
                        #print diff,pre_a,self.a
                        if diff < self.epsilon:
                              flag=0
                        pre_a=copy.deepcopy(self.a)
                  if flag==0:
                        print(it,"break")
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
                  #最初始状态״̬
                  j=self.__randJ(i)
                  Ej=self.__calE(self.xL[j],self.yL[j])
                  return j,Ej

      def __randJ(self,i):#随机选择一个j出来,j的范围为[0,len(self.xL)-1]
            j=i
            while(j==i):
                  j=random.randint(0,len(self.xL)-1)
            return j
      #可行域选择
      def __calLH(self,pre_a,j,i):
            if(self.yL[j]!= self.yL[i]):
                #可行域为L = max[0,a2-a1];H = min[C,C+a2-a1]
                  return (max(0,pre_a[j]-pre_a[i]),min(self.C,self.C-pre_a[i]+pre_a[j]))
            else:
                #可行域为L = max[0,a2+a1-C];H = min[C,a2+a1]
                  return (max(0,-self.C+pre_a[i]+pre_a[j]),min(self.C,pre_a[i]+pre_a[j]))
              
      #Ei表示预测值与真实值之差为
      def __calE(self,x,y):
            #print x,y
            #真实值-预测值
            y_,q=self.predict(x)
            return y_-y

      def __calW(self):
            #print(len(self.trainData[0][0]))
            #给参数的权重赋值，在初始化的时候已经设置过了可以干掉
            #self.w=[0 for i in range(len(self.trainData[0][0]))]
            for i in range(len(self.trainData)):
                  for j in range(len(self.w)):
                        #w=∑i=1->m αi*yi*xi
                        print(self.w[j],self.a[i],self.yL[i],self.xL[i][j])
                        self.w[j]+=self.a[i]*self.yL[i]*self.xL[i][j]

      def __update(self):
            #更新 self.b 和 self.w
            self.__calW()#更新 self.w
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
                  #decision rule 
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
