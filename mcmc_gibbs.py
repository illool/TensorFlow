#coding=utf-8
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
#多元正态分布，方差和协方差
#cov:
#σ21   ρσ1σ2 
#ρσ1σ2 σ22
#多元正态分布
samplesource = multivariate_normal(mean=[5,-1], cov=[[1,0.5],[0.5,2]])

def p_ygivenx(x, m1, m2, s1, s2):
    #得到正态分布的随机数,均值，方差
    #每一变量的条件概率为
    #P(x2|x1)=Norm(μ2+ρσ2/σ1(x1−μ1),(1−ρ2)σ22)
    return (random.normalvariate(m2 + rho * s2 / s1 * (x - m1), math.sqrt(1 - rho ** 2) * s2))

def p_xgiveny(y, m1, m2, s1, s2):
    #得到正态分布的随机数,均值，方差
    #每一变量的条件概率为，这个可以通过公式推导出来
    #P(x1|x2)=Norm(μ1+ρσ1/σ2(x2−μ2),(1−ρ2)σ21)
    return (random.normalvariate(m1 + rho * s1 / s2 * (y - m2), math.sqrt(1 - rho ** 2) * s1))

N = 5000
K = 20
x_res = []
y_res = []
z_res = []
m1 = 5#第一个高斯分布的均值u1，对应上面的mean=[5,-1]
m2 = -1#第二个高斯分布的均值u2，对应上面的mean=[5,-1]
s1 = 1#第一个高斯分布的协方差σ1，对应上面的cov=[[1,0.5],[0.5,2]
s2 = 2#第二个高斯分布的协方差σ2，对应上面的cov=[[1,0.5],[0.5,2]

rho = 0.5#个高斯分布的协方差系数ρ，对应上面的cov=[[1,0.5],[0.5,2]
y = m2

for i in range(N):
    for j in range(K):
        #得到正态分布的随机数
        x = p_xgiveny(y, m1, m2, s1, s2)#先固定第二个高斯分布的参数，从条件分布中采样第一个高斯的样本
        y = p_ygivenx(x, m1, m2, s1, s2)#固定第一个高斯分布的参数，从条件分布中采样第二个高斯的样本
        #得到多元正态分布的概率值
        z = samplesource.pdf([x,y])
        
        x_res.append(x)
        y_res.append(y)
        z_res.append(z)

num_bins = 50
plt.hist(x_res, num_bins, normed=1, facecolor='green', alpha=0.5)
plt.hist(y_res, num_bins, normed=1, facecolor='red', alpha=0.5)
plt.scatter(x_res, z_res)
plt.scatter(y_res, z_res)
plt.title('Histogram')
plt.show()

fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0,1.5,1.5], elev=30, azim=20)
ax.scatter(x_res, y_res, z_res,marker='o')
plt.show()
