#coding=utf-8
import random
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
#matplotlib inline

print("from scipy.stats import norm就是个高斯函数的计算包")
def norm_dist_prob(theta):
    #输入x，返回高斯 概率密度函数pdf(x, loc=0, scale=1)
    #一个均值3，标准差2的正态分布
    #在这里是我们的目标函数的概率，根据采样点来得到目标函数的概率值
    y = norm.pdf(theta, loc=3, scale=2)
    return y
#设定状态转移次数阈值
T = 5000
pi = [0 for i in range(T)]
sigma = 1
t = 0
while t < T-1:
    t = t + 1
    #从条件概率分布Q(x|xt)中采样得到样本x∗
    #Random variates.rvs(loc=0, scale=1, size=1, random_state=None)就是从这个分布中抽一些样本
    #size=100是样本大小，loc＝0是均值，scal=1是标准差
    #模拟转移矩阵
    #该样本是依赖上一次的抽样结果pi[t-1]的，用上一次的的样本作为均值生成样本，pi_star是依赖上一次的样本的状态，从而构成一个马尔可夫链
    pi_star = norm.rvs(loc=pi[t - 1], scale=sigma, size=1, random_state=None)
    #print(pi_star[0])
    #print("pi",pi[t - 1])
    #根据目标函数的概率值来计算aij
    alpha = min(1, (norm_dist_prob(pi_star[0]) / norm_dist_prob(pi[t - 1])))
    #根据aij和0，1均匀分布来计算拒绝和接受
    u = random.uniform(0, 1)
    if u < alpha:
        #接受，使用新生成的样本更新样本库
        pi[t] = pi_star[0]
        #print(norm_dist_prob(pi[t]))
    else:
        #拒绝，使用上一次的样本更新样本库，转台转移失败
        pi[t] = pi[t - 1]
        #print(norm_dist_prob(pi[t]))


print("在例子里，我们的目标平稳分布是一个均值3，标准差2的正态分布，\n而选择的马尔可夫链状态转移矩阵Q(i,j)的条件转移概率是以i为均值,方差1的正态分布在位置j的值。\n这个例子仅仅用来让大家加深对M-H采样过程的理解。毕竟一个普通的一维正态分布用不着去用M-H采样来获得样本。")
#正态分布N(3,2)概率密度函数某个pi对应的值
plt.scatter(pi, norm.pdf(pi, loc=3, scale=2))
#plt.scatter(pi, norm.pdf(pi, loc=3, scale=1))
num_bins = 50
plt.hist(pi, num_bins, normed=1, facecolor='red', alpha=0.7)
plt.show()
