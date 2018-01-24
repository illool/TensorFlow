# encoding=utf-8

import numpy as np
'''
EM算法的入门简单例子：

已知有三枚硬币A，B，C，假设抛掷A，B，C出现正面的概率分别为π，p，q。
单次实验的过程是:

    首先抛掷硬币A，如果A出现正面选择硬币B，否则，选择硬币C。
    抛掷所选择的硬币，正面输出1，反面输出0。

重复上述单词实验n次，需要估计抛掷硬币A，B，C出现正面的概率π，p，q。其中每次实验步骤1的抛掷结果不可见，可见的是所挑选硬币的抛掷结果。
http://blog.csdn.net/sajiahan/article/details/53106642
'''

def generate_observe_sequence(n):
    return (np.random.random(size=n)> 0.35).astype(np.int)
#观测值，[π, p, q]
def Estep(observe_list, theta):
#计算在参数π(i), p(i), q(i)下观测数据y来自投掷硬币B的概率
    def sample_mu(y):
        #P(y|θ)=πp^y(1−p)^(1−y)+(1−π)q^y(1−q)^(1−y)
        up_1 = theta[0] * np.power(theta[1], y) * np.power((1-theta[1]),(1-y))
        up_2 = (1-theta[0]) * np.power(theta[2], y) * np.power((1-theta[2]),(1-y))
        return up_1/(up_1 + up_2)

    return [sample_mu(y) for y in observe_list]
#观测值，[自投掷硬币B的概率s]
def MStep(observe_list, mus):
    p = [0.0, 0.0, 0.0]
    #π
    p[0] = sum(mus)/len(mus)
    #p
    p[1] = sum([mus[i] * observe_list[i] for i in range(len(observe_list))])/sum(mus)
    #q
    p[2] = sum([(1-mus[i]) * observe_list[i] for i in range(len(observe_list))])/sum([1-mu for mu in mus])
    return p[:]

if __name__ == "__main__":
    records = []
    #π,p,q
    theta = [0.3, 0.6, 0.7]
    #结束条件设置
    m = 1e-7
    #加入到list
    records.append(theta)
    #观测值
    observe_list = [1,1,0,1,0,0,1,0,1,1]
    #observe_list = generate_observe_sequence(5)
    #打印出初始的π p q
    #print(theta)
    #开始迭代
    while True:
        mus = Estep(observe_list, theta)
        print(mus)
        new_theta = MStep(observe_list, mus)
        print(new_theta)
        records.append(new_theta)
        err = 0
        for old, new in zip(theta, new_theta):
            err += np.abs(old-new)
        print(err)
        #结束条件
        if err < m:
            break
        theta = new_theta[:]
    print("###########################")
    for record in records:
        print(record)
        #pass
