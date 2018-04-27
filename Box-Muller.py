#coding=utf-8
#随机采样系列3：Box-Muller方法产生正太分布
import math
import time
import numpy as np
import matplotlib.pylab as plt
import random
class Samples:
    def __init__(self):
        pass
    def rand(self, num, seed = 1):
        m = math.pow(2, 32)
        a = 214013
        c = 2531011
        i = 1
        x = np.zeros(num)
        x[0] = seed
        while(i < num):
            x[i] = (a * x[i-1] + c) % m
            i += 1
        return x
    def uniform(self, num, seed = 1):
        m = math.pow(2, 32)
        print("m:",m)
        x = self.rand(num, seed)
        return x / m
    def normal(self, num):
        t = int(time.time()) 
        print("t",t)
        u1 = self.uniform(num, t)
        t1 = int(time.time())+1
        u2 = self.uniform(num, t1)
        #u1,u2服从(0,1)分布，则z0，z1服从正太分布
        z0 = np.sqrt(-2 * np.log(u1)) * np.sin(2.0 * np.pi * u2)
        z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
        #plt.hist(z0)
        #plt.show()
        return z0
if __name__=='__main__':
    s = Samples()
    x00 = s.normal(1000)
    plt.hist(x00,100)
    plt.show()
