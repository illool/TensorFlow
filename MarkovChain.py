#coding=utf-8
import numpy as np
pi_0 = np.array([.21 ,.68, .11])
pi_1 = np.array([.75 ,.15, .1])
P = np.array([[.65, .28, .07],
              [.15, .67, .18],
              [.12, .36, .52]])
x = pi_0
pi_n = [x]
for i in range(10):
    print('{:.3f} {:.3f} {:.3f}'.format(x[0], x[1], x[2]))
    x = np.dot(x, P)
    pi_n.append(x)
print("验证：")
print(pi_n[-1])
print(np.dot(pi_n[-1], P))   

xx = pi_1
pi_n = [xx]
for i in range(100):
    print('{:.3f} {:.3f} {:.3f}'.format(xx[0], xx[1], xx[2]))
    xx = np.dot(xx, P)
    pi_n.append(xx) 
