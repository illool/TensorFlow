#coding=utf-8
import random
from scipy.stats import norm
import matplotlib.pyplot as plt

def cauchy(theta):
    print("柯西分布的分布函数")
    y = 1.0 / (1.0 + theta ** 2)
    return y

print("柯西分布中采样数据")
T = 500000
sigma = 1
thetamin = -30
thetamax = 30
#theta = [0.0] * (T+1)
#theta[0] = random.uniform(thetamin, thetamax)
theta = [0 for i in range(T)]
t = 0
while t < T-1:
    t = t + 1
    theta_star = norm.rvs(loc=theta[t - 1], scale=sigma, size=1, random_state=None)
    #print theta_star
    alpha = min(1, (cauchy(theta_star[0]) / cauchy(theta[t - 1])))

    u = random.uniform(0, 1)
    if u <= alpha:
        theta[t] = theta_star[0]
    else:
        theta[t] = theta[t - 1]

ax1 = plt.subplot(211)
ax2 = plt.subplot(212) 
plt.sca(ax1)
plt.ylim(thetamin, thetamax)
plt.plot(range(T), theta, 'g-')
plt.sca(ax2)
num_bins = 50
plt.scatter(theta, list(map(cauchy,theta)))
plt.hist(theta, num_bins, normed=1, facecolor='red', alpha=0.5)
plt.show()
