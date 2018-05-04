#coding=utf-8
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
#####################
#二项分布
#####################
def test_binom_pmf():
    '''
    为离散分布
    二项分布的例子：抛掷10次硬币，恰好两次正面朝上的概率是多少？
    '''
    n = 10#独立实验次数
    p = 0.5#每次正面朝上概率
    k = np.arange(0,11)#0-10次正面朝上概率
    binomial = stats.binom.pmf(k,n,p)
    print(binomial)#概率和为1
    print(sum(binomial))
    print(binomial[2])

    plt.plot(k, binomial,'o-')
    plt.title('Binomial: n=%i , p=%.2f' % (n,p),fontsize=15)
    plt.xlabel('Number of successes')
    plt.ylabel('Probability of success',fontsize=15)
    plt.show()

def test_binom_rvs():
    '''
    为离散分布
    使用.rvs函数模拟一个二项随机变量，其中参数size指定你要进行模拟的次数。我让Python返回10000个参数为n和p的二项式随机变量
    进行10000次实验，每次抛10次硬币，统计有几次正面朝上，最后统计每次实验正面朝上的次数
    '''
    binom_sim = data = stats.binom.rvs(n=10,p=0.3,size=10000)
    print(len(binom_sim))
    print("mean: %g" % np.mean(binom_sim))
    print("SD: %g" % np.std(binom_sim,ddof=1))

    plt.hist(binom_sim,bins=10,normed=True)
    plt.xlabel('x')
    plt.ylabel('density')
    plt.show()
#####################
#泊松分布
#####################
def test_poisson_pmf():
    '''
    泊松分布的例子：已知某路口发生事故的比率是每天2次，那么在此处一天内发生4次事故的概率是多少？
    泊松分布的输出是一个数列，包含了发生0次、1次、2次，直到10次事故的概率。
    '''
    rate = 2
    n = np.arange(0,10)
    y = stats.poisson.pmf(n,rate)
    print(y)
    plt.plot(n, y, 'o-')
    plt.title('Poisson: rate=%i' % (rate), fontsize=15)
    plt.xlabel('Number of accidents')
    plt.ylabel('Probability of number accidents', fontsize=15)
    plt.show()

def test_poisson_rvs():
    '''
    模拟1000个服从泊松分布的随机变量
    '''
    data = stats.poisson.rvs(mu=2, loc=0, size=1000)
    print("mean: %g" % np.mean(data))
    print("SD: %g" % np.std(data, ddof=1))

    rate = 2
    n = np.arange(0,10)
    y = stats.poisson.rvs(n,rate)
    print(y)
    plt.plot(n, y, 'o-')
    plt.title('Poisson: rate=%i' % (rate), fontsize=15)
    plt.xlabel('Number of accidents')
    plt.ylabel('Probability of number accidents', fontsize=15)
    plt.show()
#####################
#正态分布
#####################
def test_norm_pmf():
    '''
    正态分布是一种连续分布，其函数可以在实线上的任何地方取值。
    正态分布由两个参数描述：分布的平均值μ和方差σ2 。
    '''
    mu = 0#mean
    sigma = 1#standard deviation
    x = np.arange(-5,5,0.1)
    y = stats.norm.pdf(x,0,1)
    print(y)
    plt.plot(x, y)
    plt.title('Normal: $\mu$=%.1f, $\sigma^2$=%.1f' % (mu,sigma))
    plt.xlabel('x')
    plt.ylabel('Probability density', fontsize=15)
    plt.show()

#####################
#beta分布
#####################
def test_beta_pmf():
    '''
    β分布是一个取值在 [0, 1] 之间的连续分布，它由两个形态参数α和β的取值所刻画。
    β分布的形状取决于α和β的值。贝叶斯分析中大量使用了β分布。
    '''
    a = 2.0#
    b = 5.0
    x = np.arange(0.01,1,0.01)
    y = stats.beta.pdf(x,a,b)
    print(y)
    plt.plot(x, y)
    plt.title('Beta: a=%.1f, b=%.1f' % (a,b))
    plt.xlabel('x')
    plt.ylabel('Probability density', fontsize=15)
    plt.show()

#####################
#指数分布（Exponential Distribution）
#####################
def test_exp():
    '''
    指数分布是一种连续概率分布，用于表示独立随机事件发生的时间间隔。
    比如旅客进入机场的时间间隔、打进客服中心电话的时间间隔、中文维基百科新条目出现的时间间隔等等。
    '''
    lambd = 0.5#
    x = np.arange(0,15,0.1)
    y =lambd * np.exp(-lambd *x)
    print(y)
    plt.plot(x, y)
    plt.title('Exponential: $\lambda$=%.2f' % (lambd))
    plt.xlabel('x')
    plt.ylabel('Probability density', fontsize=15)
    plt.show()

def test_expon_rvs():
    '''
    指数分布下模拟1000个随机变量。scale参数表示λ的倒数。函数np.std中，参数ddof等于标准偏差除以 $n-1$ 的值。
    '''
    data = stats.expon.rvs(scale=2, size=1000)
    print("mean: %g" % np.mean(data))
    print("SD: %g" % np.std(data, ddof=1))

    plt.hist(data, bins=20, normed=True)
    plt.xlim(0,15)
    plt.title('Simulating Exponential Random Variables')
    plt.show()

if __name__=='__main__':
    test_expon_rvs()
    test_exp()
    test_beta_pmf()
    test_norm_pmf()
    test_poisson_rvs()
    test_poisson_pmf()
    test_binom_rvs()
    test_binom_pmf()
