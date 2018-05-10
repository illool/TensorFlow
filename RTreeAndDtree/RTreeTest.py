# coding: utf-8
import numpy as np
from numpy import *
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
import matplotlib.pyplot as plt


def plotfigure(X, X_test, y, yp):
    plt.figure()
    plt.scatter(X, y, c="k", label="data")
    plt.plot(X_test, yp, c="r", label="max_depth=5", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend(loc='upper right')
    plt.show()
    #plt.savefig('./res.png', format='png')


x = np.linspace(-5, 5, 200)
siny = np.sin(x)
X = x.reshape(-1, 1)
#X = mat(x).T
y = siny + np.random.rand(1, len(siny)) * 0.5
y = np.array(y.tolist()[0])
clf = DecisionTreeRegressor(max_depth=7)
rf = ensemble.RandomForestRegressor(n_estimators=2000)  # 这里使用20个决策树
clf.fit(X, y)
rf.fit(X, y)

X_test = np.arange(-5.0, 5.0, 0.05)[:, np.newaxis]
#yp = clf.predict(X_test)
yp = rf.predict(X_test)

plotfigure(x, X_test, y, yp)
