#coding=utf-8
import numpy as np  
from sklearn.svm import SVR  
import matplotlib.pyplot as plt  

###############################################################################  
# Generate sample data  
X = np.sort(5 * np.random.rand(400, 1), axis=0)  #产生40组数据，每组一个数据，axis=0决定按列排列，=1表示行排列  
y = np.sin(X).ravel()   #np.sin()输出的是列，和X对应，ravel表示转换成行  

###############################################################################  
# Add noise to targets  
y[::5] += 3 * (0.5 - np.random.rand(80))  

###############################################################################  
# Fit regression model  
svr_rbf10 = SVR(kernel='rbf',C=100, gamma=10.0)  
svr_rbf1 = SVR(kernel='rbf', C=100, gamma=0.1)  
svr_rbf1 = SVR(kernel='rbf', C=100, gamma=0.1)  
#svr_lin = SVR(kernel='linear', C=1e3)  
#svr_poly = SVR(kernel='poly', C=1e3, degree=3)  
y_rbf10 = svr_rbf10.fit(X, y).predict(X)  
y_rbf1 = svr_rbf1.fit(X, y).predict(X) 
#y_lin = svr_lin.fit(X, y).predict(X)  
#y_poly = svr_poly.fit(X, y).predict(X)  

###############################################################################  
# look at the results  
lw = 2 #line width  
plt.scatter(X, y, color='darkorange', label='data')  
plt.hold('on')  
plt.plot(X, y_rbf10, color='navy', lw=lw, label='RBF gamma=10.0')  
plt.plot(X, y_rbf1, color='c', lw=lw, label='RBF gamma=1.0')  
#plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')  
#plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')  
plt.xlabel('data')  
plt.ylabel('target')  
plt.title('Support Vector Regression')  
plt.legend()  
plt.show()
