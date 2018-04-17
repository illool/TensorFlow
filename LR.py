# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

def generate_data():
    X = np.linspace(-3, 3, 200).reshape(200, 1)
    y = 2 * X*X*X + 3 + (np.random.rand(200, 1)*50)
    return X, y

X, y = generate_data()

fig = plt.figure(figsize=(12, 8))

ax = fig.add_subplot(1,1,1)
ax.scatter(X, y, c= 'r')
ax.plot(X, 2 * X*X*X + 3, c='g', linewidth=2)
plt.show()

def learn_para(X, y, batch_size = 5, epoch_num  = 5):
    
    batch_num = int(X.shape[0] / batch_size)
    print(type(batch_num))
    X = X[:batch_size * batch_num]
    y = y[:batch_size * batch_num]
    
    cost = 0
    w = 0
    b = 0
    learn_rate = 0.05

    for i in range(epoch_num):
        
        X_y = np.concatenate((X, y), axis=1)
        np.random.shuffle(X_y)
        X, y = X_y[:, 0].reshape(200, 1), X_y[:, 1].reshape(200, 1)
        for index in range(0, len(X), batch_size):
            batch_X = X[index: index+batch_size].reshape(batch_size, 1)
            batch_y = y[index: index+batch_size].reshape(batch_size, 1)
            #yield batch_X, batch_y
            
            predict = w * batch_X*batch_X*batch_X + b
            cost = np.power((predict - batch_y), 2).sum()/(2*batch_size)
            
            w = w - learn_rate * (-(batch_y-predict)*batch_X).mean()
            b = b - learn_rate * (-(batch_y-predict)).mean()
    return w, b

w, b = learn_para(X, y)
predict = w * X*X*X + b

fig = plt.figure(figsize=(12, 8))

ax = fig.add_subplot(1,1,1)
ax.scatter(X, y, c= 'r')
ax.plot(X, 2 * X*X*X + 3, c='g', linewidth=2)
ax.plot(X, predict, c='y', linewidth=2)
plt.show()
