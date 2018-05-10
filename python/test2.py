# coding: utf-8
import matplotlib.pyplot as plt 
import numpy as np 

def loadDataSet():
    dataMat10 = []; dataMat11 = [];dataMat12 = [];dataMat13 = [];dataMat20 = [];dataMat21 = [];dataMat22 = [];dataMat23 = [];labelMat1 = [];labelMat2 = [] 
    dict1 = {};dict2 = {}
    fr = open('F:\\PythonPro\\LoadData\\src\\test2.txt')
    for line in fr.readlines():
        lineArr = line.strip("\n").split("\t")
        if '2016' in lineArr[0]:
            #dataMat1.append(float(lineArr[1]))
            #labelMat1.append(int(lineArr[0][4:8]))
            dict1[int(lineArr[0][4:8])] = [float(lineArr[1]),float(lineArr[2]),float(lineArr[3]),float(lineArr[4])]
            #print(dict1)
        else:
            #dataMat2.append(float(lineArr[1]))
            #labelMat2.append(int(lineArr[0][4:8]))
            dict2[int(lineArr[0][4:8])] = [float(lineArr[1]),float(lineArr[2]),float(lineArr[3]),float(lineArr[4])]
            #print(dict2)
    key1 = sorted(dict1.keys())
    key2 = sorted(dict2.keys())
    keys = sorted(list(set(key2+key1)))
    #print(len(keys))
    
    for key in keys:
        #print(key)
        value1 = dict1.get(key, [0,0,0,0])
        value2 = dict2.get(key, [0,0,0,0])
        dataMat10.append(value1[0])
        dataMat11.append(value1[1])
        dataMat12.append(value1[2])
        dataMat13.append(value1[3])
        dataMat20.append(value2[0])
        dataMat21.append(value2[1])
        dataMat22.append(value2[2])
        dataMat23.append(value2[3])
        #print(value2,value1)
    print(len(dataMat10),len(dataMat20))
    
    return dataMat10, dataMat11,dataMat12,dataMat13,dataMat20,dataMat21,dataMat22,dataMat23,keys
 
def call_back(event):
    info = 'name:{}\n button:{}\n x,y:{},{}\n xdata,ydata:{}{}'.format(event.name, event.button,event.x, event.y,event.xdata, event.ydata)
    
    #info = 'name:{}'.format()
    #text = ax.text(event.xdata, event.ydata, 'event', ha='center', va='center', fontdict={'size': 10})
    text.set_text(info)
    fig.canvas.draw_idle() 

def onpick3(event):
        ind = event.ind
        print('onpick3 scatter:', ind, np.take(key, ind), np.take(dataMat, ind))
        info = 'time,sale:{}{}'.format(np.take(key, ind), np.take(dataMat, ind))
        text.set_text(info)
        fig.canvas.draw_idle()

if __name__ == "__main__": 
    dataMat10, dataMat11,dataMat12,dataMat13,dataMat20,dataMat21,dataMat22,dataMat23,keys = loadDataSet()
    x=np.arange(0,len(dataMat10),1)
    y=np.arange(0,len(dataMat20),1)
    
    
    fig, ax = plt.subplots()
    text = ax.text(0, 30000, 'event', ha='center', va='center', fontdict={'size': 10})
    
    #fig.canvas.mpl_connect('motion_notify_event', call_back)
    #plt.gcf().canvas.mpl_connect('motion_notify_event', call_back)
    ax.plot(x,dataMat10,'ko--')
    #ax.plot(x,dataMat11,'ko--')
    #ax.plot(x,dataMat12,'ko--')
    #ax.plot(x,dataMat13,'ko--')
    
    ax.plot(y,dataMat20,'go--')
    #ax.plot(y,dataMat21,'g*--')
    #ax.plot(y,dataMat22,'g*--')
    #ax.plot(y,dataMat23,'g*--')
    key = keys+keys
    xy=np.append(x,y)
    print(len(xy))
    dataMat = dataMat10+dataMat20
    print(len(dataMat))
    width = 0.15
    plt.xticks(np.arange(len(keys)) + 1.0*width, keys)
    axis = plt.gca().xaxis
    for label in axis.get_ticklabels():
        label.set_color("k")
        label.set_rotation(45)
        label.set_fontsize(10)
    col1 = ax.scatter(xy, dataMat,picker=True)    
    fig.canvas.mpl_connect('pick_event', onpick3)
    
    fig1, ax1 = plt.subplots()
    text1 = ax1.text(0, 30000, 'event', ha='center', va='center', fontdict={'size': 10})
    
    #fig.canvas.mpl_connect('motion_notify_event', call_back)
    #plt.gcf().canvas.mpl_connect('motion_notify_event', call_back)
    ax1.plot(x,dataMat12,'ko--')
    #ax.plot(x,dataMat11,'ko--')
    #ax.plot(x,dataMat12,'ko--')
    #ax.plot(x,dataMat13,'ko--')
    
    ax1.plot(y,dataMat22,'go--')
    #ax.plot(y,dataMat21,'g*--')
    #ax.plot(y,dataMat22,'g*--')
    #ax.plot(y,dataMat23,'g*--')
    key = keys+keys
    xy=np.append(x,y)
    print(len(xy))
    dataMat = dataMat11+dataMat21
    print(len(dataMat))
    width = 0.15
    plt.xticks(np.arange(len(keys)) + 1.0*width, keys)
    axis = plt.gca().xaxis
    for label in axis.get_ticklabels():
        label.set_color("k")
        label.set_rotation(45)
        label.set_fontsize(10)
    col1 = ax1.scatter(xy, dataMat,picker=True)    
    fig1.canvas.mpl_connect('pick_event', onpick3)
    #index = 0
    #for item in keys:
    #    plt.text(item, y[index]+0.05, '%.0f' % y[index], ha='center', va= 'bottom',fontsize=7)
    #    print(item)
    #    index = index+1
    #x1 = dataMat1
    #y1 = keys
    #index = 0
    #for item in y1:
    #    plt.plot([dataMat1[index], dataMat1[index],], [0, item,], 'k--', linewidth=2.5)
    #    plt.scatter([dataMat1[index], ], [item, ], s=50, color='b')
    #    #plt.annotate(r'$2x+1=%s$' % item, xy=(dataMat1[index], item), xycoords='data', xytext=(+30, -30),textcoords='offset points', fontsize=16,arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
    #    index = index+1
    plt.show()
