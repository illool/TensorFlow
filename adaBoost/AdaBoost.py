# coding:UTF-8
import numpy as np
from numpy import *
import matplotlib.pyplot as plt


def loadSimpData():
    datMat = matrix([[1.,  2.1],
                     [2.,  1.1],
                     [1.3,  1.],
                     [1.,  1.],
                     [2.,  1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    # get number of fields
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

# 特征：dimen，分类的阈值是 threshVal,分类对应的大小值是threshIneq


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # just classify the data
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

# 构建一个简单的单层决策树，作为弱分类器
# D作为每个样本的权重，作为最后计算error的时候多项式乘积的作用
# 三层循环
# 第一层循环，对特征中的每一个特征进行循环，选出单层决策树的划分特征
# 对步长进行循环，选出阈值
# 对大于，小于进行切换


def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))  # numSteps作为迭代这个单层决策树的步长
    minError = inf  # init error sum, to +infinity
    for i in range(n):  # loop over all dimensions
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()  # 第i个特征值的最大最小值
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):  # loop over all range in current dimension
            for inequal in ['lt', 'gt']:  # go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                # call stump classify with i, j, lessThan
                predictedVals = stumpClassify(
                    dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr  # calc total error multiplied by D
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the
                # weighted error is %.3f" % (i, threshVal, inequal,
                # weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

# 基于单层决策树的AdaBoost的训练过程
# numIt 循环次数，表示构造40个单层决策树


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # init D to all equal
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        # 根据当前额D分布算出最优的阀值
        bestStump, error, classEst = buildStump(
            dataArr, classLabels, D)  # build Stump
        # print "D:",D.T
        # calc alpha, throw in max(error,eps) to account for error=0
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # store Stump Params in Array
        # print "classEst: ",classEst.T
        # exponent for D calc, getting messy
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))  # Calc New D for next iteration
        D = D / D.sum()
        # calc training error of all classifiers, if this is 0 quit for loop
        # early (use break)
        aggClassEst += alpha * classEst
        # print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones(
            (m, 1)))  # 这里还用到一个sign函数，主要是将概率可以映射到-1,1的类型
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    # do stuff similar to last aggClassEst in adaBoostTrainDS
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])  # call stump classify
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return sign(aggClassEst)


def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)  # cursor
    ySum = 0.0  # variable to calculate AUC
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()  # get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        # draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", ySum * xStep)


'''
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    #基于单层决策树的AdaBoost训练过程    
    weakClfArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        # 每一次循环 只有样本的权值分布 D 发生变化
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print(" D: ", D.T)

        # 计算弱分类器的权重
        alpha = float(0.5 * np.log((1 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClfArr.append(bestStump)
        print("classEst: ", classEst.T)

        # 更新训练数据集的权值分布
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()

        # 记录对每个样本点的类别估计的累积值
        aggClassEst += alpha * classEst
        print("aggClassEst: ", aggClassEst.T)

        # 计算分类错误率
        aggErrors = np.multiply(np.sign(aggClassEst) !=
                                np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate, "\n")

        # 如果完全正确，终止迭代
        if errorRate == 0.0:
            break
    return weakClfArr
'''

if __name__ == '__main__':
    print(__doc__)
    datMat, classLabels = loadSimpData()
    #plt.scatter(datMat[:, 0], datMat[:, 1], c=classLabels,markers=classLabels, s=200, cmap="Paired")
    print(adaBoostTrainDS(datMat, classLabels, 9))
