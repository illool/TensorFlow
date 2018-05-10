# coding: utf-8
import math
import operator
import plotID3Tree


def calcShannonEnt(dataset):
    # 计算信息熵
    # H(x)=−∑(i=1->n) p(xi)*log2p(xi)=∑(i=1->n)p(xi)*log2(1/p(xi))
    # 熵越高，表示混合的数据越多,信息量就越少
    numEntries = len(dataset)  # 样本总数
    labelCounts = {}  # 标签做key,
    for featVec in dataset:
        currentLabel = featVec[-1]  # 标签是featVec最后一个，详见CreateDataSet
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # 统计相应标签下的数目，就是累加

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 每个标签出现的概率
        shannonEnt -= prob * math.log(prob, 2)  # p(xi)*log2p(xi) 累减所以是"-="
    return shannonEnt


def CreateDataSet():
    # 不浮出水面是否可以生存(no surfacing)     是否有脚蹼(flippers)     是否属于鱼类(类标签)
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def splitDataSet(dataSet, axis, value):
    # 定义按照某个特征进行划分的函数splitDataSet
    # 输入三个变量(待划分的数据集，特征编号，分类值)
    # 第几个特征(axis)的取值(value)是什么，分为一组
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)

    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    # 定义按照最大信息增益划分数据的函数
    # 那个特征的信息熵增益越高就返回那个特征的index
    # 如果选择一个特征后，信息增益最大(信息不确定性减少的程度最大)，那么我们就选取这个特征。
    numberFeatures = len(dataSet[0]) - 1  # 有多少个特征
    baseEntropy = calcShannonEnt(dataSet)  # 香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numberFeatures):  # 遍历所有的特征
        featList = [example[i] for example in dataSet]  # 得到某个特征下所有值(某列)
        uniqueVals = set(featList)  # set无重复的属性特征值
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))  # 即p(t),概率
            # 对各子集香农熵求和，某个特征的熵*这个特征的出现概率
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # 计算信息增益
        if(infoGain > bestInfoGain):
            # 最大信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature  # 返回特征值


def majorityCnt(classList):
    # 投票表决代码
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),  # Python3.5中：iteritems变为items
                              key=operator.itemgetter(1), reverse=True)  # operator.itemgetter(1)，获取对象的第一个域的值
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    # 将所有的标签放在classList中去
    classList = [example[-1] for example in dataSet]
    # print(classList.count(classList[0]))
    # classList中所有的值都相等，则类别相同，停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # dataSet每条记录长度为1(只有类目标签没有特征)，返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 按照信息增益最高选取分类特征属性
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 返回分类的特征序号
    bestFeatLabel = labels[bestFeat]  # 该特征的label
    myTree = {bestFeatLabel: {}}  # 构建树的字典
    del(labels[bestFeat])  # 从labels的list中删除该label
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:  # bestFeat的取值情况有几种(uniqueVals)
        subLabels = labels[:]  # 子集合
        # 构建数据的子集合，并进行递归
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLables, testVec):
    # 输入三个变量(决策树，属性特征标签，测试的数据)
    firstStr = list(inputTree.keys())[0]  # 获取树的第一个特征属性
    secondDict = inputTree[firstStr]  # 树的分支，子集合Dict
    featIndex = featLables.index(firstStr)  # 获取决策树第一层在featLables中的位置
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLables, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    # 存储
    import pickle
    fw = open(filename, 'wb')  # pickle默认方式是二进制，需要制定'wb'
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    # 读取
    import pickle
    fr = open(filename, 'rb')  # 需要制定'rb'，以byte形式读取
    return pickle.load(fr)


if __name__ == '__main__':
    myDat, labels = CreateDataSet()
    print(myDat)
    print(labels)

    Shannon = calcShannonEnt(myDat)
    print(Shannon)

    print(myDat[0][1])
    retDataSet = splitDataSet(myDat, 0, myDat[0][0])
    print(retDataSet)
    bestDataSet = chooseBestFeatureToSplit(myDat)
    print(bestDataSet)
    myTree = createTree(myDat, labels)
    print(myTree)

    classList = [example[-1] for example in myDat]
    count = majorityCnt(classList)

    predit = classify(myTree, ['no surfacing', 'flippers'], [1, 0])
    print(predit)

    plotID3Tree.maincreatePlot(myTree)
