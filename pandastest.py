# encoding=utf-8
import pandas as pd
data = [[1,2,3],[4,5,6]]
index = [0,1]
columns=['a','b','c']
df = pd.DataFrame(data=data, index=index, columns=columns)
print(df.loc[1])

data = [[1,2,3],[4,5,6]]  
index = ['d','e']  
columns=['a','b','c']  
df = pd.DataFrame(data=data, index=index, columns=columns)  
print(df.loc['d'])
'''
data = [[1,2,3],[4,5,6]]  
index = ['d','e']  
columns=['a','b','c']  
df = pd.DataFrame(data=data, index=index, columns=columns)  
print(df.loc['a'])
'''

data = [[1,2,3],[4,5,6]]  
index = ['d','e']  
columns=['a','b','c']  
df = pd.DataFrame(data=data, index=index, columns=columns)  
print(df.loc['d':])

data = [[1,2,3],[4,5,6]]  
index = ['d','e']  
columns=['a','b','c']  
df = pd.DataFrame(data=data, index=index, columns=columns)  
print(df.loc['d',['b','c']])

data = [[1,2,3],[4,5,6]]  
index = ['d','e']  
columns=['a','b','c']  
df = pd.DataFrame(data=data, index=index, columns=columns)  
print(df.loc[:,['c']])
'''
data = [[1,2,3],[4,5,6]]  
index = ['d','e']  
columns=['a','b','c']  
df = pd.DataFrame(data=data, index=index, columns=columns)  
print(df.loc[1])
'''
'''
data = [[1,2,3],[4,5,6]]  
index = ['d','e']  
columns=['a','b','c']  
df = pd.DataFrame(data=data, index=index, columns=columns)  
print(df.iloc['a']) 
'''
data = [[1,2,3],[4,5,6]]  
index = ['d','e']  
columns=['a','b','c']  
df = pd.DataFrame(data=data, index=index, columns=columns)  
print(df.iloc[0:])

data = [[1,2,3],[4,5,6]]  
index = ['d','e']  
columns=['a','b','c']  
df = pd.DataFrame(data=data, index=index, columns=columns)  
print(df.iloc[:,[1]])

data = [[1,2,3],[4,5,6]]  
index = ['d','e']  
columns=['a','b','c']  
df = pd.DataFrame(data=data, index=index, columns=columns)  
print(df.ix[1]) 

data = [[1,2,3],[4,5,6]]  
index = ['d','e']  
columns=['a','b','c']  
df = pd.DataFrame(data=data, index=index, columns=columns)  
print(df.ix['e'])
