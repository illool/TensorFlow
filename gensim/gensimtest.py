#coding=utf-8
from gensim import corpora, models, similarities
import logging
'''
gensim中的必须理解的概念有：
1 raw strings 原始字符串
2 corpora 语料库
3 sparse vectors 稀疏向量
4 vector space model 向量空间模型
5 transformation 转换，指由稀疏向量组成的稀疏矩阵生成某个向量空间模型。
6 index 索引 
'''

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
documents = ["Shipment of gold damaged in a fire",
             "Delivery of silver arrived in a silver truck",
             "Shipment of gold arrived in a truck"]
print(documents)

texts = [[word for word in document.lower().split()] for document in documents]
print(texts)
# corpora.Dictionary 对象,可以理解为python中的字典对象, 其Key是字典中的词，其Val是词对应的唯一数值型I
# 构造方法Dictionary(documents=None, prune_at=2000000)
dictionary = corpora.Dictionary(texts)#生成字典
print(dictionary)
print(dictionary.token2id)
print("########dictionary信息##########")
print(str(dictionary))
print("字典，{单词id，在多少文档中出现}")
print(dictionary.dfs) #字典，{单词id，在多少文档中出现}
print("文档数目")  
print(dictionary.num_docs) #文档数目 
print("dictionary.items()")
print(dict(dict(dictionary.items()))) #  
print("字典，{单词id，对应的词}")  
print(dict(dictionary.id2token)) #字典，{单词id，对应的词}  
print("字典，{词，对应的单词id}")  
print(dict(dictionary.token2id)) #字典，{词，对应的单词id}  
print("所有词的个数") 
print(dictionary.num_pos) #所有词的个数  
print("每个文件中不重复词个数的和")  
print(dictionary.num_nnz) #每个文件中不重复词个数的和  
print("########doc2bow##########")  
result, missing = dictionary.doc2bow(documents, allow_update=False, return_missing=True)  
print("词袋b，列表[(单词id，词频)]") 
print(result)
print("不在字典中的词及其词频，字典[(单词，词频)]")
print(dict(missing))
print("########bow信息##########")
for id, freq in result:  
    print(id, dictionary.id2token[id], freq)
print("########dictionary信息##########")  
#过滤文档频率大于no_below，小于no_above*num_docs的词  
dictionary.filter_extremes(no_below=1, no_above=0.5, keep_n=10)

corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus)

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)
