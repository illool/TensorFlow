#coding=utf-8
from gensim import corpora, models, similarities
import gensim

import logging  
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents = ["Human machine interface for lab abc computer applications",  
              "A survey of user opinion of computer system response time",  
              "The EPS user interface management system",  
              "System and human system engineering testing of EPS",  
              "Relation of user perceived response time to error measurement",  
              "The generation of random binary unordered trees",  
              "The intersection graph of paths in trees",  
              "Graph minors IV Widths of trees and well quasi ordering",  
              "Graph minors A survey"]  

# 去除停用词并分词  
# 译者注：这里只是例子，实际上还有其他停用词  
# 处理中文时，请借助 Py结巴分词 https://github.com/fxsjy/jieba  
stoplist = set('for a of the and to in'.split())  
texts = [[word for word in document.lower().split() if word not in stoplist]  for document in documents]  
  
# 去除仅出现一次的单词  
from collections import defaultdict  
frequency = defaultdict(int)
for text in texts:  
    for token in text:  
        frequency[token] += 1  
 
texts = [[token for token in text if frequency[token] > 1]  
         for text in texts]  
 
from pprint import pprint   # pretty-printer  
pprint(texts) 

dictionary = corpora.Dictionary(texts)
dictionary.save('deerwester.dict') # 把字典保存起来，方便以后使用
print(dictionary)
#此时又12个不同的单词
print(dictionary.token2id)

new_doc = "Human computer interaction"
#将new_doc分词后使用dictionary(字典)中的词来计数，输出的词为每个词对应dictionary(字典)中的id和在new_doc中出现的次数
new_vec = dictionary.doc2bow(new_doc.lower().split())
#computer”(id 0) 和“human”(id 1)各出现一次
print(new_vec) # "interaction"没有在dictionary中出现，因此忽略 

corpus = [dictionary.doc2bow(text) for text in texts] 
corpora.MmCorpus.serialize('deerwester.mm', corpus) # 存入硬盘，以备后需  
print(corpus)

#读取文本的方式生成语料库
class MyCorpus(object):
    def __iter__(self):
        for line in open('mycorpus.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())

corpus_memory_friendly = MyCorpus() # 没有将整个语料库载入内存
print(corpus_memory_friendly) 

for vector in corpus_memory_friendly:
    print(vector) 
    
# 收集所有符号的统计信息  
dictionary = corpora.Dictionary(line.lower().split() for line in open('mycorpus.txt'))
# 收集停用词和仅出现一次的词的id 
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist  
            if stopword in dictionary.token2id]  
#Python3.5中：iteritems变为items
## 得到仅出现一次的词
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]  
print("once_ids",once_ids)
dictionary.filter_tokens(stop_ids + once_ids) # 删除停用词和仅出现一次的词  
print(dictionary.token2id)  
dictionary.compactify() # 消除id序列在删除词后产生的不连续的缺口  
print(dictionary.token2id)  

from gensim import corpora  
# 创建一个玩具级的语料库  
corpus = [[(1, 0.5)], []]  # 让一个文档为空，作为它的heck
#将corpus写入到corpus.mm
corpora.MmCorpus.serialize('corpus.mm', corpus)
#保存为不同的格式
corpora.SvmLightCorpus.serialize('corpus.svmlight', corpus)  
corpora.BleiCorpus.serialize('corpus.lda-c', corpus)  
corpora.LowCorpus.serialize('corpus.low', corpus) 

#读取一个.mm文件
corpus = corpora.MmCorpus('corpus.mm')  
print(corpus)
# 将语料库全部导入内存的方法 
# 调用list()将会把所有的序列转换为普通Python List 
print(list(corpus))
# 另一种利用流接口，一次只打印一个文档  
for doc in corpus:  
    print(doc)  
    
 #想将这个 Matrix Market格式的语料库存为Blei’s LDA-C格式   
corpora.BleiCorpus.serialize('corpus.lda-c', corpus)  

#Gensim包含了许多高效的工具函数来帮你实现语料库与numpy矩阵之间互相转换
#corpus = gensim.matutils.Dense2Corpus(numpy_matrix)
#numpy_matrix = gensim.matutils.corpus2dense(corpus, num_terms=number_of_corpus_features) 
#以及语料库与scipy稀疏矩阵之间的转换：
#corpus = gensim.matutils.Sparse2Corpus(scipy_sparse_matrix)
scipy_csc_matrix = gensim.matutils.corpus2csc(corpus) 

