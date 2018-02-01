import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models, similarities

dictionary = corpora.Dictionary.load('deerwester.dict')
corpus = corpora.MmCorpus('deerwester.mm')
print(corpus)
print(list(corpus))

# 文档从一种向量表示方式转换到另一种
# 将语料库中隐藏的结构发掘出来，发现词语之间的关系，并且利用这些结构、关系使用一种新的、更有语义价值的（这是我们最希望的）方式描述其中的文档。
# 使得表示方式更加简洁。这样不仅能提高效率（新的表示方法一般消耗较少的资源）还能提高效果（忽略了边际数据趋势、降低了噪音）。

# 我们使用了前一个教程中用过的语料库来初始化（训练）这个转换模型。不同的转换可能需要不同的初始化参数；在Tfidf案例中，“训练”仅仅是遍历提供的语料库然后计算所有属性的文档频率（译者注：在多少文档中过）。训练其他模型，例如潜在语义分析或隐含狄利克雷分配，更加复杂，因此耗时也多
# 第一步 -- 初始化一个模型
tfidf = models.TfidfModel(corpus)

doc_bow = [(0, 1), (1, 2)]
print(tfidf[doc_bow])# 第二步 -- 使用模型转换向量

corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)
    
    
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # 初始化一个LSI转换
corpus_lsi = lsi[corpus_tfidf] # 在原始语料库上加上双重包装: bow->tfidf->fold-in-lsi   
lsi.print_topics(2) 

for doc in corpus_lsi:
    print(doc)
    
    
lsi.save('model.lsi') # same for tfidf, lda, ...
lsi = models.LsiModel.load('model.lsi')

#Gensim实现了几种常见的向量空间模型算法：

#词频-逆文档频（Term Frequency * Inverse Document Frequency， Tf-Idf）
#model = tfidfmodel.TfidfModel(bow_corpus, normalize=True)

#潜在语义索引（Latent Semantic Indexing，LSI，or sometimes LSA）
#model = lsimodel.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=300)

#随机映射（Random Projections，RP）
#model = rpmodel.RpModel(tfidf_corpus, num_topics=500)

#隐含狄利克雷分配（Latent Dirichlet Allocation, LDA）
#model = ldamodel.LdaModel(bow_corpus, id2word=dictionary, num_topics=100)

#分层狄利克雷过程（Hierarchical Dirichlet Process，HDP）
#model = hdpmodel.HdpModel(bow_corpus, id2word=dictionary)
