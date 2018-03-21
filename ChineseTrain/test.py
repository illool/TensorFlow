import tensorflow as tf    
import numpy as np    
import os    
from collections import Counter    
import librosa
from train import get_wav_label
from train import get_wav_files
from train import get_wav_length
from train import speech_to_text
    
from joblib import Parallel, delayed  

if __name__ == "__main__":
    
    wav_files = get_wav_files()  
    wav_files, labels = get_wav_label()  
    print(u"样本数 ：", len(wav_files))  
  
    all_words = []  
    for label in labels:  
        # 字符分解  
        all_words += [word for word in label]  
  
    counter = Counter(all_words)  
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  
  
    words, _ = zip(*count_pairs)  
    words_size = len(words)  
    print(u"词汇表大小：", words_size)  
  
    word_num_map = dict(zip(words, range(len(words))))  
  
    # 当字符不在已经收集的words中时，赋予其应当的num，这是一个动态的结果  
    to_num = lambda word: word_num_map.get(word, len(words))  
  
    # 将单个file的标签映射为num 返回对应list,最终all file组成嵌套list  
    labels_vector = [list(map(to_num, label)) for label in labels]  
  
    label_max_len = np.max([len(label) for label in labels_vector])  
    print(u"最长句子的字数:" + str(label_max_len))  
  
    # 下面仅仅计算了语音特征相应的最长的长度。  
    # 如果仅仅是计算长度是否需要施加变换后计算长度？  
    parallel_read = False  
    if parallel_read:  
        wav_max_len = np.max(Parallel(n_jobs=7)(delayed(get_wav_length)(wav) for wav in wav_files))  
    else:  
        wav_max_len = 673  
    print("最长的语音", wav_max_len)  
  
    batch_size = 1 
    n_batch = len(wav_files) // batch_size
  
    X = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, 20])
  
    # 实际mfcc中的元素并非同号，不严格的情况下如此得到序列长度也是可行的  
    sequence_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(X, reduction_indices=2), 0.), tf.int32), reduction_indices=1)  
  
    Y = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
  
    #train_speech_to_text_network(wav_max_len)  #训练
    
    wav_file = "./D4_750.wav"
    print (wav_file)
    speech_to_text(wav_file)
