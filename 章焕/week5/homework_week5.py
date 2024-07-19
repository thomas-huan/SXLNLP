#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
import torch
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict



#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters,n_init='auto')  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    centers = kmeans.cluster_centers_ # [42,100]
    labels = kmeans.labels_ # [42,100]

    sentence_label_dict = defaultdict(list)
    sentence_center_dict = defaultdict(list)
    for sentence, label, vector in zip(sentences, labels, vectors):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
        sentence_center_dict[label].append(vector)         #同标签所有向量放一起

    filter_dict = {}
    for label, vectors in sentence_center_dict.items():
        #取label对应的行
        vectorCenter = centers[label, :]
        distanceData = [] #欧氏距离的数组
        for v in vectors:
            #计算欧式距离
            distance = np.linalg.norm(torch.FloatTensor(vectorCenter) - torch.FloatTensor(v))
            distanceData.append(distance)
        average = np.mean(distanceData)#计算平均值
        filter_dict[label] = average

    filter_dict = dict(sorted(filter_dict.items(), key=lambda item: item[1]))
    filter_list = list(filter_dict.keys())[:10]

    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        if label in filter_list:
            print("欧式距离平均值：%s" % filter_dict[label])
            for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
                print(sentences[i].replace(" ", ""))
            print("---------")
        else:
            print("欧式距离平均值：%s, 已被过滤" %filter_dict[label])
            print("---------")

if __name__ == "__main__":
    main()

