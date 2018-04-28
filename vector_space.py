# -*- encoding:UTF-8 -*-
import sys
from sklearn.datasets.base import Bunch
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


# 读取文件
def _readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content


# 读取bunch对象
def _readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch


# 写入bunch对象
def _writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)


# 这个函数用于创建TF-IDF词向量空间
def vector_space(stopword_path, bunch_path, space_path, train_tfidf_path=None):
    stpwrdlst = _readfile(stopword_path).splitlines()   # 读取停用词
    bunch = _readbunchobj(bunch_path)   # 导入分词后的词向量bunch对象
    for i in range(len(bunch.contents)):
        bunch.contents[i] = bunch.contents[i].decode('utf-8', 'ignore')
    # 构建tf-idf词向量空间对象
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                       vocabulary={})
    # 使用TfidfVectorizer初始化向量空间模型
    # 这里面有TF-IDF权重矩阵还有我们要得词向量空间坐标轴信息vocabulary

    if train_tfidf_path is not None:
        trainbunch = _readbunchobj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5,
                                     vocabulary=trainbunch.vocabulary)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
    else:
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)
        # 此时tdm里面存储的就是tf-idf权值矩阵
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfspace.vocabulary = vectorizer.vocabulary_
    _writebunchobj(space_path, tfidfspace)
    print("tf-idf词向量空间实例创建成功！！！")


if __name__ == '__main__':
    # print("占位！")
    stopword_path = "train_word_bag/hlt_stop_words.txt"  # 停用词表的路径
    bunch_path = "train_word_bag/train_set.dat"  # 导入训练集Bunch的路径
    space_path = "train_word_bag/tfidfspace.dat"  # 词向量空间保存路径
    vector_space(stopword_path, bunch_path, space_path)

    bunch_path = "test_word_bag/test_set.dat"  # 导入测试集Bunch的路径
    space_path = "test_word_bag/testspace.dat"  # 词向量空间保存路径
    train_tfidf_path = "train_word_bag/tfidfspace.dat"
    vector_space(stopword_path, bunch_path, space_path, train_tfidf_path)




