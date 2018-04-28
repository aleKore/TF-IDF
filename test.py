# -*- encoding:UTF-8 -*-
import sys
import os
import pickle
from sklearn.datasets.base import Bunch
import jieba
import codecs
# from pandas.io.common import file_path_to_url
# from blaze.server.client import content
sys.path.append("../")


# 使用jieba分词
def cuttest(test_sent):
    result = jieba.cut(test_sent)
    print(" / ".join(result))
# 以上为test.py自带代码，以下为自己添加代码


# 用Bunch数据结构来表示数据
def corpus2Bunch(wordbag_path, seg_path):
    catelist = os.listdir(seg_path)
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(catelist)
    for mydir in catelist:
        class_path = seg_path + mydir + "/"
        file_list = os.listdir(class_path)
        for file_path in file_list:
            fullname = class_path + file_path
            bunch.label.append(mydir)
            bunch.filenames.append(fullname)
            bunch.contents.append(ReadFile(fullname))
    # for k in range(len(bunch.contents)):
    #     bunch.contents[k] = bunch.contents[k].encode('utf-8', 'ignore')
    with open(wordbag_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)
    print("构建文本对象结束！！！")


# 保存文件
def SaveFile(savepath, content):
    content = content.encode("utf-8")
    with open(savepath, "wb") as fp:
        fp.write(content)
        fp.close()


# 读取文件
def ReadFile(path):
    with open(path, "rb") as fp:
        content = fp.read()
        fp.close()
    return content


# 读取bunch对象
def _readbunchobj(path):
    # with codecs.open(path, 'rb', 'utf-8') as file_obj:
    #     bunch = pickle.load(file_obj)
    # return bunch
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch


# 复制corpuscle_path类别目录到test_corpus_path
def Dir_Copy(corpus_path, test_corpus_path):
    catelist = os.listdir(corpus_path)
    for mydir in catelist:
        class_path = test_corpus_path + mydir + "/"
        if not os.path.exists(class_path):
            os.makedirs(class_path)


# 删除seg_path中与test_seg_path同名的文件
def File_Delete(seg_path, test_seg_path):
    catelist = os.listdir(test_seg_path)
    for mydir in catelist:
        class_path = test_seg_path + mydir + "/"
        file_list = os.listdir(class_path)
        for file_path in file_list:
            fullname = seg_path + mydir + "/" + file_path
            if os.path.isfile(fullname):
                os.remove(fullname)


# 用jieba进行中文语料分词
def Corpus_Segment(corpus_path, seg_path, stop_word_path):
    '''
    corpus_path是未分词语料库路径
    seg_path是分词后语料库存储路径
    '''

    # 把停用词做成字典
    stopwords = {}
    fstop = open(stop_word_path, 'r')
    for eachWord in fstop:
        stopwords[eachWord.strip()] = eachWord.strip()
    fstop.close()

    catelist = os.listdir(corpus_path)  # 获取corpuscle_path下的所有子目录
    '''
    其中子目录的名字就是类别名，例如：
    train_corpus/art/21.txt中，'train_corpus/'是corpus_path,'art'是catelist
    '''
    # 获取每个目录（类别）下所有的文件
    for mydir in catelist:
        '''
        这里的mydir是train_corpus/art/21.txt中的art(即catelist中的一个类别)
        '''
        class_path = corpus_path + mydir + "/"
        seg_dir = seg_path + mydir + "/"

        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)

        file_list = os.listdir(class_path)
        for file_path in file_list:
            fullname = class_path + file_path
            content = ReadFile(fullname)
            content = content.decode('GBK', 'ignore')
            content = content.replace('\r\n', '')
            content = content.replace(" ", "")
            content_seg = jieba.cut(content)
            # 去停用词
            outstr = ''
            for word in content_seg:
                if word not in stopwords:
                    if word != '\t':
                        outstr += word
                        outstr += " "
            # content_seg = content_seg.encode('utf-8')
            SaveFile(seg_dir + file_path, outstr)
            # SaveFile(seg_dir + file_path, " ".join(content_seg))
    print("中文语料分词结束！！！")


# 停用词文件去重复
def sorte_stopwords(stop_word_path):
    with open(stop_word_path, 'r') as inputfp:
        # d = sorted(set(inputfp.read()))
        d = inputfp.read().split("\n")
        d = sorted(set(d), key=d.index)
        d = "\n".join(d)
    print(d)
    inputfp.close()
    with open("./train_word_bag/hlt_stop_word.txt", 'w') as outputfp:
        outputfp.write(d)
    outputfp.close()


if __name__ == "__main__":
    # 停用词去重复
    # stop_word_path = "./train_word_bag/hlt_stop_word.txt"
    # contents = sorte_stopwords(stop_word_path)
    # 对训练集进行分词
    # train_corpus_path = "./train_corpus/"
    # train_seg_path = "./train_corpus_seg/"
    # Corpus_Segment(train_corpus_path, train_seg_path, stop_word_path)

    # 对测试集进行分词
    # test_corpus_path = "./test_corpus/"
    # test_seg_path = "./test_corpus_seg/"
    # Corpus_Segment(test_corpus_path, test_seg_path, stop_word_path)

    # 对训练集进行Bunch化操作
    # wordbag_path = "./train_word_bag/train_set.dat"
    # seg_path = "./train_corpus_seg/"
    # corpus2Bunch(wordbag_path, seg_path)

    # # 对测试集进行Bunch化操作
    # wordbag_path = "./test_word_bag/test_set.dat"
    # seg_path = "./test_corpus_seg/"
    # corpus2Bunch(wordbag_path, seg_path)

    # 测试bunch内容是否准确
    # bunch_path = "./test_word_bag/test_set.dat"
    # stopwords_path = "./train_word_bag/hlt_stop_words.txt"
    # bunch = _readbunchobj(bunch_path)
    # contents = bunch.contents
    # for i in range(4):
    #     bunch.contents[i] = bunch.contents[i].decode('utf-8', 'ignore')
    #     print(bunch.contents[i])
    #     print("\n")

    # 测试分词后单独去停用词
    # stopwords = {}
    # fstop = open(stopwords_path, 'r')
    # stwd = str(fstop.read())
    # for eachWord in stwd:
    #     stopwords[eachWord.strip()] = eachWord.strip()
    # fstop.close()
    # # 读取jieba分词后的bunch数据结构，从而获取bunch.contents
    # bunch = _readbunchobj(bunch_path)
    # contents = str(bunch.contents[0].decode('utf-8', 'ignore')).split()
    # contents = list(contents)
    # # print(contents)
    # for word in contents:
    #     if word in stopwords:
    #         contents.remove(word)
    # print(contents)

    # for i in range(len(bunch.contents)):
    #     bunch.contents[i].encode()
    # 测试tfidfspace向量空间内容是否准确
    # space_path = "./train_word_bag/tfidfspace.dat"
    # tfidfspace = _readbunchobj(space_path)
    # for i in range(1):
    #     print(tfidfspace.vocabulary)
        # print(tfidfspace.target_name)
        # print(tfidfspace.tdm)

    '''cuttest("杩欐槸涓�涓几鎵嬩笉瑙佷簲鎸囩殑榛戝銆傛垜鍙瓩鎮熺┖锛屾垜鐖卞寳浜紝鎴戠埍Python鍜孋++銆�")
     cuttest('鐪嬩笂鍘籭phone8鎵嬫満鏍峰紡寰堣禐,鍞环699缇庡厓,閿�閲忔定浜�5%涔堬紵')
'''