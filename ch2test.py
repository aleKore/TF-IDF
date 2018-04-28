# -*- encoding:utf-8 -*-
import jieba  # 导入jieba模块
import re
import os
import test
# from test import _readbunchobj
import jieba.posseg as pseg
# jieba.load_userdict("newdict.txt")  # 加载自定义词典


def splitSentence(stop_word_path, inputFile_path, outputFile_path):
    # 把停用词做成字典
    stopwords = {}
    fstop = open(stop_word_path, 'r')
    for eachWord in fstop:
        stopwords[eachWord.strip()] = eachWord.strip()
    fstop.close()
    # 读取jieba分词后的bunch数据结构，从而获取bunch.contents
    bunch = test._readbunchobj(inputFile_path)
    contents = bunch.contents
    for i in range(len(bunch.contents)):
        contents[i] = str(contents[i].decode('utf-8', 'ignore')).split()
    contents = list(contents)
    outstr = ''
    for word in contents[0]:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    outputs = open(outputFile_path, 'w')
    outputs.write(outstr)
    print(outputs)
    outputs.closed
    # fin = open(inputFile_path, 'r')  # 以读的方式打开文件
    # fout = open(outputFile_path, 'w')  # 以写得方式打开文件
    # jieba.enable_parallel(4)  # 并行分词
    # for eachLine in fin:
    #     line = eachLine.strip().decode('utf-8', 'ignore')  # 去除每行首尾可能出现的空格，并转为Unicode进行处理
    #     line1 = re.sub("[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+".decode("utf8"), "".decode("utf8"),
    #                    line)
    #     wordList = list(jieba.cut(line1))  # 用结巴分词，对每行内容进行分词
    #     outStr = ''
    #     for word in wordList:
    #         if word not in stopwords:
    #             outStr += word
    #             outStr += ' '
    #     fout.write(outStr.strip().encode('utf-8') + '\n')  # 将分词好的结果写入到输出文件
    # fin.close()
    # fout.close()


if __name__ == "__main__":
    stop_word_path = "./train_word_bag/hlt_stop_words.txt"
    bunch_path = "./test_word_bag/test_set.dat"
    save_path = "./test_word_bag/test_set_nostopwords.dat"
    splitSentence(stop_word_path, bunch_path, save_path)
    # i = 0
    # stopwords = {}
    # fstop = open(stop_word_path, 'r')
    # for eachWord in fstop.read():
    #     stopwords[eachWord.strip()] = eachWord.strip()
    # for key in stopwords:
    #     i += 1
    #     if stopwords[key] == '即':
    #         print(i)
    #     print(stopwords[key])

