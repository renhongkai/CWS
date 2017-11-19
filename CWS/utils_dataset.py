# -*- coding: utf-8 -*-
# @Time    : 2017/11/16 19:02
# @Author  : renhongkai
# @Email   : 13753198867@163.com
# @Software: PyCharm
import torch

# """
#     问题：通过对pku训练数据使用结巴分词，对比查看了下结果跟原来的分词差别很大
#     找到测试数据的分词结果了！！！！
#     对数据进行分词,用于计算正确率等（将模型训练出的分词结果和原来的分词结果进行对比计算得到正确率）
# """
# def word_segment(sourceFileName,targetFileName):
#     f = open(sourceFileName, 'r')
#     f_wordsegment = open(targetFileName, 'a')
#     lines = f.readlines()
#     for line in lines:
#         result = jieba.cut(line)  # 默认是精准模式
#         f_wordsegment.write(" ".join(result))
#     f.close()
#     f_wordsegment.close()

"""
    将分好词的文件处理成（0 迈 B）这种形式
"""
def word_BIO(sourceFileName,targetFileName):
    Puntuation = ['，', '。', '？', '、', '（', '）', '《', '》', '：', '；', '！', '“', '”', '——', '———']
    f = open(sourceFileName, 'r')
    f_BIO = open(targetFileName, 'a')
    lines = [line.rstrip() for line in f.readlines()]  # 去掉列表中每一行的\n
    for line in lines:
        writeList = []  #作用：往文件中写入时，可以根据元素的下标，将下标写入
        for word in line.split():
            ws_str = ''
            if word in Puntuation:
                ws_str = word + ' ' + 'O' + '\n'
                writeList.append(ws_str)
            else:
                for i in range(len(word)):
                    if i == 0:
                        ws_str = word[i] + ' ' + 'B' + "\n"
                        writeList.append(ws_str)
                    else:
                        ws_str = word[i] + ' ' + 'I' + '\n'
                        writeList.append(ws_str)
        for index, item in enumerate(writeList):
            f_BIO.write(str(index) + ' ' + item)
    f.close()
    f_BIO.close()

# word_BIO("./icwb2-data/training/pku_training.utf8","./icwb2-data/training/pku_training_BIO.utf8")
# word_BIO("./icwb2-data/testing/pku_test_gold.utf8","./icwb2-data/testing/pku_test_gold_BIO.utf8")

"""
    得到句子seq中的每个字在字典表word_to_ix中的下标    
"""
def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        if w in to_ix.keys():
            idxs.append(to_ix[w])
        else:
            idxs.append(0)
    # tensor = torch.LongTensor(idxs)
    # return torch.autograd.Variable(tensor)
    return idxs

"""
    从训练数据中读取所有的字，放入word_to_ix中
"""
def character_index(training_data):
    word_to_ix = {}  # 所有词及对应的下标
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix

"""
    将pku_training_BIO.utf8中的数据处理成[(['迈','向'],['B','I']), (['喜','欢'],['B','I'])]这种格式
    以下代码有个问题：最后一个句子无法写入，后续再调整...问题已经解决！！！
"""
def generate_train_data(train_data_filename):
    training_data = []
    f = open(train_data_filename, 'r')
    lines = [line.rstrip() for line in f.readlines()]  # 去掉列表中每一行的\n
    characterList = []
    tagList = []
    for line in lines:
        character_tag_list = line.split()
        if (line == lines[len(lines) - 1]):
            training_data.append((characterList, tagList))
        if (character_tag_list[0] == '0'):
            if len(characterList) != 0:
                training_data.append((characterList, tagList))
            characterList = [character_tag_list[1]]
            tagList = [character_tag_list[2]]
        else:
            characterList.append(character_tag_list[1])
            tagList.append(character_tag_list[2])
    f.close()
    word_to_ix = character_index(training_data)
    tag_to_ix = {"B": 0, "I": 1, "O": 2, "<START>": 3, "<STOP>": 4}  # 标记及其下标
    # 将training_data处理成torch能识别的Dataset格式
    training_data_Dataset = [[],[]]
    for sentence, tags in training_data:
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = [tag_to_ix[t] for t in tags]
        training_data_Dataset[0].append(sentence_in)
        training_data_Dataset[1].append(targets)

    return training_data,training_data_Dataset

# generate_train_data("./icwb2-data/training/pku_training_BIO.utf8")
# generate_train_data("./icwb2-data/training/pku_training_BIO_test.utf8")

"""
    获取测试集中的所有数据,格式：[['共', '同'],['你', '好']...]
"""
def generate_test_data(test_data_filename):
    f = open(test_data_filename, 'r')
    lines = [line.rstrip() for line in f.readlines()]
    test_data = []
    test_data_item = []
    for line in lines:
        character_tag_list = line.split()
        if character_tag_list[0] == '0':
            if len(test_data_item) != 0:
                test_data.append(test_data_item)
            test_data_item = [character_tag_list[1]]
        else:
            test_data_item.append(character_tag_list[1])
    return test_data


