# -*- coding: utf-8 -*-
# @Time    : 2017/11/16 18:54
# @Author  : renhongkai
# @Email   : 13753198867@163.com
# @Software: PyCharm

import torch
import utils
import pickle
# import word_segement_model_batch

if __name__=='__main__':
    # 获取测试集中的所有数据
    test_data = utils.generate_test_data("./icwb2-data/testing/pku_test_gold_BIO_test.utf8")
    training_data = utils.generate_train_data("./icwb2-data/training/pku_training_BIO_test.utf8")
    word_to_ix = utils.character_index(training_data)  # 所有词及对应的下标

    # 提取出保存的整个网络
    bilstm_crf = torch.load('word_segment_model.pkl')

    for item in test_data:
        prediction = bilstm_crf(utils.prepare_sequence(item, word_to_ix))
        print(type(prediction))

