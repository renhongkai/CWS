# -*- coding: utf-8 -*-
# @Time    : 2017/11/15 18:37
# @Author  : renhongkai
# @Email   : 13753198867@163.com
# @Software: PyCharm

"""
    计算正确率、召回率、F值
"""

import pandas

line=[]
file=open(r'./icwb2-data/testing/pku_test_gold_BIO.utf8','r')
# file_model_test = open('./icwb2-data/testing/test_model_100epoch.txt','r')
file_model_test = open('./icwb2-data/testing/testall_model_10epoch.txt','r')
file_lines = [line.rstrip() for line in file.readlines()]
file_model_test_lines = [line.rstrip() for line in file_model_test.readlines()]
print(len(file_lines))
print(len(file_model_test_lines))
for i in range(len(file_lines)):
    line.append((file_lines[i] + ' ' +file_model_test_lines[i]).split())
df=pandas.DataFrame(line,columns=['index','character','tag','model_tag'])

correct=df[df.tag==df.model_tag]
# print(correct)

for i in ('B','I','O'):
    # 准确率
    R=sum(correct.model_tag==i)/sum(df.tag==i)
    # print(R)
    # 召回率
    P=sum(correct.model_tag==i)/(sum(df.model_tag==i))
    # print(P)
    # F值
    F=R*P*2/(R+P)
    print(i,':\n','正确率：',R,' 召回率：',P,' F值：',F)

file.close()
file_model_test.close()
