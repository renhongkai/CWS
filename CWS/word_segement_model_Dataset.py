# -*- coding: utf-8 -*-
# @Time    : 2017/11/15 8:38
# @Author  : renhongkai
# @Email   : 13753198867@163.com
# @Software: PyCharm

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import utils_dataset
import torch.utils.data as Data

torch.manual_seed(1)  #设置随机种子

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

"""
    创建模型
"""
class BiLSTM_CRF(nn.Module):  #继承torch.nn.Module类
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim): #下面调用时model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size   #训练数据中字的个数
        self.tag_to_ix = tag_to_ix    #tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}    #标记及其下标
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)   #字向量初始化的时候第一个参数要是所有字的个数
        print("==============",self.word_embeds)  #Embedding(29, 5)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,num_layers=1, bidirectional=True)  #第一个参数为输入的特征维度，第二个参数为隐状态的特征维度。num_layers表示rnn层的个数，bidirectional – 如果为True，将会变成一个双向RNN，默认为False。

        # Maps the output of the LSTM into tag space. 将lstm的输出映射到标记空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j. 参数的转换矩阵，
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        # print("======",self.transitions)
        # print("------------",self.transitions.data[tag_to_ix[START_TAG], :])
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        # print("改变后的transitions数据1：", self.transitions)
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        # print("改变后的transitions数据2：",self.transitions)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

'''
    运行训练
'''
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
training_data,training_data_dataset = utils_dataset.generate_train_data("./icwb2-data/training/pku_training_BIO.utf8")
word_to_ix = utils_dataset.character_index(training_data)  #所有词及对应的下标

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}    #标记及其下标

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
print(model)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
# # Check predictions before training
precheck_sent = utils_dataset.prepare_sequence("充 满 希 望".split(), word_to_ix)
precheck_tags = torch.LongTensor([tag_to_ix[t] for t in "B I B I".split()])
print(model(precheck_sent))

# 先转换成torch能识别的Dataset
torch_dataset = Data.TensorDataset(data_tensor=torch.LongTensor(training_data_dataset[0]), target_tensor=torch.LongTensor(training_data_dataset[1]))
# 将torch_dataset放入DataLoader中
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=64 ,      # mini batch size
    shuffle=True,               # random shuffle for training    每次打乱数据的顺序，False则不打乱
    num_workers=2,              # subprocesses for loading data  多线程来读取数据
)
for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data
    print("第",epoch,"轮")
    for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
        model.zero_grad()
        neg_log_likelihood = model.neg_log_likelihood(batch_x, batch_y)
        neg_log_likelihood.backward()
        optimizer.step()
torch.save(model, 'word_segment_model.pkl')  # save entire net  保存整个网络
# 获取测试集中的所有数据
test_data = utils_dataset.generate_test_data("./icwb2-data/testing/pku_test_gold_BIO.utf8")
f = open("./icwb2-data/testing/test_model_100epoch.txt",'a')
tag_ind = ['B','I','O','','']
for item in test_data:
    print("句子：",item)
    prediction = model(utils_dataset.prepare_sequence(item, word_to_ix))
    print("预测：",prediction[1])
    for tag_index in prediction[1]:
        f.write(tag_ind[tag_index] + '\n')