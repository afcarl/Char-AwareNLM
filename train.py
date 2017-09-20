#! /usr/bin/env python

import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import data_loader


# Parameters
# ==================================================
dtype = torch.FloatTensor

# Data loading params
train_file = "./data/ptb/train.txt"
valid_file = "./data/ptb/valid.txt"
test_file = "./data/ptb/test.txt"

# Model Hyperparameters
cnn_d = 15 # dimensionality of character embeddings
cnn_w = [1,2,3,4,5,6] # filter widths
cnn_h = [25,50,75,100,125,150] # number of filter matrices
sum_h = sum(cnn_h)
#ch## = [0 : pad, 1 : unk] 
#wd## = [0 : UNK, 1 : END]

# Training Parameters
batch_size = 20
time_step = 35
num_epochs = 35
learning_rate = 1.0 
evaluate_every = 5

# Data Preparation
# ===========================================================
# Load data
print("Loading data...")
word2id, id2word, char2id, id2char, word_maxlen, word_train, word_valid, word_test, char_train, char_valid, char_test, wordlen_train, wordlen_valid, wordlen_test = data_loader.load_data(train_file, valid_file, test_file)

word_train = list(word_train)[1:]+[0] # 0 : END
word_valid = list(word_valid)[1:]+[0] # 0 : END
word_test = list(word_test)[1:]+[0] # 0 : END
print("Train/Validation/Test: {:d}/{:d}/{:d}".format(len(word_train), len(word_valid), len(word_test)))
#print("==================================================================================")

class Conv2d(nn.Module):
    # input : (batch + time_step) * cnn_d * word_maxlen
    # torch.max( (batch + time_step) * (25*kernel_size) * (word_maxlen-kernel_size+1)
    # output : (batch + time_step) * (25*kernel_size)
    def __init__(self, kernel_sizes, in_channels=cnn_d):
        super(Conv2d, self).__init__()

        # attributes:
        self.kernel_sizes = kernel_sizes
        self.in_channels = in_channels
        self.out_channels = 25*kernel_sizes

        # modules:
        self.conv1d = nn.ModuleList([nn.Conv1d(self.in_channels, out_channels=25*kernel_size,
                                    kernel_size=kernel_size, stride=1) 
                                    for kernel_size in self.kernel_sizes])

    def forward(self, words):
        # RuntimeError : tensors are on different GPUs
        for i, conv in enumerate(self.conv1d):
            f_k = torch.max(F.tanh(conv(words)), 2)[0]

            if i == 0:
                y_k = f_k 
            else:
                y_k = torch.cat((y_k, f_k), dim=1)
        return y_k
        
class HW(nn.Module):
    # input : (batch + time_step) * (25*kernel_size)
    # output : (batch + time_step) * (25*kernel_size)
    def __init__(self, in_features):
        super(HW, self).__init__()
        self.in_features = in_features

        # modules:
        self.wh = nn.Linear(in_features, in_features)
        self.wt = nn.Linear(in_features, in_features)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_k):
        h = self.relu(self.wh(y_k))
        t = self.sigmoid(self.wt(y_k))
        z = torch.mul(t, h) + torch.mul((1-t), y_k)
        return z
        
class LSTM(nn.Module):
    # input : time_step * batch * (25*kernel_size)
    # output : batch  * (25*kernel_size)
    def __init__(self, in_features):
        super(LSTM, self).__init__()

        # attributes:
        self.in_features = in_features
        self.out_features = in_features 
        self.num_layers = 1

        # modules:
        self.lstm = nn.LSTM(self.in_features, self.out_features, self.num_layers, dropout=0.5)

    def forward(self, x):
        # Need to seperate by time_step
        batch_size = x.size()[1]
        h0 = Variable(torch.zeros(1, batch_size, self.out_features))
        c0 = Variable(torch.zeros(1, batch_size, self.out_features))
        h, _ = self.lstm(x, (h0, c0))
        return h[time_step-1]

class NLL(nn.Module):
    # input : batch * (25*kernel_size)
    # output : 1
    def __init__(self, in_features):
        super(NLL, self).__init__()

        # attributes:
        self.in_features = in_features
        self.V = len(id2word)

        # modules:
        self.linear = nn.Linear(self.in_features, self.V)
        self.crossentropy = nn.CrossEntropyLoss()
        self.logsoftmax = nn.LogSoftmax()
        self.nnl = nn.NLLLoss()

    def forward(self, h, word_embed):
        hp = self.linear(h)
        NLL = self.crossentropy(hp, word_embed[time_step:])

        return NLL

char_weight = nn.Embedding(len(id2char), cnn_d).type(dtype)

cnn_models = Conv2d(cnn_w)
hw_model = HW(sum_h)
lstm_model = LSTM(sum_h)

nll_model = NLL(sum_h)

def parameters():

    params = []
    for model in [cnn_models, hw_model, lstm_model, nll_model]:
        params += list(model.parameters())

    return params

def make_charmask(word_maxlen, cnn_d, word_len):
    one = [1]*cnn_d
    zero = [0]*cnn_d
    words = []
    for c in word_len:
        words.append(one*c + zero*(word_maxlen-c))

    # (batch + time_step) * word_maxlen * cnn_d
    # [[1 1 1 ... 1 0 0 0 ... 0]]
    return Variable(torch.from_numpy(np.asarray(words)), requires_grad=False).type(dtype)

optimizer = torch.optim.SGD(parameters(), lr=learning_rate)

def run(word_embed, char_embed, word_len, step):
    if len(word_embed) < time_step:
        return 0

    # (batch + time_step) x (word_maxlen) x (cnn_d)
    char_embed = Variable(torch.from_numpy(np.asarray(char_embed))).type(torch.LongTensor)
    char_embed = char_weight(char_embed).view(-1, word_maxlen, cnn_d)
    char_mask = make_charmask(word_maxlen, cnn_d, word_len).view(-1, word_maxlen, cnn_d)
    char_embed = torch.transpose(torch.mul(char_embed, char_mask), 1, 2)

    # (batch + time_step) 
    word_embed = Variable(torch.from_numpy(np.asarray(word_embed)).type(torch.LongTensor))

    # CNN
    # (batch + time_step) * cnn_d * word_maxl -> (batch + time_step) * (25*kernel_size)
    y_k = cnn_models(char_embed)

    # High Way
    # (batch + time_step) * (25*kernel_size) -> (batch + time_step) * (25*kernel_size)
    x = hw_model(y_k) 

    # Slicing
    # -> time_step * batch * (25*kernel_size)
    for i in xrange(x.size()[0]-time_step):
        if i == 0:
            x_t = x[i:i+time_step]
        else:
            x_t = torch.cat((x_t, x[i:i+time_step]), dim=0)
    x_t = x_t.view(-1, time_step, sum_h).permute(1,0,2)

    # LSTM
    # time_step * batch * (25*kernel_size) -> batch * (25*kernel_size)
    h = lstm_model(x_t)

    # NLL
    # batch * (25*kernel_size) -> 1
    NLL = nll_model(h, word_embed)

    PPL = torch.exp(torch.div(NLL, time_step))

    if (step != 0):
        return torch.sum(PPL)

    optimizer.zero_grad()
    NLL.backward()
    optimizer.step()
    
def print_score(batches, step):
    total_ppl = 0.0

    for batch in batches:
        word_batch, char_batch, wordlen_batch = zip(*batch)
        batch_ppl = run(word_batch, char_batch, wordlen_batch, step=step)
        total_ppl += batch_ppl

    print("ppl: ", total_ppl)

for i in xrange(num_epochs):
    # Training
    train_batches = data_loader.train_batch_iter(list(zip(word_train, char_train, wordlen_train)), batch_size, time_step)
    for j, train_batch in enumerate(train_batches):
        word_batch, char_batch, wordlen_batch = zip(*train_batch)
        run(word_batch, char_batch, wordlen_batch, step=1)
        if (j+1) % 4300 == 0:
            print("batch #{:d}/{}: ".format((j+1)/4300, 10)), datetime.datetime.now()

    print("epoch #{:d}: ".format(i+1)), datetime.datetime.now()
    # Evaludation
    if (i+1) % evaluate_every == 0:
        print("=======================")
        print("Evaluation at epoch #{:d}: ".format(i+1))
        validation_batches = data_loader.validation_batch_iter(list(zip(word_valid, char_valid, wordlen_valid)),batch_size)
        print_score(validation_batches, step=2)

# Testing
print("=======================")
print("Training End..")
print("Test: ")
test_batches = data_loader.validation_batch_iter(list(zip(word_test, char_test, wordlen_test)), batch_size, time_step)
print_score(test_batches, step=3)
