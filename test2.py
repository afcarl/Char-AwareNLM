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
gtype = torch.cuda.FloatTensor

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
batch_size = 35 
num_epochs = 35
learning_rate = 1.
evaluate_every = 5

ex_ppl = 9999.

# Data Preparation
# ===========================================================
# Load data
print("Loading data...")
word2id, id2word, char2id, id2char, word_maxlen, word_train, word_valid, word_test, char_train, char_valid, char_test, wordlen_train, wordlen_valid, wordlen_test = data_loader.load_data(train_file, valid_file, test_file)

# Target word preprcessing
word_train = list(word_train)[1:]+[0] # 0 : END
word_valid = list(word_valid)[1:]+[0] # 0 : END
word_test = list(word_test)[1:]+[0] # 0 : END
print("Train/Validation/Test: {:d}/{:d}/{:d}".format(len(word_train), len(word_valid), len(word_test)))
#print("==================================================================================")
###############################################################################################

class Conv2d(nn.Module):
    # input : (batch) * cnn_d * word_maxlen
    # torch.max( (batch) * (25*kernel_size) * (word_maxlen-kernel_size+1)
    # output : (batch) * (25*kernel_size)
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
    # input : (batch) * (25*kernel_size)
    # output : (batch) * (25*kernel_size)
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
    # input : (batch) * 1 * (25*kernel_size)
    # output : batch  * (25*kernel_size)
    def __init__(self, in_features):
        super(LSTM, self).__init__()

        # attributes:
        self.in_features = in_features
        self.hidden_features = 300
        self.out_features = in_features 
        self.num_layers = 2

        # modules:
        self.lstm = nn.LSTM(self.in_features, self.out_features, self.num_layers, dropout=0.5)

    def forward(self, x, hn):
        # Need to seperate by time_step
        h, (ht, ct) = self.lstm(x, hn)
        h0 = Variable(ht.data.cpu())
        c0 = Variable(ct.data.cpu())

        return h.view(-1, sum_h)

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
        # Have to size_average=False
        self.crossentropy = nn.CrossEntropyLoss(size_average=False)

    def forward(self, h, word_embed):
        hp = self.linear(h)
        word_embed = Variable(word_embed).cuda()
        NLL = self.crossentropy(hp, word_embed)

        return NLL

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

    # (batch) * word_maxlen * cnn_d
    # [[1 1 1 ... 1 0 0 0 ... 0]...]
    return Variable(torch.from_numpy(np.asarray(words)).type(gtype), requires_grad=False)

###############################################################################################
def run(word_embed, char_embed, word_len, step):

    global h0, c0
    optimizer.zero_grad()

    # (batch) x (word_maxlen) x (cnn_d)
    char_embed = Variable(torch.from_numpy(np.asarray(char_embed)).type(torch.cuda.LongTensor))
    char_embed = char_weight(char_embed).view(-1, word_maxlen, cnn_d)
    char_mask = make_charmask(word_maxlen, cnn_d, word_len).view(-1, word_maxlen, cnn_d)
    char_embed = torch.transpose(torch.mul(char_embed, char_mask), 1, 2)

    # (batch) 
    word_embed = torch.from_numpy(np.asarray(word_embed)).type(torch.LongTensor)

    # CNN
    # (batch) * cnn_d * word_maxl -> (batch) * (25*kernel_size)
    y_k = cnn_models(char_embed)

    # High Way
    # (batch) * (25*kernel_size) -> (batch) * 1 * (25*kernel_size)
    x = hw_model(y_k).view(-1, 1, sum_h) 

    # LSTM
    # (batch) * 1 * (25*kernel_size) -> batch * (25*kernel_size)
    h = lstm_model(x, (h0, c0))

    # NLL
    # batch * (25*kernel_size) -> 1
    batch_NLL = nll_model(h, word_embed)
    NLL = batch_NLL/len(word_embed)

    if (step != 1):
        return batch_NLL.data.cpu().numpy()[0]

    NLL.backward()
    optimizer.step()
    
    return batch_NLL.data.cpu().numpy()[0] 

def print_score(batches, step):
    global ex_ppl, learning_rate, optimizer
    total_nll = 0.0

    h0 = Variable(torch.zeros(2, 1, sum_h)).cuda()
    c0 = Variable(torch.zeros(2, 1, sum_h)).cuda()
    for batch in batches:
        word_batch, char_batch, wordlen_batch = zip(*batch)
        batch_nll = run(word_batch, char_batch, wordlen_batch, step=step)
        total_nll += batch_nll

    if step == 2:
        len_word = len(word_valid)
    elif step == 3:
        len_word = len(word_test)

    # No Front padding
    PPL = np.exp(total_nll/len_word)

    # New learning rate
    if PPL+1 > ex_ppl:
        learning_rate = learning_rate*0.5
        optimizer = torch.optim.SGD(parameters(), lr=learning_rate)
    ex_ppl = PPL

    print("ppl: ", PPL)

###############################################################################################
char_weight = nn.Embedding(len(id2char), cnn_d).type(gtype)

cnn_models = Conv2d(cnn_w).cuda()
hw_model = HW(sum_h).cuda()
lstm_model = LSTM(sum_h).cuda()
nll_model = NLL(sum_h).cuda()
optimizer = torch.optim.SGD(parameters(), lr=learning_rate)

for i in xrange(num_epochs):
    # Training
    # Every 35 words
    train_batches = data_loader.test_train_batch_iter(list(zip(word_train, char_train, wordlen_train)), batch_size)
    h0 = Variable(torch.zeros(2, 1, sum_h)).cuda()
    c0 = Variable(torch.zeros(2, 1, sum_h)).cuda()
    batch_nll = 0.
    for j, train_batch in enumerate(train_batches):
        word_batch, char_batch, wordlen_batch = zip(*train_batch)
        batch_nll += run(word_batch, char_batch, wordlen_batch, step=1)
        if (j+1) % 5000 == 0:
            print("batch #{:d}: ".format(j+1)), "batch_ppl :", np.exp(batch_nll/j/batch_size), datetime.datetime.now()

    # Validation 
    # In every epoch (for updating a learning rate)
    print("=======================")
    print("Evaluation at epoch #{:d}: ".format(i+1))
    validation_batches = data_loader.test_validation_batch_iter(list(zip(word_valid, char_valid, wordlen_valid)),batch_size)
    print_score(validation_batches, step=2)

cnn_models.eval()
hw_model.eval()
lstm_model.eval()
nll_model.eval()

# Testing
print("=======================")
print("Training End..")
print("Test: ")
test_batches = data_loader.test_validation_batch_iter(list(zip(word_test, char_test, wordlen_test)), batch_size)
print_score(test_batches, step=3)
