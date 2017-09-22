import numpy as np

word2id = {}
id2word = []
char2id = {}
id2char = []

word_maxlen = 0

# 0 : PAD or END, 1 : UNK
def char_index(word):
    word_embed = [char2id.get(char) if char in char2id else 1 for char in word] + [0]*(word_maxlen-len(word))

    return word_embed

def load_data(train_file, valid_file, test_file):
    train = []
    valid = []
    test = []

    train_f = open(train_file, "r")
    lines = train_f.readlines()

    for line in lines:
        tokens = line.strip().split(" ")
        train.append(tokens)
    train = [token for line in train
                   for token in line]
    train_f.close()

    valid_f = open(valid_file, "r")
    lines = valid_f.readlines()

    for line in lines:
        tokens = line.strip().split(" ")
        valid.append(tokens)
    valid = [token for line in valid 
                   for token in line]
    valid_f.close()
    
    test_f = open(test_file, "r")
    lines = test_f.readlines()

    for line in lines:
        tokens = line.strip().split(" ")
        test.append(tokens)
    test = [token for line in test 
                  for token in line]
    test_f.close()

    global word_maxlen

    char2id['pad'] = len(char2id)
    id2char.append('pad')
    char2id['unk'] = len(char2id)
    id2char.append('unk')
    word2id['END'] = len(word2id)
    id2word.append('END')
    word2id['UNK'] = len(word2id)
    id2word.append('UNK')

    for word in train:
        word_maxlen = word_maxlen if word_maxlen>len(word) else len(word)
        if word == '<unk>':
            continue

        if word2id.get(word) == None:
            word2id[word] = len(word2id)
            id2word.append(word)

        for char in word:
            if char2id.get(char) == None:
                char2id[char] = len(char2id)
                id2char.append(char)

    char_train = []
    word_train = []
    wordlen_train = []
    for word in train:
        if word == '<unk>':
            word_train.append(1)
            char_train.append([1] + [0]*(word_maxlen-1))
            wordlen_train.append(1)
        else:
            word_train.append(word2id.get(word))
            char_train.append(char_index(word))
            wordlen_train.append(len(word))

    char_valid = []
    word_valid = []
    wordlen_valid = []
    for word in valid:
        if word == '<unk>':
            word_valid.append(1)
            char_valid.append([1] + [0]*(word_maxlen-1))
            wordlen_valid.append(1)
        else:
            word_valid.append(word2id.get(word))
            char_valid.append(char_index(word))
            wordlen_valid.append(len(word))

    char_test = []
    word_test = []
    wordlen_test = []
    for word in test:
        if word == '<unk>':
            word_test.append(1)
            char_test.append([1] + [0]*(word_maxlen-1))
            wordlen_test.append(1)
        else:
            word_test.append(word2id.get(word))
            char_test.append(char_index(word))
            wordlen_test.append(len(word))

    return word2id, id2word, char2id, id2char, word_maxlen, word_train, word_valid, word_test, char_train, char_valid, char_test, wordlen_train, wordlen_valid, wordlen_test

def train_batch_iter(data, batch_size, time_step):
    data = np.array(data, dtype=object)
    data_size = len(data)
    num_batches = int((data_size-time_step)/batch_size) + (1 if ((data_size-time_step)%batch_size)!=0 else 0)
    for batch_num in xrange(num_batches):
        start_index = batch_num * batch_size
        end_index = min(start_index + batch_size + time_step, data_size)
        # How to seperate each batches
        yield data[start_index:end_index]

def validation_batch_iter(data, batch_size, time_step):
    data = np.asarray(data, dtype=object)
    data_size = len(data)
    num_batches = int((data_size-time_step)/batch_size) + (1 if ((data_size-time_step)%batch_size)!=0 else 0)
    for batch_num in xrange(num_batches):
        start_index = batch_num * batch_size
        end_index = min(start_index + batch_size + time_step, data_size)
        yield data[start_index:end_index]

def test_train_batch_iter(data, batch_size):
    data = np.array(data, dtype=object)
    data_size = len(data)
    num_batches = int(data_size/batch_size) + 1
    for batch_num in xrange(num_batches):
        start_index = batch_num * batch_size
        end_index = min(start_index + batch_size, data_size)
        # How to seperate each batches
        yield data[start_index:end_index]

def test_validation_batch_iter(data, batch_size, time_step):
    data = np.asarray(data, dtype=object)
    data_size = len(data)
    num_batches = int(data_size/batch_size) + 1
    for batch_num in xrange(num_batches):
        start_index = batch_num * batch_size
        end_index = min(start_index + batch_size, data_size)
        yield data[start_index:end_index]
