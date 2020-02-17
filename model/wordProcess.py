import os
import re
import pdb
import nltk
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

def showNLine(file, num = 5):
    with open(file, 'r') as f:
        count = 0
        for line in f:
            print(line)
            count += 1
            if count >= num:
                break

def readCorpus(file, w2i, t2i, test = False):
    assert os.path.exists(file)

    with open(file, 'r') as f:
        for line in f:
            if len(line.strip()) > 0:
                tag, sentence = line.lower().strip().split("|||")
                sentence = preprocess_text(sentence)
                # pdb.set_trace()
                if test:
                    yield ([w2i[x] for x in sentence.split(" ") if len(x) > 0])
                else:
                    yield ([w2i[x] for x in sentence.split(" ") if len(x) > 0], t2i[tag])

def preprocess_text(sentence):

    # remove punctuation and numbers
    sentence = re.sub('[^a-zA-Z]', " ", sentence)
    # remove single character
    sentence = re.sub(r"\s+[a-zA-Z]\s+", " ", sentence)
    # remove multiple spaces
    sentent = re.sub(r"\s+", " ", sentence)
    return sentence

# def clean_str(string, TREC=False):
#     """
#     Tokenization/string cleaning for all datasets except for SST.
#     Every dataset is lower cased except for TREC
#     """
#     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"!", " ! ", string)
#     string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip() if TREC else string.strip().lower()


def padSentence(content):
    ct = []
    MAXLEN = max(map(lambda sentence: len(sentence), content))
    for line in content:
        if len(line) < MAXLEN:
            pad = ["<pad>"] * (MAXLEN - len(line))
            line += pad
        ct.append(line)
    return ct

class CorpusDataset(Dataset):
    def __init__(self, dataDict, padding = False, seq_len = None):
        if padding == True and seq_len is not None:
            self.data = [torch.tensor(x[0] + [0] * (seq_len - len(x[0]))) for x in dataDict if len(x[0]) > 0]
            self.target = [torch.tensor(x[1]) for x in dataDict if len(x[0]) > 0]
        else:
            self.data = [torch.tensor(x[0]) for x in dataDict if len(x[0]) > 0]
            self.target = [torch.tensor(x[1]) for x in dataDict if len(x[0]) > 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.target[idx]

        return data, target

# #
class TestDataset(Dataset):
    def __init__(self, data, padding = False, seq_len = None):
        if padding == True and seq_len is not None:
            self.data = [torch.tensor(x + [0] * (seq_len - len(x))) for x in data]
        else:
            self.data = [torch.tensor(x) for x in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data


def collate_line(seqline):
    inputs, targets = zip(*seqline)
    lens = [len(seq) for seq in inputs]
    seq_order = sorted(range(len(lens)), key = lens.__getitem__, reverse=True)
    # pdb.set_trace()
    input = [inputs[i] for i in seq_order]
    target = [targets[i] for i in seq_order]
    return input, target














