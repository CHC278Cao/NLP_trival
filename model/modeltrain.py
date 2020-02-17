import os
import random
import time
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
from wordProcess import *
from classifyModel1 import *
from RNNClassify import *


def train_CNN(train, valid, test):
    model = CNNModel(nwords, 100, 0.1, ntags)
    print(model)
    for param in model.parameters():
        print(param.data.size())

    trainMAXLEN = max([len(seq[0]) for seq in train])
    validMAXLEN = max([len(seq[0]) for seq in valid])
    testMAXLEN = max([len(seq[0]) for seq in test])
    MAXLEN = max(trainMAXLEN, validMAXLEN, testMAXLEN)
    print("trainMAXLEN is {}, validMAXLEN is {}, testMAXLEN is {}".format(trainMAXLEN, validMAXLEN, testMAXLEN))
    times = MAXLEN // 8
    MAXLEN = 8 * (times + 1) if (times % 2) == 1 else 8 * times
    print("MAXLEN is {}".format(MAXLEN))
    trainDataset = CorpusDataset(train, padding=True, seq_len=MAXLEN)
    validDataset = CorpusDataset(valid, padding=True, seq_len=MAXLEN)
    testDataset = CorpusDataset(test, padding=True, seq_len=MAXLEN)
    traindataset_args = dict(shuffle=True, batch_size=64, collate_fn=collate_line)
    validdataset_args = dict(shuffle = True, batch_size = 64, collate_fn = collate_line)
    testdataset_args = dict(shuffle = False, batch_size = 64)
    trainloader = DataLoader(trainDataset, **traindataset_args)
    validloader = DataLoader(validDataset, **validdataset_args)
    testloader = DataLoader(testDataset, **testdataset_args)


    epochs = 50
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    tolerate = 3
    count = 0
    best_valid_loss = float("inf")
    saver = "CNNModel1.pt"
    for i in range(epochs):
        start = time.time()
        train_loss = train_epoch(model, trainloader, device, optimizer, criterion)
        valid_loss, valid_acc = valid_epoch(model, validloader, device,  criterion)
        print("Time: {}".format(start))
        print("Training Loss: {0:.4f}".format(train_loss))
        print("Validing Loss: {0:.4f}, Accuracy is {1:.4f}".format(valid_loss, valid_acc))
        print("-" * 80)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            count = 0
            torch.save(model.state_dict(), saver)
        else:
            count += 1
            if count >= tolerate:
                print("Early Stopping")
                break


def train_RNN(train, valid, test, device):
    model = RNNModel(nwords, 100, 64, 2, ntags, 0.1, device)
    print(model)
    for param in model.parameters():
        print(param.data.size())

    trainDataset = CorpusDataset(train, padding = False)
    validDataset = CorpusDataset(valid, padding = False)
    testDataset = CorpusDataset(test, padding = False)

    traindataset_args = dict(shuffle=True, batch_size=64, collate_fn=collate_line)
    validdataset_args = dict(shuffle=True, batch_size=64, collate_fn=collate_line)
    testdataset_args = dict(shuffle=False, batch_size=64)
    trainloader = DataLoader(trainDataset, **traindataset_args)
    validloader = DataLoader(validDataset, **validdataset_args)
    testloader = DataLoader(testDataset, **testdataset_args)


    epochs = 50
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    criterion = nn.CrossEntropyLoss()


    tolerate = 3
    count = 0
    best_valid_loss = float("inf")
    saver = "RNNModel1.pt"
    for i in range(epochs):
        start = time.time()
        train_loss = train_epoch(model, trainloader, device, optimizer, criterion)
        valid_loss, valid_acc = valid_epoch(model, validloader, device,  criterion)
        print("Time: {}".format(start))
        print("Training Loss: {0:.4f}".format(train_loss))
        print("Validing Loss: {0:.4f}, Accuracy is {1:.4f}".format(valid_loss, valid_acc))
        print("-" * 80)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            count = 0
            torch.save(model.state_dict(), saver)
        else:
            count += 1
            if count >= tolerate:
                print("Early Stopping")
                break




if __name__ == '__main__':
    w2i = defaultdict(lambda: len(w2i))
    t2i = defaultdict(lambda: len(t2i))
    PAD = w2i['pad']
    UNK = w2i['unk']

    dataDir = os.path.join('..', 'topicclass')
    trainfile = os.path.join(dataDir, "topicclass_train.txt")
    validfile = os.path.join(dataDir, 'topicclass_valid.txt')
    testfile = os.path.join(dataDir, 'topicclass_test.txt')

    train = list(readCorpus(trainfile, w2i, t2i))
    valid = list(readCorpus(validfile, w2i, t2i))
    test = list(readCorpus(testfile, w2i, t2i))

    nwords = len(w2i)
    ntags = len(t2i) - 1
    print("nwords is {}, ntags is {}".format(nwords, ntags))

    # this park if for RNN model, so we will use pack_pad_seq to pad data in each batch
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    train_RNN(train, valid, test, device)



























