import os
import re
import pdb
import nltk
import torch
import numpy as np
from torch.utils.data import Dataset

class DataProcess(object):
    def __init__(self, word2Ind, tag2Ind):
        """
            Create a dataprocess class to process data
        :param word2Ind: type: dict, word to index
        :param tag2Ind: type: dict, tag to index
        """
        self.word2Ind = word2Ind
        self.tag2Ind = tag2Ind

        if len(word2Ind) == 0 and len(tag2Ind) == 0:
            self.word2Ind['pad'] = 0
            self.word2Ind['unk'] = 1

        self.dirpath = os.path.join('..', 'topicclass')

    def read_corpus(self, file, testfile=False):
        """
            read files and generate token
        :param file:
        :param testfile:
        :return:
        """
        filepath = os.path.join(self.dirpath, file)
        # pdb.set_trace()
        assert (os.path.exists(filepath)), "file doens't exist"
        with open(filepath, 'r') as f:
            for line in f:
                if len(line.strip()) == 0:
                    continue
                tag, sentence = line.lower().strip().split("|||")
                sentence = self._preprocess_text(sentence)
                if len(sentence.strip()) == 0:
                    continue
                # pdb.set_trace()
                if testfile:
                    yield ([self.word2Ind[x] for x in sentence.split(" ") if len(x) > 0])
                else:
                    yield ([self.word2Ind[x] for x in sentence.split(" ") if len(x) > 0], self.tag2Ind[tag])

    def _preprocess_text(self, sentence):
        # remove punctuation and numbers
        sentence = re.sub('[^a-zA-Z]', " ", sentence)
        # remove single character
        sentence = re.sub(r"\s+[a-zA-Z]\s+", " ", sentence)
        # remove multiple spaces
        sentent = re.sub(r"\s+", " ", sentence)
        return sentence

    @staticmethod
    def show_line(file, line_count=5):
        count = 0
        with open(file, 'r') as f:
            for line in f:
                print(line)
                count += 1
                if count >= line_count:
                    break


class CorpusDataset(Dataset):
    def __init__(self, dataDict, padding = False, seq_len = None, test_flag = False):
        self.test_flag = test_flag

        if self.test_flag:
            self.data = dataDict
        else:
            self.data, self.target = zip(*dataDict)

        if padding == True and seq_len is not None:
            self.data = [data + [0] * (seq_len - len(data)) for data in self.data]

    def _convert_to_tensor(self, data):
        return torch.tensor(data, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self._convert_to_tensor(self.data[idx])
        if self.test_flag:
            return data
        else:
            target = self._convert_to_tensor(self.target[idx])
            return data, target


def collate_line(seqline):
    inputs, targets = zip(*seqline)
    lens = [len(seq) for seq in inputs]
    seq_order = sorted(range(len(lens)), key = lens.__getitem__, reverse=True)
    # pdb.set_trace()
    input = [inputs[i] for i in seq_order]
    target = [targets[i] for i in seq_order]
    return input, target














