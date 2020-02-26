import os
import re
import pdb
import nltk
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict

class data_bert_process(object):
    def __init__(self, tokenizer, max_seq_len = None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.data_dir = os.path.join('..', 'topicclass')
        self.tag2Ind = defaultdict()


    def load_file(self, filepath, test_flag = False):
        filepath = os.path.join(self.data_dir, filepath)
        assert (os.path.exists(filepath)), "file doens't exist"

        # content = []
        with open(filepath, 'r') as f:
            for line in f:
                if len(line.strip()) == 0:
                    continue
                tag, sentence = line.lower().strip().split("|||")
                if len(sentence.strip()) == 0:
                    continue

                token = self.tokenizer.tokenize(sentence)
                token = token[:self.max_seq_len-2]
                token = ['[CLS]'] + token + ['[SEP]']
                # token_len = len(token)
                token_ids = self.tokenizer.convert_tokens_to_ids(token)
                if test_flag:
                    yield token_ids
                else:
                    label = self._load_label(tag)
                    yield (token_ids, label)


    def _load_label(self, tag):
        if tag not in self.tag2Ind:
            self.tag2Ind[tag] = len(self.tag2Ind)

        return self.tag2Ind[tag]

    def get_num_labels(self):
        return len(set(self.tag2Ind))

    def get_labelmap(self):
        id2label = dict()
        for k, v in self.tag2Ind.items():
            id2label[v] = k
        return id2label


class model_dataset(Dataset):
    def __init__(self, data, max_seq_len, test_flag = False):
        self.test_flag = test_flag
        self.max_seq_len = max_seq_len
        # pdb.set_trace()
        if self.test_flag:
            pdb.set_trace()
            self.token_ids = data
        else:
            self.token_ids, self.label = zip(*data)

    def _convert_to_tensor(self, data):
        return torch.tensor(data, dtype=torch.long)

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        token_ids = self.token_ids[idx]
        input_ids = token_ids + [0] * (self.max_seq_len - len(token_ids))
        mask_ids = [1] * len(token_ids) + [0] * (self.max_seq_len - len(token_ids))
        segment_ids = [0] * self.max_seq_len
        label_ids = self.label[idx]

        input_ids = self._convert_to_tensor(input_ids)
        mask_ids = self._convert_to_tensor(mask_ids)
        segment_ids = self._convert_to_tensor(segment_ids)
        if self.test_flag:
            label_ids = []
        else:
            label_ids = self._convert_to_tensor(label_ids)

        return {
            "input_ids": input_ids,
            "mask_ids": mask_ids,
            "segment_ids": segment_ids,
            "label_ids": label_ids
        }











