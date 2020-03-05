
import numpy as np
import re
import pdb

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class model_dataset(Dataset):
    def __init__(self, qtitle, qbody, answer, tokenizer, max_seq_len, targets = None):
        self.qtitle = qtitle
        self.qbody = qbody
        self.answer = answer

        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.targets = targets

    def __len__(self):
        return len(self.qtitle)

    def __getitem__(self, idx):
        question_title = self._clean_text(str(self.qtitle[idx]))
        question_body = self._clean_text(str(self.qbody[idx]))
        answer = self._clean_text(str(self.answer[idx]))

        text_a = question_title + " " + question_body
        text_b = answer

        token_a, token_b = self._truncate_text(text_a, text_b)
        token_a = ['[CLS]'] + token_a + ['[SEP]']
        token_b = token_b + ['[SEP]']
        token = token_a + token_b
        input_ids = self.tokenizer.convert_tokens_to_ids(token)

        padding = [0] * (self.max_seq_len - len(input_ids))
        ids = input_ids + padding
        attention_mask = [1] * len(input_ids) + padding
        token_type_ids = [0] * len(token_a) + [1] * len(token_b) + padding

        if self.targets is not None:
            targets = self.targets[idx, :]
            return {
                "ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(attention_mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "targets": torch.tensor(targets, dtype=torch.float)
            }
        else:
            return {
                "ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(attention_mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long)
            }


    def _truncate_text(self, text_a, text_b):
        """

        :param text_a:
        :param text_b:
        :return:
        """
        token_a = self.tokenizer.tokenize(text_a)
        token_b = self.tokenizer.tokenize(text_b)

        if len(token_a) + len(token_b) > self.max_seq_len - 3:
            if len(token_a) >= len(token_b):
                if len(token_a) >= self.max_seq_len - 3:
                    resize_a = min(len(token_a) // 2, self.max_seq_len // 2)
                    resize_b = self.max_seq_len - resize_a - np.random.randint(low=3, high=10)
                    token_a = token_a[:resize_a]
                    token_b = token_b[:resize_b]

                else:
                    resize_a = self.max_seq_len - len(token_b) - np.random.randint(low=3, high=10)
                    token_a = token_a[:resize_a]
            else:
                if len(token_b) >= self.max_seq_len - 3:
                    resize_b = min(len(token_b) // 2, self.max_seq_len // 2)
                    resize_a = self.max_seq_len - resize_b - np.random.randint(low=3, high=10)
                    token_a = token_a[:resize_a]
                    token_b = token_b[:resize_b]
                else:
                    resize_b = self.max_seq_len - len(token_a) - np.random.randint(low=3, high=10)
                    token_b = token_b[:resize_b]


        return token_a, token_b

    def _clean_text(self, text):

        text = text.lower().strip()
        text = self._remove_whitespace(text)
        # text = self._remove_html(text)
        text = self._remove_url(text)
        # text = self._remove_hash(text)
        text = self._remove_punctucations(text)
        return text

    def _remove_url(self, text):
        text = re.sub(r'https?:\/\/.*\/\w*', '', text)
        text = re.sub(r'http(\S)+', '', text)
        text = re.sub(r'http ...', '', text)
        text = re.sub(r'(RT|rt)[ ]*@[ ]*[\S]+', '', text)
        text = re.sub(r'RT[ ]?@', '', text)
        text = re.sub(r'@[\S]+', '', text)
        return text

    def _remove_punctucations(self, text):
        text = ' '.join(re.sub("[\.\,\!\?\:\;\-\=\/\|\'\(\']", " ", text).split())
        return text

    def _remove_html(self, text):
        text = re.sub(r'\&\w*;', '', text)
        return text

    def _remove_hash(self, text):
        text = re.sub(r'#', '', text)
        return text

    def _remove_whitespace(self, text):
        text = re.sub(r'\s\s+', '', text)
        text = re.sub(r'[ ]{2, }', ' ', text)
        return text








