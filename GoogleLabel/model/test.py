import os
import pdb
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn import model_selection

if __name__ == "__main__":
    model_path = '/Users/caochangjian/Downloads/virtualEnv/Bert/bert_base_uncased'
    vocab_path = os.path.join(model_path, 'vocab.txt')

    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    text_a = "we need to change the input and/or the output slightly"
    text_b = "we can see what should be the input and the output of the model"
    inputs = tokenizer.encode_plus(text_a, text_b, max_length=128)
    input_ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    print(input_ids, len(input_ids))
    print(mask)
    print(token_type_ids)
    token_a = tokenizer.tokenize(text_a)
    ids = tokenizer.convert_tokens_to_ids(token_a)
    print(ids)
    print(len(ids))
    print(len(input_ids), len(mask))
    print(len([x for x in token_type_ids if x == 0]))