

import os
import torch
import torch.nn as nn

import transformers
from transformers import BertConfig

class BertBaseUncased(nn.Module):
    def __init__(self, bert_path, output_size, dropout, train_mode = True):
        super(BertBaseUncased, self).__init__()
        self.bert_path = bert_path
        if train_mode:
            self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        else:
            bertcofig = BertConfig.from_json_file(os.path.join(bert_path, 'bert_config.json'))
            self.bert = transformers.BertModel(config=bertcofig)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, output_size)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        out = self.dropout(o2)
        out = self.linear(out)

        return out


