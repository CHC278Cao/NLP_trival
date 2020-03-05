
import os
import pdb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import transformers
from transformers import BertConfig, BertTokenizer
from . import BertBaseUncased
from . import Preprocess


def predict(test_df, target_cols, vocab_path, model_path):
    test_idx = test_df.qa_id.values
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    BATCH_SIZE = 4
    MAX_SEQ_LEN = 320
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_outputs = None
    for FOLD in range(5):
        test_dataset = Preprocess.model_dataset(test_df.question_title.values, test_df.question_body.values,
                                             test_df.answer.values, tokenizer, MAX_SEQ_LEN, targets=None)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

        # for i, test_loader_sample in enumerate(test_data_loader):
        #     print(i, test_loader_sample["ids"].shape, test_loader_sample["mask"].shape,
        #           test_loader_sample["token_type_ids"].shape)

        weight_path = f'data/record/model_{FOLD}_{MAX_SEQ_LEN}.bin'
        model = BertBaseUncased.BertBaseUncased(bert_path=model_path, output_size=30, dropout=0.2, train_mode=False)
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        model.eval()
        model.to(device)

        with torch.no_grad():
            fin_outputs = []
            for batch_idx, data in enumerate(test_data_loader):
                ids = data["ids"].to(device)
                mask = data["mask"].to(device)
                token_type_ids = data["token_type_ids"].to(device)

                outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                fin_outputs.append(outputs.cpu().detach().numpy())
            fin_outputs = np.vstack(fin_outputs)

        pdb.set_trace()
        if model_outputs is None:
            model_outputs = fin_outputs
        else:
            model_outputs += fin_outputs

    model_outputs /= 5

    cols = ["qa_id"] + target_cols
    sub = pd.DataFrame(np.column_stack((test_idx, model_outputs)), columns=cols)
    sub["qa_id"] = sub["qa_id"].astype("int")
    return sub




