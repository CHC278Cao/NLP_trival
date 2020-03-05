
import os
import pdb
import time
import numpy as np
import pandas as pd
from scipy import stats

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn import model_selection

from . import Preprocess
from . import BertBaseUncased
from . import predict



def train_epoch(model, data_loader, optimizer, criterion, device, scheduler = None):
    model.train()
    model.to(device)
    running_loss = 0.0

    for batch_idx, data in enumerate(data_loader):

        ids = data["ids"].to(device)
        mask = data["mask"].to(device)
        token_type_ids = data["token_type_ids"].to(device)
        targets = data["targets"].to(device)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = criterion(outputs, targets)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if (batch_idx % 50) == 49:
            print(f"batch_idx =  {batch_idx+1}, loss = {running_loss / (batch_idx * 16)}")

        del data

    return running_loss / len(data_loader)

def valid_epoch(model, data_loader, criterion, device):
    model.eval()
    model.to(device)
    running_loss = 0.0
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)
            targets = data["targets"].to(device)

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            fin_targets.append(targets.cpu().detach().numpy())
            fin_outputs.append(outputs.cpu().detach().numpy())

    return running_loss / len(data_loader), np.vstack(fin_outputs), np.vstack(fin_targets)
    # return running_loss / len(data_loader)


def run(train_df, target_cols, vocab_path, model_path):
    tokenizer = BertTokenizer.from_pretrained(vocab_path)

    EPOCHS = 10
    BATCH_SIZE = 4
    MAX_SEQ_LEN = 320
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kf = model_selection.KFold(n_splits=5, random_state=42)
    for split_idx, (train_idx, valid_idx) in enumerate(kf.split(X = train_df)):
        pdb.set_trace()
        Xtrain = train_df.loc[train_idx]
        Xvalid = train_df.loc[valid_idx]
        Xtrain_targets = Xtrain[target_cols].values
        Xvalid_targets = Xvalid[target_cols].values
        train_dataset = Preprocess.model_dataset(Xtrain.question_title.values, Xtrain.question_body.values,
                                      Xtrain.answer.values, tokenizer, MAX_SEQ_LEN, targets=Xtrain_targets)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True,
                                                        drop_last=True)

        valid_dataset = Preprocess.model_dataset(Xvalid.question_title.values, Xvalid.question_body.values,
                                      Xvalid.answer.values, tokenizer, MAX_SEQ_LEN, targets=Xvalid_targets)
        valid_data_loader = torch.utils.data.DataLoader(valid_dataset, BATCH_SIZE, shuffle=True,
                                                        drop_last=True)


        for i, train_loader_sample in enumerate(train_data_loader):
            print(i, train_loader_sample["ids"].shape, train_loader_sample["mask"].shape,
                  train_loader_sample["token_type_ids"].shape, train_loader_sample["targets"].shape)

        for i, test_loader_sample in enumerate(valid_data_loader):
            print(i, test_loader_sample["ids"].shape, test_loader_sample["mask"].shape,
                  test_loader_sample["token_type_ids"].shape)

        pdb.set_trace()

        lr = 2e-5
        num_train_steps = int(len(train_dataset) / BATCH_SIZE * EPOCHS)
        PATIENT = 3
        BEST_VALID_LOSS = float("inf")

        model = BertBaseUncased.BertBaseUncased(bert_path=model_path, output_size=30, dropout=0.2)
        optimizer = AdamW(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps
        )

        start_time = time.time()
        cnt = 0
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_data_loader, optimizer, criterion, device, scheduler)
            end_time = time.time()
            print(f"epoch = {epoch}, train_loss = {train_loss}, time = {end_time - start_time}")
            valid_loss, fin_outputs, fin_targets = valid_epoch(model, valid_data_loader, criterion, device)
            print(f"epoch = {epoch}, valid_loss = {valid_loss}, time = {time.time() - end_time}")

            spear = []
            for jj in range(fin_outputs.shape[1]):
                p1 = list(fin_targets[:, jj])
                p2 = list(fin_outputs[:, jj])
                coef, _ = np.nan_to_num(stats.spearmanr(p1, p2))
                spear.append(coef)
            spear = np.mean(spear)
            print(f"epoch = {epoch}, spearman = {spear}")

            if valid_loss < BEST_VALID_LOSS:
                BEST_VALID_LOSS = valid_loss
                cnt = 0
                torch.save(model.state_dict(), f"model_{split_idx}.bin")

            else:
                cnt += 1
                if cnt > PATIENT:
                    print("Early stopping")
                    break


if __name__ == '__main__':
    train_path = os.environ.get("TRAIN_FILE")
    test_path = os.environ.get("TEST_FILE")
    submission_path = os.environ.get("SUBMISSION")
    model_path = os.environ.get("MODEL_PATH")
    mode = os.environ.get('MODE')

    train_df = pd.read_csv(train_path).reset_index(drop=True)
    # print(train_df.head())
    # print(train_df.columns)

    test_df = pd.read_csv(test_path).reset_index(drop=True)
    # print(test_df.head())
    # print(test_df.columns)
    #
    # submission = pd.read_csv(submission_path)
    # print(submission.head())
    pdb.set_trace()
    model_path = '/Users/caochangjian/Downloads/virtualEnv/Bert/bert_base_uncased'
    vocab_path = os.path.join(model_path, 'vocab.txt')
    target_cols = [x for x in train_df.columns if x not in test_df.columns]
    print(target_cols)

    if mode == "train":
        run(train_df, target_cols, vocab_path, model_path)
    elif mode == "test":
        sub = predict.predict(test_df, target_cols, vocab_path, model_path)








