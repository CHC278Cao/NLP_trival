import os
import logging


import torch
import torch.nn as nn
# from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from transformers import BertTokenizer, BertForSequenceClassification

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}


def train_epoch(model, dataloader, optimizer, criterion, device, args, scheduler = None):
    model.train()
    model.to(device)
    running_loss = 0.0
    for batch_idx, data in enumerate(dataloader):
        # pdb.set_trace()
        input_ids = data["input_ids"].to(device)
        input_mask = data["mask_ids"].to(device)
        segment_ids = data["segment_ids"].to(device)
        label_ids = data["label_ids"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, segment_ids, input_mask)[0]
        loss = criterion(logits, label_ids)
        loss.backward()
        running_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args['--clip_grad']))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if (batch_idx + 1) % int(args['--log_every']) == 0:
            print("batch_idx: {}, Loss: {}".format(batch_idx+1,
                                                   running_loss / ((batch_idx + 1) * int(args['--batch_size']))))

    return running_loss / len(dataloader)

def valid_epoch(model, dataloader, criterion, device):
    model.eval()
    model.to(device)
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            input_ids = data["input_ids"].to(device)
            input_mask = data["mask_ids"].to(device)
            segment_ids = data["segment_ids"].to(device)
            label_ids = data["label_ids"].to(device)

            logits = model(input_ids, segment_ids, input_mask)[0]
            loss = criterion(logits, label_ids)
            running_loss += loss.item()
            pred = torch.argmax(logits, dim = 1)
            cor = pred.eq(label_ids).sum()
            correct += cor.item()

    return running_loss / len(dataloader), correct / len(dataloader)


