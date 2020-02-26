"""
Usage:
    run.py train --trainfile=<file> --validfile=<file> [options]
    run.py predict --testfile<file> [options]

Options:
    -h  --help                                       show this screen
    --trainfile=<file>                               train source file
    --validfile=<file>                               valid source file
    --testfile=<file>                                test source file
    --seed=<int>                                     seed [default: 42]
    --max_seq_len=<int>                              maximum of sequence length [default: 64]
    --batch_size=<int>                               batch_size [default: 32]
    --clip_grad=<float>                              gradient clipping [default: 5.0]
    --max_epoch=<int>                                max epochs [default: 50]
    --log_every=<int>                                log every [default: 100]
    --model=<str>                                    petrained model [default: bert-base-uncased]
    --patient=<int>                                  number of iterations to decay learning rate [default: 5]
    --lr=<float>                                     learning rate [default: 2e-5]
    --lr_decay=<float>                               learning decay [default: 0.5]
    --lr_decay_epoch=<int>                           learning decay after each period [default: 10]
    --dropout=<float>                                dropout rate [default: 0.3]
    --input_dir=<file>                               input file directory
    --out_dir=<file>                                 output file directory
    --model_save_path=<file>                         model_save_path

"""


import os
import sys
import random
import time
from docopt import docopt
import pdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dataProcess import data_bert_process, model_dataset
from bertTextModel import train_epoch, valid_epoch
from transformers import BertTokenizer, BertForSequenceClassification, \
    BertModel, AdamW, BertConfig, get_linear_schedule_with_warmup


PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}


def generate_bert_token(args, dataBertProcess):

    trainfile = args['--trainfile']
    validfile = args['--validfile']
    # testfile = args['--testfile']

    # model = args['--model']
    max_seq_len = int(args['--max_seq_len'])
    batch_size = int(args['--batch_size'])

    pdb.set_trace()
    trainData = list(dataBertProcess.load_file(trainfile))
    validData = list(dataBertProcess.load_file(validfile))
    # testData = list(dataBertProcess.load_file(testfile, test_flag=True))

    # num_labels = dataBertProcess.get_num_labels()
    # print("Number of labels is {}".format(num_labels))
    # id2Label = dataBertProcess.get_labelmap()
    # id2Label = pd.DataFrame.from_dict(id2Label, columns=['id', 'label'])
    # id2label_file = os.path.join(args['--out_dir'], 'id2label.csv')
    # id2Label.to_csv(id2label_file, index=False)

    trainDataset = model_dataset(trainData, max_seq_len)
    validDataset = model_dataset(validData, max_seq_len)
    # testDataset = model_dataset(testData, MAX_SEQ_LEN, test_flag = True)

    train_args = dict(shuffle = True, batch_size = batch_size)
    valid_args = dict(shuffle = True, batch_size = batch_size)
    # test_args = dict(shuffle = False, batch_size = 32)
    train_dataloader = DataLoader(trainDataset, **train_args)
    valid_dataloader = DataLoader(validDataset, **valid_args)
    # test_dataloader = DataLoader(testDataset, **test_args)

    return train_dataloader, valid_dataloader


# Get all of the model's parameters as a list of tuples.
def print_model(model):
    print(model)

    params = list(model.named_parameters())
    print('The BERT model has {:} different named parameters.\n'.format(len(params)))
    print('==== Embedding Layer ====\n')
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

def train(args, device):
    trainfile = args['--trainfile']
    validfile = args['--validfile']
    # testfile = args['--testfile']
    pdb.set_trace()
    model = args['--model']
    max_seq_len = int(args['--max_seq_len'])
    batch_size = int(args['--batch_size'])
    max_epoch = int(args['--max_epoch'])
    tokernizer = BertTokenizer.from_pretrained(model, do_lower_case=True)
    dataBertProcess = data_bert_process(tokenizer=tokernizer, max_seq_len = max_seq_len)

    train_dataloader, valid_dataloader = generate_bert_token(args, dataBertProcess)
    num_labels = dataBertProcess.get_num_labels()
    print("Number of labels is {}".format(num_labels))
    id2Label = dataBertProcess.get_labelmap()
    # id2Label = pd.DataFrame.from_dict(id2Label, columns=['id', 'label'])
    # id2label_file = os.path.join(args['--out_dir'], 'id2label.csv')
    # id2Label.to_csv(id2label_file, index=False)

    pdb.set_trace()
    model = BertForSequenceClassification.from_pretrained(
        model,
        num_labels = num_labels,
        output_attentions = False,
        output_hidden_states = False,
    )
    print_model(model)


    num_training_steps = int(int(args['--lr_decay_epoch']) * len(train_dataloader) / batch_size)

    optimizer = AdamW(model.parameters(), lr = float(args['--lr']))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps= num_training_steps
    )
    criterion = nn.CrossEntropyLoss()

    model_save_path = args['--model_save_path']
    start_time = time.time()
    epoch = 0
    patience = 0
    hist_valid_socre = []
    while True:
        epoch += 1
        train_loss = train_epoch(model = model, dataloader=train_dataloader, optimizer=optimizer,
                                 criterion = criterion, device = device, args = args, scheduler = scheduler)
        print("Epoch: {}, Loss: {}, time elapsed: {2:%.2f}".format(epoch, train_loss, time.time() - start_time))
        print("begin validation ... ", file = sys.stderr)
        valid_loss, valid_acc = valid_epoch(model = model, dataloader = valid_dataloader,
                                            criterion = criterion, device = device)

        is_better = len(hist_valid_socre) == 0 or valid_acc > max(hist_valid_socre)
        hist_valid_socre.append(valid_acc)

        if is_better:
            patience = 0
            print("save currently the best model to {}".format(model_save_path), file = sys.stderr)
            model.save(model_save_path)
            torch.save(optimizer.state_dict(), model_save_path + ".optim")

        elif patience < int(args['--patience']):
            patience +=1
            print("hit patience {}".format(patience))

            if patience == int(args['--patience']):
                print("reach maximum number of num_patience, early stopping")
                exit(0)

        if epoch == int(args['--max_epoch']):
            print('reached maximum number of epochs!', file=sys.stderr)
            exit(0)


def set_seed(args):
    seed = int(args['--seed'])
    pdb.set_trace()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main():
    pdb.set_trace()
    args = docopt(__doc__)

    set_seed(args)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


    if args['train']:
        train(args, device)
    # elif args['perdict']:
    #     predict(args, device)


if __name__ == '__main__':
    main()


