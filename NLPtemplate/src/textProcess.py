
import re
import numpy as np
import pdb

def clean_text(text):
    text = text.lower().strip()
    text = remove_whitespace(text)
    text = remove_html(text)
    text = remove_url(text)
    text = remove_hash(text)
    text = remove_punctucations(text)
    return text


def remove_url(text):
    text = re.sub(r'https?:\/\/.*\/\w*', '', text)
    text = re.sub(r'http(\S)+', '', text)
    text = re.sub(r'http ...', '', text)
    text = re.sub(r'(RT|rt)[ ]*@[ ]*[\S]+', '', text)
    text = re.sub(r'RT[ ]?@', '', text)
    text = re.sub(r'@[\S]+', '', text)
    return text


def remove_punctucations(text):
    text = ' '.join(re.sub("[\.\,\!\?\:\;\-\=\/\|\'\(\']", " ", text).split())
    return text


def remove_html(text):
    text = re.sub(r'\&\w*;', '', text)
    return text


def remove_hash(text):
    text = re.sub(r'#', '', text)
    return text


def remove_whitespace(text):
    text = re.sub(r'\s\s+', '', text)
    text = re.sub(r'[ ]{2, }', ' ', text)
    return text


def cal_loss(y, y_preds, eps = 1e-15, label_size = None):
    pdb.set_trace()
    if len(y.shape) == 1:
        if label_size is None:
            actual = np.zeros((y.shape[0], y_preds.shape[1]))
        else:
            actual = np.zeros((y.shape[0], label_size))
            all_labels = set(list(range(label_size)))
            idx_loss = list(all_labels.difference(set(y)))[0]
            new_preds = np.zeros((y.shape[0], label_size))
            new_preds[:, :idx_loss] = y_preds[:, :idx_loss]
            new_preds[:, idx_loss+1:] = y_preds[:, idx_loss:]
            y_preds = new_preds

        for idx, val in enumerate(y):
            actual[idx, val] = 1

    pdb.set_trace()
    clip = np.clip(y_preds, eps, 1 - eps)
    rows = y.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota
