import os
import pdb
import joblib
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics

from .textProcess import clean_text, cal_loss
from .sklearnModel import log_classify

def train(train_df, valid_df, test_df):

    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df.target.values) + list(valid_df.target.values))
    pdb.set_trace()

    # process target label
    train_df.loc[:, 'target'] = lbl.transform(train_df.target.values)
    valid_df.loc[:, 'target'] = lbl.transform(valid_df.target.values)
    label_size = len(set(train_df["target"].unique() + valid_df["target"].unique()))

    # process text content
    train_df["text"].apply(clean_text)
    valid_df["text"].apply(clean_text)
    test_df["text"].apply(clean_text)
    print(train_df.head())
    print(valid_df.head())

    ctv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), stop_words='english')
    ctv.fit(list(train_df.text.values) + list(valid_df.text.values))

    fin_outputs = None
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X=train_df["text"],
                                                               y=train_df["target"].values)):
        X_train, y_train = train_df["text"].iloc[train_idx], train_df["target"].iloc[train_idx]
        X_valid, y_valid = train_df["text"].iloc[valid_idx], train_df["target"].iloc[valid_idx]

        X_train = ctv.transform(X_train)
        X_valid = ctv.transform(X_valid)

        valid_target = valid_df["target"].values
        valid = ctv.transform(valid_df["text"].values)
        X_test = ctv.transform(test_df["text"].values)

        clf, valid_data, test_data = log_classify(X_train, y_train, X_valid, y_valid,
                                              label_size=label_size, valid_data=valid, test_data=X_test)

        # test validation
        valid_preds = clf.predict_proba(valid_data)
        valid_possibility = np.argmax(valid_preds, axis=1)
        pdb.set_trace()
        valid_score = metrics.f1_score(valid_target, valid_possibility, average="weighted")
        print(f"For Valid_df, F1 score = {valid_score}")
        valid_loss = cal_loss(valid_target, valid_preds, label_size=label_size)
        print(f"Loss = {valid_loss}")

        # joblib.dump(clf, f"model_{fold_idx}.pkl")

        out = clf.predict_proba(test_data)
        test_preds = np.argmax(out, axis=1)

        if fin_outputs is None:
            fin_outputs = test_preds
        else:
            fin_outputs += test_preds

    fin_outputs /= 5
    return fin_outputs