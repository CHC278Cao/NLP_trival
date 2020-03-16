
import pdb
import numpy as np
import xgboost as xgb
import lightgbm as lgbm

from sklearn import preprocessing, decomposition
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import pipeline

from .textProcess import cal_loss


def log_classify(X_train, y_train, X_valid, y_valid, label_size, valid_data, test_data):
    clf = linear_model.LogisticRegression(C=0.05, multi_class="multinomial", max_iter=1000)

    pdb.set_trace()
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_valid)
    possibility = np.argmax(preds, axis=1)
    f1_score = metrics.f1_score(y_valid, possibility, average="weighted")
    print(f"F1 score = {f1_score}")
    loss = cal_loss(y_valid, preds, label_size=label_size)
    print(f"Loss = {loss}")

    return clf, valid_data, test_data


def naivebayes_classify(X_train, y_train, X_valid, y_valid, label_size, valid_data, test_data):
    clf = MultinomialNB()

    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_valid)
    possibility = np.argmax(preds, axis=1)
    f1_score = metrics.f1_score(y_valid, possibility, average="weighted")
    print(f"F1 score = {f1_score}")
    loss = cal_loss(y_valid, preds, label_size=label_size)
    print(f"Loss = {loss}")

    return clf, valid_data, test_data


def svm_classify(X_train, y_train, X_valid, y_valid, label_size, valid_data, test_data):
    clf = SVC(C=1.0, probability=True)
    svd = decomposition.TruncatedSVD(n_components=350)
    svd.fit(X_train)
    X_train_svd = svd.transform(X_train)
    X_valid_svd = svd.transform(X_valid)
    valid_data = svd.transform(valid_data)
    test_data = svd.transform(test_data)

    scl = preprocessing.StandardScaler()
    scl.fit(X_valid_svd)
    X_train_svd_scl = scl.transform(X_train_svd)
    X_valid_svd_scl = scl.transform(X_valid_svd)
    valid_data = scl.transform(valid_data)
    test_data = scl.transform(test_data)

    clf.fit(X_train_svd_scl, y_train)
    preds = clf.predict_proba(X_valid_svd_scl)
    possibility = np.argmax(preds, axis=1)
    f1_score = metrics.f1_score(y_valid, possibility, average="weighted")
    print(f"F1 score = {f1_score}")
    loss = cal_loss(y_valid, preds, label_size=label_size)
    print(f"Loss = {loss}")

    return clf, valid_data, test_data


def xgb_classify(X_train, y_train, X_valid, y_valid, label_size, valid_data, test_data):
    clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                            subsample=0.8, nthread=10, learning_rate=0.1)

    clf.fit(X_train.tocsc(), y_train)
    preds = clf.predict_proba(X_valid.tocsc())
    possibility = np.argmax(preds, axis=1)
    f1_score = metrics.f1_score(y_valid, possibility, average="weighted")
    print(f"F1 score = {f1_score}")
    loss = cal_loss(y_valid, preds, label_size=label_size)
    print(f"Loss = {loss}")

    return clf, valid_data, test_data


def grid_search(X_train, y_train, X_valid, y_valid, label_size):
    mll_score = metrics.make_scorer(cal_loss, greater_is_better=False, needs_proba=True)

    # make transformer
    svd = decomposition.TruncatedSVD()
    scl = preprocessing.StandardScaler()

    # create model
    lr_model = linear_model.LogisticRegression()

    # create pipeline
    clf = pipeline.Pipeline([('svd', svd),
                             ('scl', scl),
                             ('lr_model', lr_model)])
    params = {
        'svd__n_components': [300, 350, 400],
        'lr_model__C': [0.05, 0.1, 0.5, 1.0],
        'lr_model__penalty': ['l1', 'l2'],
    }

    model = GridSearchCV(estimator=clf, param_grid=params, scoring=mll_score,
                         verbose=100, n_jobs=-1, iid=True, refit=True, cv=5)
    model.fit(X_train, y_train)

    print("Best score: {:.3f}".format(model.best_score_))
    print("Best parameters set: ")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


