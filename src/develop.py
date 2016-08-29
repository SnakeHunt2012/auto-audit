#!/usr/bin/env python
# -*- coding: utf-8 -*-

from json import dumps, loads
from codecs import open
from numpy import array
from pickle import dump, load
from urlparse import urlparse
from argparse import ArgumentParser
from random import shuffle
from scipy.sparse import coo_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

def sparse_matrix(url_list, feature_list, label_list, dim, index_list):

    row_list = []
    column_list = []
    value_list = []

    for row_index in xrange(len(index_list)):
        for item in feature_list[index_list[row_index]]:
            assert len(item) == 2
            row_list.append(row_index)
            column_list.append(item[0])
            value_list.append(item[1])
    assert len(row_list) == len(column_list) == len(value_list)

    X_matrix = coo_matrix((value_list, (row_list, column_list)), shape=(len(index_list), dim))
    y_array = array([label_list[index] for index in index_list], dtype="int")

    return [url_list[index] for index in index_list], X_matrix, y_array

#def load_data(path, netloc_dict):
def load_data(path):
    
    good_label = 0
    bad_label = 1
    
    url_list = []
    feature_list = [] # sparse storage: ith row -> [(j, data), (j, data), ... ]
    label_list = []
    
    good_count = 0
    bad_count = 0

    with open(path, 'r') as fd:
        data_dict = loads(fd.read())
        url_list = data_dict["url_list"]
        feature_list = data_dict["feature_list"]
        label_list = data_dict["label_list"]
        feature_dim = data_dict["feature_dim"]
    
    for label in label_list:
        if label == good_label:
            good_count += 1
        if label == bad_label:
            bad_count += 1
    
    print len(url_list) , len(feature_list) , len(label_list) , (good_count + bad_count)
    assert len(url_list) == len(feature_list) == len(label_list) == (good_count + bad_count)
    
    good_index_list = [index for index in xrange(len(label_list)) if label_list[index] == good_label]
    bad_index_list = [index for index in xrange(len(label_list)) if label_list[index] == bad_label]
    assert len(good_index_list) == good_count
    assert len(bad_index_list) == bad_count

    shuffle(good_index_list)
    shuffle(bad_index_list)

    assert good_count > bad_count
    good_index_list = good_index_list[:bad_count]

    index_list = good_index_list + bad_index_list
    shuffle(index_list)

    threshold = int(0.8 * len(index_list))
    train_index_list = index_list[:threshold]
    validate_index_list = index_list[threshold:]

    url_train, X_train, y_train = sparse_matrix(url_list, feature_list, label_list, feature_dim, train_index_list)
    url_validate, X_validate, y_validate = sparse_matrix(url_list, feature_list, label_list, feature_dim, validate_index_list)

    return url_train, X_train, y_train, url_validate, X_validate, y_validate
    
def main():

    parser = ArgumentParser()
    parser.add_argument("data_file", help = "data_file (input)")
    #parser.add_argument("netloc_file", help = "netloc_file (input)")
    parser.add_argument("score_file", help = "score_file (output)")
    parser.add_argument("--load-matrix", help = "matrix file to load (input)")
    parser.add_argument("--dump-matrix", help = "matrix file to dump (output)")
    parser.add_argument("--dump-model", help = "model file to dump (output)")
    args = parser.parse_args()
    
    data_file = args.data_file
    #netloc_file = args.netloc_file
    score_file = args.score_file
    matrix_load_path = args.load_matrix
    matrix_dump_path = args.dump_matrix
    model_dump_path = args.dump_model

    #print "loading url feature dict ..."
    #with open(netloc_file, 'r') as fd:
    #    netloc_dict = loads(fd.read())
    #print "loading url feature dict done"

    print "loading data ..."
    if matrix_load_path:
        with open(matrix_load_path, "rb") as fd:
            url_train, X_train, y_train, url_validate, X_validate, y_validate = load(fd)
    else:
        #X_train, y_train, X_validate, y_validate = load_data(data_file, netloc_dict)
        url_train, X_train, y_train, url_validate, X_validate, y_validate = load_data(data_file)
    print X_train.shape, y_train.shape, X_validate.shape, y_validate.shape
    print "loading data done."

    if matrix_dump_path:
        print "dumping data ..."
        with open(matrix_dump_path, "wb") as fd:
            dump((url_train, X_train, y_train, url_validate, X_validate, y_validate), fd)
        print "dumping data done"

    rf = RandomForestClassifier(
        n_estimators=1000,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=20,
        min_weight_fraction_leaf=0.0,
        max_features='auto',
        max_leaf_nodes=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=50,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None
    )

    print "training ..."
    rf.fit(X_train, y_train)
    print "training done"

    assert rf.classes_.tolist() == [0, 1]

    pred_train = rf.predict(X_train)
    pred_validate = rf.predict(X_validate)
    proba_train = rf.predict_proba(X_train)
    proba_validate = rf.predict_proba(X_validate)

    acc_train = accuracy_score(y_train, pred_train)
    acc_validate = accuracy_score(y_validate, pred_validate)
    auc_train = roc_auc_score(1 - y_train, proba_train[:, 0])
    auc_validate = roc_auc_score(1 - y_validate, proba_validate[:, 0])

    score_dict = {
        "url_train": url_train,
        "y_train": y_train.tolist(),
        "proba_train": proba_train[:, 0].tolist(),
        "url_validate": url_validate,
        "y_validate": y_validate.tolist(),
        "proba_validate": proba_validate[:, 0].tolist(),
        "acc_train": acc_train,
        "acc_validate": acc_validate,
        "auc_train": auc_train,
        "auc_validate": auc_validate,
        "feature_importance": rf.feature_importances_.tolist()
    }
    print "dumping socre ..."
    with open(score_file, 'w') as fd:
        fd.write(dumps(score_dict, indent = 4))
    print "dumping socre done"

    print "dumping model ..."
    with open(model_dump_path, "wb") as fd:
        dump(rf, fd)
    print "dumping model done"
    
if __name__ == "__main__":
    
    main()

