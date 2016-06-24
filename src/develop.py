#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import stdout
from codecs import open
from numpy import array
from argparse import ArgumentParser
from random import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve

def load_data(path):
    
    dim = 0
    record_list = []
    good_label = 0
    bad_label = 1
    good_count = 0
    bad_count = 0
    with open(path, 'r') as fd:
        for line in fd:
            splited_line = line.strip().split()
            assert len(splited_line) > 0
            if dim == 0:
                dim = len(splited_line)
            assert len(splited_line) == dim
            record = [int(val) for val in splited_line]
            if record[-1] == good_label:
                good_count +=1
            if record[-1] == bad_label:
                bad_count += 1
            record_list.append(record)
    assert (good_count + bad_count) == len(record_list)
    
    good_index_list = []
    bad_index_list = []
    for index in range(len(record_list)):
        if record_list[index][-1] == good_label:
            good_index_list.append(index)
        if record_list[index][-1] == bad_label:
            bad_index_list.append(index)
    assert (len(good_index_list) + len(bad_index_list)) == len(record_list)
    assert (len(good_index_list) > len(bad_index_list))

    shuffle(good_index_list)
    index_list = good_index_list[:len(bad_index_list)] + bad_index_list
    shuffle(index_list)
    record_list = [record_list[index] for index in index_list]

    split = int(0.8 * len(record_list))
    record_array = array(record_list)
    X_train, y_train = (record_array[:split, :-1], record_array[:split, -1])
    X_validate, y_validate = (record_array[split:, :-1], record_array[split:, -1])

    return X_train, y_train, X_validate, y_validate

def main():

    parser = ArgumentParser()
    parser.add_argument("data_file", help = "data_file")
    args = parser.parse_args()
    
    data_file = args.data_file

    X_train, y_train, X_validate, y_validate = load_data(data_file)

    rf = RandomForestClassifier(
        n_estimators=10,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='auto',
        max_leaf_nodes=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=48,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None
    )

    rf.fit(X_train, y_train)

    assert rf.classes_.tolist() == [0, 1]
    proba_train = rf.predict_proba(X_train)[:, 0]
    proba_validate = rf.predict_proba(X_validate)[:, 0]

    precision_train, recall_train, threshold_train = precision_recall_curve(y_train, proba_train)
    precision_validate, recall_validate, threshold_validate = precision_recall_curve(y_validate, proba_validate)

    stdout.write("%s\n" % " ".join([str(val) for val in precision_train.tolist()]))
    stdout.write("%s\n" % " ".join([str(val) for val in recall_train.tolist()]))
    stdout.write("%s\n" % " ".join([str(val) for val in threshold_train.tolist()]))
    stdout.write("%s\n" % " ".join([str(val) for val in precision_validate.tolist()]))
    stdout.write("%s\n" % " ".join([str(val) for val in recall_validate.tolist()]))
    stdout.write("%s\n" % " ".join([str(val) for val in threshold_validate.tolist()]))
    
if __name__ == "__main__":
    
    main()
