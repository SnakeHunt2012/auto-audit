#!/usr/bin/env
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from json import loads
from numpy import array, sum, mean, std
from codecs import open
from argparse import ArgumentParser
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

def main():

    parser = ArgumentParser()
    parser.add_argument("score_file", help = "score_file in json format")
    args = parser.parse_args()
    
    score_file = args.score_file
    with open(score_file, 'r') as fd:
        score_dict = loads(fd.read())
        
    y_train = 1 - array(score_dict["y_train"])
    proba_train = array(score_dict["proba_train"])
    y_validate = 1 - array(score_dict["y_validate"])
    proba_validate = array(score_dict["proba_validate"])

    precision_train, recall_train, threshold_train = precision_recall_curve(y_train, proba_train)
    precision_validate, recall_validate, threshold_validate = precision_recall_curve(y_validate, proba_validate)

    fpr_train, tpr_train, _ = roc_curve(y_train, proba_train)
    fpr_validate, tpr_validate, _ = roc_curve(y_validate, proba_validate)

    precision_train = precision_train.tolist()
    recall_train = recall_train.tolist()
    threshold_train = threshold_train.tolist()

    precision_validate = precision_validate.tolist()
    recall_validate = recall_validate.tolist()
    threshold_validate = threshold_validate.tolist()

    remaining_train = [sum(proba_train > threshold) for threshold in threshold_train]
    remaining_validate = [sum(proba_validate > threshold) for threshold in threshold_validate]
    
    precision_train.pop(-1)
    recall_train.pop(-1)
    precision_validate.pop(-1)
    recall_validate.pop(-1)

    plt.clf()
    plt.plot(recall_train, precision_train, '-', color='g', label="training set")
    plt.plot(recall_validate, precision_validate, '-', color='b', label="validation set")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc = "lower left")
    plt.show()
    
    plt.clf()
    plt.plot(threshold_train, precision_train, '-', color='g', label="training set")
    plt.plot(threshold_validate, precision_validate, '-', color='b', label="validation set")
    plt.xlabel("Threshold")
    plt.ylabel("Precision")
    plt.grid()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc = "lower left")
    plt.show()
    
    #plt.clf()
    #plt.plot(threshold_train, remaining_train, 'g', label="training set")
    #plt.plot(threshold_validate, remaining_validate, 'b', label="validation set")
    #plt.xlabel("Threshold")
    #plt.ylabel("Remaining")
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0, len(y_train)])
    #plt.legend(loc = "lower left")
    #plt.show()

    plt.clf()
    #threshold_validate_mean = mean(threshold_validate)
    #threshold_validate_std = std(threshold_validate)
    plt.plot(threshold_validate, remaining_validate, '-', color='b', label="validation set")
    plt.xlabel("Threshold")
    plt.ylabel("Remaining")
    plt.grid()
    plt.xlim([0.0, 1.0])
    plt.ylim([0, len(y_validate)])
    plt.legend(loc = "lower left")
    plt.show()

    #plt.clf()
    #plt.plot(precision_train, remaining_train, label="Remainin-Precision curve (on training set)")
    #plt.xlabel("Threshold")
    #plt.ylabel("Remaining")
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0, len(y_train)])
    #plt.legend(loc = "lower left")
    #plt.show()
    #
    #plt.clf()
    #plt.plot(precision_validate, remaining_validate, label="Remaining-Precision curve (on validation set)")
    #plt.xlabel("Threshold")
    #plt.ylabel("Remaining")
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0, len(y_validate)])
    #plt.legend(loc = "lower left")
    #plt.show()
    #
    plt.clf()
    plt.plot(fpr_train, tpr_train, '-', color='g', label="training set")
    plt.plot(fpr_validate, tpr_validate, '-', color='b', label="validation set")
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.grid()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc = "lower left")
    plt.show()
    #
    #plt.clf()
    #plt.xlabel("fpr")
    #plt.ylabel("tpr")
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.0])
    #plt.legend(loc = "lower left")
    #plt.show()
    
if __name__ == "__main__":

    main()
