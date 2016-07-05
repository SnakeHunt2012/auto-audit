#!/usr/bin/env python
# -*- coding: utf-8 -*-

from re import compile
from sys import stdin
from time import time
from json import loads
from jieba import cut
from codecs import open
from pickle import load
from argparse import ArgumentParser
from numpy import array
from numpy.linalg import norm
from scipy.sparse import coo_matrix

def duration(timer):

    second = (timer) % 60
    minute = (timer) % 3600 / 60
    hour = (timer) / 3600
    return "%d:%02d:%02d" % (hour, minute, second)
                

def main():

    parser = ArgumentParser()
    parser.add_argument("idf_file", help = "idf_dict file in json format")
    parser.add_argument("template_file", help = "template_dict file in json format")
    parser.add_argument("model_file", help = "model file in pickle format")
    args = parser.parse_args()

    idf_file = args.idf_file
    tempalte_file = args.template_file
    model_file = args.model_file

    image_sub = compile("\[img\][^\[\]]+\[/img\]")
    br_sub = compile("\[br\]")

    with open(idf_file, 'r') as fd:
        idf_dict = loads(fd.read())
        idf_dict = dict((key.encode("utf-8"), idf_dict[key]) for key in idf_dict) # word -> idf
    with open(tempalte_file, 'r') as fd:
        index_dict = loads(fd.read())
        index_dict = dict((int(key), index_dict[key].encode("utf-8")) for key in index_dict) # index -> word
    word_dict = dict((index_dict[key], key) for key in index_dict) # word -> index
    word_list = [index_dict[index] for index in xrange(len(word_dict))] # [word ...]

    with open(model_file, 'rb') as fd:
        rf = load(fd)


    timer_one = 0
    timer_two = 0
    timer_three = 0
    timer_four = 0
    timer_five = 0
    timer_total = 0
    for line in stdin:
        total_flag = time()
        
        splited_line = line.strip().split("\t")
        assert len(splited_line) == 3
        url, title, content = splited_line

        local_flag = time()
        content = image_sub.sub("", content)
        content = br_sub.sub("\n", content)
        timer_one += time() - local_flag

        local_flag = time()
        seg_list = [seg.encode("utf-8") for seg in cut(content)]
        timer_two += time() - local_flag

        local_flag = time()
        tf_dict = {}
        for word in seg_list:
            if word not in tf_dict:
                tf_dict[word] = 0
            tf_dict[word] += 1 # word -> word_count
        if len(seg_list) > 0:
            for word in tf_dict:
                tf_dict[word] = float(tf_dict[word]) / len(seg_list) # word -> tf
        timer_three += time() - local_flag

        local_flag = time()
        feature = [0] * len(word_list)
        for word in tf_dict:
            if (word in word_dict) and (word in idf_dict):
                feature[word_dict[word]] = tf_dict[word] * idf_dict[word]
        feature_norm = norm(feature)
        if feature_norm > 0:
            feature = [value / feature_norm for value in feature]
        column_list = []
        value_list = []
        for column_index in xrange(len(feature)):
            if feature[column_index] != 0:
                column_list.append(column_index)
                value_list.append(feature[column_index])
        row_list = [0] * len(value_list)
        feature = coo_matrix((value_list, (row_list, column_list)), shape=(1, len(feature)))
        timer_four = time() - local_flag

        local_flag = time()
        proba_test = rf.predict_proba(feature)
        timer_five += time() - local_flag
        print url, proba_test
        
        timer_total += time() - total_flag
    print "%s\tone:%.4f\ttwo:%.4f\tthree:%.4f\tfour:%.4f\tfive:%.5f" % (duration(timer_total),
                                                                        float(timer_one) / timer_total,
                                                                        float(timer_two) / timer_total,
                                                                        float(timer_three) / timer_total,
                                                                        float(timer_four) / timer_total,
                                                                        float(timer_five) / timer_total)
 

if __name__ == "__main__":

    main()
