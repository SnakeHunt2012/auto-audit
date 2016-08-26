#!/da1/huangjingwen/software/anaconda/bin/python
# -*- coding: utf-8 -*-

from re import compile
from sys import stdin, path
from time import time
from json import loads
from jieba import cut, enable_parallel
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

def load_banned_set(banned_file):

    banned_list = []
    with open(banned_file, 'r') as fd:
        banned_list = loads(fd.read())
    banned_set = set(word.encode("utf-8") for word in banned_list)
    return banned_set

def main():

    parser = ArgumentParser()
    parser.add_argument("idf_file", help = "idf_dict file in json format")
    parser.add_argument("template_file", help = "template_dict file in json format")
    parser.add_argument("banned_file", help = "banned dict file")
    parser.add_argument("model_file", help = "model file in pickle format")
    args = parser.parse_args()

    idf_file = args.idf_file
    tempalte_file = args.template_file
    banned_file = args.banned_file
    model_file = args.model_file

    enable_parallel(6)

    image_sub = compile("\[img\][^\[\]]+\[/img\]")
    br_sub = compile("\[br\]")

    banned_set = load_banned_set(banned_file)

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
    row_index = 0
    row_list = []
    column_list = []
    value_list = []
    url_list = []
    url_banned_dict = {}
    
    total_flag = time()
    for line in stdin:
        
        splited_line = line.strip().split("\t")
        #assert len(splited_line) == 3
	if len(splited_line) != 3:
		continue
        url, title, content = splited_line

        local_flag = time()
        content = image_sub.sub("", content)
        content = br_sub.sub("\n", content)
        timer_one += time() - local_flag
        
        # segment
        local_flag = time()
        seg_list = [seg.encode("utf-8") for seg in cut(content)]
        timer_two += time() - local_flag

        # banned check
        banned_count = 0
        for word in seg_list:
            if word in banned_set:
                banned_count += 1
        url_banned_dict[url] = banned_count
        
        local_flag = time()
        tf_dict = {}
        for word in seg_list:
            if word not in tf_dict:
                tf_dict[word] = 1
            else:
                tf_dict[word] += 1 # word -> word_count
        if len(seg_list) > 0:
            for word in tf_dict:
                tf_dict[word] = float(tf_dict[word]) / len(seg_list) # word -> tf
        timer_three += time() - local_flag
        
        local_flag = time()
        feature_dict = {}
        for word in tf_dict:
            if (word in word_dict) and (word in idf_dict):
                feature_dict[word_dict[word]] = tf_dict[word] * idf_dict[word]
        feature_norm = norm([feature_dict[key] for key in feature_dict])
        for column_index in feature_dict:
            row_list.append(row_index)
            column_list.append(column_index)
            value_list.append(feature_dict[column_index] / feature_norm)
        url_list.append(url)
        row_index += 1
        timer_four = time() - local_flag

    local_flag = time()
    feature = coo_matrix((value_list, (row_list, column_list)), shape=(len(url_list), len(word_list)))
    try:
        proba_test = rf.predict_proba(feature)
    except ValueError:
	return
    timer_five += time() - local_flag
    assert len(url_list) == proba_test.shape[0]
    for url, (proba, _) in zip(url_list, proba_test.tolist()):
        banned_count = 0
        if url in url_banned_dict:
            banned_count = url_banned_dict[url]
        label = 0 if (proba > 0.60 and banned_count < 5) else 1
        print "%s\t%d" % (url, label)
    timer_total += time() - total_flag
        
    #print "%s\tone:%.4f\ttwo:%.4f\tthree:%.4f\tfour:%.4f\tfive:%.5f" % (duration(timer_total),
    #                                                                    float(timer_one) / timer_total,
    #                                                                    float(timer_two) / timer_total,
    #                                                                    float(timer_three) / timer_total,
    #                                                                    float(timer_four) / timer_total,
    #                                                                    float(timer_five) / timer_total)

if __name__ == "__main__":

    main()
 
