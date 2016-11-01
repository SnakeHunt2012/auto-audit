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

white_url_list = [
        "www.people.com.cn",
        "www.xinhuanet.com",
        "www.news.cn",
        "www.xinhua.org",
        "www.china.com.cn",
        "www.china.org.cn",
        "www.cri.cn",
        "www.chinadaily.com.cn",
        "www.cntv.cn",
        "www.cctv.com",
        "www.cctv.com.cn",
        "www.cntv.com.cn",
        "www.cctv.cn",
        "www.youth.cn",
        "www.cycnet.com",
        "www.cycnet.com",
        "www.taiwan.cn",
        "www.chinataiwan.org",
        "www.tibet.cn",
        "www.cnr.cn",
        "www.chinanews.com",
        "www.chinanews.com.cn",
        "www.cyol.net",
        "www.cyol.com"
]

def duration(timer):

    second = (timer) % 60
    minute = (timer) % 3600 / 60
    hour = (timer) / 3600
    return "%d:%02d:%02d" % (hour, minute, second)

def load_banned_dict(banned_file):

    banned_dict = {}
    with open(banned_file, 'r') as fd:
        banned_dict = loads(fd.read())
    return banned_dict

def load_political_dict(political_file):

    political_dict = {}
    with open(political_file, 'r') as fd:
        political_dict = loads(fd.read())
    return political_dict
    
def main():

    parser = ArgumentParser()
    parser.add_argument("idf_file", help = "idf_dict file in json format")
    parser.add_argument("template_file", help = "template_dict file in json format")
    parser.add_argument("banned_file", help = "banned dict file")
    parser.add_argument("political_file", help = "political dict file in json format")
    parser.add_argument("model_file", help = "model file in pickle format")
    args = parser.parse_args()

    idf_file = args.idf_file
    tempalte_file = args.template_file
    banned_file = args.banned_file
    political_file = args.political_file
    model_file = args.model_file

    enable_parallel(6)

    image_sub = compile("\[img\][^\[\]]+\[/img\]")
    br_sub = compile("\[br\]")

    banned_dict = load_banned_dict(banned_file)
    
    political_dict = load_political_dict(political_file)
    political_name_set = set(name.encode("utf-8") for name in political_dict["name_list"])
    political_verb_set = set(verb.encode("utf-8") for verb in political_dict["verb_list"])
    
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
        banned_score = 0
        for word in seg_list:
            if word.decode("utf-8") in banned_dict:
                banned_score += banned_dict[word.decode("utf-8")]
        url_banned_dict[url] = banned_score

        # political check
        political_name_flag = False
        political_verb_flag = False
        for word in seg_list:
            if word in political_name_set:
                political_name_flag = True
                break
        for word in seg_list:
            if word in political_verb_set:
                political_verb_flag = True
                break
        
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
        banned_score = 0
        if url in url_banned_dict:
            banned_score = url_banned_dict[url]
        label = 0 if (proba > 0.59 and banned_score <= 10 and not (political_name_flag and political_verb_flag)) else 1
        print "%s\t%d" % (url, label)
    timer_total += time() - total_flag
        

if __name__ == "__main__":

    main()
 
