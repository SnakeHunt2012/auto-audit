#!/usr/bin/env python
# -*- coding: utf-8 -*-

from re import compile
from sys import stdin, path
from time import time
from json import loads, dumps
from jieba import cut, enable_parallel
from codecs import open
from pickle import load
from urllib import unquote
from urlparse import urlparse, parse_qs
from argparse import ArgumentParser
from numpy import array
from numpy.linalg import norm
from scipy.sparse import coo_matrix
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler


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
    
class HTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):

        url, title, content = "", "", ""
        query_dict = parse_qs(unquote(self.path))
        if "title" in query_dict:
            title = parse_qs(unquote(self.path))["title"][0]
        if "content" in query_dict:
            content = parse_qs(unquote(self.path))["content"][0]
            content = image_sub.sub("", content)
            content = br_sub.sub("\n", content)
        if "url" in query_dict:
            url = parse_qs(unquote(self.path))["url"][0]

        seg_list = [seg.encode("utf-8") for seg in cut(title + " " + content)]
        
        # banned check
        banned_score = 0
        for word in seg_list:
            if word.decode("utf-8") in banned_dict:
                banned_score += banned_dict[word.decode("utf-8")]

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
        tf_dict = {}
        for word in seg_list:
            if word not in tf_dict:
                tf_dict[word] = 1
            else:
                tf_dict[word] += 1 # word -> word_count
        if len(seg_list) > 0:
            for word in tf_dict:
                tf_dict[word] = float(tf_dict[word]) / len(seg_list) # word -> tf
        
        feature_dict = {}
        for word in tf_dict:
            if (word in word_dict) and (word in idf_dict):
                feature_dict[word_dict[word]] = tf_dict[word] * idf_dict[word]
        feature_norm = norm([feature_dict[key] for key in feature_dict])
        
        row_list = []
        column_list = []
        value_list = []
        for column_index in feature_dict:
            row_list.append(0)
            column_list.append(column_index)
            value_list.append(feature_dict[column_index] / feature_norm)
        
        feature = coo_matrix((value_list, (row_list, column_list)), shape=(1, len(word_list)))
        try:
            proba_test = rf.predict_proba(feature)
        except ValueError, error:
            raise error
        
        res_dict = {"proba": proba_test[0][0], "banned_score": banned_score, "political_name_flag": political_name_flag, "political_verb_flag": political_verb_flag}
        res_str = dumps(res_dict)

        # end
        self.send_response(200)
        self.send_header("Content-type", "text/html;charset=utf-8")
        self.send_header("Content-Length", str(len(res_str)))
        self.end_headers()
        self.wfile.write(res_str)


image_sub = compile("\[img\][^\[\]]+\[/img\]")
br_sub = compile("\[br\]")

idf_file = "./download/idf-new.json"
tempalte_file = "./download/template-new.json"
banned_file = "./download/banned-dict.json"
political_file = "./download/political-dict.json"
model_file = "./download/model-new.pickle"

enable_parallel(6)

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

server_address = ("", 4321)
http_deamon = HTTPServer(server_address, HTTPRequestHandler)
http_deamon.serve_forever()

