#!/usr/bin/env python
# -*- coding: utf-8

from re import compile
from json import loads, dumps, dump
from math import sqrt
from codecs import open
from numpy import log
from jieba import cut, enable_parallel
from urlparse import urlparse
from argparse import ArgumentParser
from progressbar import ProgressBar

def load_template_dict(template_file):

    with open(template_file, 'r') as fd:
        template_dict = loads(fd.read())
    word_idf_dict = dict((key.encode("utf-8"), value) for key, value in template_dict["word_idf_dict"].iteritems())
    word_index_dict = dict((key.encode("utf-8"), value) for key, value in template_dict["word_index_dict"].iteritems())
    assert len(word_idf_dict) == len(word_index_dict)
    return word_idf_dict, word_index_dict

def main():

    parser = ArgumentParser()
    parser.add_argument("url_file", help = "url_file (input)")
    parser.add_argument("content_file", help = "content_file (input)")
    parser.add_argument("template_file", help = "template file to load (input)")
    parser.add_argument("data_file", help = "data_file (output)")
    args = parser.parse_args()
    
    url_file = args.url_file
    content_file = args.content_file
    data_file = args.data_file
    template_file = args.template_file
    
    url_re = compile("(?<=<url:)[^>]*(?=>)")
    content_re = compile("(?<=<content:)[^>]*(?=>)")
    image_sub = compile("\[img\][^\[\]]+\[/img\]")
    br_sub = compile("\[br\]")

    tag_dict = {"0": 0, "1": 1, "2": 1, "3": 0, "4": 1, "5": 1}

    word_idf_dict, word_index_dict = load_template_dict(template_file)
    
    # url_label_dict
    print "aggregating url dict (url -> label) ..."
    url_label_dict = {} # url -> label
    with open(url_file, 'r') as fd:
        for line in fd:
            splited_line = line.strip().split("\t")
            if len(splited_line) != 4:
                continue
            url, email, tag, timestamp = splited_line
            if url not in url_label_dict:
                url_label_dict[url] = tag_dict[tag]
    print "aggregating url dict (url -> label) done."

    # doc_list
    print "aggregating doc list [(url, seg_list)] ..."
    enable_parallel(24)
    doc_list = [] # [(url, seg_list), (url, seg_list), ...]
    with open(content_file, 'r') as fd:
        progress = ProgressBar(maxval = len(url_label_dict)).start()
        counter = 0
        for line in fd:
            if not line.startswith("<flag:0>"):
                continue
            line = line.strip()
            
            result = url_re.search(line)
            if not result:
                continue
            url = result.group(0)

            result = content_re.search(line)
            if not result:
                continue
            content = result.group(0)

            content = image_sub.sub("", content)
            content = br_sub.sub("\n", content)
            seg_list = [seg.encode("utf-8") for seg in cut(content)]

            doc_list.append((url, seg_list))
            
            counter += 1
            if (counter < len(url_label_dict)):
                progress.update(counter)
        progress.finish()
    print "aggregating doc list [(url, seg_list)] done"

    # tfidf
    print "dumping feature ..."
    with open(data_file, 'w') as fd:
        url_list = []
        feature_list = []
        label_list = []
        
        progress = ProgressBar(maxval = len(doc_list)).start()
        counter = 0
        for url, seg_list in doc_list:
            word_tf_dict = {} 
            for word in seg_list:
                if word not in word_tf_dict:
                    word_tf_dict[word] = 0
                word_tf_dict[word] += 1 # word -> word_count
            for word in word_tf_dict:
                word_tf_dict[word] = float(word_tf_dict[word]) / len(seg_list)
                
            feature_dict = {}
            for word in word_tf_dict:
                if (word in word_index_dict) and (word in word_idf_dict):
                    feature_dict[word_index_dict[word]] = word_tf_dict[word] * word_idf_dict[word]
            norm = sqrt(sum([value * value for value in feature_dict.itervalues()]))
            index_value_list = [(index, float(value) / norm) for index, value in feature_dict.iteritems()]

            url_list.append(url)
            feature_list.append(index_value_list)
            label_list.append(url_label_dict[url])
            
            counter += 1
            progress.update(counter)
        progress.finish()
        
        assert len(url_list) == len(feature_list) == len(label_list)
        dump({"url_list": url_list, "feature_list": feature_list, "label_list": label_list, "feature_dim": len(word_index_dict)}, fd)
    print "dumping feature done"
    
if __name__ == "__main__":

    main()

