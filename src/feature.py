#!/usr/bin/env python
# -*- coding: utf-8

from re import compile
from json import loads, dumps, dump
from math import sqrt
from codecs import open
from numpy import log
from jieba import cut
from urlparse import urlparse
from argparse import ArgumentParser
from progressbar import ProgressBar

def main():

    parser = ArgumentParser()
    parser.add_argument("url_file", help = "url_file (input)")
    parser.add_argument("content_file", help = "content_file (input)")
    parser.add_argument("data_file", help = "data_file (output)")
    parser.add_argument("--load-idf", help = "idf file to load (input)")
    parser.add_argument("--dump-idf", help = "idf file to dump (output)")
    parser.add_argument("--load-template", help = "idf file to load (input)")
    parser.add_argument("--dump-template", help = "idf file to dump (output)")
    args = parser.parse_args()
    
    url_file = args.url_file
    content_file = args.content_file
    data_file = args.data_file
    load_idf_file = args.load_idf
    dump_idf_file = args.dump_idf
    load_template_file = args.load_template
    dump_template_file = args.dump_template

    url_re = compile("(?<=<url:)[^>]*(?=>)")
    content_re = compile("(?<=<content:)[^>]*(?=>)")
    image_sub = compile("\[img\][^\[\]]+\[/img\]")
    br_sub = compile("\[br\]")

    tag_dict = {"0": 0, "1":1, "2":1, "3":0, "4":1, "5":1}
    
    # url_dict
    print "aggregating url dict (url -> label) ..."
    url_dict = {} # url -> label
    with open(url_file, 'r') as fd:
        for line in fd:
            splited_line = line.strip().split("\t")
            if len(splited_line) != 4:
                continue
            url, email, tag, timestamp = splited_line
            if url not in url_dict:
                url_dict[url] = tag_dict[tag]
    print "aggregating url dict (url -> label) done."

    # doc_list
    print "aggregating doc list [(url, seg_list)] ..."
    doc_list = [] # [(url, seg_list), (url, seg_list), ...]
    with open(content_file, 'r') as fd:
        progress = ProgressBar(maxval = len(url_dict)).start()
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
            if (counter < len(url_dict)):
                progress.update(counter)
        progress.finish()
    print "aggregating doc list [(url, seg_list)] done"

    if load_idf_file and load_template_file:
        with open(load_idf_file, 'r') as fd:
            idf_dict = loads(fd.read())
        idf_dict = dict((key.encode("utf-8"), idf_dict[key]) for key in idf_dict)
        with open(load_template_file, 'r') as fd:
            index_dict = loads(fd.read())
        index_dict = dict((int(key), index_dict[key].encode("utf-8")) for key in index_dict)
        word_dict = dict((index_dict[key], key) for key in index_dict)
        word_list = [index_dict[index] for index in xrange(len(word_dict))]
    else:
        # df_dict
        print "aggregating df dict (word -> df) ..."
        df_dict = {}
        
        progress = ProgressBar(maxval = len(doc_list)).start()
        counter = 0
        for url, seg_list in doc_list:
            word_set = set(seg_list)
            for word in word_set:
                if word not in df_dict:
                    df_dict[word] = 0
                df_dict[word] += 1                                          # word -> doc_count
            counter += 1
            progress.update(counter)
        progress.finish()
        print "aggregating df dict (word -> df) done"
    
        # idf_dict
        print "aggregating df dict (word -> idf) ..."
        idf_dict = {}
        
        progress = ProgressBar(maxval = len(df_dict)).start()
        counter = 0
        for word in df_dict:
            if df_dict[word] > 10:
                idf_dict[word] = log(float(len(doc_list)) / df_dict[word])               # word -> idf
            counter += 1
            progress.update(counter)
        progress.finish()
        word_list = list(idf_dict)
        word_dict = dict((word_list[index], index) for index in xrange(len(word_list)))  # word -> index
        index_dict = dict((index, word_list[index]) for index in xrange(len(word_list))) # index -> word
        print "aggregating idf dict (word -> idf) done"
        
    if dump_idf_file:
        # dump idf dict
        print "dumping idf (word -> idf) ..."
        with open(dump_idf_file, 'w') as fd:
            fd.write(dumps(idf_dict, indent=4, ensure_ascii=False))
        print "dumping idf (word -> idf) done"

    if dump_template_file:
        # dump index->word dict
        print "dumping template (index -> word) ..."
        with open(dump_template_file, 'w') as fd:
            fd.write(dumps(index_dict, indent=4, ensure_ascii=False))
        print "dumping template (index -> word) done"

    # tfidf
    print "dumping feature ..."
    with open(data_file, 'w') as fd:
        url_list = []
        feature_list = []
        label_list = []
        
        progress = ProgressBar(maxval = len(doc_list)).start()
        counter = 0
        for url, seg_list in doc_list:
            tf_dict = {} 
            for word in seg_list:
                if word not in tf_dict:
                    tf_dict[word] = 0
                tf_dict[word] += 1 # word -> word_count
            for word in tf_dict:
                tf_dict[word] = float(tf_dict[word]) / len(seg_list)
                
            feature_dict = {}
            for word in tf_dict:
                if (word in word_dict) and (word in idf_dict):
                    feature_dict[word_dict[word]] = tf_dict[word] * idf_dict[word]
            norm = sqrt(sum([value * value for value in feature_dict.itervalues()]))
            index_value_list = [(index, float(value) / norm) for index, value in feature_dict.iteritems()]

            url_list.append(url)
            feature_list.append(index_value_list)
            label_list.append(url_dict[url])
            
            counter += 1
            progress.update(counter)
        progress.finish()
        
        assert len(url_list) == len(feature_list) == len(label_list)
        dump({"url_list": url_list, "feature_list": feature_list, "label_list": label_list, "feature_dim": len(word_dict)}, fd)
    print "dumping feature done"
    
if __name__ == "__main__":

    main()

