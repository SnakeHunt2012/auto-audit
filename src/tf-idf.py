#!/usr/bin/env python
# -*- coding: utf-8

from re import compile
from json import dumps
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
    good_netloc_dict = {
        "finance.chinanews.com": 198, 
        "finance.sina.cn": 2370, 
        "finance.ifeng.com": 635, 
        "finance.sina.com.cn": 1379, 
        "finance.china.com.cn": 102, 
        "finance.eastmoney.com": 729, 
        "news.hexun.com": 798, 
        "sports.ifeng.com": 898, 
        "sports.eastday.com": 172, 
        "sports.qq.com": 868, 
        "sports.163.com": 2119, 
        "we.sportscn.com": 208, 
        "slide.sports.sina.com.cn": 320, 
        "sports.sina.com.cn": 3109, 
        "pic.sports.sohu.com": 264, 
        "sports.sohu.com": 788, 
        "sports.sina.cn": 1276,
        "www.dongqiudi.com": 1331,
        "tech.hexun.com": 935, 
        "tech.huanqiu.com": 293, 
        "tech.163.com": 356, 
        "tech.ifeng.com": 460, 
        "m.techweb.com.cn": 533, 
        "tech.sina.cn": 165, 
        "tech.gmw.cn": 227, 
        "www.techweb.com.cn": 201, 
        "tech.sina.com.cn": 545, 
        "tech.china.com": 611, 
        "tech.qq.com": 160,
        "www.ifanr.com": 165,
        "www.leiphone.com": 122,
        "www.cnsoftnews.com": 315, 
        "news.mydrivers.com": 785, 
        "news.qudong.com": 148, 
        "www.cnsoftnews.com": 315, 
    }
    bad_netloc_dict = {
        "news.cnr.cn": 232, 
        "news.163.com": 6810, 
        "www.chinanews.com": 549, 
        "news.sina.cn": 775, 
        "news.youth.cn": 483, 
        "news.cyol.com": 227, 
        "news.vdfly.com": 113, 
        "news.focus.cn": 128, 
        "news.dahe.cn": 415, 
        "news.ubetween.com": 116, 
        "news.ifeng.com": 3588, 
        "news.szzaix.com": 145, 
        "news.k618.cn": 452, 
        "news.southcn.com": 299, 
        "news.gmw.cn": 929, 
        "cnews.chinadaily.com.cn": 353, 
        "news.china.com.cn": 101, 
        "news.china.com": 248, 
        "news.xinhuanet.com": 1249, 
        "news.qq.com": 201, 
        "news.bitauto.com": 364, 
        "news.7k7k.com": 104, 
        "news.sina.com.cn": 2367, 
        "m.news.cn": 184, 
        "news.wmxa.cn": 108, 
        "xhpfm.news.zhongguowangshi.com:8091": 184, 
        "news.e23.cn": 403, 
        "mil.news.sina.com.cn": 248, 
        "news.eastday.com": 181, 
        "pic.news.sohu.com": 117, 
        "news.sohu.com": 871, 
    }

    good_netloc_set = set(good_netloc_dict)
    bad_netloc_set = set(bad_netloc_dict)

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
            progress.update(counter)
        progress.finish()
    print "aggregating doc list [(url, seg_list)] done"

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

    print "aggregating df dict (word -> idf) ..."
    idf_dict = {}
    
    progress = ProgressBar(maxval = len(df_dict)).start()
    counter = 0
    for word in df_dict:
        if df_dict[word] > 100:
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
        progress = ProgressBar(maxval = len(doc_list)).start()
        counter = 0
        for url, seg_list in doc_list:
            tag = None
            if urlparse(url).netloc in good_netloc_set:
                tag = 0
            elif urlparse(url).netloc in bad_netloc_set:
                tag = 1
            if tag is None:
                continue
                
            tf_dict = {} 
            for word in seg_list:
                if word not in tf_dict:
                    tf_dict[word] = 0
                tf_dict[word] += 1 # word -> word_count
            for word in tf_dict:
                tf_dict[word] = float(tf_dict[word]) / len(seg_list)
                
            feature = [0] * len(word_list)
            for word in tf_dict:
                if (word in word_dict) and (word in idf_dict):
                    feature[word_dict[word]] = tf_dict[word] * idf_dict[word]

            fd.write("%s\t%s %d\n" % (url, " ".join([str(value) for value in feature]), tag))
            counter += 1
            progress.update(counter)
        progress.finish()
    print "dumping feature done"
    
if __name__ == "__main__":

    main()

