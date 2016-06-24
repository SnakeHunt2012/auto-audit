#!/usr/bin/env python
# -*- coding: utf-8

from sys import stdin, stdout
from codecs import open
from numpy import array, zeros
from re import compile
from argparse import ArgumentParser
from jieba import cut
from progressbar import ProgressBar

def main():

    parser = ArgumentParser()
    parser.add_argument("correlate_file", help = "url_file")
    parser.add_argument("url_file", help = "url_file")
    parser.add_argument("content_file", help = "content_file")
    parser.add_argument("data_file", help = "data_file")
    args = parser.parse_args()
    
    correlate_file = args.correlate_file
    url_file = args.url_file
    content_file = args.content_file
    data_file = args.data_file

    url_re = compile("(?<=<url:)[^>]*(?=>)")
    content_re = compile("(?<=<content:)[^>]*(?=>)")

    tag_dict = {"0": 0, "1":1, "2":1, "3":0, "4":1, "5":1}

    correlate_set = set([])
    url_dict = {}
    feature_list = []
    label_list = []
    
    with open(correlate_file, 'r') as fd:
        for line in fd:
            splited_line = line.strip().split()
            if len(splited_line) != 1:
                continue
            correlate_set.add(splited_line.pop(0))

    correlate_list = list(correlate_set)
    correlate_dict = dict((correlate_list[i], i) for i in range(len(correlate_list)))

    with open(url_file, 'r') as fd:
        for line in fd:
            splited_line = line.strip().split("\t")
            if len(splited_line) != 4:
                continue
            url, email, tag, timestamp = splited_line
            if url not in url_dict:
                url_dict[url] = {"tag": tag}

    with open(content_file, 'r') as fd:
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

            if url in url_dict:
                url_dict[url]["content"] = content
    
    with open(data_file, 'w') as fd:
        progress = ProgressBar(maxval = len(url_dict)).start()
        counter = 1
        for url in url_dict:
            if "content" in url_dict[url]:
                #print url_dict[url]["tag"], url_dict[url]["content"]
                seg_list = cut(url_dict[url]["content"])
                feature = [0] * len(correlate_list)
                for seg in seg_list:
                    if seg.encode("utf-8") in correlate_set:
                        feature[correlate_dict[seg.encode("utf-8")]] += 1
                #feature_list.append(feature)
                #label_list.append(tag_dict[url_dict[url]["tag"]])
                data = feature + [tag_dict[url_dict[url]["tag"]]]
                fd.write("%s\n" % " ".join([str(val) for val in data]))
            counter += 1
            progress.update(counter)
        progress.finish()

if __name__ == "__main__":

    main()

