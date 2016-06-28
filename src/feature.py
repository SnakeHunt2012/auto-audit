#!/usr/bin/env python
# -*- coding: utf-8

from re import compile
from json import dumps
from codecs import open
from numpy import array, zeros, sum, mean, var
from jieba import cut
from argparse import ArgumentParser
from progressbar import ProgressBar

def main():

    parser = ArgumentParser()
    parser.add_argument("correlate_file", help = "correlate_file")
    parser.add_argument("url_file", help = "url_file")
    parser.add_argument("content_file", help = "content_file")
    parser.add_argument("data_file", help = "data_file")
    parser.add_argument("template_file", help = "template_file")
    args = parser.parse_args()
    
    correlate_file = args.correlate_file
    url_file = args.url_file
    content_file = args.content_file
    data_file = args.data_file
    template_file = args.template_file

    url_re = compile("(?<=<url:)[^>]*(?=>)")
    content_re = compile("(?<=<content:)[^>]*(?=>)")
    image_re = compile("(?<=\[img\])[^\[\]]+(?=\[/img\])")
    image_sub = compile("\[img\][^\[\]]+\[/img\]")
    br_split = compile("\[br\]")

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
    index_dict = dict((i, correlate_list[i]) for i in range(len(correlate_list)))

    with open(template_file, 'w') as fd:
        fd.write(dumps(index_dict, indent = 4, ensure_ascii = False))

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

            image_list = []
            result = image_re.findall(content)
            for group in result:
                image_list.append(group)

            content = image_sub.sub("", content)
            sentence_list = [sentence for sentence in br_split.split(content) if not (len(sentence) == 0 or sentence.isspace())]

            if url in url_dict:
                url_dict[url]["image_list"] = image_list
                url_dict[url]["sentence_list"] = sentence_list
    
    with open(data_file, 'w') as fd:
        progress = ProgressBar(maxval = len(url_dict)).start()
        counter = 0
        for url in url_dict:
            if "image_list" in url_dict[url] and "sentence_list" in url_dict[url]:
                feature = [0] * len(correlate_set)
                
                image_list = url_dict[url]["image_list"]
                sentence_list = url_dict[url]["sentence_list"]
                
                # sentence_count
                sentence_count = len(sentence_list)

                # word_count & word_mean & word_var and correlate_feature
                length_list = []
                for sentence in sentence_list:
                    seg_list = cut(sentence)
                    for seg in seg_list:
                        if seg.encode("utf-8") in correlate_set:
                            feature[correlate_dict[seg.encode("utf-8")]] += 1
                    length_list.append(sum([1 for seg in seg_list]))
                word_count = sum(length_list)
                word_mean = mean(length_list)
                word_var = var(length_list)

                # image_count
                image_count = len(image_list)

                
                feature.extend([sentence_count, word_count, word_mean, word_var, image_count])
                data = feature + [tag_dict[url_dict[url]["tag"]]]
                fd.write("%s\t%s\n" % (url, " ".join([str(val) for val in data])))
            counter += 1
            progress.update(counter)
        progress.finish()

if __name__ == "__main__":

    main()

