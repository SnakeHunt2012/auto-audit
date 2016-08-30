#!/usr/bin/env python
# -*- coding: utf-8 -*-

from json import load, dump
from numpy import mean
from argparse import ArgumentParser

def load_template_dict(template_file):

    with open(template_file, 'r') as fd:
        template_dict = load(fd)
    word_idf_dict = dict((key.encode("utf-8"), value) for key, value in template_dict["word_idf_dict"].iteritems())
    word_index_dict = dict((key.encode("utf-8"), value) for key, value in template_dict["word_index_dict"].iteritems())
    assert len(word_idf_dict) == len(word_index_dict)
    return word_idf_dict

def main():

    parser = ArgumentParser()
    parser.add_argument("template_file_one", help = "template in json format (songbo input)")
    parser.add_argument("template_file_two", help = "template in json format (jingwen input)")
    parser.add_argument("template_file_merge", help = "template in json format (output)")
    args = parser.parse_args()

    template_file_one = args.template_file_one
    template_file_two = args.template_file_two
    template_file_merge = args.template_file_merge

    word_idf_dict_one = load_template_dict(template_file_one)
    word_idf_dict_two = load_template_dict(template_file_two)

    mean_one = mean(word_idf_dict_one.values())
    mean_two = mean(word_idf_dict_two.values())
    max_one = max(word_idf_dict_one.values())
    max_two = max(word_idf_dict_two.values())

    word_idf_dict = dict((key, value) for key, value in word_idf_dict_one.iteritems())
    for word in word_idf_dict_two:
        word_idf_dict[word] = max_one

    word_list = list(word_idf_dict)
    word_index_dict = dict((word_list[index], index) for index in xrange(len(word_list))) # word -> index
    index_word_dict = dict((index, word_list[index]) for index in xrange(len(word_list))) # index -> word

    # check
    for word in word_idf_dict_one:
        assert word in word_idf_dict
    for word in word_idf_dict_two:
        assert word in word_idf_dict

    template_dict = {"word_idf_dict": word_idf_dict,
                     "word_index_dict": word_index_dict,
                     "index_word_dict": index_word_dict}
    with open(template_file_merge, 'w') as fd:
        dump(template_dict, fd, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":

    main()
