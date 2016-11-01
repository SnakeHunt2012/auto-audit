#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>

#include <math.h>
#include <stdlib.h>
#include <assert.h>

#include "config.h"

#include "common.h"

using namespace std;

DMatrixHandle load_X(SparseMatrix &sparse_matrix)
{
    unsigned long *indptr;
    unsigned int *indices;
    float *data;
    sparse_matrix.get_data(&indptr, &indices, &data);
    
    unsigned int nindptr = sparse_matrix.get_nindptr();
    unsigned int nelem = sparse_matrix.get_nelem();
        
    DMatrixHandle matrix_handel;
    int error_code = XGDMatrixCreateFromCSR(indptr, indices, data, nindptr + 1, nelem, &matrix_handel);
    if (error_code) throw runtime_error("error: XGDMatrixCreateFromCSR failed");
    return matrix_handel;
}

float *load_y(vector<int> &label_vec)
{
    float *label_array = (float *) malloc(sizeof(float) * label_vec.size());
    for (size_t i = 0; i < label_vec.size(); ++i) {
        label_array[i] = (float) label_vec[i];
    }
    return label_array;
}

void load_data_file(const char *data_file, GlobalDict &global_dict, vector<string> &url_vec, SparseMatrix &sparse_matrix, vector<int> &label_vec)
{
    ifstream input(data_file);
    if (!input)
        throw runtime_error("error: unable to open input file: " + string(data_file));

    regex_t record_regex = compile_regex("^(.*)\t(.*)\t(.*)\t(.*)$");
    regex_t image_regex = compile_regex("\\[img\\][^\\[]*\\[/img\\]");
    regex_t br_regex = compile_regex("\\[br\\]");

    qss::segmenter::Segmenter *segmenter;
    load_segmenter("/home/huangjingwen/work/news-content/mod_content/mod_segment/conf/qsegconf.ini", &segmenter);
    //load_segmenter("./qsegconf.ini", &segmenter);

    string line;
    while (getline(input, line)) {
        string tag = regex_search(&record_regex, 1, line);
        string url = regex_search(&record_regex, 2, line);
        string title = regex_search(&record_regex, 3, line);
        string content = regex_search(&record_regex, 4, line);
        content = regex_replace(&image_regex, " ", content);
        content = regex_replace(&br_regex, " ", content);
        if (content.size() > 50000)
            content = content.substr(0, 50000);
        
        // parse label from tag
        int label = atoi(tag.c_str());
        
        vector<string> content_seg_vec;
        vector<string> title_seg_vec;
        segment(segmenter, content, content_seg_vec);
        segment(segmenter, title, title_seg_vec);

        // assemble tf
        int term_count = 0;
        map<string, int> word_tf_map; // term_count_map
        for (vector<string>::const_iterator iter = content_seg_vec.begin(); iter != content_seg_vec.end(); ++iter) {
            map<string, int>::iterator word_tf_iter;
            if ((word_tf_iter = word_tf_map.find(*iter)) == word_tf_map.end())
                word_tf_iter = word_tf_map.insert(word_tf_iter, map<string, int>::value_type(*iter, 0));
            word_tf_iter->second += 1;
            //word_tf_map[*iter] += 1;
            term_count += 1;
        }
        for (vector<string>::const_iterator iter = title_seg_vec.begin(); iter != title_seg_vec.end(); ++iter) {
            map<string, int>::iterator word_tf_iter;
            if ((word_tf_iter = word_tf_map.find(*iter)) == word_tf_map.end())
                word_tf_iter = word_tf_map.insert(word_tf_iter, map<string, int>::value_type(*iter, 0));
            word_tf_iter->second += 10;
            //word_tf_map[*iter] += 10;
            term_count += 10;
        }
        
        // assemble feature (feature -> value)
        map<string, double> feature_value_map;
        for (map<string, int>::const_iterator iter = word_tf_map.begin(); iter != word_tf_map.end(); ++iter) {
            const string &word = iter->first;
            const double &tf = (double)iter->second / term_count; // the result is equla to not divided by term_count for final feature values
            map<string, double>::const_iterator word_idf_iter = global_dict.word_idf_map.find(word);
            if (word_idf_iter != global_dict.word_idf_map.end())
                feature_value_map[word] = tf * word_idf_iter->second;
        }
        normalize(feature_value_map);
        
        // assemble feature (index -> value)
        map<int, double> index_value_map;
        for (map<string, double>::const_iterator feature_value_iter = feature_value_map.begin(); feature_value_iter != feature_value_map.end(); ++feature_value_iter) {
            map<string, int>::const_iterator word_index_iter = global_dict.word_index_map.find(feature_value_iter->first);
            if (word_index_iter != global_dict.word_index_map.end())
                index_value_map[word_index_iter->second] = feature_value_iter->second;
        }
        
        url_vec.push_back(url);
        sparse_matrix.push_back(index_value_map);
        label_vec.push_back(label);
    }
    assert(url_vec.size() == sparse_matrix.get_nindptr());
    assert(label_vec.size() == sparse_matrix.get_nindptr());

    regex_free(&record_regex);
    regex_free(&image_regex);
    regex_free(&br_regex);
}

void load_data_file(const char *data_file, GlobalDict &global_dict, vector<string> &url_vec, vector<vector<string> > &title_vec, vector<vector<string> > &content_vec, vector<string> &tag_vec)
{
    //ifstream input(data_file);
    //if (!input)
    //    throw runtime_error("error: unable to open input file: " + string(data_file));
    //
    //regex_t tag_regex = compile_regex("(<LBL>)(.*)(</LBL>)");
    //regex_t title_regex = compile_regex("(<TITLE>)(.*)(</TITLE>)");
    //regex_t url_regex = compile_regex("(<URL>)(.*)(</URL>)");
    //regex_t content_regex = compile_regex("(<CONTENT>)(.*)(</CONTENT>)");
    //regex_t image_regex = compile_regex("\\[img\\][^\\[]*\\[/img\\]");
    //regex_t br_regex = compile_regex("\\[br\\]");
    //
    //qss::segmenter::Segmenter *segmenter;
    //load_segmenter("/home/huangjingwen/work/news-content/mod_content/mod_segment/conf/qsegconf.ini", &segmenter);
    ////load_segmenter("./qsegconf.ini", &segmenter);
    //
    //string line;
    //while (getline(input, line)) {
    //    string tag = regex_search(&tag_regex, 2, line);
    //    string title = regex_search(&title_regex, 2, line);
    //    string url = regex_search(&url_regex, 2, line);
    //    string content = regex_search(&content_regex, 2, line);
    //    content = regex_replace(&image_regex, " ", content);
    //    content = regex_replace(&br_regex, " ", content);
    //
    //    // parse tag
    //    string::size_type spliter_index = tag.rfind("|");
    //    if (spliter_index != string::npos) {
    //        tag = string(tag, spliter_index + 1, tag.size());
    //    }
    //    tag = global_dict.sub_parent_map[tag];
    //
    //    vector<string> title_seg_vec;
    //    segment(segmenter, title, title_seg_vec);
    //
    //    vector<string> content_seg_vec;
    //    segment(segmenter, content, content_seg_vec);
    //
    //    url_vec.push_back(url);
    //    title_vec.push_back(title_seg_vec);
    //    content_vec.push_back(content_seg_vec);
    //    tag_vec.push_back(tag);
    //}
    //
    //regex_free(&tag_regex);
    //regex_free(&title_regex);
    //regex_free(&url_regex);
    //regex_free(&content_regex);
    //regex_free(&image_regex);
    //regex_free(&br_regex);
}

void load_segmenter(const char *conf_file, qss::segmenter::Segmenter **segmenter)
{
    qss::segmenter::Config::get_instance()->init(conf_file);
    *segmenter = qss::segmenter::CreateSegmenter();
    if (!segmenter)
        throw runtime_error("error: loading segmenter failed");
}

void segment(qss::segmenter::Segmenter *segmenter, const string &line, vector<string> &seg_res)
{
    int buffer_size = line.size() * 2;
    char *buffer = (char *) malloc(sizeof(char) * buffer_size);
    int res_size = segmenter->segmentUtf8(line.c_str(), line.size(), buffer, buffer_size);
    
    stringstream ss(string(buffer, res_size));
    for (string token; ss >> token; seg_res.push_back(token)) ;
    
    free(buffer);
}

void normalize(map<string, double> &feature_value_map)
{
    double feature_norm = 0;
    for (map<string, double>::const_iterator iter = feature_value_map.begin(); iter != feature_value_map.end(); ++iter)
        feature_norm += iter->second * iter->second;
    feature_norm = sqrt(feature_norm);
    for (map<string, double>::iterator iter = feature_value_map.begin(); iter != feature_value_map.end(); ++iter)
        iter->second /= feature_norm;
}

void reduce_word_count(const vector<string> &key_vec, map<string, int> &key_count_map, int weight)
{
    for (vector<string>::const_iterator iter = key_vec.begin(); iter != key_vec.end(); ++iter)
        key_count_map[*iter] += weight;
}

string parse_netloc(const string &url)
{
    regex_t netloc_regex = compile_regex("(//)([^/]*)(/)?");
    string netloc = regex_search(&netloc_regex, 2, url);
    regex_free(&netloc_regex);
    return netloc;
}

void parse_pred(const float *proba_array, int label_count, int *pred, float *proba)
{
    assert(label_count > 1);

    *pred = 0;
    float denominator = 0.0;
    for (size_t offset = 0; offset < label_count; ++offset) {
        denominator += exp(*(proba_array + offset));
        if (*(proba_array + offset) > *(proba_array + *pred))
            *pred = offset;
    }
    *proba = exp(*(proba_array + *pred)) / denominator;
}

regex_t compile_regex(const char *pattern)
{
    regex_t regex;
    int error_code = regcomp(&regex, pattern, REG_EXTENDED);
    if (error_code != 0) {
        size_t length = regerror(error_code, &regex, NULL, 0);
        char *buffer = (char *) malloc(sizeof(char) * length);
        (void) regerror(error_code, &regex, buffer, length);
        string error_message = string(buffer);
        free(buffer);
        throw runtime_error(string("error: unable to compile regex '") + pattern + "', message: " + error_message);
    }
    return regex;
}

string regex_search(const regex_t *regex, int field, const string &line)
{
    regmatch_t match_res[field + 1];
    int error_code = regexec(regex, line.c_str(), field + 1, match_res, 0);
    if (error_code != 0) {
        size_t length = regerror(error_code, regex, NULL, 0);
        char *buffer = (char *) malloc(sizeof(char) * length);
        (void) regerror(error_code, regex, buffer, length);
        string error_message = string(buffer);
        free(buffer);
        throw runtime_error(string("error: unable to execute regex, message: ") + error_message);
    }
    return string(line, match_res[field].rm_so, match_res[field].rm_eo - match_res[field].rm_so);
}

string regex_replace(const regex_t *regex, const string &sub_string, const string &ori_string)
{
    regmatch_t match_res;
    int error_code;
    string res_string = ori_string;
    while ((error_code = regexec(regex, res_string.c_str(), 1, &match_res, 0)) != REG_NOMATCH) {
        if (error_code != 0) {
            size_t length = regerror(error_code, regex, NULL, 0);
            char *buffer = (char *) malloc(sizeof(char) * length);
            (void) regerror(error_code, regex, buffer, length);
            string error_message = string(buffer);
            free(buffer);
            throw runtime_error(string("error: unable to execute regex, message: ") + error_message);
        }
        res_string = string(res_string, 0, match_res.rm_so) + sub_string + string(res_string, match_res.rm_eo, res_string.size() - match_res.rm_eo);
    }
    return res_string;
}

void regex_free(regex_t *regex)
{
    regfree(regex);
}
