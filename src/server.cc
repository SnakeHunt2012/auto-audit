#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <set>

#include <stdio.h>
#include <stdlib.h>
#include <syslog.h>
#include <string.h>

#include <fcgi_stdio.h>
#include <curl/curl.h>

#include "common.h"
#include "classifier.h"

using namespace std;

string parse_query(CURL *, const regex_t *, const char *);

void load_white_url_set(const char *white_url_file, set<string> &white_url_set);
void load_pornographic_set(const char *pornographic_file, set<string> &pornographic_set);
void load_sensitive_map(const char *sensitive_file, map<string, int> &sensitive_map);
void load_political_set(const char *political_name_file, const char *political_verb_file, set<string> &political_name_set, set<string> &political_verb_set);

bool white_url_check(const set<string> &white_url_set, const string &url, string &message);
bool pornographic_check(const set<string> &porngraphic_set, const vector<string> &title_seg_vec, const vector<string> &content_seg_vec, string &message);
bool sensitive_check(const map<string, int> &sensitive_map, const vector<string> &title_seg_vec, const vector<string> &content_seg_vec, string &message);
bool political_check(const set<string> &political_name_set, const set<string> &political_verb_set, const vector<string> &title_seg_vec, string &message);
void model_check(Classifier &classifier, const vector<string> &title_seg_vec, const vector<string> &content_seg_vec, int *pred, float *proba);

int main()
{
    CURL *curl = curl_easy_init();
    regex_t url_regex = compile_regex("url=([^&]*)(&|$)");
    regex_t title_regex = compile_regex("title=([^&]*)(&|$)");
    regex_t content_regex = compile_regex("content=([^&]*)(&|$)");
    qss::segmenter::Segmenter *segmenter;

    const char *log_file = "server-error.log";
    const char *template_file = "./dict/template-dict.json";
    const char *model_file = "./model";
    const char *segment_config_file = "/home/huangjingwen/work/auto-audit/qmodule/segment-2.2.1/conf/qsegconf.ini";
    const char *white_url_file = "./download/white-url-dict.tsv";
    const char *pornographic_file = "./download/pornographic-dict.tsv";
    const char *sensitive_file = "./download/sensitive-dict.tsv";
    const char *political_name_file = "./download/political-name-dict.tsv";
    const char *political_verb_file = "./download/political-verb-dict.tsv";

    ofstream info_log(log_file);
    Classifier classifier(template_file, model_file);
    load_segmenter(segment_config_file, &segmenter);

    set<string> white_url_set;
    load_white_url_set(white_url_file, white_url_set);

    set<string> pornographic_set;
    load_pornographic_set(pornographic_file, pornographic_set);

    map<string, int> sensitive_map;
    load_sensitive_map(sensitive_file, sensitive_map);

    set<string> political_name_set, political_verb_set;
    load_political_set(political_name_file, political_verb_file, political_name_set, political_verb_set);

    while (FCGI_Accept() >= 0)
    {
        char *query = getenv("QUERY_STRING");
        string url = parse_query(curl, &url_regex, query);
        string title = parse_query(curl, &title_regex, query);
        string content = parse_query(curl, &content_regex, query);

        vector<string> title_seg_vec, content_seg_vec;
        segment(segmenter, title, title_seg_vec);
        segment(segmenter, content, content_seg_vec);

        string white_url_check_message;
        bool white_url_check_flag = white_url_check(white_url_set, url, white_url_check_message);
        if (white_url_check_flag) {
            printf("Content-type:application/json\r\n\r\n");
            printf("{\"pred\": 0, \"message\": \"%s\"}", white_url_check_message.c_str());
            info_log << "{\"url\": \"" << url << "\", \"pred\": 0, \"message\": \"}" << white_url_check_message <<  "\"}" << endl;
            continue;
        }

        string pornographic_check_message;
        bool pornographic_check_flag = pornographic_check(pornographic_set, title_seg_vec, content_seg_vec, pornographic_check_message);
        if (pornographic_check_flag) {
            printf("Content-type:application/json\r\n\r\n");
            printf("{\"pred\": 1, \"message\": \"%s\"}", pornographic_check_message.c_str());
            info_log << "{\"url\": \"" << url << "\", \"pred\": 0, \"message\": \"}" << white_url_check_message <<  "\"}" << endl;
            continue;
        }

        string sensitive_check_message;
        bool sensitive_check_flag = sensitive_check(sensitive_map, title_seg_vec, content_seg_vec, sensitive_check_message);
        if (sensitive_check_flag) {
            printf("Content-type:application/json\r\n\r\n");
            printf("{\"pred\": 1, \"message\": \"%s\"}", sensitive_check_message.c_str());
            info_log << "{\"url\": \"" << url << "\", \"pred\": 0, \"message\": \"}" << white_url_check_message <<  "\"}" << endl;
            continue;
        }

        string political_check_message;
        bool political_check_flag = political_check(political_name_set, political_verb_set, title_seg_vec, political_check_message);
        if (political_check_flag) {
            printf("Content-type:application/json\r\n\r\n");
            printf("{\"pred\": 1, \"message\": \"%s\"}", political_check_message.c_str());
            info_log << "{\"url\": \"" << url << "\", \"pred\": 0, \"message\": \"}" << white_url_check_message <<  "\"}" << endl;
            continue;
        }

        int pred; float proba;
        model_check(classifier, title_seg_vec, content_seg_vec, &pred, &proba);
        if (pred == 1) {
            proba = 1 - proba;
        }
        pred = proba > 0.1 ? 0 : 1;

        printf("Content-type:application/json\r\n\r\n");
        printf("{\"pred\": \"%d\", \"message\": \"%f\"}", pred, proba);
        info_log << "{\"url\": \"" << url << "\", \"pred\": 0, \"message\": \"}" << white_url_check_message <<  "\"}" << endl;
        //ostringstream oss;
        //for (vector<string>::const_iterator iter = title_seg_vec.begin(); iter != title_seg_vec.end(); ++iter) {
        //    oss << *iter;
        //    oss << " ";
        //}
        //printf("{\"query\":\"%s\", \"url\":\"%s\", \"title\":\"%s\", \"content\":\"%s\", \"pred\": \"%d\", \"proba\": \"%f\", \"res\": \"%s\"}",
        //       query, url.c_str(), title.c_str(), content.c_str(), pred, proba, oss.str().c_str());
    }

    info_log.close();

    regex_free(&url_regex);
    regex_free(&title_regex);
    regex_free(&content_regex);
    return 0;
}

void load_white_url_set(const char *white_url_file, set<string> &white_url_set)
{
    ifstream file(white_url_file);
    
    string line;
    while (getline(file, line))
        white_url_set.insert(line);
    
    file.close();
}

void load_pornographic_set(const char *pornographic_file, set<string> &pornographic_set)
{
    ifstream file(pornographic_file);

    string line;
    while (getline(file, line))
        pornographic_set.insert(line);
    
    file.close();
}

void load_sensitive_map(const char *sensitive_file, map<string, int> &sensitive_map)
{
    regex_t regex = compile_regex("^([^\t]*)\t([^\t]*)$");
    ifstream file(sensitive_file);

    string line;
    while (getline(file, line)) {
        string sensitive_word = regex_search(&regex, 1, line);
        string sensitive_score = regex_search(&regex, 2, line);
        sensitive_map[sensitive_word] = atoi(sensitive_score.c_str());
    }
    
    file.close();
    regex_free(&regex);
}

void load_political_set(const char *political_name_file, const char *political_verb_file, set<string> &political_name_set, set<string> &political_verb_set)
{
    ifstream name_file(political_name_file), verb_file(political_verb_file);

    string line;
    while (getline(name_file, line))
        political_name_set.insert(line);
    while (getline(verb_file, line))
        political_verb_set.insert(line);

    name_file.close();
    verb_file.close();
}

bool white_url_check(const set<string> &white_url_set, const string &url, string &message)
{
    string netloc = parse_netloc(url);
    set<string>::const_iterator iter = white_url_set.find(netloc);
    if(iter != white_url_set.end()) {
        message = "netloc in white url list";
        return true;
    }
    return false;
}

bool pornographic_check(const set<string> &pornographic_set, const vector<string> &title_seg_vec, const vector<string> &content_seg_vec, string &message)
{
    for (vector<string>::const_iterator seg_iter = title_seg_vec.begin(); seg_iter != title_seg_vec.end(); ++seg_iter) {
        set<string>::const_iterator pornographic_iter = pornographic_set.find(*seg_iter);
        if (pornographic_iter != pornographic_set.end()) {
            message = "porongraphie word in title: ";
            message += *pornographic_iter;
            return true;
        }
    }
    for (vector<string>::const_iterator seg_iter = content_seg_vec.begin(); seg_iter != content_seg_vec.end(); ++seg_iter) {
        set<string>::const_iterator pornographic_iter = pornographic_set.find(*seg_iter);
        if (pornographic_iter != pornographic_set.end()) {
            message = "porongraphie word in content: ";
            message += *pornographic_iter;
            return true;
        }
    }
    return false;
}

bool sensitive_check(const map<string, int> &sensitive_map, const vector<string> &title_seg_vec, const vector<string> &content_seg_vec, string &message)
{
    ostringstream oss;
    oss << "sensitive words:";
    int sensitive_counter = 0;
    for (vector<string>::const_iterator seg_iter = title_seg_vec.begin(); seg_iter != title_seg_vec.end(); ++seg_iter) {
        map<string, int>::const_iterator sensitive_iter = sensitive_map.find(*seg_iter);
        if (sensitive_iter != sensitive_map.end()) {
            sensitive_counter += sensitive_iter->second;
            oss << " ";
            oss << sensitive_iter->first;
        }
    }
    for (vector<string>::const_iterator seg_iter = content_seg_vec.begin(); seg_iter != content_seg_vec.end(); ++seg_iter) {
        map<string, int>::const_iterator sensitive_iter = sensitive_map.find(*seg_iter);
        if (sensitive_iter != sensitive_map.end()) {
            sensitive_counter += sensitive_iter->second;
            oss << " ";
            oss << sensitive_iter->first;
        }
    }
    if (sensitive_counter >= 30) {
        oss << ", totaly score: ";
        oss << sensitive_counter;
        message = oss.str();
        return true;
    }
    return false;
}

bool political_check(const set<string> &political_name_set, const set<string> &political_verb_set, const vector<string> &title_seg_vec, string &message)
{
    ostringstream oss;
    bool political_name_flag = false, political_verb_flag = false;
    for (vector<string>::const_iterator seg_iter = title_seg_vec.begin(); seg_iter != title_seg_vec.end(); ++seg_iter) {
        set<string>::const_iterator political_iter = political_name_set.find(*seg_iter);
        if (political_iter != political_name_set.end()) {
            political_name_flag = true;
            oss << "political name: " << *political_iter;
            break;
        }
    }
    for (vector<string>::const_iterator seg_iter = title_seg_vec.begin(); seg_iter != title_seg_vec.end(); ++seg_iter) {
        set<string>::const_iterator political_iter = political_verb_set.find(*seg_iter);
        if (political_iter != political_verb_set.end()) {
            political_verb_flag = true;
            oss << ", " << "political verb: " << *political_iter;
            break;
        }
    }
    //if (political_name_flag && political_verb_flag) {
    //    message = oss.str();
    //    return true;
    //}
    if (political_name_flag) {
        message = oss.str();
        return true;
    }
    return false;
}

void model_check(Classifier &classifier, const vector<string> &title_seg_vec, const vector<string> &content_seg_vec, int *pred, float *proba)
{
    map<string, int> title_reduce_map, content_reduce_map;
    reduce_word_count(title_seg_vec, title_reduce_map, 10);
    reduce_word_count(content_seg_vec, content_reduce_map, 1);
    classifier.classify(title_reduce_map, content_reduce_map, pred, proba);
}

string parse_query(CURL *curl, const regex_t *regex, const char *query)
{
    int length;
    string encoded_str = regex_search(regex, 1, query);
    char *decoded_arr = curl_easy_unescape(curl, encoded_str.c_str(), strlen(encoded_str.c_str()), &length);
    string decoded_str(decoded_arr);
    free(decoded_arr);
    return decoded_str;
}
