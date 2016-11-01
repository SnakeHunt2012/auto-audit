#include <iostream>
#include <fstream>
#include <stdexcept>

#include <stdlib.h>

#include "json/json.h"
#include "json/json-forwards.h"

#include "global_dict.h"

using namespace std;

GlobalDict::GlobalDict(const char *template_file)
    : word_count(0)
{
    load_template_file(template_file);
}

void GlobalDict::load_template_file(const char *template_file)
{
    Json::Value template_dict;
    ifstream input(template_file);
    if (!input)
        throw runtime_error("error: unable to open input file: " + string(template_file));
    input >> template_dict;

    const Json::Value word_idf_dict = template_dict["word_idf_dict"];
    const Json::Value word_index_dict = template_dict["word_index_dict"];
    const Json::Value index_word_dict = template_dict["index_word_dict"];

    vector<string> key_vec;

    key_vec = word_idf_dict.getMemberNames();
    for (vector<string>::const_iterator iter = key_vec.begin(); iter != key_vec.end(); ++iter)
        word_idf_map[*iter] = atof(word_idf_dict[*iter].asString().c_str());

    key_vec = word_index_dict.getMemberNames();
    for (vector<string>::const_iterator iter = key_vec.begin(); iter != key_vec.end(); ++iter)
        word_index_map[*iter] = word_index_dict[*iter].asInt();

    key_vec = index_word_dict.getMemberNames();
    for (vector<string>::const_iterator iter = key_vec.begin(); iter != key_vec.end(); ++iter)
        index_word_map[atoi(((string) (*iter)).c_str())] = index_word_dict[*iter].asString();

    word_count = 0;
    for (map<int, string>::const_iterator iter = index_word_map.begin(); iter != index_word_map.end(); ++iter)
        ++word_count;
}
