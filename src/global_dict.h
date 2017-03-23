#ifndef GLOBAL_DICT_H_
#define GLOBAL_DICT_H_

#include <string>
#include <vector>
#include <map>

class GlobalDict {
public:
    
    GlobalDict(const char *);
    ~GlobalDict() {}

    int get_word_count(void) const { return word_count; }
    
    std::map<std::string, double> word_idf_map;
    std::map<std::string, int> word_index_map; 
    std::map<int, std::string> index_word_map;

private:

    void load_template_file(const char *);
    int word_count;
};

#endif  // GLOBAL_DICT_H_
