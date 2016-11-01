#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>

#include <argp.h>
#include <assert.h>

#include "xgboost/c_api.h"

#include "common.h"
#include "global_dict.h"
#include "sparse_matrix.h"

using namespace std;

const char *argp_program_version = "auto-audit develop 0.1";
const char *argp_program_bug_address = "<SnakeHunt2012@gmail.com>";

static char prog_doc[] = "Develop and dump model for classification in auto-audit."; /* Program documentation. */
static char args_doc[] = "TEMPLATE_FILE TRAIN_FILE VALIDATE_FILE MODEL_FILE"; /* A description of the arguments we accept. */

/* Keys for options without short-options. */
#define OPT_DEBUG       1
#define OPT_PROFILE     2

/* The options we understand. */
static struct argp_option options[] = {
    {"verbose",     'v',             0,             0, "produce verbose output"},
    {"quite",       'q',             0,             0, "don't produce any output"},
    {"silent",      's',             0,             OPTION_ALIAS},

    {0,0,0,0, "The following options are about output format:" },
    
    {"output",      'o',             "FILE",        0, "output information to FILE instead of standard output"},
    {"debug",       OPT_DEBUG,       0,             0, "output debug information"},
    {"profile",     OPT_PROFILE,     0,             0, "output profile information"},
    
    { 0 }
};

/* Used by main to communicate with parse_opt. */
struct arguments
{
    char *template_file, *train_file, *validate_file, *model_file, *output_file;
    bool verbose, silent, debug, profile;
};

/* Parse a single option. */
static error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    struct arguments *arguments = (struct arguments *) state->input;
    switch (key)
        {
        case 'v':
            arguments->verbose = true;
            break;
        case 'q': case 's':
            arguments->silent = true;
            break;
        case 'o':
            arguments->output_file = arg;
            break;
        case OPT_DEBUG:
            arguments->debug = true;
            break;
        case OPT_PROFILE:
            arguments->profile = true;
            break;
            
        case ARGP_KEY_ARG:
            if (state->arg_num == 0) arguments->template_file = arg;
            if (state->arg_num == 1) arguments->train_file = arg;
            if (state->arg_num == 2) arguments->validate_file = arg;
            if (state->arg_num == 3) arguments->model_file = arg;
            if (state->arg_num >= 4) argp_usage(state);
            break;
            
        case ARGP_KEY_END:
            if (state->arg_num < 4) argp_usage(state);
            break;
            
        case ARGP_KEY_NO_ARGS:
            argp_usage(state);
            break;
            
        default:
            return ARGP_ERR_UNKNOWN;
        }
    return 0;
}

/* Our argp parser. */
static struct argp argp = { options, parse_opt, args_doc, prog_doc };

int main(int argc, char *argv[])
{
    struct arguments arguments;
    arguments.template_file = NULL;
    arguments.train_file = NULL;
    arguments.validate_file = NULL;
    arguments.model_file = NULL;
    arguments.output_file = NULL;
    arguments.verbose = true;
    arguments.silent = false;
    arguments.debug = false;
    arguments.profile = false;
    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    GlobalDict global_dict(arguments.template_file);

    vector<string> url_train, url_validate;
    SparseMatrix matrix_train, matrix_validate;
    vector<int> label_train, label_validate;
    
    load_data_file(arguments.train_file, global_dict, url_train, matrix_train, label_train);
    load_data_file(arguments.validate_file, global_dict, url_validate, matrix_validate, label_validate);

    // debug
    cout << "url_train.size(): " << url_train.size() << endl;
    cout << "url_validate.size(): " << url_validate.size() << endl;
    cout << "matrix_train.get_nindptr(): " << matrix_train.get_nindptr() << endl;
    cout << "matrix_train.get_nelem(): " << matrix_train.get_nelem() << endl;
    cout << "matrix_validate.get_nindptr(): " << matrix_validate.get_nindptr() << endl;
    cout << "matrix_validate.get_nelem(): " << matrix_validate.get_nelem() << endl;
    cout << "label_train.size(): " << label_train.size() << endl;
    cout << "label_validate.size(): " << label_validate.size() << endl;

    DMatrixHandle X_train = load_X(matrix_train), X_validate = load_X(matrix_validate);
    float *y_train = load_y(label_train);

    if (XGDMatrixSetFloatInfo(X_train, "label", y_train, matrix_train.get_nindptr()))
        throw runtime_error("error: XGDMatrixSetUIntInfo failed");

    BoosterHandle classifier;
    XGBoosterCreate(&X_train, 1, &classifier);
    
    // general parameters
    XGBoosterSetParam(classifier, "booster", "gbtree");
    XGBoosterSetParam(classifier, "silent", "0");
    XGBoosterSetParam(classifier, "nthread", "5");
    
    // booster parameters
    XGBoosterSetParam(classifier, "eta", "0.3");
    XGBoosterSetParam(classifier, "min_child_weight", "10");
    //XGBoosterSetParam(classifier, "max_depth", "6");         // ignored if define max_leaf_nodes
    XGBoosterSetParam(classifier, "max_leaf_nodes", "100");   // ignore max_depth
    XGBoosterSetParam(classifier, "gamma", "0");
    XGBoosterSetParam(classifier, "max_delta_step", "0");    // usually not needed
    XGBoosterSetParam(classifier, "sub_sample", "1");        // the fraction of observations to be randomly samples for each tree
    XGBoosterSetParam(classifier, "colsample_bytree", "0.8");  // the fraction of columns to be randomly samples for each tree
    XGBoosterSetParam(classifier, "colsample_bylevel", "0.02"); // the subsample ratio of columns for each split, in each level
    XGBoosterSetParam(classifier, "lambda", "1");  // L1 regularization weight, many data scientists don't use it often
    XGBoosterSetParam(classifier, "alpha", "0");   // L2 regularization weight, used in case of very high dimensionality so that the algorithm runs faster
    XGBoosterSetParam(classifier, "scale_pos_weight", "1");  // a value greater than 0 should be used in case of high class imbalance as it helps in faster convergence

    // learning parameters
    XGBoosterSetParam(classifier, "objective", "multi:softmax");
    XGBoosterSetParam(classifier, "num_class", "2");
    //XGBoosterSetParam(classifier, "eval_metric", "merror"); // default according to objective
    XGBoosterSetParam(classifier, "seed", "0");

    // train
    for (int iter = 0; iter < 400; ++iter) {
        cout << "iter: " << iter << endl;
        XGBoosterUpdateOneIter(classifier, iter, X_train);
    }

    // dump model
    if (XGBoosterSaveModel(classifier, arguments.model_file))
        throw runtime_error("error: dumping model failed");

    // dump validate
    unsigned long y_proba_length_train, y_proba_length_validate;
    const float *y_proba_train, *y_proba_validate;
    unsigned long true_counter, false_counter;
    
    if (XGBoosterPredict(classifier, X_train, 1, 0, &y_proba_length_train, &y_proba_train))
         throw runtime_error("error: XGBoosterPredict failed");
    
    if (XGBoosterPredict(classifier, X_validate, 1, 0, &y_proba_length_validate, &y_proba_validate))
       throw runtime_error("error: XGBoosterPredict failed");
    
    int label_count = 2;
    
    true_counter = 0;
    false_counter = 0;
    for (size_t i = 0; i < y_proba_length_train; i += label_count) {
        int pred;
        float proba;
        parse_pred(y_proba_train + i, label_count, &pred, &proba);

        assert(i % label_count == 0);
        int index = i / label_count;
        
        if (label_train[index] != pred)
            ++false_counter;
        else
            ++true_counter;
        cout << "url: " << url_train[index] << "\t" << "label: " << label_train[index] << "\t" << "y_pred_train: " << pred << "\t" << "proba: " << proba << endl;
    }
    cout << "acc on training set: " << (double) true_counter / (true_counter + false_counter) << endl;

    true_counter = 0;
    false_counter = 0;
    for (size_t i = 0; i < y_proba_length_validate; i += label_count) {
        int pred;
        float proba;
        parse_pred(y_proba_validate + i, label_count, &pred, &proba);

        assert(i % label_count == 0);
        int index = i / label_count;
        
        if (label_validate[index] != pred)
            ++false_counter;
        else
            ++true_counter;
        cout << "url: " << url_validate[index] << "\t" << "label: " << label_validate[index] << "\t" << "y_pred_validate: " << pred << "\t" << "proba: " << proba << endl;
    }
    cout << "acc on validateing set: " << (double) true_counter / (true_counter + false_counter) << endl;

    return 0;
}
