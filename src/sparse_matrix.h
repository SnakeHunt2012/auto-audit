#ifndef SPARSE_MATRIX_H_
#define SPARSE_MATRIX_H_

#include <vector>
#include <map>

#include <stdlib.h>

class SparseMatrix {
public:
    
    SparseMatrix()
        : need_recompile(true), indptr(NULL), indices(NULL), data(NULL)
    {}
    ~SparseMatrix() {
        if (indptr) free(indptr);
        if (indices) free(indices);
        if (data) free(data);
    }

    void push_back(const std::map<int, double> &);
    void get_data(unsigned long **, unsigned int **, float **);
    unsigned long get_nindptr() const;
    unsigned long get_nelem() const;

private:

    std::vector<unsigned long>indptr_vec;
    std::vector<unsigned int>indices_vec;
    std::vector<double> data_vec;

    bool need_recompile;

    unsigned long *indptr;
    unsigned int *indices;
    float *data;
};

#endif  // SPARSE_MATRIX_H_
