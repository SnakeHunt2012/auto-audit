#include "sparse_matrix.h"

using namespace std;

void SparseMatrix::push_back(const map<int, double> &index_value_map)
{
    need_recompile = true;
    indptr_vec.push_back(data_vec.size());
    for (map<int, double>::const_iterator iter = index_value_map.begin(); iter != index_value_map.end(); ++iter) {
        indices_vec.push_back(iter->first);
        data_vec.push_back(iter->second);
    }
}

void SparseMatrix::get_data(unsigned long **indptr, unsigned int **indices, float **data)
{
    if (need_recompile) {
        if (this->indptr) free(this->indptr);
        if (this->indices) free(this->indices);
        if (this->data) free(this->data);
        
        this->indptr = (unsigned long *) malloc(sizeof(unsigned long) * indptr_vec.size() + 1);
        for (size_t i = 0; i < indptr_vec.size(); ++i)
            this->indptr[i] = indptr_vec[i];
        this->indptr[indptr_vec.size()] = data_vec.size();
        
        this->indices = (unsigned int *) malloc(sizeof(unsigned int) * indices_vec.size());
        for (size_t i = 0; i < indices_vec.size(); ++i)
            this->indices[i] = indices_vec[i];
        
        this->data = (float *) malloc(sizeof(float) * data_vec.size());
        for (size_t i = 0; i < data_vec.size(); ++i)
            this->data[i] = data_vec[i];

        need_recompile = false;
    }
    
    *indptr = this->indptr;
    *indices = this->indices;
    *data = this->data;
}

unsigned long SparseMatrix::get_nindptr() const
{
    return indptr_vec.size();
}

unsigned long SparseMatrix::get_nelem() const
{
    return data_vec.size();
}

