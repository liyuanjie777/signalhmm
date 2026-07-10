/*
*  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-08-07
 *  License: MIT
 */

#pragma once
#include "variable.h"
#include <vector>

class MixMatrix {
public:
    MixMatrix(const int row, const int col, const Real init_val);
    MixMatrix(const int* x, const int* y, const int num, const int row, const int col, const Real init_val);
    MixMatrix(const MixMatrix& other);
    ~MixMatrix();
    SparseVectorView get_row(const int i);
    void transpose();
    void set_values(const MixMatrix& src);
private:
    const int _row;
    const int _col;
    const size_t _num;
    bool _sparse;
    bool _transpose = false;
    std::vector<Real> _val;
    // sparse matrix csr
    std::vector<int> _indices_csr;
    std::vector<int> _ptr_csr;

    //dense matrix
    std::vector<int> _indices_dense;

};
