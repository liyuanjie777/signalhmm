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
    MixMatrix(const int row, const int col);
    MixMatrix(const int* x, const int* y, const int num, const int row, const int col);
    ~MixMatrix();
    int get_row(const int i, Real*& val, const int*& icol);
private:
    const int _row;
    const int _col;
    //sparse matrix csr
    std::vector<Real> _val_csr;
    std::vector<int> _indices_csr;
    std::vector<int> _ptr_csr;
    const int _num;
    //dense matrix
    std::vector<Real> _val_dense;
    std::vector<int> _indices_dense;
    const bool _sparse;
};
