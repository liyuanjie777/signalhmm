/*
*  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-08-07
 *  License: MIT
 */
#include "sparsematrix.h"
#include <stdexcept>

MixMatrix::MixMatrix(const int row, const int col) : _row(row), _col(col), _num(row * col),
_val_dense(row * col, -std::numeric_limits<Real>::infinity()), _indices_dense(col), _sparse(false) {
    for (int i = 0; i < col; ++i) {
        _indices_dense[i] = i;
    }
}

MixMatrix::MixMatrix(const int* x, const int* y, const int num, const int row, const int col) :
    _row(row), _col(col), _val_csr(num, -std::numeric_limits<Real>::infinity()), _indices_csr(num, 0),
    _ptr_csr(row + 1, 0), _num(num), _sparse(true) {
    std::vector<int> idx(num);
    std::vector<int> x_sort(num);
    for (int i = 0; i < num; ++i)
    {
        idx[i] = i;
    }
    std::sort(idx.begin(), idx.end(),
        [&](const int i, const int j)
        {
            if (x[i] != x[j])
                return x[i] < x[j];
            return y[i] < y[j];
        });
    for (int i = 0; i < num; ++i)
    {
        x_sort[i] = x[idx[i]];
        _indices_csr[i] = y[idx[i]];
        _ptr_csr[x[idx[i]] + 1]++;
    }
    for (int i = 0; i < _num; ++i)
    {
        _ptr_csr[i + 1] += _ptr_csr[i];
    }
}

int MixMatrix::get_row(const int i, Real*& val, const int*& icol) {
    if (i >= _row) {
        throw std::out_of_range("index out of range");
    }
    int length = 0;
    if (_sparse) {
        val = &_val_csr[_ptr_csr[i]];
        icol = &_indices_csr[_ptr_csr[i]];
        length = _ptr_csr[i + 1] - _ptr_csr[i];
    }
    else {
        const size_t offset = static_cast<size_t>(i) * static_cast<size_t>(_col);
        val = &_val_dense[offset];
        icol = _indices_dense.data();
        length = _col;
    }
    return length;
}
