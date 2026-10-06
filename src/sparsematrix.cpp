/*
*  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-08-07
 *  License: MIT
 */
#include "sparsematrix.h"
#include <stdexcept>

MixMatrix::MixMatrix(const int row, const int col, const Real init_val) : _row(row), _col(col), _num(static_cast<size_t>(row) * static_cast<size_t>(col)),
_sparse(false), _val(_num, init_val), _indices_dense(col) {
    for (int i = 0; i < col; ++i) {
        _indices_dense[i] = i;
    }
}

MixMatrix::MixMatrix(const int* x, const int* y, const int num, const int row, const int col, const Real init_val) :
    _row(row), _col(col), _num(num), _sparse(true),
    _val(num, init_val), _indices_csr(num, 0), _ptr_csr(row + 1, 0) {
    std::vector<int> idx(num);
    std::vector<int> x_sort(num);
    for (int i = 0; i < num; ++i) {
        idx[i] = i;
    }
    std::sort(idx.begin(), idx.end(),
        [&](const int i, const int j)
        {
            if (x[i] != x[j])
                return x[i] < x[j];
            return y[i] < y[j];
        });
    for (int i = 0; i < num; ++i) {
        x_sort[i] = x[idx[i]];
        _indices_csr[i] = y[idx[i]];
        _ptr_csr[x[idx[i]] + 1]++;
    }
    for (int i = 0; i < _row; ++i) {
        _ptr_csr[i + 1] += _ptr_csr[i];
    }
}

MixMatrix::MixMatrix(const MixMatrix& other): _row(other._row), _col(other._col), _num(other._num), _sparse(other._sparse) {
    _val = other._val;
    _indices_dense = other._indices_dense;
    _indices_csr = other._indices_csr;
    _ptr_csr = other._ptr_csr;
}

SparseVectorView  MixMatrix::get_row(const int i) {
    if (i < 0 || i >= _row) {
        throw std::out_of_range("row index out of range");
    }
    int length = 0;
    SparseVectorView result;
    if (_sparse) {
        result.data = &_val[_ptr_csr[i]];
        result.indices = &_indices_csr[_ptr_csr[i]];
        result.size = _ptr_csr[i + 1] - _ptr_csr[i];
    }
    else {
        const size_t offset = static_cast<size_t>(i) * static_cast<size_t>(_col);
        result.data = &_val[offset];
        result.indices = _indices_dense.data();
        result.size = _col;
    }
    return result;
}

void MixMatrix::set_values(const MixMatrix& src) {
    if (src._num != _num || src._col != _col || src._row != _row) {
        throw std::out_of_range("number of rows do not match");
    }
    memcpy(_val.data(), src._val.data(), src._num * sizeof(Real));
}

void MixMatrix::transpose() {
    int n = _row;
    int m = _col;
    const int nnz = _ptr_csr[n];
    if (_transpose) {
        n = _col;
        m = _row;
    }
    std::vector<int> new_row_ptr(m + 1, 0);
    std::vector<int> new_col_idx(nnz);
    std::vector<Real> new_values(nnz);
    for (int i = 0; i < nnz; ++i) new_row_ptr[_indices_csr[i] + 1]++;

    for (int i = 0; i < m; ++i) new_row_ptr[i + 1] += new_row_ptr[i];

    std::vector<int> current_pos = new_row_ptr;
    for (int i = 0; i < n; ++i) {
        for (int j = _ptr_csr[i]; j < _ptr_csr[i + 1]; ++j) {
            const int col = _indices_csr[j];
            const int dest = current_pos[col]++;
            new_col_idx[dest] = i;
            new_values[dest] = _val[j];
        }
    }

    _ptr_csr = std::move(new_row_ptr);
    _indices_csr = std::move(new_col_idx);
    _val = std::move(new_values);
    _transpose = !_transpose;
}
