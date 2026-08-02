/*
 *  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-08-07
 *  License: MIT
 */
#include "sparsetransition.h"
#include "mymath.hpp"
#include <algorithm>

SparseTransition::SparseTransition(const int* x, const int* y, const Real* val, const int num, const int dim)
{
    dim_ = dim;
    num_ = num;
    val_csr.resize(num, 0);
    val_csc.resize(num, 0);
    indices_csr.resize(num, 0);
    indices_csc.resize(num, 0);
    csr_csc_map.resize(num, 0);
    ptr_csr.resize(dim + 1, 0);
    ptr_csc.resize(dim + 1, 0);
    cache_.resize(num, -std::numeric_limits<Real>::infinity());

    std::vector<int> idx(num);
    std::vector<int> x_sort(num);
    for (int i = 0; i < num; ++i)
    {
        idx[i] = i;
    }
    std::sort(idx.begin(), idx.end(),
        [&](size_t i, size_t j)
        {
            if (x[i] != x[j])
                return x[i] < x[j];
            return y[i] < y[j];
        });
    for (int i = 0; i < num; ++i)
    {
        val_csr[i] = std::log(val[idx[i]]);
        x_sort[i] = x[idx[i]];
        indices_csr[i] = y[idx[i]];
        ptr_csr[x[idx[i]] + 1]++;
    }
    for (int i = 0; i < dim; ++i)
    {
        ptr_csr[i + 1] += ptr_csr[i];
    }

    for (int i = 0; i < num; ++i)
    {
        idx[i] = i;
    }
    std::sort(idx.begin(), idx.end(),
        [&](size_t i, size_t j)
        {
            if (indices_csr[i] != indices_csr[j])
                return indices_csr[i] < indices_csr[j];
            return x_sort[i] < x_sort[j];
        });
    for (int i = 0; i < num; ++i)
    {
        val_csc[i] = val_csr[idx[i]];
        indices_csc[i] = x_sort[idx[i]];
        ptr_csc[indices_csr[idx[i]] + 1]++;
        csr_csc_map[idx[i]] = i;
    }
    for (int i = 0; i < dim; ++i)
    {
        ptr_csc[i + 1] += ptr_csc[i];
    }
    normalize();
}

Real SparseTransition::log_transition(int i, int j) const
{
    int start = ptr_csr[i];
    int end = ptr_csr[i + 1];
    Real res = -std::numeric_limits<Real>::infinity();
    for (int col = start; col < end; ++col)
    {
        if (indices_csr[col] == j)
        {
            res = val_csr[col];
            break;
        }
    }
    return res;
}

SparseVectorView SparseTransition::get_row(const int i) {
    if (i >= dim_) {
        throw std::out_of_range("index out of range");
    }
    SparseVectorView result;
    result.data = &val_csr[ptr_csr[i]];
    result.indices = &indices_csr[ptr_csr[i]];
    result.size = ptr_csr[i + 1] - ptr_csr[i];
    return result;
}

SparseVectorView SparseTransition::get_col(const int j)
{
    if (j >= dim_) {
        throw std::out_of_range("index out of range");
    }
    SparseVectorView result;
    result.data = &val_csc[ptr_csc[j]];
    result.indices = &indices_csc[ptr_csc[j]];
    result.size = ptr_csc[j + 1] - ptr_csc[j];
    return result;
}

void SparseTransition::mulMV(const SparseVectorView& idata, Real* odata) const {
    std::fill_n(odata, dim_, -std::numeric_limits<Real>::infinity());
    for (int j = 0; j < idata.size; ++j) {
        const int col= idata.indices[j];
        const int start = ptr_csc[col];
        const int end = ptr_csc[col + 1];
        if (start == end) {
            continue;
        }
        for (int row = start; row < end; ++row) {
            const int i_dense = indices_csc[row];
            odata[i_dense] = logsumexp2x(odata[i_dense], val_csc[row] + idata.data[j]);
        }
    }
}
void SparseTransition::mulVM(const SparseVectorView& idata, Real* odata) const {
    std::fill_n(odata, dim_, -std::numeric_limits<Real>::infinity());
    for (int i = 0; i < idata.size; ++i) {
        const int row = idata.indices[i];
        const int start = ptr_csr[row];
        const int end = ptr_csr[row + 1];
        if (start == end) {
            continue;
        }
        for (int col = start; col < end; ++col) {
            const int j_dense = indices_csr[col];
            odata[j_dense] = logsumexp2x(odata[j_dense], val_csr[col] + idata.data[i]);
        }
    }
}

void SparseTransition::epsilon_M_step(const Real* idata, int n)
{
    for (size_t i = 0; i < num_; ++i)
    {
        std::vector<Real> tmp(n);
        for (size_t j = 0; j < n; ++j)
        {
            tmp[j] = idata[j * num_ + i];
        }
        cache_[i] = logsumexp2x(cache_[i], logsumexp(tmp.data(), n));
    }
}

void SparseTransition::epsilon_E_step(const SparseVectorView& alpha, const SparseVectorView& beta, Real* odata) const
{
    std::vector<Real> tmp(num_, -std::numeric_limits<Real>::infinity());
    std::copy_n(val_csr.data(), num_, tmp.data());
    for (int i = 0;i < alpha.size; ++i) {
        const int i_dense = alpha.indices[i];
        const int start = ptr_csr[i_dense];
        const int end = ptr_csr[i_dense + 1];
        int pointer = start;
        for (int j = 0; j < beta.size; ++j) {
            const int j_dense = beta.indices[j];
            while ((pointer < end) && (indices_csr[pointer] < j_dense)) {
                pointer++;
            }
            if (pointer >= end) {
                break;
            }
            if (indices_csr[pointer] == j_dense) {
                tmp[pointer] += alpha.data[i] + beta.data[j];
            }
        }
    }
    log_normalize(tmp.data(), tmp.size());
    for (int i = 0; i < num_; ++i)
    {
        odata[i] = logsumexp2x(odata[i], tmp[i]);
    }
}

void SparseTransition::normalize()
{
    std::vector<Real> res;
    for (int i = 0; i < dim_; ++i)
    {
        res.clear();
        const int start = ptr_csr[i];
        const int end = ptr_csr[i + 1];
        for (int col = start; col < end; ++col)
        {
            res.push_back(val_csr[col]);
        }
        if (res.size() > 0)
        {
            log_normalize(res.data(), res.size());
            memcpy(&val_csr[start], res.data(), res.size() * sizeof(Real));
        }
    }
    for (int i = 0; i < num_; ++i)
    {
        val_csc[csr_csc_map[i]] = val_csr[i];
    }
}

void SparseTransition::update(Real alpha)
{
    for (size_t i = 0; i < num_; ++i)
    {
        val_csr[i] = logsumexp2x(std::log(Real(1.0) - alpha) + val_csr[i], std::log(alpha) + cache_[i]);
    }
    normalize();
    cache_.assign(num_, -std::numeric_limits<Real>::infinity());
}

void SparseTransition::save(int* x, int* y, Real* val)
{
    int icol = 0;
    for (size_t i = 0; i < num_; ++i)
    {
        y[i] = indices_csr[i];
        val[i] = std::exp(val_csr[i]);
        while (i >= ptr_csr[icol + 1] && icol < dim_)
            icol++;
        x[i] = icol;
    }
}

void SparseTransition::fill(const int* x, const int* y, const Real* val, const int num, const int dim)
{
    dim_ = dim;
    num_ = num;
    val_csr.resize(num, 0);
    val_csc.resize(num, 0);
    indices_csr.resize(num, 0);
    indices_csc.resize(num, 0);
    csr_csc_map.resize(num, 0);
    ptr_csr.resize(dim + 1, 0);
    ptr_csc.resize(dim + 1, 0);
    cache_.resize(num, -std::numeric_limits<Real>::infinity());

    std::vector<int> idx(num);
    std::vector<int> x_sort(num);
    for (int i = 0; i < num; ++i)
    {
        idx[i] = i;
    }
    std::sort(idx.begin(), idx.end(),
        [&](size_t i, size_t j)
        {
            if (x[i] != x[j])
                return x[i] < x[j];
            return y[i] < y[j];
        });
    for (int i = 0; i < num; ++i)
    {
        val_csr[i] = std::log(val[idx[i]]);
        x_sort[i] = x[idx[i]];
        indices_csr[i] = y[idx[i]];
        ptr_csr[x[idx[i]] + 1]++;
    }
    for (int i = 0; i < dim; ++i)
    {
        ptr_csr[i + 1] += ptr_csr[i];
    }

    for (int i = 0; i < num; ++i)
    {
        idx[i] = i;
    }
    std::sort(idx.begin(), idx.end(),
        [&](size_t i, size_t j)
        {
            if (indices_csr[i] != indices_csr[j])
                return indices_csr[i] < indices_csr[j];
            return x_sort[i] < x_sort[j];
        });
    for (int i = 0; i < num; ++i)
    {
        val_csc[i] = val_csr[idx[i]];
        indices_csc[i] = x_sort[idx[i]];
        ptr_csc[indices_csr[idx[i]] + 1]++;
        csr_csc_map[idx[i]] = i;
    }
    for (int i = 0; i < dim; ++i)
    {
        ptr_csc[i + 1] += ptr_csc[i];
    }
    normalize();
}