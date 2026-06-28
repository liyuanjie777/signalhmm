/*
 *  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-08-07
 *  License: MIT
 */
#include "sparsetransition.h"
#include "mymath.hpp"
#include <algorithm>

SparseTransition::SparseTransition(const int* x, const int* y, const Real* val, int num, int dim)
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

void SparseTransition::getrow(int i, std::vector<int>& id, std::vector<Real>& val) const
{
    id.clear();
    val.clear();
    int start = ptr_csr[i];
    int end = ptr_csr[i + 1];
    for (int col = start; col < end; ++col)
    {
        val.push_back(val_csr[col]);
        id.push_back(indices_csr[col]);
    }
    return;
}

void SparseTransition::getcol(int j, std::vector<int>& id, std::vector<Real>& val) const
{
    id.clear();
    val.clear();
    int start = ptr_csc[j];
    int end = ptr_csc[j + 1];
    for (int col = start; col < end; ++col)
    {
        val.push_back(val_csc[col]);
        id.push_back(indices_csc[col]);
    }
    return;
}

void SparseTransition::mulMV(const Real* idata, Real* odata) const
{
    for (int i = 0; i < dim_; ++i)
    {
        int start = ptr_csr[i];
        int end = ptr_csr[i + 1];
        if (start == end)
        {
            odata[i] = -std::numeric_limits<Real>::infinity();
            continue;
        }
        Real acc = -std::numeric_limits<Real>::infinity();
        for (int col = start; col < end; ++col)
        {
            acc = logsumexp2x(acc, val_csr[col] + idata[indices_csr[col]]);
        }
        odata[i] = acc;
    }
    return;
}
void SparseTransition::mulVM(const Real* idata, Real* odata) const
{
    for (int j = 0; j < dim_; ++j)
    {
        int start = ptr_csc[j];
        int end = ptr_csc[j + 1];
        if (start == end)
        {
            odata[j] = -std::numeric_limits<Real>::infinity();
            continue;
        }
        Real acc = -std::numeric_limits<Real>::infinity();
        for (int col = start; col < end; ++col)
        {
            acc = logsumexp2x(acc, val_csc[col] + idata[indices_csc[col]]);
        }
        odata[j] = acc;
    }
    return;
}

void SparseTransition::epsilon_accumulate(const Real* idata, int n)
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
    return;
}

void SparseTransition::epsilon_mstep(const Real* alpha, const Real* beta, Real* odata) const
{
    std::vector<Real> tmp(num_, -std::numeric_limits<Real>::infinity());
    for (int i = 0; i < dim_; ++i)
    {
        int start = ptr_csr[i];
        int end = ptr_csr[i + 1];
        if (start == end)
            continue;
        for (int col = start; col < end; ++col)
        {
            tmp[col] = val_csr[col] + alpha[i] + beta[indices_csr[col]];
        }
    }
    log_normalize(tmp.data(), tmp.size());
    for (int i = 0; i < num_; ++i)
    {
        odata[i] = logsumexp2x(odata[i], tmp[i]);
    }
    return;
}

void SparseTransition::normalize()
{
    std::vector<Real> res;
    for (int i = 0; i < dim_; ++i)
    {
        res.clear();
        int start = ptr_csr[i];
        int end = ptr_csr[i + 1];
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
        val_csr[i] = logsumexp2x(val_csr[i], std::log(alpha) + cache_[i]);
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

void SparseTransition::setValues(const int* x, const int* y, const Real* val, int num, int dim)
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