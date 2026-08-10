/*
 *  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-08-07
 *  License: MIT
 */
#include "histogrammodel.h"
#include "mymath.hpp"

HistogramModel::HistogramModel(const Real* low_val, const Real* high_val, int* bins_num, int dim)
{
    dim_ = dim;
    low_val_.resize(dim_);
    high_val_.resize(dim_);
    strides_.resize(dim_);
    bins_.resize(dim_);
    width_.resize(dim_);
    memcpy(low_val_.data(), low_val, sizeof(Real) * dim_);
    memcpy(high_val_.data(), high_val, sizeof(Real) * dim_);
    memcpy(bins_.data(), bins_num, sizeof(int) * dim_);

    for (int i = 0; i < dim_; ++i)
    {
        width_[i] = (high_val_[i] - low_val_[i]) / static_cast<Real>(bins_[i]);
        strides_[i] = (i == 0) ? 1 : strides_[i - 1] * static_cast<size_t>(bins_[i - 1]);
    }
    total_num_ = size_t(strides_.back()) * bins_.back();
    count_.resize(total_num_, -50.0);
    log_normalize(count_.data(), count_.size());
    count_accumulate_.resize(total_num_, -50.0);
    return;
}

void HistogramModel::log_emission(const Real* obs, Real* log_probs, int num) const
{
    for (int i = 0; i < num; ++i)
    {
        size_t global_idx = 0;
        for (int j = 0; j < dim_; ++j)
        {
            Real x = obs[i * dim_ + j];
            int idx = static_cast<int>(std::floor((x - low_val_[j]) / width_[j]));
            idx = (idx >= bins_[j]) ? bins_[j] - 1 : idx;
            idx = (idx < 0) ? 0 : idx;
            global_idx += idx * strides_[j];
        }
        log_probs[i] = count_[global_idx];
    }
    return;
}

void HistogramModel::score(const Real* obs, Real* log_probs, int num) const
{
    for (int i = 0; i < num; ++i)
    {
        size_t global_idx = 0;
        for (int j = 0; j < dim_; ++j)
        {
            Real x = obs[i * dim_ + j];
            int idx = static_cast<int>(std::floor((x - low_val_[j]) / width_[j]));
            idx = (idx >= bins_[j]) ? bins_[j] - 1 : idx;
            idx = (idx < 0) ? 0 : idx;
            global_idx += idx * strides_[j];
        }
        log_probs[i] = count_[global_idx];
    }
    return;
}

void HistogramModel::update(Real alpha)
{
    log_normalize(count_accumulate_.data(), count_accumulate_.size());
    count_ = count_accumulate_;
    std::fill(count_accumulate_.begin(), count_accumulate_.end(), -50.0);
    return;
}

void HistogramModel::accumulate(const Real* obs, const Real* gamma, int num)
{
    for (int i = 0; i < num; ++i)
    {
        size_t global_idx = 0;
        for (int j = 0; j < dim_; ++j)
        {
            Real x = obs[i * dim_ + j];
            int idx = static_cast<int>(std::floor((x - low_val_[j]) / width_[j]));
            idx = (idx >= bins_[j]) ? bins_[j] - 1 : idx;
            idx = (idx < 0) ? 0 : idx;
            global_idx += idx * strides_[j];
        }
        count_accumulate_[global_idx] = logsumexp2x(count_accumulate_[global_idx], gamma[i]);
    }
    return;
}

std::vector<Real> HistogramModel::save()
{
    std::vector<Real> res = count_;
    for (size_t i = 0; i < res.size(); ++i)
    {
        res[i] = std::exp(res[i]);
    }
    return res;
}