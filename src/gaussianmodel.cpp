/*
 *  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-08-07
 *  License: MIT
 */
#include "gaussianmodel.h"
#include "mymath.hpp"

GaussianModel::GaussianModel(int dim)
{
    dim_ = dim;
    mean_.resize(dim_, 0.0);
    chol_cov_.resize(dim_ * dim_, 0.0);
    for (int i = 0; i < dim_; ++i)
    {
        for (int j = 0; j < dim_; ++j)
        {
            if (i == j)
                chol_cov_[i * dim_ + j] = 1.0;
        }
    }
    log_norm_ = -Real(dim_) / 2 * std::log(2 * PI);
    weight_ = 0.0;
    sum_w_ = 0.0;
    sum_w2_ = 0.0;
    M2_.resize(dim_ * dim_, 0.0);
    mean_new_.resize(dim_, 0.0);
    return;
}

void GaussianModel::log_emission(const Real* obs, Real* log_probs, int num) const
{
    Eigen::Map<const Mat> L(chol_cov_.data(), dim_, dim_);
    Eigen::Map<const Vec> mu(mean_.data(), dim_);
    for (size_t i = 0; i < num; ++i)
    {
        Eigen::Map<const Vec> x(obs + i * dim_, dim_);
        Vec delta = x - mu;
        Vec y = L.triangularView<Eigen::Lower>().solve(delta);
        log_probs[i] = -0.5 * y.squaredNorm() + log_norm_;
    }

    return;
}

void GaussianModel::score(const Real* obs, Real* distance, int num) const
{
    Eigen::Map<const Mat> L(chol_cov_.data(), dim_, dim_);
    Eigen::Map<const Vec> mu(mean_.data(), dim_);
    for (size_t i = 0; i < num; ++i)
    {
        Eigen::Map<const Vec> x(obs + i * dim_, dim_);
        Vec delta = x - mu;
        Vec y = L.triangularView<Eigen::Lower>().solve(delta);
        distance[i] = 5.0 - y.squaredNorm();
    }

    return;
}

void GaussianModel::reset()
{
    std::fill(mean_.begin(), mean_.end(), 0.0);
    std::fill(chol_cov_.begin(), chol_cov_.end(), 0.0);
    for (int i = 0; i < dim_; ++i)
    {
        for (int j = 0; j < dim_; ++j)
        {
            if (i == j)
                chol_cov_[i * dim_ + j] = 1.0;
        }
    }
    log_norm_ = -Real(dim_) / 2 * std::log(2 * PI);
    weight_ = 0.0;
    sum_w_ = 0.0;
    sum_w2_ = 0.0;
    std::fill(mean_new_.begin(), mean_new_.end(), 0.0);
    std::fill(M2_.begin(), M2_.end(), 0.0);
}

void GaussianModel::update(Real alpha)
{
    Eigen::Map<const Mat> M2(M2_.data(), dim_, dim_);
    Eigen::Map<Vec> mu(mean_.data(), dim_);
    Eigen::Map<Vec> mu_new(mean_new_.data(), dim_);
    Eigen::Map<Mat> L(chol_cov_.data(), dim_, dim_);
    if (sum_w_ <= Real(0))
    {
        reset();
        return;
    }
    const Real n_eff = sum_w_ - (sum_w2_ / sum_w_);
    if (!(n_eff > Real(0)))
    {
        reset();
        return;
    }
    mu = mu_new;
    Mat cov = M2 / n_eff;
    cov = Real(0.5) * (cov + cov.transpose());
    Eigen::LLT<Mat> llt(cov);
    if (llt.info() != Eigen::Success)
    {
        reset();
        return;
    }
    L = llt.matrixL();
    Real log_det = 2.0 * cov.diagonal().array().log().sum();
    log_norm_ = -0.5 * (dim_ * std::log(2.0 * PI) + log_det);
    weight_ = sum_w_;
    sum_w_ = 0.0;
    sum_w2_ = 0.0;
    std::fill(M2_.begin(), M2_.end(), 0.0);
    std::fill(mean_new_.begin(), mean_new_.end(), 0.0);
    return;
}

void GaussianModel::log_accumulate(const Real* x, const Real* gamma, int num)
{
    Eigen::Map<Mat> M2(M2_.data(), dim_, dim_);
    Eigen::Map<Vec> mu(mean_new_.data(), dim_);
    Vec delta;
    for (size_t i = 0; i < num; ++i)
    {
        double val = std::exp(double(gamma[i]));
        sum_w_ += val;
        if (sum_w_ == 0)
            continue;
        sum_w2_ += val * val;
        Eigen::Map<const Vec> data(x + i * dim_, dim_);
        delta = data - mu;
        mu += Real(val / sum_w_) * delta;
        M2.noalias() += val * (delta * (data - mu).transpose());
    }
    return;
}

std::vector<Real> GaussianModel::save()
{
    std::vector<Real> res;
    for (int i = 0; i < dim_; ++i)
    {
        res.push_back(mean_[i]);
    }
    for (int i = 0; i < dim_ * dim_; ++i)
    {
        res.push_back(chol_cov_[i]);
    }
    res.push_back(log_norm_);
    res.push_back(weight_);
    return res;
}

void GaussianModel::setValues(const Real* values, int n)
{
    memcpy(mean_.data(), values, dim_ * sizeof(Real));
    memcpy(chol_cov_.data(), &values[dim_], dim_ * dim_ * sizeof(Real));
    log_norm_ = values[n - 2];
    weight_ = values[n - 1];
}