/*
 *  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2026-06-27
 *  License: MIT
 */
#include "gaussianmixmodel.h"
#include "mymath.hpp"
#include <algorithm>

GaussianMixModel::GaussianMixModel(int npeak)
{
    npeak_ = npeak;
    means_.resize(npeak_, 0.0);
    for (int i = 0; i < npeak_; ++i) {
        means_[i] = -1.0 + 2.0 / (Real(npeak_ - 1));
    }
    vars_.resize(npeak_, 0.01);
    log_weights_.resize(npeak, std::log(1.0 / Real(npeak_)));
    sum_k_gamma_.resize(npeak_, 0.0);
    sum_k_x_.resize(npeak_, 0.0);
    sum_k_xx_.resize(npeak, 0.0);
    return;
}

void GaussianMixModel::log_emission(const Real* obs, Real* log_probs, int num) const
{
    std::vector<Real> log_vars(npeak_);
    for (int k = 0; k < npeak_; ++k) {
        log_vars[k] = -0.5 * std::log(2 * PI * vars_[k]) + log_weights_[k];
    }
    for (size_t i = 0; i < num; ++i)
    {
        Real log_pdf= -std::numeric_limits<Real>::infinity();
        for (int k = 0; k < npeak_; ++k) {
            Real diff = obs[i] - means_[k];
            Real log_pdf_k = -0.5 * diff * diff / vars_[k] + log_vars[k];
            log_pdf = logsumexp2x(log_pdf, log_pdf_k);
        }
        log_probs[i] = log_pdf;
    }
    return;
}

void GaussianMixModel::score(const Real* obs, Real* distance, int num) const
{
    std::vector<Real> log_vars(npeak_);
    for (int k = 0; k < npeak_; ++k) {
        log_vars[k] = -0.5 * std::log(2 * PI * vars_[k]) + log_weights_[k];
    }
    for (size_t i = 0; i < num; ++i)
    {
        Real log_pdf= -std::numeric_limits<Real>::infinity();
        for (int k = 0; k < npeak_; ++k) {
            Real diff = obs[i] - means_[k];
            Real log_pdf_k = -0.5 * diff * diff / vars_[k] + log_vars[k];
            log_pdf = logsumexp2x(log_pdf, log_pdf_k);
        }
        distance[i] = -log_pdf;
    }
    return;
}

void GaussianMixModel::reset()
{
    std::fill(sum_k_gamma_.begin(), sum_k_gamma_.end(), 0.0);
    std::fill(sum_k_x_.begin(), sum_k_x_.end(), 0.0);
    std::fill(sum_k_xx_.begin(), sum_k_xx_.end(), 0.0);
}

void GaussianMixModel::update(Real alpha)
{
    Real log_alpha = std::log(alpha);
    Real log_beta = std::log(1 - alpha);
    double sum = 0.0;
    for (int k = 0; k < npeak_; ++k) sum += sum_k_gamma_[k];
    for (int k = 0; k < npeak_; ++k) {
        if (sum_k_gamma_[k] > 0.0) {
            Real mu_new = sum_k_x_[k] / sum_k_gamma_[k];
            Real var_new = (sum_k_xx_[k] / sum_k_gamma_[k]) - (mu_new * mu_new);
            var_new = std::max(var_new, Real(1e-9));
            means_[k] = (1.0 - alpha) * means_[k] + alpha * mu_new;
            vars_[k]  = (1.0 - alpha) * vars_[k]  + alpha * var_new;
            log_weights_[k] = logsumexp2x(log_beta + log_weights_[k], log_alpha + Real(sum_k_gamma_[k] / sum));
        }
    }
    reset();
    return;
}; 

void GaussianMixModel::log_accumulate(const Real* x, const Real* gamma, int num)
{
    for (int k=0; k < npeak_; ++k) {
        double var = vars_[k];
        double norm_const = 1.0 / std::sqrt(2.0 * M_PI * var);
        double inv_2var = -0.5 / var;
        double mu = means_[k];
        double weight = std::exp(log_weights_[k]);
        for (size_t i = 0; i < num; ++i) {
            double xval = x[i];
            double diff = xval - mu;
            double pdf_k = norm_const * std::exp(diff * diff * inv_2var);
            double r_ik = std::exp(double(gamma[i])) * (weight * pdf_k); 
            sum_k_gamma_[k] += r_ik;
            sum_k_x_[k] += r_ik * xval;
            sum_k_xx_[k] += r_ik * xval * xval;
        }
    }
    return;
}

std::vector<Real> GaussianMixModel::save()
{
    std::vector<Real> res;
    for (int k = 0; k < npeak_; ++k)
    {
        res.push_back(std::exp(log_weights_[k]));
        res.push_back(means_[k]);
        res.push_back(std::sqrt(vars_[k]));
    }
    return res;
}

void GaussianMixModel::setValues(const Real* values, int n)
{
    for (int k = 0; k < npeak_; ++k)
    {
        log_weights_[k] = std::log(values[k * 3]);
        means_[k] = values[k * 3 + 1];
        vars_[k] = values[k * 3 + 2] * values[k * 3 + 2];
    }
}