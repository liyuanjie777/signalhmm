/*
 *  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2026-06-27
 *  License: MIT
 */
#include "gaussianmixmodel.h"
#include "mymath.hpp"
#include <algorithm>
#include <stdexcept>
#include <Eigen/Dense>
#include <random>

bool computeCholesky(const Real* A_ptr, Real* L_ptr, int D) {
    Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(A_ptr, D, D);
    Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> L(L_ptr, D, D);
    Eigen::LLT<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> llt;
    llt.compute(A);
    if (llt.info() != Eigen::Success) {
        return false;
    }
    L = llt.matrixL();
    return true;
}

GaussianMixModel::GaussianMixModel(const int n_components, const int dim): n_components_(n_components), dim_(dim) {
    const int kd = n_components_ * dim_;
    const int kdd = kd * dim_;
    const int dd = dim_ * dim_;

    means_.resize(kd, 0.0);
    vars_.resize(kdd, 0.0);
    chol_cov_.resize(kdd, 0.0);
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<Real> mean_dist(-2.0, 2.0);
    std::uniform_real_distribution<Real> var_dist(0.14, 0.15);
    for(int k = 0; k < n_components_; k++) {
        Real* mean = means_.data() + k * dim_;
        for(int d = 0; d < dim_; d++) {
            //mean[d] = mean_dist(gen);
            mean[d] = 0.0;
        }
        Real* cov = vars_.data() + k * dd;
        Real* chol = chol_cov_.data() + k * dd;
        for(int i = 0; i < dim_; i++) {
            for(int j = 0; j < dim_; j++) {
                cov[i * dim_ + j] = 0;
                chol[i * dim_ + j] = 0;
            }
        }

        for(int d = 0; d < dim_; d++) {
            Real sigma = var_dist(gen);
            cov[d * dim_ + d] = sigma;
            chol[d * dim_ + d] = std::sqrt(sigma);
        }
    }
    log_weights_.resize(n_components_,std::log(1.0 / static_cast<Real>(n_components_)));
    sum_k_gamma_.resize(n_components_,0.0);
    sum_k_x_.resize(kd,0.0);
    sum_k_xx_.resize(kdd,0.0);
}

void GaussianMixModel::loglikelihood(const Real* obs, Real* log_probs) const {
    const int D = dim_;
    const int K = n_components_;
    const Real log_2pi = std::log(2.0 * M_PI);
    thread_local std::vector<Real> y_cache;
    y_cache.assign(D, 0.0);
    for (int k = 0; k < K; ++k) {
        const Real* L_ptr = chol_cov_.data() + k * (D * D);
        const Real* mu = means_.data() + k * D;
        Real sum_log_diag = 0.0;
        for (int d = 0; d < D; ++d) {
            Real l_dd = L_ptr[d * D + d];
            l_dd = std::max(l_dd, static_cast<Real>(1e-12));
            sum_log_diag += std::log(l_dd);
        }
        const Real component_constants = -0.5 * D * log_2pi - sum_log_diag + log_weights_[k];
        Real mahalanobis_dist = 0.0;
        for (int r = 0; r < D; ++r) {
            Real diff_r = obs[r] - mu[r];
            for (int c = 0; c < r; ++c) {
                diff_r -= L_ptr[r * D + c] * y_cache[c];
            }
            Real l_rr = std::max(L_ptr[r * D + r], static_cast<Real>(1e-12));
            y_cache[r] = diff_r / l_rr;
            mahalanobis_dist += y_cache[r] * y_cache[r];
        }
        log_probs[k] = component_constants - 0.5 * mahalanobis_dist;
    }
}

void GaussianMixModel::emission(const Real* obs, Real* probs, const int num) const
{
    const int K = n_components_;
    const int D = dim_;
    thread_local std::vector<Real> log_pdf_k;
    log_pdf_k.resize(K);
    for (int i = 0; i < num; ++i) {
        const Real* x = obs + i * D;
        this->loglikelihood(x, log_pdf_k.data());
        Real total_log_pdf = -std::numeric_limits<Real>::infinity();
        for (int k = 0; k < K; ++k) {
            const Real weighted_log_pdf = log_pdf_k[k];
            total_log_pdf = logsumexp2x(total_log_pdf, weighted_log_pdf);
        }
        probs[i] = total_log_pdf;
    }
}

void GaussianMixModel::score(const Real* obs, Real* probs, int num) const
{
    const int K = n_components_;
    const int D = dim_;
    std::vector<Real> log_pdf_k(K);
    for (int i = 0; i < num; ++i) {
        const Real* x = obs + i * D;
        this->loglikelihood(x, log_pdf_k.data());
        Real total_log_pdf = -std::numeric_limits<Real>::infinity();
        for (int k = 0; k < K; ++k) {
            const Real weighted_log_pdf = log_pdf_k[k];
            total_log_pdf = logsumexp2x(total_log_pdf, weighted_log_pdf);
        }
        probs[i] = total_log_pdf;
    }
}

void GaussianMixModel::reset()
{
    std::fill(sum_k_gamma_.begin(), sum_k_gamma_.end(), 0.0);
    std::fill(sum_k_x_.begin(), sum_k_x_.end(), 0.0);
    std::fill(sum_k_xx_.begin(), sum_k_xx_.end(), 0.0);
}

void GaussianMixModel::update() {
    const int D = dim_;
    const int K = n_components_;
    double total_gamma_sum = 0.0;
    for (int k = 0; k < K; ++k) {
        total_gamma_sum += sum_k_gamma_[k];
    }
    if (total_gamma_sum <= 1e-8) {
        reset();
        return;
    }
    std::vector<Real> mu_new(D);
    std::vector<Real> cov_new(D * D);
    std::vector<Real> L_new(D * D);
    for (int k = 0; k < K; ++k) {
        Real gamma_k = sum_k_gamma_[k];
        if (gamma_k > 0) {
            cov_new.assign(D * D, 0.0);
            L_new.assign(D * D, 0.0);
            const Real inv_gamma_k = 1.0 / gamma_k;
            for (int d = 0; d < D; ++d) {
                mu_new[d] = sum_k_x_[k * D + d] * inv_gamma_k;
            }
            const Real* xx_ptr = sum_k_xx_.data() + k * (D * D);
            for (int r = 0; r < D; ++r) {
                for (int c = 0; c < D; ++c) {
                    const Real E_xx = xx_ptr[r * D + c] * inv_gamma_k;
                    cov_new[r * D + c] = E_xx - mu_new[r] * mu_new[c];
                }
            }
            if (computeCholesky(cov_new.data(), L_new.data(), D)) {
                Real* chol_ptr = chol_cov_.data() + k * (D * D);
                Real* cov_ptr = vars_.data() + k * (D * D);
                Real* mean_ptr = means_.data() + k * D;
                for (int i = 0; i < D; ++i) {
                    mean_ptr[i] = mu_new[i];
                }
                for (int idx = 0; idx < D * D; ++idx) {
                    cov_ptr[idx] = cov_new[idx];
                    chol_ptr[idx] = L_new[idx];
                }
                const Real weight_prob = gamma_k / total_gamma_sum;
                log_weights_[k] =std::log(weight_prob);
            }
        }
    }
    reset();
    return;
};

void GaussianMixModel::update(GaussianMixModel** em_list, const int size) {
    const int D = em_list[0]->dim_;
    const int K = em_list[0]->n_components_;
    std::vector<Real> cov_new(K* D * D, 0.0);
    std::vector<Real> L_new(K * D * D, 0.0);
    std::vector<Real> gamma_all(size, 0.0);
    std::vector<Real> gamma_all_k(K, 0.0);
    std::vector<Real> sum_x(K * D, 0.0);
    std::vector<Real> sum_xx(K * D * D, 0.0);

    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < K; ++k) {
            gamma_all[i] += em_list[i]->sum_k_gamma_[k];
            gamma_all_k[k] += em_list[i]->sum_k_gamma_[k];
            for (int d = 0; d < D; ++d) {
                sum_x[k * D + d] += em_list[i]->sum_k_x_[k * D + d];
            }
            for (int d = 0; d < D * D; ++d) {
                sum_xx[k * D * D + d] += em_list[i]->sum_k_xx_[k * D * D + d];
            }
        }
    }
    for (int k = 0; k < K; ++k) {
        const int koff = k * D * D;
        if (gamma_all_k[k] > 0.0) {
            const Real gamma_all_k_inv = 1.0 / gamma_all_k[k];
            for (int r = 0; r < D; ++r) {
                for (int c = 0; c < D; ++c) {
                    cov_new[koff + r * D + c] = sum_xx[koff + r * D + c] * gamma_all_k_inv
                    - sum_x[k * D + r] * sum_x[k * D + c] * gamma_all_k_inv * gamma_all_k_inv;
                }
            }
        }
    }
    for (int k = 0; k < K; ++k) {
        const int koff = k * D * D;
        if (computeCholesky(cov_new.data() + koff, L_new.data() + koff, D)) {
            for (int i = 0; i < size; ++i) {
                GaussianMixModel * em = em_list[i];
                Real gamma_k = em->sum_k_gamma_[k];
                if (gamma_k > 0.0) {
                    for (int d = 0; d < D; ++d) {
                        em->means_[k * D + d] = em->sum_k_x_[k * D + d] / gamma_k;
                    }
                }
                for (int idx = 0; idx < D * D; ++idx) {
                    em->vars_[koff + idx] = cov_new[koff + idx];
                    em->chol_cov_[koff + idx] = L_new[koff + idx];
                }
                const Real weight_prob = gamma_k / gamma_all[i];
                em->log_weights_[k] =std::log(weight_prob);
            }
        }
    }
    for (int i = 0; i < size; ++i) {
        GaussianMixModel * em = em_list[i];
        em->reset();
    }
    return;
};

void GaussianMixModel::accumulate(const Real* x_ptr, const Real* gamma, const int num) {
    const int K = n_components_;
    const int D = dim_;
    thread_local std::vector<Real> scores;
    scores.resize(K);
    for (int i = 0; i < num; ++i) {
        const Real* x_i = x_ptr + i * D;
        const Real g_i = gamma[i];
        this->loglikelihood(x_i, scores.data());
        log_normalize(scores.data(), scores.size());
        for (int k = 0; k < K; ++k) {
            double r_ik = std::exp(g_i + scores[k]);
            sum_k_gamma_[k] += r_ik;
            Real* sum_x_k = sum_k_x_.data() + k * D;
            for (int d = 0; d < D; ++d) {
                sum_x_k[d] += r_ik * x_i[d];
            }
            Real* sum_xx_k = sum_k_xx_.data() + k * (D * D);
            for (int r = 0; r < D; ++r) {
                for (int c = 0; c < D; ++c) {
                    sum_xx_k[r * D + c] += r_ik * x_i[r] * x_i[c];
                }
            }
        }
    }
}

std::vector<Real> GaussianMixModel::save() {
    std::vector<Real> res;
    const int total_size = n_components_ * (dim_ * dim_ + dim_ + 1);
    res.reserve(total_size);
    for (int k = 0; k < n_components_; ++k) {
        res.push_back(std::exp(log_weights_[k]));
        for (int d = 0; d < dim_; ++d) {
            res.push_back(means_[k * dim_ + d]);
        }
        for (int r = 0; r < dim_; ++r) {
            for (int c = 0; c < dim_; ++c) {
                res.push_back(vars_[k * dim_ * dim_ + r * dim_ + c]);
            }
        }
    }
    return res;
}

void GaussianMixModel::setValues(const Real* values, int n)
{
    if (const int total_size = n_components_ * (dim_ * dim_ + dim_ + 1); n != total_size) {
        throw std::runtime_error("GaussianMixModel::setValues(), length n not match with peak number");
    }
    for (int k = 0; k < n_components_; ++k) {
        const Real* data = values + k * (dim_ * dim_ + dim_ + 1);
        log_weights_[k] = std::log(data[0]);
        for (int d = 0; d < dim_; ++d) {
            means_[k * dim_ + d] = data[d + 1];
        }
        for (int r = 0; r < dim_; ++r) {
            for (int c = 0; c < dim_; ++c) {
                vars_[k * dim_ * dim_ + r * dim_ + c] = data[r * dim_ + c + dim_ + 1];
            }
        }

        Real* chol_ptr = chol_cov_.data() + k * (dim_ * dim_);
        Real* cov_ptr = vars_.data() + k * (dim_ * dim_);
        computeCholesky(cov_ptr, chol_ptr, dim_);
    }
}