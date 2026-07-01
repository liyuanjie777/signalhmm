/*
 *  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-08-07
 *  License: MIT
 */
#include "hmm.h"
#include "mymath.hpp"
#include <omp.h>

HMM::HMM(const std::vector<EmissionModel*>& emit, TransitionModel* transit, std::vector<Real>& prob_pi, const int dim, const int state,
         const int coo_num, const char* method)
    : dim_(dim), state_(state), coo_num_(coo_num), method_(method), emit_(emit), transit_(transit),
      log_pi_(prob_pi)
{
    log_pi_cache_.resize(log_pi_.size(), -std::numeric_limits<Real>::infinity());
    for (int i = 0; i < log_pi_.size(); ++i)
        log_pi_[i] = std::log(log_pi_[i]);
    log_normalize(log_pi_.data(), log_pi_.size());
    for (int i = 0; i < emit_.size(); ++i)
    {
        EmissionModel* ptr_i = emit_[i];
        bool found = false;
        for (auto& group : shared_emit_)
        {
            int j = group[0];
            if (emit_[j] == ptr_i)
            {
                group.push_back(i);
                found = true;
                break;
            }
        }
        if (!found)
        {
            shared_emit_.push_back({i});
        }
    }
}

Real HMM::update(const Real rate)
{
    if (std::strchr(method_, 'e') != nullptr)
    {
        for (auto & k : shared_emit_)
        {
            emit_[k[0]]->update(1.0);
        }
    }
    if (std::strchr(method_, 't') != nullptr)
    {
        transit_->update(rate);
    }
    if (std::strchr(method_, 'p') != nullptr)
    {
        log_pi_ = log_pi_cache_;
        log_normalize(log_pi_.data(), log_pi_.size());
        log_pi_cache_.assign(log_pi_cache_.size(), -std::numeric_limits<Real>::infinity());
    }
    const Real res = residual_;
    residual_ = 0;
    return res;
}

void HMM::infer(const Real* data, int* label, const size_t* batch, const int nbatch) const {
#pragma omp parallel for
    for (int i = 0; i < nbatch; i++)
    {
        viterbi(&data[batch[i] * dim_], &label[batch[i]], batch[i + 1] - batch[i]);
    }
}

void HMM::EM_step(const Real* data, const size_t* batch, const int nbatch)
{
    epsilon_.resize(coo_num_ * nbatch, -std::numeric_limits<Real>::infinity());
    gamma_.resize(state_ * batch[nbatch], -std::numeric_limits<Real>::infinity());
    Real total_residual = 0;
#pragma omp parallel for reduction(+ : total_residual)
    for (int i = 0; i < nbatch; ++i)
    {
        const int n_i = batch[i + 1] - batch[i];
        const Real* data_i = data + batch[i] * dim_;
        Real* gamma_i = gamma_.data() + state_ * batch[i];
        Real* epsilon_i = nullptr;
        if (std::strchr(method_, 't') != nullptr)
        {
            epsilon_i = epsilon_.data() + coo_num_ * i;
        }

        std::vector<Real> beta(state_, 0.0);
        std::vector<Real> beta_next(state_, -std::numeric_limits<Real>::infinity());
        std::vector<Real> emit_log_prob(n_i * state_);
        Real residual = 0;
        // forward
        for (int t = 0; t < n_i; ++t)
        {
            size_t offset = t * state_;
            for (size_t j = 0; j < state_; ++j)
            {
                emit_[j]->log_emission(&data_i[t * dim_], &emit_log_prob[offset + j], 1);
            }
        }
        memcpy(gamma_i, emit_log_prob.data(), sizeof(Real) * state_);
        vec_add(gamma_i, log_pi_.data(), state_);
        residual += log_normalize(gamma_i, state_);
        for (int t = 1; t < n_i; ++t)
        {
            const size_t id = t * state_;
            const size_t id_pre = (t - 1) * state_;
            transit_->mulVM(&gamma_i[id_pre], &gamma_i[id]);
            vec_add(&gamma_i[id], &emit_log_prob[id], state_);
            residual += log_normalize(&gamma_i[id], state_);
        }
        vec_add(&gamma_i[(n_i - 1) * state_], beta.data(), state_);
        log_normalize(&gamma_i[(n_i - 1) * state_], state_);
        for (int t = n_i - 2; t >= 0; --t)
        {
            std::swap(beta, beta_next);
            const size_t id = t * state_;
            const size_t id_next = (t + 1) * state_;
            vec_add(beta_next.data(), &emit_log_prob[id_next], state_);
            transit_->mulMV(beta_next.data(), beta.data());
            if (std::strchr(method_, 't') != nullptr)
            {
                transit_->epsilon_mstep(&gamma_i[id], beta_next.data(), epsilon_i);
            }
            log_normalize(beta.data(), state_);
            vec_add(&gamma_i[id], beta.data(), state_);
            log_normalize(&gamma_i[id], state_);
        }
        total_residual += residual / n_i;
    }

    if (std::strchr(method_, 'e') != nullptr)
    {
#pragma omp parallel for
        for (int k = 0; k < shared_emit_.size(); ++k)
        {
            std::vector<Real> gamma_j(batch[nbatch], -std::numeric_limits<Real>::infinity());
            for (int t = 0; t < batch[nbatch]; ++t)
            {
                for (int s = 0; s < shared_emit_[k].size(); ++s)
                {
                    int j = shared_emit_[k][s];
                    gamma_j[t] = logsumexp2x(gamma_j[t], gamma_[t * state_ + j]);
                }
            }
            emit_[shared_emit_[k][0]]->log_accumulate(data, gamma_j.data(), batch[nbatch]);
        }
    }
    if (std::strchr(method_, 't') != nullptr)
    {
        transit_->epsilon_accumulate(epsilon_.data(), nbatch);
    }
    if (std::strchr(method_, 'p') != nullptr)
    {
        for (int i = 0; i < nbatch; ++i)
        {
            for (int j = 0; j < state_; ++j)
            {
                log_pi_cache_[j] = logsumexp2x(log_pi_cache_[j], gamma_[batch[i] * state_ + j]);
            }
        }
    }
    residual_ += total_residual;
}

void HMM::viterbi(const Real* data, int* label, const int n) const {
    std::vector<int> col_id;
    std::vector<Real> col_val;
    std::vector<Real> row(state_);
    std::vector<Real> row_pre(state_);
    const size_t N = n * state_;
    std::vector<size_t> dp(N);
    std::vector<Real> emit_log_prob(N);
    for (int t = 0; t < n; ++t)
    {
      const size_t offset = t * state_;
        for (size_t j = 0; j < state_; ++j)
        {
            emit_[j]->log_emission(&data[t * dim_], &emit_log_prob[offset + j], 1);
        }
    }
    std::fill(dp.begin(), dp.end(), 0);
    // max_forward
    for (int j = 0; j < state_; j++)
    {
        row_pre[j] = emit_log_prob[j] + log_pi_[j];
    }
    log_normalize(row_pre.data(), row_pre.size());

    for (int i = 1; i < n; i++)
    {
      const size_t id = i * state_;
        size_t id_pre = (i - 1) * state_;
        for (int j = 0; j < state_; j++)
        {
          const Real log_prob = emit_log_prob[id + j];
            transit_->getcol(j, col_id, col_val);
            for (int k = 0; k < col_val.size(); ++k)
            {
                col_val[k] += row_pre[col_id[k]];
            }
            const int maxk =
                std::distance(col_val.begin(), std::max_element(col_val.begin(), col_val.end()));
            dp[id + j] = col_id[maxk];
            row[j] = col_val[maxk] + log_prob;
        }
        std::swap(row, row_pre);
    }

    // search
    const auto it = std::max_element(row_pre.begin(), row_pre.end());
    int j = std::distance(row_pre.begin(), it);
    for (int i = n - 1; i >= 0; --i)
    {
        size_t id = i * size_t(state_) + j;
        label[i] = j;
        j = dp[id];
    }
    return;
}