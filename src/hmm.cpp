/*
 *  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-08-07
 *  License: MIT
 */
#include "hmm.h"
#include "mymath.hpp"
#include "sparsematrix.h"
#include "signalmath.hpp"
#include <omp.h>

HMM::HMM(const std::vector<EmissionModel*>& emit, TransitionModel* transit, std::vector<Real>& prob_pi, const char* method)
    : method_(method), emit_(emit), transit_(transit), log_pi_(prob_pi) {
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
        for (auto & k : emit_)
        {
            k->update(1.0);
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

void HMM::infer(const std::vector<std::vector<Real>>& datas, std::vector<std::vector<int>>& labels, const int data_dim) const {
    if (labels.size() != datas.size()) {
        labels.resize(datas.size());
    }
    for (int i = 0; i < labels.size(); ++i) {
        labels[i].resize(datas[i].size() / data_dim);
        std::fill_n(labels[i].begin(), labels[i].size(), -1);
    }
#pragma omp parallel for
    for (int i = 0; i < datas.size(); i++)
    {
        viterbi(datas[i].data(), labels[i].data(), labels[i].size(), data_dim);
    }
}

void HMM::EM_step(const std::vector<std::vector<Real>>& datas, const int data_dim) {
    const int coo_num = transit_->num_values();
    const int state = transit_->num_states();
    std::vector<Real> epsilons(coo_num * datas.size(), -std::numeric_limits<Real>::infinity());
    std::vector<MixMatrix> gammas;
    for (int i = 0; i < datas.size(); ++i) {
        const int data_size = datas[i].size() / data_dim;
        std::vector<int> coo_x;
        std::vector<int> coo_y;
        uniform_align(state, data_size, 20, coo_x, coo_y);
        gammas.emplace_back(coo_x.data(), coo_y.data(), coo_x.size(), data_size, state, -std::numeric_limits<Real>::infinity());
    }
    Real total_residual = 0;
    //E-step
#pragma omp parallel for reduction(+ : total_residual)
    for (int i = 0; i < datas.size(); ++i) {
        const Real* data = datas[i].data();
        const int data_size = datas[i].size() / data_dim;
        MixMatrix& gamma = gammas[i];
        MixMatrix emit_log_prob(gamma);

        Real* epsilon = nullptr;
        if (std::strchr(method_, 't') != nullptr)
        {
            epsilon = epsilons.data() + coo_num * i;
        }

        SparseVector beta;
        SparseVector beta_next;
        SparseVectorView alpha_pre;
        SparseVectorView alpha;
        Real residual = 0;
        // forward
        for (int t = 0; t < data_size; ++t)
        {
            alpha = gamma.get_row(t);
            alpha_pre = emit_log_prob.get_row(t);
            for (int j = 0; j < alpha.size; ++j)
            {
                emit_[alpha.indices[j]]->log_emission(&data[t * data_dim], &alpha.data[j], 1);
                alpha_pre.data[j] = alpha.data[j];
            }
        }
        alpha_pre = gamma.get_row(0);
        for (int id = 0; id < alpha_pre.size; ++id) {
            alpha_pre.data[id] += log_pi_[alpha_pre.indices[id]];
        }
        residual += log_normalize(alpha_pre.data, alpha_pre.size);
        for (int t = 1; t < data_size; ++t)
        {
            alpha = gamma.get_row(t);
            alpha_pre = gamma.get_row(t - 1);
            transit_->mulVM(alpha_pre, alpha);
            residual += log_normalize(alpha.data, alpha.size);
        }
        //backward
        beta = emit_log_prob.get_row(data_size - 1);
        std::fill_n(beta.data, beta.size, 0);
        for (int t = data_size - 2; t >= 0; --t)
        {
            beta_next = emit_log_prob.get_row(t + 1);
            vec_add(beta_next.data, beta.data, beta.size);
            beta = emit_log_prob.get_row(t);
            std::fill_n(beta.data, beta.size, -std::numeric_limits<Real>::infinity());
            alpha_pre = beta;
            alpha = beta_next;
            transit_->mulMV(alpha, alpha_pre);
            alpha = gamma.get_row(t);
            if (std::strchr(method_, 't') != nullptr)
            {
                transit_->epsilon_E_step(alpha, alpha_pre, epsilon);
            }
            log_normalize(beta.data, beta.size);
            vec_add(alpha.data, beta.data, beta.size);
            log_normalize(alpha.data, alpha.size);
        }
        total_residual += residual / data_size;
    }
    // M-step
    if (std::strchr(method_, 'p') != nullptr)
    {
        for (int i = 0; i < datas.size(); ++i)
        {
            SparseVectorView vec = gammas[i].get_row(0);
            for (int t = 0; t < vec.size; ++t)
            {
                const int j = vec.indices[t];
                log_pi_cache_[j] = logsumexp2x(log_pi_cache_[j], vec.data[t]);
            }
        }
    }
    if (std::strchr(method_, 'e') != nullptr)
    {
        std::vector<Real> data_all;
        for (int i = 0; i < datas.size(); ++i) {
            data_all.insert(data_all.end(), datas[i].begin(), datas[i].end());
            gammas[i].transpose();
        }
#pragma omp parallel for
        for (int k = 0; k < shared_emit_.size(); ++k)
        {
            std::vector<Real> gamma(data_all.size() / data_dim, -std::numeric_limits<Real>::infinity());
            Real* gamma_ptr = gamma.data();
            for (int s = 0; s < shared_emit_[k].size(); ++s)
            {
                std::fill_n(gamma.begin(), data_all.size() / data_dim, -std::numeric_limits<Real>::infinity());
                int j = shared_emit_[k][s];
                for (int t = 0; t < datas.size(); ++t)
                {
                    SparseVectorView vec = gammas[t].get_row(j);
                    for (int t = 0; t < vec.size; ++t) {
                        gamma_ptr[vec.indices[t]] = vec.data[t];
                    }
                    gamma_ptr += datas[t].size() / data_dim;
                }
                emit_[j]->log_accumulate(data_all.data(), gamma.data(), gamma.size());
            }
        }
    }
    if (std::strchr(method_, 't') != nullptr)
    {
        transit_->epsilon_M_step(epsilons.data(), datas.size());
    }
    residual_ += total_residual;
}

void HMM::viterbi(const Real* data, int* label, const int n, const int data_dim) const {
    const int coo_num = transit_->num_values();
    const int state = transit_->num_states();
    std::vector<Real> row(state);
    std::vector<Real> row_pre(state);
    const size_t N = n * state;
    std::vector<int> dp(N);
    std::vector<Real> emit_log_prob(N);
    for (int t = 0; t < n; ++t)
    {
      const size_t offset = t * state;
        for (size_t j = 0; j < state; ++j)
        {
            emit_[j]->log_emission(&data[t * data_dim], &emit_log_prob[offset + j], 1);
        }
    }
    std::fill(dp.begin(), dp.end(), 0);
    // max_forward
    for (int j = 0; j < state; j++)
    {
        row_pre[j] = emit_log_prob[j] + log_pi_[j];
    }
    log_normalize(row_pre.data(), row_pre.size());

    for (int i = 1; i < n; i++)
    {
      const size_t id = i * state;
        for (int j = 0; j < state; j++)
        {
          const Real log_prob = emit_log_prob[id + j];
            SparseVector vec;
            vec = transit_->get_col(j);
            for (int k = 0; k < vec.size; ++k)
            {
                vec.data[k] += row_pre[vec.indices[k]];
            }
            const int maxk =
                std::distance(vec.data, std::max_element(vec.data, vec.data + vec.size));
            dp[id + j] = vec.indices[maxk];
            row[j] = vec.data[maxk] + log_prob;
        }
        std::swap(row, row_pre);
    }

    // search
    const auto it = std::max_element(row_pre.begin(), row_pre.end());
    int j = std::distance(row_pre.begin(), it);
    for (int i = n - 1; i >= 0; --i) {
        const size_t id = i * static_cast<size_t>(state) + j;
        label[i] = j;
        j = dp[id];
    }
    return;
}