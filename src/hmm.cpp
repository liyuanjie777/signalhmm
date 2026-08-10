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

#include "gaussianmixmodel.h"

HMM::HMM(const std::vector<EmissionModel*>& emit, TransitionModel* transit, std::vector<Real>& prob_pi)
    :emit_(emit), transit_(transit), pi_(prob_pi) {
    pi_cache_.assign(pi_.size(), -std::numeric_limits<Real>::infinity());
    for (int i = 0; i < emit_.size(); ++i) {
        EmissionModel* ptr_i = emit_[i];
        bool found = false;
        for (auto& group : shared_emit_) {
            int j = group[0];
            if (emit_[j] == ptr_i) {
                group.push_back(i);
                found = true;
                break;
            }
        }
        if (!found) {
            shared_emit_.push_back({i});
        }
    }
}

Real HMM::update(const char* method) {

    if (std::strchr(method, 'e') != nullptr) {
        if (std::strchr(method, 's') != nullptr) {
            std::vector<GaussianMixModel*> em_list;
            for (auto & k : shared_emit_) {
                em_list.push_back(dynamic_cast<std::vector<GaussianMixModel *>::value_type>(emit_[k[0]]));
            }
            GaussianMixModel::update(em_list.data() + 1, em_list.size() - 1);
        }
        for (auto & k : emit_) {
            k->update();
        }
    }
    if (std::strchr(method, 't') != nullptr) {
        transit_->update();
    }
    if (std::strchr(method, 'p') != nullptr) {
        log_normalize(pi_cache_.data(), pi_cache_.size());
        for (int i = 0; i < pi_.size(); ++i) {
            pi_[i] = pi_cache_[i];
        }
        pi_cache_.assign(pi_cache_.size(), -std::numeric_limits<Real>::infinity());
    }
    const Real res = residual_;
    residual_ = 0;
    return res;
}

void HMM::infer(const std::vector<std::vector<Real>>& datas, std::vector<std::vector<int>>& labels, const int data_dim, std::vector<Real>& scores) const {
    scores.clear();
    scores.assign(datas.size(), 0);
#pragma omp parallel for
    for (int i = 0; i < datas.size(); i++) {
        const int data_size = datas[i].size() / data_dim;
        if (labels[i].size() != data_size) {
            labels[i].resize(data_size);
        }
        scores[i] = viterbi(datas[i].data(), labels[i].data(), labels[i].size(), data_dim);
    }
}

void HMM::EM_step(const std::vector<std::vector<Real>>& datas, const int data_dim, const char* method, const std::vector<std::vector<std::vector<int>>>& adj_lists) {
    const int coo_num = transit_->num_values();
    const int state = transit_->num_states();
    std::vector<Real> epsilons(coo_num * datas.size(), -std::numeric_limits<Real>::infinity());
    std::vector<MixMatrix> gammas;

    for (int t = 0; t < adj_lists.size(); ++t) {
        std::vector<int> coo_x;
        std::vector<int> coo_y;
        for (int i = 0; i < adj_lists[t].size(); ++i) {
            for (int j = 0; j < adj_lists[t][i].size(); ++j) {
                coo_x.push_back(i);
                coo_y.push_back(adj_lists[t][i][j]);
            }
        }
        gammas.emplace_back(coo_x.data(), coo_y.data(), coo_x.size(), adj_lists[t].size(), state, -std::numeric_limits<Real>::infinity());
    }

    Real total_residual = 0;
    //E-step
#pragma omp parallel for reduction(+ : total_residual)
    for (int i = 0; i < datas.size(); ++i) {
        const Real* data = datas[i].data();
        const int data_size = datas[i].size() / data_dim;
        MixMatrix& gamma = gammas[i];

        Real* epsilon = nullptr;
        if (std::strchr(method, 't') != nullptr) {
            epsilon = epsilons.data() + coo_num * i;
        }

        SparseVector beta;
        SparseVectorView beta_view;
        SparseVectorView alpha_pre;
        SparseVectorView alpha;
        std::vector<Real> dense_collector(state);
        std::vector<Real> coeff(data_size, 0.0);
        // forward
        for (int t = 0; t < data_size; ++t) {
            alpha = gamma.get_row(t);
            for (int j = 0; j < alpha.size; ++j) {
                emit_[alpha.indices[j]]->emission(&data[t * data_dim], &alpha.data[j], 1);
            }
        }
        MixMatrix emit_prob(gamma);
        alpha_pre = gamma.get_row(0);
        alpha_pre.add(pi_.data());
        coeff[0] = log_normalize(alpha_pre.data, alpha_pre.size);
        for (int t = 1; t < data_size; ++t) {
            alpha = gamma.get_row(t);
            alpha_pre = gamma.get_row(t - 1);
            transit_->mulVM(alpha_pre, dense_collector.data());
            alpha.add(dense_collector.data());
            coeff[t] = log_normalize(alpha.data, alpha.size);
        }
        //backward
        beta = emit_prob.get_row(data_size - 1);
        std::fill_n(beta.data, beta.size, 0.0);
        for (int t = data_size - 2; t >= 0; --t) {
            beta_view = emit_prob.get_row(t + 1);
            vec_add(beta.data, beta_view.data, beta.size);
            beta_view = beta;
            vec_add(beta_view.data, -coeff[t + 1], beta_view.size);
            alpha = gamma.get_row(t);
            if (std::strchr(method, 't') != nullptr) {
                transit_->epsilon_E_step(alpha, beta_view, epsilon);
            }
            transit_->mulMV(beta_view, dense_collector.data());
            beta = emit_prob.get_row(t);
            beta_view = beta;
            beta_view.set(dense_collector.data());
            vec_add(alpha.data, beta_view.data, beta_view.size);
            log_normalize(alpha.data, alpha.size);
        }
        Real residual = 0;
        for(int t = 0;t < data_size; t++) {
            residual += coeff[t] / data_size;
        }
        total_residual += residual;
    }
    // M-step
    if (std::strchr(method, 'p') != nullptr) {
        for (int i = 0; i < datas.size(); ++i) {
            SparseVectorView vec = gammas[i].get_row(0);
            for (int t = 0; t < vec.size; ++t) {
                const int j = vec.indices[t];
                pi_cache_[j] = logsumexp2x(pi_cache_[j], vec.data[t]);
            }
        }
    }
    if (std::strchr(method, 'e') != nullptr)
    {
        std::vector<Real> data_all;
        for (int i = 0; i < datas.size(); ++i) {
            data_all.insert(data_all.end(), datas[i].begin(), datas[i].end());
            gammas[i].transpose();
        }
#pragma omp parallel for
        for (int k = 0; k < shared_emit_.size(); ++k) {
            int total_frames = data_all.size() / data_dim;
            std::vector<Real> gamma(total_frames, -std::numeric_limits<Real>::infinity());
            for (int s = 0; s < shared_emit_[k].size(); ++s) {
                int frame_offset = 0;
                int j = shared_emit_[k][s];
                for (int b = 0; b < datas.size(); ++b) {
                    SparseVectorView vec = gammas[b].get_row(j);
                    for (int t = 0; t < vec.size; ++t) {
                        Real gamma_t = gamma[frame_offset + vec.indices[t]];
                        gamma[frame_offset + vec.indices[t]] = logsumexp2x(gamma_t, vec.data[t]);
                    }
                    frame_offset += datas[b].size() / data_dim;
                }
            }
            int j = shared_emit_[k][0];
            emit_[j]->accumulate(data_all.data(), gamma.data(), gamma.size());
        }
    }
    if (std::strchr(method, 't') != nullptr) {
        transit_->epsilon_M_step(epsilons.data(), datas.size());
    }
    residual_ += total_residual;
}

Real HMM::viterbi(const Real* data, int* label, const int n, const int m) const {
    const int state = transit_->num_states();
    std::vector<Real> row(state);
    std::vector<Real> row_pre(state);
    const size_t N = n * state;
    std::vector dp(N, 0);
    //MixMatrix backward(n, state, 0.0f);
    std::vector<Real> emit_prob(N);
    //std::vector<Real> dense_collector(state);
    for (int t = 0; t < n; ++t) {
        const size_t offset = t * state;
        for (size_t j = 0; j < state; ++j) {
            emit_[j]->emission(&data[t * m], &emit_prob[offset + j], 1);
        }
    }
    //for (int  t = n - 2; t >= 0; --t) {
    //    SparseVectorView beta = backward.get_row(t);
    //    const SparseVectorView beta_next = backward.get_row(t + 1);
    //    beta.set(beta_next.data);
    //    const Real * emit = emit_prob.data() + (t + 1) * state;
    //    beta.add(emit);
    //    transit_->mulMV(beta, dense_collector.data());
    //    beta.set(dense_collector.data());
    //}
    // max_forward
    for (int j = 0; j < state; j++) {
        row_pre[j] = emit_prob[j] + pi_[j];
    }
    Real score = log_normalize(row_pre.data(), row_pre.size());
    for (int i = 1; i < n; i++) {
        //SparseVectorView back_score = backward.get_row(i);
        const size_t id = i * state;
        for (int j = 0; j < state; j++) {
            const Real prob = emit_prob[id + j];
            SparseVector vec;
            vec = transit_->get_col(j);
            for (int k = 0; k < vec.size; ++k) {
                //vec.data[k] += row_pre[vec.indices[k]] + back_score.data[vec.indices[k]];
                vec.data[k] += row_pre[vec.indices[k]];
            }
            const int maxk =
                std::distance(vec.data, std::max_element(vec.data, vec.data + vec.size));
            dp[id + j] = vec.indices[maxk];
            //row[j] = vec.data[maxk] + prob - back_score.data[vec.indices[maxk]];
            row[j] = vec.data[maxk] + prob;
        }
        score += log_normalize(row.data(), row.size());
        std::swap(row, row_pre);
    }

    // search
    std::fill_n(label, n, 0);
    const auto it = std::max_element(row_pre.begin(), row_pre.end());
    score += *it;
    int j = std::distance(row_pre.begin(), it);
    for (int i = n - 1; i >= 0; --i) {
        const size_t id = i * static_cast<size_t>(state) + j;
        label[i] = j;
        j = dp[id];
    }
    return score;
}