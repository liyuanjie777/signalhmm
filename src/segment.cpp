//
// Created by yuanjie on 8/8/26.
//
#include "segment.h"
#include "signalmath.hpp"

std::vector<Real> _pelt_fpt(int max_run_length, const Real mu, const Real var) {
    std::vector<Real> cost_time(max_run_length + 1);
    const Real lambda = mu * mu * mu / var;
    for (int L = 1; L <= max_run_length; ++L) {
        const Real t = static_cast<Real>(L);
        cost_time[L] = 0.5 * std::log(2.0 * M_PI * t * t * t / lambda) + lambda * (t - mu) * (t - mu) / (2.0 * mu * mu * t);
    }
    return cost_time;
}

Real _pelt_var(const Real* sum1, const Real* sum2, const int i, const int j) {
    const Real n = j - i;
    const Real s1 = (sum1[j] - sum1[i]) / n;
    const Real s2 = (sum2[j] - sum2[i]) / n;
    Real var = s2  - s1 * s1;
    return var;
}

void pelt(const std::vector<Real>& data, std::vector<int>& mv, const Real speed_mean, const Real speed_var, const int max_run_length) {
    std::vector<Real> cost_time = _pelt_fpt(max_run_length + 1, speed_mean, speed_var);
    mv.clear();
    mv.reserve(data.size());
    const int n = data.size();
    const Real min_var = min_variance(data.data(), data.size(), 6);
    const Real low_bound = *std::min_element(cost_time.begin() + 1, cost_time.end());
    std::vector<Real> sum1(n + 1, 0.0);
    std::vector<Real> sum2(n + 1, 0.0);
    Real* sum1_ptr = sum1.data() + 1;
    Real* sum2_ptr = sum2.data() + 1;
    for (int i = 0; i < n; i++) {
        sum1_ptr[i] = sum1_ptr[i - 1] + data[i];
        sum2_ptr[i] = sum2_ptr[i - 1] + data[i] * data[i];
    }
    std::vector<int> R;
    R.reserve(n);
    std::vector<Real> F_extend(n + 1, 0.0);
    std::vector<int> cp(n, -1);
    Real* F = F_extend.data() + 1;
    R.push_back(-1);
    for (int j = 0; j < n; ++j) {
        Real best_cost = std::numeric_limits<Real>::max();
        int best_cp = -1;
        for (const auto i : R) {
            if (j - i > max_run_length) {continue;}
            Real cost_seg = std::max(_pelt_var(sum1_ptr, sum2_ptr, i, j), min_var);
            cost_seg = 0.5 * static_cast<Real>(j - i) * (std::log(2.0 * M_PI * cost_seg) + 1.0);
            cost_seg += cost_time[j - i] + F[i];
            if (cost_seg < best_cost) {
                best_cp = i;
                best_cost = cost_seg;
            }
        }
        F[j] = best_cost;
        cp[j] = best_cp;

        int ptr = 0;
        for (int k = 0; k < R.size(); ++k) {
            const int i = R[k];
            Real cost_lb = 0.5 * static_cast<Real>(j - i) * (std::log(2.0 * M_PI * min_var) + 1.0) + low_bound;
            if ((j - i < max_run_length) && (F[i] + cost_lb < F[j])) {
                R[ptr] = R[k];
                ptr++;
            }
        }
        R.resize(ptr);
        R.push_back(j);
    }
    std::vector<int> tmp;
    int idx = n - 1;
    while (idx >= 0) {
        tmp.push_back(idx);
        idx = cp[idx];
    }
    std::reverse(tmp.begin(), tmp.end());
    mv.assign(n, -1);
    idx = 0;
    for (int i = 0; i < n; ++i) {
        mv[i] = idx;
        if (i == tmp[idx]) {
            idx++;
        }
    }
    return;
}
