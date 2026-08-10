#pragma once
#include "fileio.h"
#include "mymath.hpp"
#include "variable.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <utility>

struct dps {
    int j;
    float cost;
    int run;
};

void getTopK(std::vector<dps>& states, const int k) {
    if (k >= states.size()) return;
    std::ranges::nth_element(
    states,
    states.begin() + k,
    std::ranges::less{},
    &dps::cost
    );
    states.resize(k);
}

int argmin(const dps* data, const int n) {
    float min_val = std::numeric_limits<float>::max();
    int min_id = 0;
    for (int i = 0; i < n; ++i) {
        if (data[i].cost < min_val) {
            min_val = data[i].cost;
            min_id = i;
        }
    }
    return min_id;
}

template <typename T>
T min_std(const T* data, const int n, const int window) {
    T mu = 0.0;
    T m2 = 0.0;
    for (int i = 0; i < window; i++) {
        T delta = data[i] - mu;
        mu += delta / static_cast<T>(i + 1);
        T delta2 = data[i] - mu;
        m2 += delta * delta2;
    }
    T min_var = m2;
    for (int i = window; i < n; i++) {
        T x_old = data[i - window];
        T x_new = data[i];
        T mu_old = mu;
        mu += (x_new - x_old) / static_cast<T>(window);
        m2 += (x_new - mu) * (x_new - mu_old) - (x_old - mu) * (x_old - mu_old);
        if (m2 < min_var) {
            min_var = m2;
        }
    }
    return std::sqrt(min_var / static_cast<T>(window));
}

template <typename T>
std::vector<T> hazard_vector(const int R, const T jump_mean, const T jump_sigma) {
    std::vector<T> out(R * 2);
    const T mu = jump_mean;
    const T lambda = (mu * mu * mu) / (jump_sigma * jump_sigma);
    std::vector<T> log_pdf(R, -INFINITY);
    for (int r = 1; r < R; r++) {
        const T t = static_cast<T>(r);
        log_pdf[r] = log_pdf_ig(t, mu, lambda);
    }
    std::vector<T> logS(R, -INFINITY);
    T running = -INFINITY;
    for (int r = R - 1; r >= 1; r--) {
        running = (running == -INFINITY)
            ? log_pdf[r]
            : logaddexp(running, log_pdf[r]);

        logS[r] = running;
    }
    for (int r = 1; r < R; r++) {
        T log_jump = log_pdf[r] - logS[r];
        out[r * 2] = log_jump;
        out[r * 2 + 1] = log1mexp(log_jump);
    }
    return out;
}

template <typename T>
std::vector<T> sliding_variance(const T* data, const int n, const int window) {
    std::vector<T> var(n);
    T sum = 0;
    T sum2 = 0;
    for(int i=0;i<n;i++) {
        sum += data[i];
        sum2 += data[i] * data[i];
        if(i >= window) {
            sum -= data[i - window];
            sum2 -= data[i - window] * data[i - window];
        }
        int count = std::min(i + 1, window);
        T mean = sum / count;
        T v = sum2 / count - mean * mean;
        var[i] = std::max(0.0, v);
    }
    return var;
}

template <typename T>
void mccp(const std::vector<T>& data, std::vector<int>& mv, const vector<T>& state_current, const int beam_size, const std::vector<float>& hazard_table) {
    //forward
    const int max_run_length = hazard_table.size() / 2;
    const int data_length = data.size();
    const int state_length = state_current.size();
    size_t dp_size = data_length * state_length;
    std::vector<int> dp(dp_size, -1);
    std::vector<dps> pre_states(state_length);
    std::vector<dps> cur_states;
    std::vector<dps> tmp_states(beam_size);
    cur_states.reserve(state_length);

    for (int j = 0; j < state_length; ++j) {
        const float current = state_current[j * 2];
        const float var = state_current[j * 2 + 1];
        const float diff = data[0] - current;
        pre_states[j].cost = diff * diff / var;
        pre_states[j].run = 1;
        pre_states[j].j = j;
    }
    for (int i = 1; i < data_length; ++i) {
        const int dp_offset = i * state_length;
        const float current = data[i];
        getTopK(pre_states, beam_size);
        cur_states.resize(state_length);
        for (int j = 0; j < state_length; j++) {
            const float mean = state_current[j * 2];
            const float var = state_current[j * 2 + 1];
            float diff = current - mean;
            diff = diff * diff / var;
            for (int k = 0; k < beam_size; ++k) {
                tmp_states[k].cost = diff;
                tmp_states[k].run = 1;
                tmp_states[k].j = j;
                float cost_move = 0.0f;
                float cost_stay = 0.0f;
                if (pre_states[k].run < max_run_length) {
                    cost_move = hazard_table[pre_states[k].run * 2];
                    cost_stay = hazard_table[pre_states[k].run * 2 + 1];
                }
                if (pre_states[k].j == j) {
                    tmp_states[k].cost += pre_states[k].cost + cost_stay;
                    tmp_states[k].run += pre_states[k].run;
                }
                else {
                    tmp_states[k].cost += pre_states[k].cost + cost_move;
                }
            }
            const int min_k = argmin(tmp_states.data(), tmp_states.size());
            cur_states[j] = tmp_states[min_k];
            dp[dp_offset + j] = pre_states[min_k].j;
        }
        std::swap(cur_states, pre_states);
    }


    std::fill_n(mv, data_length, -1);
    int max_col = argmin(pre_states.data(), pre_states.size());
    for (int i = data_length - 1; i >= 0; --i) {
        const int dp_offset = i * state_length;
        mv[i] = max_col;
        max_col = dp[dp_offset + max_col];
    }
    return;
}

void PELT(std::vector<Real>& x, std::vector<int>& path, const Real beta = 0.9)
{
    int n = x.size();
    Real mu;
    Real sigmal;
    norm_data(x.data(), n, mu, sigmal);
    std::vector<Real> sum1(n + 1, 0.0);
    std::vector<Real> sum2(n + 1, 0.0);
    std::vector<Real> sum3(n + 1, 0.0);
    std::vector<Real> sum4(n + 1, 0.0);
    Real c1 = 0;
    Real c2 = 0;
    Real c3 = 0;
    Real c4 = 0;
    for (int i = 0; i < n; ++i)
    {
        Real y = x[i] - c1;
        Real t = sum1[i] + y;
        c1 = (t - sum1[i]) - y;
        sum1[i + 1] = t;

        y = std::pow(x[i], 2) - c2;
        t = sum2[i] + y;
        c2 = (t - sum2[i]) - y;
        sum2[i + 1] = t;

        y = std::pow(x[i], 3) - c3;
        t = sum3[i] + y;
        c3 = (t - sum3[i]) - y;
        sum3[i + 1] = t;

        y = std::pow(x[i], 4) - c4;
        t = sum4[i] + y;
        c4 = (t - sum4[i]) - y;
        sum4[i + 1] = t;
    }

    std::vector<int> R;
    std::vector<Real> F(n, std::numeric_limits<Real>::max());
    std::vector<int> cp(n, -1);
    F[0] = 0;
    for (int i = 1; i < n; ++i)
    {
        int max_id = -1;
        for (auto j : R)
        {
            Real cost =
                JB_cost(sum1.data(), sum2.data(), sum3.data(), sum4.data(), j, i + 1) + beta + F[j];
            if (cost < F[i])
            {
                max_id = j;
                F[i] = cost;
            }
        }
        while (!R.empty() && F[i] < F[R.back()])
        {
            R.pop_back();
        }
        R.push_back(i);
        cp[i] = max_id;
    }

    R.clear();
    for (int i = n - 1; i >= 0; i = cp[i])
    {
        R.push_back(i + 1);
    }
    R.push_back(0);
    int j = 0;
    for (int i = 0; i < n; i++)
    {
        if (i == R[j + 1])
        {
            ++j;
        }
        path[i] = j;
    }
    return;
}
