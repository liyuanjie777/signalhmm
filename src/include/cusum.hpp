
/*
 *  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-08-15
 *  License: MIT
 */

#pragma once
#include <vector>
#include "variable.h"

template <typename T>
T _cusum_min_std(const T* data, const int n, const int window) {
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
void cumsum(const std::vector<T>& data, std::vector<int>& mv, const T sigma, const T h, const int window_size = 12) {
    T logp, logn = 0;
    T gpos = 0.0;
    T gneg = 0.0;
    T anchor = 0;
    T m1 = data[0];
    T m2 = 0;
    T min_std = _cusum_min_std(data.data(), data.size(), window_size);
    int idx = 0;
    mv.clear();
    mv.reserve(data.size());
    mv.push_back(0);
    for (int i = 1; i < data.size(); i++) {
        T delta = data[i] - m1;
        m1 += delta / static_cast<T>(i - anchor + 1);
        T delta2 = data[i] - m1;
        m2 += delta * delta2;
        T stddev = std::max(min_std, std::sqrt(m2 / (i - anchor + 1.0)));
        logp = sigma / stddev * (data[i] - m1 - sigma * stddev / 2);
        logn = -1 * sigma / stddev * (data[i] - m1 + sigma * stddev / 2);
        const T new_gpos = gpos + logp;
        if (new_gpos <= 0) {
            gpos = 0;
        }
        else {
            gpos = new_gpos;
        }
        const T new_gneg = gneg + logn;
        if (new_gneg <= 0) {
            gneg = 0;
        }
        else {
            gneg = new_gneg;
        }
        if (gpos >= h || gneg >= h) {
            idx++;
            anchor = i;
            gpos = 0;
            gneg = 0;
            m1 = data[i];
            m2 = 0.0;
        }
        mv.push_back(idx);
    }
    return;
}
