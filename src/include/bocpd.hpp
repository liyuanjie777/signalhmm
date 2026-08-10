#pragma once
#include "fileio.h"
#include "mymath.hpp"
#include "variable.h"
#include <algorithm>
#include <cmath>
#include <cstring>

Real bocpd_log_prob(const Real * sum, const Real * sum2, const int i, const int j) {
    const int n = j - i;
    if (n <= 0) {
        return 0;
    }
    const Real s = sum[j] - sum[i];
    const Real q = sum2[j] - sum2[i];
    const Real mean = s / n;
    const Real var = q / n - mean * mean;
    return -static_cast<Real>(0.5) * n * std::log(static_cast<Real>(2.0) * M_PI * M_E * var);
}

void BOCPD(const std::vector<Real>& x, std::vector<int>& path, const std::vector<std::pair<Real, Real>>& hazard_table, const int max_run_length = 200, const int window_size = 10) {
    // log harzard function
    int n = x.size();
    int N = (n + 1) * (max_run_length + 1);
    std::vector<Real> s1(n + 1);
    std::vector<Real> s2(n + 1);
    std::vector<Real> logpr(N);
    std::vector<Real> logpr_break(n);

    s1[0] = 0;
    s2[0] = 0;
    for (int i = 0; i < n; i++) {
        s1[i + 1] = x[i] + s1[i];
        s2[i + 1] = x[i] * x[i] + s2[i];
    }
    for (int i = 0; i < n; i++) {
        if (i + skip < n)
        {
            mu0[i] = (s1[i + skip] - s1[i]) / skip;
            Real var0 = (s2[i + skip] - s2[i]) / skip - mu0[i] * mu0[i];
            var0 = (var0 <= 0) ? 1e-6 : var0;
            beta0[i] = var0 * (alpha - 1.0);
        }
        else
        {
            mu0[i] = mu0[n - skip - 1];
            beta0[i] = beta0[n - skip - 1];
        }
    }
    logpr[0] = 0.0;
    for (int i = 1; i <= n; ++i) {
        int id = i * (max_run_length + 1);
        size_t id_pre = (i - 1) * (max_run_length + 1);
        size_t max_i = (i <= max_run_length) ? i : max_run_length;
        Real likeli_change = log_student_t_predictive(s1.data(), s2.data(), x[i - 1], int(i - 1),
            int(i - 1), mu0[i - 1], kappa, alpha, beta0[i - 1]);
        for (size_t j = 1; j <= max_i; ++j)
        {
            Real likeli_stay = log_student_t_predictive(s1.data(), s2.data(), x[i - 1], i - j,
                i - 1, mu0[i - 1], kappa, alpha, beta0[i - 1]);
            logpr[id + j] = logpr[id_pre + j - 1] + log_stay + likeli_stay;
            logpr_break[j - 1] = logpr[id_pre + j - 1] + log_change + likeli_change;
        }
        logpr[id] = logsumexp(logpr_break.data(), max_i);
        log_normalize(logpr.data() + id, max_i + 1);
    }
    // recall
    std::vector<int> change_points;
    size_t i = n;
    while (i > 0)
    {
        size_t id = i * (max_run_length + 1);
        size_t max_i = (i <= max_run_length) ? i : max_run_length;
        int r;
        r = argmax(logpr.data() + id, max_i + 1);
        if (r == 0)
        {
            i--;
            continue;
        }
        int cp = i - r;
        i = cp;
        change_points.push_back(cp);
    }
    std::reverse(change_points.begin(), change_points.end());
    change_points.push_back(n);
    int j = 0;
    for (int i = 0; i < n; i++)
    {
        if (i == change_points[j + 1])
        {
            ++j;
        }
        path[i] = j;
    }
    return;
}

void test_bocpd(Real koff = 0.0013)
{
    std::string fn = "./test.dat";
    std::vector<float> _data = read_dat(fn);
    std::vector<Real> data(_data.size());
    for (int i = 0; i < _data.size(); ++i)
        data[i] = _data[i];
    std::vector<int> path(data.size());
    BOCPD(data, path, koff);
    std::string fn_dat = "./path.dat";
    save_dat(fn_dat, path);
}
