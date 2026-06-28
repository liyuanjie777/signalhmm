#pragma once
#include "fileio.h"
#include "mymath.hpp"
#include "variable.h"
#include <algorithm>
#include <cmath>
#include <cstring>

Real log_student_t_predictive(Real* sum, Real* sum2, const Real xt, const int i, const int j,
    const Real mu0, const Real kappa0, const Real alpha0, const Real beta0)
{
    Real PI = 3.14159265358979323846;
    Real n = j - i;
    if (n == 0)
    {
        Real nu0 = 2.0 * alpha0;
        Real scale2 = beta0 * (kappa0 + 1.0) / (alpha0 * kappa0);
        Real logt1 = std::lgamma((nu0 + 1.0) / 2.0) - std::lgamma(nu0 / 2.0);
        Real logt2 = -0.5 * std::log(nu0 * PI * scale2);
        Real logt3 = -((nu0 + 1.0) / 2.0) * std::log(1.0 + std::pow(xt - mu0, 2) / (nu0 * scale2));
        return logt1 + logt2 + logt3;
    }
    Real mean = (sum[j] - sum[i]) / n;
    Real var = (sum2[j] - sum2[i]) / n - mean * mean;
    var = std::max(var, static_cast<Real>(1e-8));
    Real kappa_n = kappa0 + n;
    Real alpha_n = alpha0 + n / 2.0;
    Real beta_n = beta0 + 0.5 * var * n + (kappa0 * n * std::pow(mean - mu0, 2)) / (2.0 * kappa_n);
    Real mu_n = (kappa0 * mu0 + n * mean) / kappa_n;
    Real nu_n = 2.0 * alpha_n;
    Real scale2 = beta_n * (kappa_n + 1) / (alpha_n * kappa_n);
    Real logt1 = std::lgamma((nu_n + 1.0) / 2.0) - std::lgamma(nu_n / 2.0);
    Real logt2 = -0.5 * std::log(nu_n * PI * scale2);
    Real logt3 = -((nu_n + 1.0) / 2.0) * std::log(1.0 + std::pow(xt - mu_n, 2) / (nu_n * scale2));
    return logt1 + logt2 + logt3;
}

void BOCPD(const std::vector<Real>& x, std::vector<int>& path, const Real koff,
    const int max_run_length = 500, const int window_size = 10)
{
    // log harzard function
    int n = x.size();
    Real alpha = 3;
    Real kappa = window_size;
    Real log_stay = std::log(1 - koff);
    Real log_change = std::log(koff);
    int skip = window_size;
    int N = (n + 1) * (max_run_length + 1);
    std::vector<Real> s1(n + 1);
    std::vector<Real> s2(n + 1);
    std::vector<Real> logpr(N);
    std::vector<Real> logpr_break(n);
    std::vector<Real> mu0(n);
    std::vector<Real> beta0(n);

    s1[0] = 0;
    s2[0] = 0;
    for (int i = 0; i < n; i++)
    {
        s1[i + 1] = x[i] + s1[i];
        s2[i + 1] = x[i] * x[i] + s2[i];
    }
    for (int i = 0; i < n; i++)
    {
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
    for (size_t i = 1; i <= n; ++i)
    {
        size_t id = i * (max_run_length + 1);
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
