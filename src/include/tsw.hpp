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
#include <iostream>
#include <unordered_map>

void TSWZJB(std::vector<Real> &x, std::vector<int> &path, const Real p = 0.9, const int w = 10)
{
    int n = x.size();
    if (n <= 2 * w || w <= 0)
        return;
    Real alpha[10] = {0.995, 0.99, 0.975, 0.95, 0.90, 0.10, 0.05, 0.025, 0.01, 0.005};
    Real df1[10] = {7.879, 6.635, 5.024, 3.841, 2.706, 0.016, 0.004, 0.001, 0.00016, 0.00004};
    Real df2[10] = {0.01, 0.02, 0.051, 0.103, 0.211, 4.605, 5.991, 7.378, 9.210, 10.597};
    Real threshold_z = 1.0;
    Real threshold_jb = 4.3;
    for (int i = 0; i < 9; ++i)
    {
        if (alpha[i] == p)
        {
            threshold_z = df1[i];
            threshold_jb = df2[i];
            break;
        }
        else if (alpha[i + 1] == p)
        {
            threshold_z = df1[i + 1];
            threshold_jb = df2[i + 1];
            break;
        }
        else if (alpha[i] > p && p > alpha[i + 1])
        {
            threshold_z = df1[i] + (p - alpha[i]) / (alpha[i + 1] - alpha[i]) * (df1[i + 1] - df1[i]);
            threshold_jb = df2[i] + (p - alpha[i]) / (alpha[i + 1] - alpha[i]) * (df2[i + 1] - df2[i]);
            break;
        }
    }
    if (p > 0.995)
    {
        threshold_z = 7.879;
        threshold_jb = 0.01;
    }
    else if (p < 0.005)
    {
        threshold_z = 0.00004;
        threshold_jb = 10.597;
    }
    Real mu;
    Real sigma;
    norm_data(x.data(), n, mu, sigma);
    std::vector<Real> zscore(n, 0);
    std::vector<Real> JB(n, 0);
    Real mean_w1 = 0;
    Real m2_w1 = 0;
    Real mean_w2 = 0;
    Real m2_w2 = 0;
    Real sum3 = 0;
    Real sum4 = 0;
    for (int i = 0; i < w; ++i)
    {
        Real delta1 = x[i] - mean_w1;
        mean_w1 += delta1 / Real(i + 1);
        m2_w1 = m2_w1 + delta1 * (x[i] - mean_w1);
        Real delta2 = x[i + w] - mean_w2;
        mean_w2 += delta2 / Real(i + 1);
        m2_w2 = m2_w2 + delta2 * (x[i + w] - mean_w2);

        sum3 += x[i + w] * x[i + w] * x[i + w];
        sum4 += x[i + w] * x[i + w] * x[i + w] * x[i + w];
    }
    for (int i = w; i < n - w; ++i)
    {
        Real delta1 = x[i] - x[i - w];
        Real delta2 = x[i + w] - x[i];
        Real mean_old1 = mean_w1;
        Real mean_old2 = mean_w2;
        mean_w1 += delta1 / Real(w);
        mean_w2 += delta2 / Real(w);
        m2_w1 += delta1 * (x[i] - mean_w1 + x[i - w] - mean_old1);
        m2_w2 += delta2 * (x[i + w] - mean_w2 + x[i] - mean_old2);
        Real val = (m2_w1 + m2_w2) / (Real(w) * Real(w - 1));
        val = (val <= 0) ? 1e-12 : val;
        zscore[i] = (mean_w1 - mean_w2) * (mean_w1 - mean_w2) / val;

        sum3 += x[i + w] * x[i + w] * x[i + w] - x[i] * x[i] * x[i];
        sum4 += x[i + w] * x[i + w] * x[i + w] * x[i + w] - x[i] * x[i] * x[i] * x[i];
        Real m2 = m2_w2 / Real(w);
        Real m3 = sum3 / w - 3 * mean_w2 * m2 - mean_w2 * mean_w2 * mean_w2;
        Real m4 = sum4 / w - 4 * mean_w2 * m3 - 6 * mean_w2 * mean_w2 * m2 - mean_w2 * mean_w2 * mean_w2 * mean_w2;
        Real skewness = m3 / std::pow(m2, 1.5);
        Real kurtosis = m4 / (m2 * m2);
        JB[i] = (w / 6.0) * (skewness * skewness + (kurtosis - 3) * (kurtosis - 3) / 4.0);
    }
    for (int i = 5500; i < 5600; i++)
    {
        std::cout << "i: " << i << ", z: " << zscore[i] << ", JB: " << JB[i] << std::endl;
    }
    int flag = -1;
    int local_best;
    std::vector<int> cp;
    cp.reserve(n);
    cp.push_back(0);
    Real th = threshold_z;
    int i = 0;
    while (i < n)
    {
        if (flag == -1 && zscore[i] >= th)
        {
            flag = i;
            local_best = i;
        }
        else if (flag != -1 && JB[i] <= threshold_jb)
        {
            flag = -1;
            cp.push_back(local_best);
            i += w;
        }
        else if (flag != -1)
        {
            local_best = (zscore[local_best] < zscore[i]) ? i : local_best;
            ++i;
        }
        else
        {
            ++i;
        }
    }
    cp.push_back(n);
    int j = 0;
    for (int i = 0; i < n; i++)
    {
        if (i == cp[j + 1])
        {
            ++j;
        }
        path[i] = j;
    }
    return;
}

void test_tsw()
{
    std::string fn = "./test.dat";
    std::vector<float> _data = read_dat(fn);
    std::vector<Real> data(_data.size());
    for (int i = 0; i < _data.size(); ++i)
        data[i] = _data[i];
    std::vector<int> path(data.size());
    TSWZJB(data, path, 0.9, 5);
    std::string fn_dat = "./path.dat";
    save_dat(fn_dat, path);
}
