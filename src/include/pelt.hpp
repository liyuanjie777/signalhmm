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
#include <utility>

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
