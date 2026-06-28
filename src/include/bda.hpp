/*
 *  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-10-05
 *  License: MIT
 */
#pragma once
#include "dtw.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace
{
    int median(const int* x, int n, int& id)
    {
        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i)
        {
            idx[i] = i;
        }
        std::sort(x, x + n, [](const int& a, const int& b) { return a < b; });
        id = idx[n / 2];
        if (n % 2 == 0)
        {

            return (x[idx[n / 2 - 1]] + x[idx[n / 2]]) / 2;
        }
        return x[idx[n / 2]];
    }
} // namespace

// BDA EM
template <typename T>
void BDA(const std::vector<T>& seqs, const std::vector<int>& batch, std::vector<T>& output, int dim)
{
    int median_id = 0;
    int n = median(batch.data(), batch.size(), median_id);
    output.resize(n);
    std::vector<int> weight(n, 0);
    std::vector<T> values(n, 0);
    std::vector<int> ptrs(batch.size(), 0);
    std::vector<int> tmp;

    // initial values
    size_t global_id = 0;
    for (int i = 0; i < batch.size(); ++i)
    {
        if (i == median_id)
            break;
        else
            global_id += size_t(batch[i]);
    }
    for (int i = 0; i < n; ++i)
    {
        output[i] = seqs[global_id + i];
    }

    // calculate dtw
    global_id = 0;
    for (int i = 0; i < batch.size(); ++i)
    {
        tmp.clear();
        DTW(&seqs[global_id], &output[0], dim, tmp);
        for (int k = 0; k < tmp.size(); k += 2)
        {
            n
        }
    }
}
