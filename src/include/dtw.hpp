/*
 *  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-08-07
 *  License: MIT
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
    template <typename T> inline T l2_dist(T* p0, T* p1, int dim)
    {
        T res = 0;
        for (int k = 0; k < dim; k++)
        {
            res += (p0[k] - p1[k]) * (p0[k] - p1[k]);
        }
        return res;
    }

    template <typename T> inline T min3(const T& a, const T& b, const T& c)
    {
        if (a <= b && a <= c)
            return a;
        if (b <= c)
            return b;
        else
            return c;
    }

    template <typename T> inline void argmin3(const T& a, const T& b, const T& c, int& i, int& j)
    {
        if (a <= b && a <= c)
        {
            i--;
            j--;
        }
        else if (b <= c)
        {
            i--;
        }
        else
        {
            j--;
        }
        return;
    }
} // namespace
template <typename T>
int DTW(const std::vector<T>& p0, const std::vector<T>& p1, int dim, std::vector<int>& warping_path)
{
    int n = p0.size();
    int m = p1.size();
    int l = 2 * (n + m - 1);
    warping_path.assign(l, 0);
    // init path array
    T* path = new T[n * m];
    path[0] = l2_dist(p0.data(), p1.data(), dim);
    for (int i = 1; i < n; i++)
    {
        size_t idx = i * m;
        path[idx] = l2_dist(p0.data() + i * dim, p1.data(), dim) + path[idx - m];
    }
    for (int j = 1; j < m; j++)
    {
        path[j] = l2_dist(p0.data(), p1.data() + j * dim, dim) + path[j - 1];
    }

    // dp fill path array
    for (int i = 1; i < n; i++)
    {
        size_t idx = i * m;
        for (int j = 1; j < m; j++)
        {
            size_t id = idx + j;
            path[id] = l2_dist(p0.data() + i * dim, p1.data() + j * dim, dim);
            path[id] += min3(path[id - m - 1], path[id - m], path[id - 1]);
        }
    }

    // back search
    int i = n - 1;
    int j = m - 1;
    warping_path[0] = i;
    warping_path[1] = j;
    int k = 2;
    while ((i > 0) || (j > 0))
    {
        size_t id = i * m + j;
        if (i > 0 && j > 0)
        {
            argmin3(path[id - m - 1], path[id - m], path[id - 1], i, j);
        }
        else if (i > 0)
        {
            i--;
        }
        else
        {
            j--;
        }
        warping_path[k] = i;
        warping_path[k + 1] = j;
        k += 2;
    }
    delete[] path;
    warping_path.resize(k);
    return k / 2;
    uint64_t* u64 = reinterpret_cast<uint64_t*>(warping_path.data());
    for (int i = 0; i < k; i++)
    {
        std::swap(u64[i], u64[k - 1 - i]);
    }
    // for (int i = 0; i < k; i++) {
    //	std::cout << "(" << warping_path[i * 2] << ", " << warping_path[i * 2 + 1] << ")" <<
    // std::endl;
    // }
}

template <typename T>
int DTW_omp(
    const std::vector<T>& p0, const std::vector<T>& p1, int dim, std::vector<int>& warping_path)
{
    int n = p0.size();
    int m = p1.size();
    int l = 2 * (n + m - 1);
    warping_path.assign(l, 0);
    // init path array
    T* path = new T[n * m];
    path[0] = l2_dist(p0.data(), p1.data(), dim);
    for (int i = 1; i < n; i++)
    {
        size_t idx = i * m;
        path[idx] = l2_dist(p0.data() + i * dim, p1.data(), dim) + path[idx - m];
    }
    for (int j = 1; j < m; j++)
    {
        path[j] = l2_dist(p0.data(), p1.data() + j * dim, dim) + path[j - 1];
    }

// dp fill path array
#pragma omp parallel
    {
        for (int s = 2; s <= n + m - 2; s++)
        {
#pragma omp for schedule(static)
            for (int i = std::max(1, s - m + 1); i <= std::min(n - 1, s - 1); i++)
            {
                int j = s - i;
                size_t id = (size_t)i * m + j;
                T d = l2_dist(p0.data() + i * dim, p1.data() + j * dim, dim);
                path[id] = d + min3(path[id - m - 1], path[id - m], path[id - 1]);
            }
        }
    }
    // back search
    int i = n - 1;
    int j = m - 1;
    warping_path[0] = i;
    warping_path[1] = j;
    int k = 2;
    while ((i > 0) || (j > 0))
    {
        size_t id = i * m + j;
        if (i > 0 && j > 0)
        {
            argmin3(path[id - m - 1], path[id - m], path[id - 1], i, j);
        }
        else if (i > 0)
        {
            i--;
        }
        else
        {
            j--;
        }
        warping_path[k] = i;
        warping_path[k + 1] = j;
        k += 2;
    }
    delete[] path;
    warping_path.resize(k);
    k /= 2;
    uint64_t* u64 = (uint64_t*)warping_path.data();
    for (int i = 0; i < k; i++)
    {
        std::swap(u64[i], u64[k - 1 - i]);
    }
    // for (int i = 0; i < k; i++) {
    //	std::cout << "(" << warping_path[i * 2] << ", " << warping_path[i * 2 + 1] << ")" <<
    // std::endl;
    // }
    return k;
}
