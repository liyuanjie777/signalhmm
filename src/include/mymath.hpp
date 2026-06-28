#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_map>
#include <vector>

template <typename T> int argmax(T* x, int n)
{
    int id = 1;
    for (int i = 2; i < n; i++)
    {
        id = (x[id] <= x[i]) ? i : id;
    }
    return id;
}

template <typename T> int argmin(T* x, int n)
{
    int id = 0;
    for (int i = 1; i < n; i++)
    {
        id = (x[id] > x[i]) ? i : id;
    }
    return id;
}

template <typename T> void mean_var(const T* sum, const T* sum2, int i, int j, T* mv)
{
    int n = j - i;
    if (n > 1)
    {
        mv[0] = sum[j] - sum[i];
        mv[0] = mv[0] / n;
        mv[1] = sum2[j] - sum2[i];
        mv[1] = (mv[1] - n * mv[0] * mv[0]) / (n - 1);
    }
    return;
}

template <typename T> T logsumexp(const T* x, int n)
{
    T max_x = *std::max_element(x, x + n);
    T sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += std::exp(x[i] - max_x);
    }
    return max_x + std::log(sum);
}

template <typename T> T logsumexp2x(const T& x, const T& y)
{
    if (x > y)
        return x + std::log1p(std::exp(y - x));
    else
        return y + std::log1p(std::exp(x - y));
}

template <typename T> T logdiff(const T& x, const T& x2)
{
    T diff = x2 - x;
    T var = x + std::log1p(-std::exp(diff));
    return var;
}

template <typename T> T log_normalize(T* x, int n)
{
    T max_x = *std::max_element(x, x + n);
    T sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += std::exp(x[i] - max_x);
    }
    T val = max_x + std::log(sum);
    for (int i = 0; i < n; i++)
    {
        x[i] -= val;
    }
    return val;
}

template <typename T> void vec_add(T* y, const T* x, const int n)
{
    for (int i = 0; i < n; i++)
    {
        y[i] += x[i];
    }
}

template <typename T> void vec_mul(T* y, const T* x, const int n)
{
    for (int i = 0; i < n; i++)
    {
        y[i] *= x[i];
    }
}

template <typename T> void vec_mean(const T* x, const int n, T* y, int dim)
{
    memset(y, dim, 0);
    T count = 0;
    for (int i = 0; i < n; ++i)
    {
        count += 1;
        for (int j = 0; j < dim; ++j)
        {
            y[j] += (x[i * dim + j] - y[j]) / count;
        }
    }
}

template <typename T> T vec_sum(const T* x, const int n)
{
    T sum = 0.0;
    T c = 0.0;
    for (int i = 0; i < n; ++i)
    {
        T y = x[i] - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

template <typename T> void norm_data(T* x, const int n, T& mu, T& sigma)
{
    T mean = 0;
    T m2 = 0;
    T delta = 0;
    for (int i = 0; i < n; ++i)
    {
        delta = x[i] - mean;
        mean += delta / T(i + 1);
        m2 += delta * (x[i] - mean);
    }
    m2 /= T(n);
    m2 = std::sqrt(m2);
    for (int i = 0; i < n; ++i)
    {
        x[i] -= mean;
        x[i] /= m2;
    }
    mu = mean;
    sigma = m2;
    return;
}

template <typename T> T normalize(T* x, const int n)
{
    T sum = vec_sum(x, n);
    if (sum == 0)
        return 1e-30;
    for (int i = 0; i < n; ++i)
    {
        x[i] /= sum;
    }
    return sum;
}

template <typename T> void smooth(T* x, const int n)
{
    std::vector<T> tmp(n);
    tmp[0] = (x[0] + x[1]) / 2;
    for (int i = 1; i < n - 1; ++i)
    {
        tmp[i] = (x[i - 1] + x[i] + x[i + 1]) / 3;
    }
    tmp[n - 1] = (x[n - 2] + x[n - 1]) / 2;
    std::copy(tmp.begin(), tmp.end(), x);
    return;
}

template <typename T>
T JB_cost(const T* sum1, const T* sum2, const T* sum3, const T* sum4, const int i, const int j)
{
    // pre sum (n + 1)
    T n = T(j - i);
    T m1 = (sum1[j] - sum1[i]) / n;
    T S2 = (sum2[j] - sum2[i]) / n;
    T S3 = (sum3[j] - sum3[i]) / n;
    T S4 = (sum4[j] - sum4[i]) / n;
    T m2 = S2 - m1 * m1;
    T m3 = S3 - 3 * m1 * S2 + 2 * m1 * m1 * m1;
    T m4 = S4 - 4 * m1 * S3 + 6 * m1 * m1 * S2 - 3 * m1 * m1 * m1 * m1;
    T S = m3 * m3 * std::pow(m2, -3);
    T K = m4 * std::pow(m2, -2) - 3;
    T JB = std::abs((n / 6.0) * (S + K * K / 4.0));
    return JB;
}

template <typename T>
T median(const T* data, const int n)
{
    if (n <= 0) return T(0);
    std::vector<T> nums(data, data + n);
    T res = 0.0;
    if (n % 2 == 0) {
        std::nth_element(nums.begin(), nums.begin() + n / 2, nums.end());
        T m1 = nums[n / 2];
        std::nth_element(nums.begin(), nums.begin() + n / 2 - 1, nums.begin() + n / 2);
        T m2 = nums[n / 2 - 1];
        res = (m1 + m2) / 2.0;
    }
    else {
        std::nth_element(nums.begin(), nums.begin() + n / 2, nums.end());
        res = nums[n / 2];
    }
    return res;
}

template <typename T>
void MAD(T* data, const int n)
{
    if (n <= 0) return;
    T med = median(data, n);
    std::vector<T> mad(data, data + n);
    for (int i = 0; i < n; ++i) {
        mad[i] = std::abs(data[i] - med);
    }
    T mad_med = median(mad.data(), n);
    for (int i = 0; i < n; ++i) {
        data[i] = (data[i] - med) / (1.4826 * mad_med);
    }
    return;
}
