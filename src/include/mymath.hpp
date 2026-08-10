#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
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

template <typename T>
T logsumexp(const T* x, const int n) {
    if(n==0) {
        return -std::numeric_limits<T>::infinity();
    }
    T max_x = x[0];
    for(int i = 1;i < n;i++) {
        if(x[i] > max_x)
            max_x = x[i];
    }
    if(std::isinf(max_x)) {
        return max_x;
    }
    T sum=0;
    for(int i = 0;i < n;i++) {
        sum += std::exp(x[i] - max_x);
    }
    return max_x + std::log(sum);
}

template <typename T> T logsumexp2x(T x, T y)
{
    if (x < y)
        std::swap(x,y);
    if (std::isinf(x))
        return x;
    return x + std::log1p(std::exp(y - x));
}

template <typename T> T logdiff(const T& x, const T& x2)
{
    T diff = x2 - x;
    T var = x + std::log1p(-std::exp(diff));
    return var;
}

template <typename T> T log_normalize(T* x, int n) {
    T max_x = x[0];
    for(int i = 1;i < n;i++) {
        if(x[i] > max_x)
            max_x = x[i];
    }
    if(std::isinf(max_x)) {
        return max_x;
    }
    T sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += std::exp(x[i] - max_x);
    }
    T val = max_x + std::log(sum);
    for (int i = 0; i < n; i++) {
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


template <typename T> void vec_add(T* y, const T x, const int n) {
    for (int i = 0; i < n; i++) {
        y[i] += x;
    }
}


template <typename T> void vec_scale(T* y, const T x, const int n)
{
    for (int i = 0; i < n; i++)
    {
        y[i] *= x;
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

template <typename T>
T vec_norm(T* x, const int n) {
    if (n <= 0) return 0;
    T sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += x[i];
    }
    if (sum == static_cast<T>(0)) {
        return static_cast<T>(0);
    }
    T inv_sum = static_cast<T>(1) / sum;

    for (int i = 0; i < n; ++i) {
        x[i] *= inv_sum;
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

template <typename T>
std::vector<T> sliding_mean_std(const T* data, const int n, const int window) {
    std::vector<T> res(n * 2, 0.0);
    const int half = window / 2;
    T mean = 0;
    T m2 = 0;
    T delta = 0;
    for (int i = 0; i < window; ++i) {
        delta = data[i] - mean;
        mean += delta / T(i + 1);
        m2 += delta * (data[i] - mean);
    }
    res[0] = data[0];
    res[1] = std::sqrt(m2 / T(window));
    for (int i = 1; i <= half; ++i) {
        res[i * 2] = data[i];
        res[i * 2 + 1] = res[1];
    }
    for (int i = half + 1; i < n - half; ++i) {
        T mean_old = mean;
        mean += (data[i + half] - data[i - half - 1]) / T(window);
        m2 += (data[i + half] - mean_old) * (data[i + half] - mean)
        - (data[i - half - 1] - mean_old) * (data[i - half - 1] - mean);
        res[i * 2] = data[i];
        res[i * 2 + 1] = std::sqrt(m2 / T(window));
    }
    for (int i = n - half; i < n; ++i) {
        res[i * 2] = data[i];
        res[i * 2 + 1] = res[2 * (n - half - 1) + 1];
    }
    return res;
}
