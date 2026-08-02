#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_map>
#include <vector>
#include <string>

inline std::vector<int> kmer_to_index(const char* seq, const int n, const int kmer)
{
    static int char_to_bit_table[128] = {0};
    char_to_bit_table['A'] = char_to_bit_table['a'] = 0;
    char_to_bit_table['T'] = char_to_bit_table['t'] = 1;
    char_to_bit_table['C'] = char_to_bit_table['c'] = 2;
    char_to_bit_table['G'] = char_to_bit_table['g'] = 3;
    std::vector<int> codes;
    if (n < kmer) {
        return codes;
    }
    size_t  code = 0;
    for (int i = 0; i < kmer; ++i) {
        code = (code << 2) | char_to_bit_table[static_cast<int>(seq[i]) & 127];
    }
    const size_t mask = (1LL << (2 * kmer)) - 1;
    codes.push_back(static_cast<int>(code));
    for (int i = kmer; i < n; ++i) {
        code = ((code << 2) | char_to_bit_table[static_cast<int>(seq[i]) & 127]) & mask;
        codes.push_back(static_cast<int>(code));
    }
    return codes;
}

inline std::string index_to_kmer(const int* codes, int n, int kmer)
{
    std::string seq = "";
    static const char table[4] = {'A', 'T', 'C', 'G'};
    size_t code = static_cast<size_t>(codes[0]);
    for (int i = kmer - 1; i >= 0; --i) {
        seq += table[code & 3];
        code >>= 2;
    }
    std::reverse(seq.begin(), seq.end());
    for (int i = kmer; i < n; ++i) {
        seq += table[codes[i] & 3];
    }
    return seq;
}

template <typename T> 
void kmer_matrix(std::vector<int>& x, std::vector<int>& y, std::vector<T>& val, int kmer, T stay) {
    const size_t nstate = (1ULL << (2 * kmer));
    const size_t mask = nstate - 1;
    for(size_t i = 0; i < nstate; ++i) {
        x.push_back(static_cast<int>(i));
        y.push_back(static_cast<int>(i));
        val.push_back(stay);
        for(int next_base = 0; next_base < 4; ++next_base) {
            size_t next_state = ((i << 2) | next_base) & mask;
            if (i == next_base) {
                val.back() += (1.0 - stay) / 4.0;
                continue;
            }
            x.push_back(static_cast<int>(i));
            y.push_back(static_cast<int>(next_state));
            val.push_back((1.0 - stay) / 4.0);
        }
    }
    return;
}

template <typename T> 
std::vector<T> kmer_current(const std::vector<T>& data, const std::vector<char>& mv) {
    std::vector<T> res;
    for (int i = 0; i < data.size(); ++i) {
        if (mv[i] == 0) {
            continue;
        }
        res.push_back(data[i]);
    }
    return res;
}

template <typename T>
void coarse_align(const char* sequence, const int n, const int kmer, const int data_length, const int band_width, std::vector<int>& coo_x, std::vector<int>& coo_y) {
    const std::vector<int> seq = kmer_to_index(sequence, n, kmer);
    const int seq_length = seq.size();
    if (data_length <= seq.size()) {
        return;
    }
    coo_x.reserve(data_length * band_width);
    coo_y.reserve(data_length * band_width);
    std::vector<bool> repeat(seq_length, false);
    const int skip = data_length / seq_length;
    int base_id = 0;
    for (int i = 0; i < data_length; ++i) {
        if (i % skip == 0) {
            base_id += 1;
        }
        base_id = (base_id >= seq_length) ? seq_length - 1 : base_id;
        std::fill(repeat.begin(), repeat.end(), false);
        for (int j = 0; j < band_width; ++j) {
            const int base_id_j = std::clamp(base_id + j - band_width / 2, 0, seq_length - 1);
            if (repeat[base_id_j]) {
                continue;
            }
            repeat[base_id_j] = true;
            coo_y.push_back(seq[base_id_j]);
            coo_x.push_back(i);
        }
    }
}

inline void uniform_align(const int sequence_length, const int data_length, const int band_width, std::vector<int>& coo_x, std::vector<int>& coo_y) {
    if (data_length <= sequence_length) {
        return;
    }
    coo_x.reserve(data_length * band_width);
    coo_y.reserve(data_length * band_width);
    const int skip = data_length / sequence_length;
    int base_id = -1;
    for (int i = 0; i < data_length; ++i) {
        if (i % skip == 0) {
            base_id += 1;
        }
        base_id = std::clamp(base_id, 0, sequence_length - 1);
        for (int j = 0; j < band_width; ++j) {
            const int base_id_j = base_id + j - band_width / 2;
            if (base_id_j < sequence_length && base_id_j >= 0) {
                coo_y.push_back(base_id_j);
                coo_x.push_back(i);
            }
        }
    }
}

inline void mv_align(const int sequence_length, const int* mv, const int data_length, const int band_width, std::vector<int>& coo_x, std::vector<int>& coo_y) {
    int max_val = std::numeric_limits<int>::min();
    int min_nonnegative = std::numeric_limits<int>::max();
    for (int i = 0; i < data_length; i++) {
        const int v = mv[i];
        if (v > max_val)
            max_val = v;
        if (v >= 0 && v < min_nonnegative)
            min_nonnegative = v;
    }
    const int query_length = max_val - min_nonnegative + 1;
    coo_x.reserve(data_length * band_width);
    coo_y.reserve(data_length * band_width);
    int skip = query_length / sequence_length;
    if (skip < 1) {
        skip = 1;
    }
    int ref_id = -1;
    int que_id = 0;
    int pre_mv = std::numeric_limits<int>::max();
    for (int i = 0; i < data_length; ++i) {
        if (mv[i] != pre_mv) {
            ++que_id;
            pre_mv = mv[i];
            if (que_id % skip == 0) {
                ++ref_id;
            }
        }
        ref_id = std::clamp(ref_id, 0, sequence_length - 1);
        for (int j = 0; j < band_width; ++j) {
            const int base_id_j = ref_id + j - band_width / 2;
            if (base_id_j < sequence_length && base_id_j >= 0) {
                coo_y.push_back(base_id_j);
                coo_x.push_back(i);
            }
        }
    }
}

