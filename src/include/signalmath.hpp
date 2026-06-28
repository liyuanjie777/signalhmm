#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_map>
#include <vector>
#include <string>

inline std::vector<int> kmer_to_index(const char* seq, int n, int kmer)
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
        code = (code << 2) | char_to_bit_table[(int)seq[i] & 127];
    }
    size_t mask = (1LL << (2 * kmer)) - 1;
    codes.push_back(static_cast<int>(code));
    for (int i = kmer; i < n; ++i) {
        code = ((code << 2) | char_to_bit_table[(int)seq[i] & 127]) & mask;
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
    size_t nstate = (1ULL << (2 * kmer));
    size_t mask = nstate - 1;
    for(size_t i = 0; i < nstate; ++i) {
        for(int next_base = 0; next_base < 4; ++next_base) {
            size_t next_state = ((i << 2) | next_base) & mask;
            x.push_back(static_cast<int>(i));
            y.push_back(static_cast<int>(next_state));
            if (next_state == i) {
                val.push_back(stay);
            } else {
                val.push_back((1.0 - stay) / 3.0);
            }
        }
    }
    return;
}
