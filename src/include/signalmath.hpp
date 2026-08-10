#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <queue>
#include <tuple>
#include <vector>

inline Real min_variance(const Real* data, const int n, const int window) {
    Real mu = 0.0;
    Real sd = 0.0;
    for (int i = 0; i < window; i++) {
        Real mu_old = mu;
        mu += (data[i] - mu) / static_cast<Real>(i + 1);
        sd += (data[i] - mu_old) * (data[i] - mu);
    }
    Real min_var = sd;
    for (int i = window; i < n; i++) {
        Real mu_old = mu;
        mu += (data[i] - data[i - window]) / static_cast<Real>(window);
        sd += (data[i - window] - mu_old) * (data[i - window] - mu) + (data[i] - mu) * (data[i] - mu_old);
        if (sd < min_var) {
            min_var = sd;
        }
    } return min_var / static_cast<Real>(window);
}

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

inline std::vector<std::string> sequence_to_kmers(const std::string& seq, const int pad_size, const char pad_char, const int kmer_len) {
    std::string padding_start(pad_size, pad_char);
    std::string padding_end(kmer_len - pad_size - 1, pad_char);
    std::string padded_seq = padding_start + seq + padding_end;
    std::vector<std::string> kmers;
    if (padded_seq.length() < static_cast<size_t>(kmer_len)) {
        return kmers;
    }
    kmers.reserve(seq.size());
    for (size_t i = 0; i < seq.size(); ++i) {
        kmers.push_back(padded_seq.substr(i, kmer_len));
    }
    return kmers;
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
    float skip = static_cast<float>(sequence_length) / static_cast<float>(query_length);
    float ref_id = 0.0f;
    int pre_mv = std::numeric_limits<int>::max();
    for (int i = 0; i < data_length; ++i) {
        if (mv[i] != pre_mv) {
            pre_mv = mv[i];
            ref_id += skip;
        }
        const int rid = std::clamp(int(ref_id), 0, sequence_length - 1);
        for (int j = 0; j < band_width; ++j) {
            const int base_id_j = rid + j - band_width / 2;
            if (base_id_j < sequence_length && base_id_j >= 0) {
                coo_y.push_back(base_id_j);
                coo_x.push_back(i);
            }
        }
    }
}

inline std::vector<int> aggregate_segments(const std::vector<int>& mv, const int sequence_length) {
    if (mv.empty() || sequence_length <= 0) {
        return {};
    }
    struct Segment {int length; int prev = -1; int next = -1; bool alive = true;};
    std::vector<Segment> segs;
    for (int i = 0; i < static_cast<int>(mv.size()); ) {
        int j = i + 1;
        while (j < static_cast<int>(mv.size()) && mv[j] == mv[i]) {
            ++j;
        }
        segs.push_back({j - i,static_cast<int>(segs.size()) - 1,static_cast<int>(segs.size()) + 1, true});
        i = j;
    }
    segs.back().next = -1;

    const int N = static_cast<int>(segs.size());
    if (N <= sequence_length) {
        std::vector<int> result;
        result.reserve(mv.size());
        int id = 0;
        for (const auto& seg : segs) {
            for (int i = 0; i < seg.length; ++i) {
                result.push_back(id);
            }
            ++id;
        }
        return result;
    }

    using Node = std::pair<int, int>;
    std::priority_queue<Node, std::vector<Node>, std::greater<>> heap;
    for (int i = 0; i < N; ++i) {
        heap.emplace(segs[i].length, i);
    }
    int count = N;
    while (count > sequence_length) {
        auto [len, idx] = heap.top();
        heap.pop();
        if (!segs[idx].alive || segs[idx].length != len) {
            continue;
        }

        const int left  = segs[idx].prev;
        const int right = segs[idx].next;
        int target;
        if (left == -1) {
            target = right;
        }
        else if (right == -1) {
            target = left;
        }
        else {
            if (segs[left].length <= segs[right].length)
                target = left;
            else
                target = right;
        }
        if (target == left) {
            segs[left].length += segs[idx].length;
            segs[left].next = segs[idx].next;
            if (segs[idx].next != -1)
                segs[segs[idx].next].prev = left;
            segs[idx].alive = false;
            heap.emplace(segs[left].length, left);
        }
        else {
            segs[right].length += segs[idx].length;
            segs[right].prev = segs[idx].prev;
            if (segs[idx].prev != -1)
                segs[segs[idx].prev].next = right;
            segs[idx].alive = false;
            heap.emplace(segs[right].length,right);
        }
        --count;
    }

    std::vector<int> result;
    result.reserve(mv.size());
    int idx = 0;
    while (idx < N && !segs[idx].alive)
        ++idx;
    int new_id = 0;
    while (idx != -1) {
        for (int i = 0; i < segs[idx].length; ++i)
            result.push_back(new_id);
        idx = segs[idx].next;
        ++new_id;
    }
    return result;
}

inline std::vector<std::vector<int>> compute_adj_list(const std::vector<std::vector<int>>& expand_seq, const std::vector<int>& mv, const int band_width) {
    const int sequence_length = expand_seq.size();
    const int data_length = mv.size();
    const std::vector<int> mv_agg = aggregate_segments(mv, sequence_length);

    std::vector<std::vector<int>> adj_list(data_length);
    for (int i = 0; i < data_length; ++i) {
        adj_list[i].reserve((band_width * 2 + 1) * expand_seq[0].size());
    }
    for (int i = 0; i < data_length; ++i) {
        int j = mv_agg[i];
        const int start = std::clamp(j - band_width, 0, sequence_length - 1);
        const int end = std::clamp(j + band_width, 0, sequence_length - 1);
        for (int k = start; k <= end; ++k) {
            for (int m = 0; m < expand_seq[k].size(); ++m) {
                adj_list[i].push_back(expand_seq[k][m]);
            }
        }
        std::sort(adj_list[i].begin(), adj_list[i].end());
    }
    return adj_list;
}