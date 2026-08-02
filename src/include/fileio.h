#pragma once
#include "variable.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>

struct Read {
    std::string uuid;
    std::string chrom;
    int spos;
    int epos;
    std::vector<Real> data;
    std::vector<int> mv;
};

struct ReadIDX {
    std::vector<size_t> poss_bytes;
    std::vector<size_t> pose_bytes;
    std::vector<std::string> uuids;
    std::vector<int> random_idx;
    int cur_id = 0;
};

std::string readFA(const std::string& fn, const std::string& target_chrom);
class ReadsFile
{
public:
    ReadsFile(){};
    void load(const std::string& fn, bool shuffle);
    Read read(const std::string& chrom);
    std::vector<Read> readChunk(const int batch, const std::string& chrom);
    void close() {
        if (_file.is_open()) {
            _file.close();
        }
        _chrom_map.clear();
        _cache_data.clear();
        _uuids.clear();
    }
    void reset() {
        for (auto& [chrom, idx] : _chrom_map) {
            idx.cur_id = 0;
        }
    }
    static void save(const std::string& fn, const std::vector<Read>& read_datas);

private:
    static std::vector<char> decompress(const char* data, size_t n);

    std::unordered_map<std::string, int> _uuids;
    std::unordered_map<std::string, ReadIDX> _chrom_map;
    std::string _filename;
    std::fstream _file;
    std::vector<char> _cache_data;
    int _compressed_level = 4;
};
