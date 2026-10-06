#pragma once
#include "variable.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>

struct Read {
    int index;
    std::string uuid;
    std::string chrom;
    std::vector<Real> data;
    std::vector<int> mv;
};

std::string readFA(const std::string& fn, const std::string& target_chrom);
class ReadsFile
{
public:
    ReadsFile(){};
    void load(const std::string& fn, bool shuffle);
    const int size () const {return _uuids.size();};
    Read read(const int i);
    std::vector<Read> readChunk(const int batch, int min_size);
    void close() {
        if (_file.is_open()) {
            _file.close();
        }
        _cache_data.clear();
        _uuids.clear();
    }
    void reset() {
        _cur_id = 0;
    }
    static void save(const std::string& fn, const std::vector<Read>& read_datas);

private:
    static std::vector<char> decompress(const char* data, size_t n);
    std::vector<std::string> _uuids;
    std::vector<std::string> _chroms;
    std::string _filename;
    std::fstream _file;
    std::vector<char> _cache_data;
    int _compressed_level = 4;
    int _cur_id = 0;
    std::vector<int> _random_idx;
    std::vector<size_t> _poss_bytes;
    std::vector<size_t> _pose_bytes;
};
