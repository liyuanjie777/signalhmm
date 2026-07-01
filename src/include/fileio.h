#pragma once
#include "variable.h"
#include <string>
#include <vector>
#include <unordered_map>

class ReadsFile
{
public:
    ReadsFile(){};
    void load(std::string& fn, int max_cache, bool shuffle);
    void readID(std::string& sequence, std::vector<Real>& data, std::vector<char>& mv, const std::string& uuid);
    void read(std::string& sequence, std::vector<Real>& data, std::vector<char>& mv);
    int readChunk(std::vector<Real>& data, std::vector<size_t>& batchs, std::vector<ChunkInfo>& chunkinfo, int chunk_size, int batch_size, int stride);
    std::string getSequence(const int id);
    void reset() {_glob_id = 0; _cur_data_i = 0;};
    static void save(const std::string& fn, const std::string& sequence, const std::vector<Real>& data, const std::vector<char>& mv, const std::string& uuid);
    int readsNumber;

private:
    static std::vector<char> decompress(const char* data, size_t n);
    void rescaleData(std::vector<float>& data, float scale, float offset);

    std::vector<size_t> _poss_bytes;
    std::vector<size_t> _pose_bytes;
    std::vector<float> _scale;
    std::vector<float> _offset;
    std::unordered_map<std::string, int> _uuids;
    std::vector<std::string> _id_to_uuids;
    std::string _filename;
    // cache data
    std::vector<char> _cache_data;
    int _s = 0;
    int _e = 0;
    int _cache_number;
    int _compressed_level = 4;
    int _glob_id = 0;
    // train dataset
    std::vector<int> _random_idx; 
    int _cur_data_i = 0;
};
