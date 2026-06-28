/*
 *  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-08-07
 *  License: MIT
 */
#include "fileio.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/mman.h>
#include <vector>
#include <random>
#include <zstd.h>

std::string read_fa(std::string& fn)
{
    std::ifstream infile(fn);
    if (!infile)
    {
        std::cerr << "Cannot open fa file.\n";
        return "";
    }
    std::string line, sequence, seq_id;
    while (std::getline(infile, line))
    {
        if (line.empty())
            continue;
        if (line[0] == '>')
        {
            seq_id = line.substr(1);
        }
        else
        {
            sequence += line;
        }
    }
    return sequence;
}

void ReadsFile::load(std::string& fn, int max_cache, bool shuffle)
{
    _poss_bytes.clear();
    _pose_bytes.clear();
    _uuids.clear();
    readsNumber = 0;
    _random_idx.clear();
    _id_to_uuids.clear();
    
    _filename = fn;
    _cache_number = max_cache;
    std::fstream file(_filename, std::ios::binary | std::ios::in);
    if (!file)
        return;
    char uuid[16];
    int number;
    while (file.read(uuid, 16))
    {
        file.read(reinterpret_cast<char*>(&number), sizeof(number));
        _poss_bytes.push_back(file.tellg());
        _pose_bytes.push_back(_poss_bytes.back() + number);
        _uuids[std::string(uuid)] = _poss_bytes.size() - 1;
        _id_to_uuids.push_back(std::string(uuid));
        file.seekg(number, std::ios::cur);
    }
    readsNumber = _uuids.size();
    _random_idx.resize(readsNumber);
    for(int i = 0; i < readsNumber; ++i) {
        _random_idx[i] = i;
    }
    if (shuffle) {
        std::mt19937 gen(42);
        std::shuffle(_random_idx.begin(), _random_idx.end(), gen);
    }
    printf("ReadsFile: %d reads\n", readsNumber);
}

void ReadsFile::readID(std::string& sequence, std::vector<Real>& data, std::vector<char>& mv, const std::string& uuid)
{   
    sequence = "";
    data.clear();
    mv.clear();
    if (_uuids.find(uuid) == _uuids.end()) {
        return;
    }
    int id = _uuids[uuid];
    std::fstream file(_filename, std::ios::binary | std::ios::in);
    if (!file)
        return;
    size_t start_bytes = _poss_bytes[id];
    size_t size_bytes = _pose_bytes[id] - start_bytes;
    if (!(id >= _s && id < _e))
    {
        _s = id;
        _e = id + _cache_number;
        _e = (_e > readsNumber) ? readsNumber : _e;
        file.seekg(start_bytes, std::ios::beg);
        _cache_data.resize(_pose_bytes[_e - 1] - _poss_bytes[_s]);
        file.read(_cache_data.data(), _cache_data.size());
    }
    size_t offset_bytes = start_bytes - _poss_bytes[_s];
    std::vector<char> odata = decompress(&_cache_data[offset_bytes], size_bytes);

    int seq_size;
    char* ptr = odata.data();
    memcpy(&seq_size, ptr, 4);
    ptr += 4;
    sequence = std::string(ptr, seq_size);
    ptr += seq_size;

    int data_size;
    memcpy(&data_size, ptr, 4);
    ptr += 4;
    std::vector<float> data_float(int(data_size / sizeof(float)));
    memcpy(data_float.data(), ptr, data_size);
    data.clear();
    for (auto it: data_float) {
        data.push_back(it);
    }
    ptr += data_size;

    int mv_size;
    memcpy(&mv_size, ptr, 4);
    ptr += 4;
    mv.resize(mv_size);
    memcpy(mv.data(), ptr, mv_size);
    return;
}

void ReadsFile::read(std::string& sequence, std::vector<Real>& data, std::vector<char>& mv)
{
    sequence = "";
    data.clear();
    mv.clear();
    int id = _random_idx[_glob_id];
    if (_glob_id >= readsNumber) {
        return;
    }
    _glob_id += 1;
    std::fstream file(_filename, std::ios::binary | std::ios::in);
    if (!file)
        return;
    size_t start_bytes = _poss_bytes[id];
    size_t size_bytes = _pose_bytes[id] - start_bytes;
    if (!(id >= _s && id < _e))
    {
        _s = id;
        _e = id + _cache_number;
        _e = (_e > readsNumber) ? readsNumber : _e;
        file.seekg(start_bytes, std::ios::beg);
        _cache_data.resize(_pose_bytes[_e - 1] - _poss_bytes[_s]);
        file.read(_cache_data.data(), _cache_data.size());
    }
    size_t offset_bytes = start_bytes - _poss_bytes[_s];
    std::vector<char> odata = decompress(&_cache_data[offset_bytes], size_bytes);

    int seq_size;
    char* ptr = odata.data();
    memcpy(&seq_size, ptr, 4);
    ptr += 4;
    sequence = std::string(ptr, seq_size);
    ptr += seq_size;

    int data_size;
    memcpy(&data_size, ptr, 4);
    ptr += 4;
    std::vector<float> data_float(int(data_size / sizeof(float)));
    memcpy(data_float.data(), ptr, data_size);
    data.clear();
    for (auto it: data_float) {
        data.push_back(it);
    }
    ptr += data_size;

    int mv_size;
    memcpy(&mv_size, ptr, 4);
    ptr += 4;
    mv.resize(mv_size);
    memcpy(mv.data(), ptr, mv_size);
    return;
}

std::vector<char> ReadsFile::decompress(const char* data, size_t n)
{
    unsigned long long decompressed_size = ZSTD_getFrameContentSize(data, n);
    if (decompressed_size == ZSTD_CONTENTSIZE_ERROR)
    {
        throw std::runtime_error("error zstd");
    }
    std::vector<char> decompressed_buffer(decompressed_size);
    size_t actual_decompressed_size =
        ZSTD_decompress(decompressed_buffer.data(), decompressed_size, data, n);
    size_t num_elements = actual_decompressed_size;
    std::vector<char> odata(num_elements);
    memcpy(odata.data(), decompressed_buffer.data(), actual_decompressed_size);
    return odata;
}

void ReadsFile::saveRead(std::string& fn, const std::string& sequence, const std::vector<float>& data, const std::vector<char>& mv, const std::string& uuid)
{
    std::fstream ofile(fn, std::ios::binary | std::ios::app | std::ios::out);
    ofile.write(uuid.data(), 16);
    size_t original_size = sequence.size() + data.size() * sizeof(float) + mv.size() + 12;
    std::vector<char> idata(original_size);

    char* ptr = idata.data();
    int size_bytes = sequence.size();
    memcpy(ptr, &size_bytes, 4);
    ptr += 4; 
    memcpy(ptr, sequence.data(), size_bytes);
    ptr += size_bytes;

    size_bytes = data.size() * sizeof(float);
    memcpy(ptr, &size_bytes, 4);
    ptr += 4;
    memcpy(ptr, data.data(), size_bytes);
    ptr += size_bytes;

    size_bytes = mv.size();
    memcpy(ptr, &size_bytes, 4);
    ptr += 4;
    memcpy(ptr, mv.data(), size_bytes);

    size_t compressed_bound = ZSTD_compressBound(original_size);
    std::vector<char> compressed_data(compressed_bound);
    size_t compressed_size = ZSTD_compress(
        compressed_data.data(), compressed_bound, idata.data(), original_size, _compressed_level);
    int compressed_size_int = compressed_size;
    ofile.write(reinterpret_cast<const char*>(&compressed_size_int), sizeof(int));
    ofile.write(compressed_data.data(), compressed_size);
    return;
}

void ReadsFile::readChunk(std::vector<Real>& data, std::vector<size_t>& batchs, std::vector<ChunkInfo>& chunkinfo) {
    data.clear();
    batchs.clear();
    batchs.push_back(0);
    chunkinfo.clear();
    if (_glob_id >= readsNumber) {
        _glob_id = 0;
        _cur_data_i = 0;
        return;
    }
    std::string sequence; 
    std::vector<Real> read_data;
    std::vector<char> mv;
    int read_id;
    if (_cur_data_i != 0) {
        _glob_id -= 1;
        read_id = _random_idx[_glob_id];
        this->read(sequence, read_data, mv);
    }
    int cur_batch = 0; 
    while(cur_batch < _max_batch_size) {
        if (read_data.size() == 0) {
            read_id = _random_idx[_glob_id];
            this->read(sequence, read_data, mv);
        }
        if (read_data.size() == 0) {
            _glob_id = 0;
            _cur_data_i = 0;
            break;
        }
        if (_cur_data_i + _max_chunk_size >= read_data.size()) {
            data.insert(data.end(), read_data.begin() + _cur_data_i, read_data.end());
            batchs.push_back(data.size());
            ChunkInfo info = {read_id, _cur_data_i, read_data.size()};
            chunkinfo.push_back(info);
            read_data.clear();
            _cur_data_i = 0;
        }
        else {
            int cur_data_j = _cur_data_i + _max_chunk_size;
            data.insert(data.end(), read_data.begin() + _cur_data_i, read_data.begin() + cur_data_j);
            batchs.push_back(data.size());
            ChunkInfo info = {read_id, _cur_data_i, cur_data_j};
            chunkinfo.push_back(info);
            _cur_data_i += _stride;
        }
        cur_batch += 1;
    }
    return;
}
