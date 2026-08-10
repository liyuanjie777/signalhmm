/*
 *  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-08-07
 *  License: MIT
 */
#include "fileio.h"
#include "mymath.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/mman.h>
#include <vector>
#include <random>
#include <zstd.h>

std::string readFA(const std::string& fn, const std::string& target_chrom)
{
    std::ifstream infile(fn);
    if (!infile) {
        std::cerr << "Cannot open fa file: " << fn << "\n";
        return "";
    }
    std::string line, sequence;
    std::string current_seq_id = "";
    bool target_found = false;
    while (std::getline(infile, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty())
            continue;
        if (line[0] == '>') {
            if (target_found) {
                break;
            }
            size_t space_pos = line.find(' ');
            if (space_pos != std::string::npos) {
                current_seq_id = line.substr(1, space_pos - 1);
            } else {
                current_seq_id = line.substr(1);
            }
            if (current_seq_id == target_chrom) {
                target_found = true;
            }
        }
        else if (target_found) {
            sequence += line;
        }
    }
    return sequence;
}

void ReadsFile::load(const std::string& fn, bool shuffle)
{
    close();
    _filename = fn;
    _file.open(_filename, std::ios::binary | std::ios::in);
    if (!_file.is_open())
        return;
    const int uuid_length = 36;
    char uuid[uuid_length];
    int number;
    std::vector<char> chrom(100);
    while (_file.read(uuid, uuid_length))
    {
        int chrom_size;
        _file.read(reinterpret_cast<char*>(&chrom_size), 4);
        _file.read(chrom.data(), chrom_size);
        std::string cur_chrom = std::string(chrom.data(), chrom_size);
        if (_chrom_map.find(cur_chrom) == _chrom_map.end()) {
            _chrom_map[cur_chrom] = {};
        }

        _file.read(reinterpret_cast<char*>(&number), sizeof(number));
        _chrom_map[cur_chrom].poss_bytes.push_back(_file.tellg());
        _chrom_map[cur_chrom].pose_bytes.push_back(_chrom_map[cur_chrom].poss_bytes.back() + number);
        _uuids[std::string(uuid, uuid_length)] = _chrom_map[cur_chrom].poss_bytes.size() - 1;
        _chrom_map[cur_chrom].uuids.push_back(std::string(uuid, uuid_length));
        _file.seekg(number, std::ios::cur);
    }
    for (auto& [chrom, idx] : _chrom_map) {
        int readsNumber = idx.uuids.size();
        idx.random_idx.resize(readsNumber);
        for(int i = 0; i < readsNumber; ++i) {
            idx.random_idx[i] = i;
        }
        if (shuffle) {
            std::mt19937 gen(42);
            std::shuffle(idx.random_idx.begin(), idx.random_idx.end(), gen);
        }
        printf("chrom: %d reads\n", readsNumber);
    }

}

Read ReadsFile::read(const std::string& chrom)
{
    Read read_data;
    read_data.chrom = chrom;
    ReadIDX& read_idx = _chrom_map[read_data.chrom];
    const int readsNumber = read_idx.uuids.size();
    if (read_idx.cur_id >= readsNumber) {
        return read_data;
    }
    const int id = read_idx.random_idx[read_idx.cur_id];
    read_idx.cur_id += 1;
    std::fstream file(_filename, std::ios::binary | std::ios::in);
    if (!file)
        return read_data;
    read_data.uuid = read_idx.uuids[id];
    size_t start_bytes = read_idx.poss_bytes[id];
    size_t size_bytes = read_idx.pose_bytes[id] - start_bytes;
    file.seekg(start_bytes, std::ios::beg);
    _cache_data.resize(size_bytes);
    file.read(_cache_data.data(), _cache_data.size());
    std::vector<char> odata = decompress(_cache_data.data(), size_bytes);

    int data_size;
    char* ptr = odata.data();

    memcpy(&data_size, ptr, 4);
    ptr += 4;
    std::vector<float> data_float(static_cast<int>(data_size / sizeof(float)));
    memcpy(data_float.data(), ptr, data_size);
    for (auto it: data_float) {
        read_data.data.push_back(it);
    }
    ptr += data_size;

    memcpy(&data_size, ptr, 4);
    ptr += 4;
    std::vector<int> data_int(static_cast<int>(data_size / sizeof(int)));
    memcpy(data_int.data(), ptr, data_size);
    read_data.mv = data_int;
    ptr += data_size;
    return read_data;
}

std::vector<char> ReadsFile::decompress(const char* data, const size_t n)
{
  const unsigned long long decompressed_size = ZSTD_getFrameContentSize(data, n);
    if (decompressed_size == ZSTD_CONTENTSIZE_ERROR)
    {
        throw std::runtime_error("error zstd");
    }
    std::vector<char> decompressed_buffer(decompressed_size);
    const size_t actual_decompressed_size =
        ZSTD_decompress(decompressed_buffer.data(), decompressed_size, data, n);
    const size_t num_elements = actual_decompressed_size;
    std::vector<char> odata(num_elements);
    memcpy(odata.data(), decompressed_buffer.data(), actual_decompressed_size);
    return odata;
}

void ReadsFile::save(const std::string& fn, const std::vector<Read>& read_datas) {
    std::fstream ofile(fn, std::ios::binary | std::ios::app | std::ios::out);
    for (const Read& read_data: read_datas) {
        std::vector<float> data(read_data.data.size());
        for (int j = 0; j < read_data.data.size(); ++j) {
            data[j] = read_data.data[j];
        }

        size_t original_size = read_data.mv.size() * sizeof(int) + data.size() * sizeof(float) + 16;
        std::vector<char> idata(original_size);

        char* ptr = idata.data();
        int size_bytes = data.size() * sizeof(float);
        memcpy(ptr, &size_bytes, 4);
        ptr += 4;
        memcpy(ptr, data.data(), size_bytes);
        ptr += size_bytes;

        size_bytes = read_data.mv.size() * sizeof(int);
        memcpy(ptr, &size_bytes, 4);
        ptr += 4;
        memcpy(ptr, read_data.mv.data(), size_bytes);
        ptr += size_bytes;

        size_t compressed_bound = ZSTD_compressBound(original_size);
        std::vector<char> compressed_data(compressed_bound);
        size_t compressed_size = ZSTD_compress(compressed_data.data(), compressed_bound, idata.data(), original_size, 4);
        int compressed_size_int = compressed_size;
        int chrom_size = read_data.chrom.size();
        ofile.write(read_data.uuid.data(), read_data.uuid.size());
        ofile.write(reinterpret_cast<char*>(&chrom_size), 4);
        ofile.write(read_data.chrom.data(), chrom_size);
        ofile.write(reinterpret_cast<const char*>(&compressed_size_int), 4);
        ofile.write(compressed_data.data(), compressed_size);
    }
    ofile.close();
    return;
}

std::vector<Read> ReadsFile::readChunk(const int batch, const std::string& chrom, int min_size) {
    std::vector<std::string> keys;
    keys.reserve(_chrom_map.size());
    for (const auto& pair : _chrom_map) {
        keys.push_back(pair.first);
    }
    int num_keys = keys.size();
    std::vector<Read> res;
    for (int i = 0; i < batch; ++i) {
        std::string key;
        if (chrom.empty()) {
            key = keys[i % num_keys];
        }
        else {
            key = chrom;
        }
        Read read_data = read(key);
        if (read_data.data.size() < min_size) {
            continue;
        }
        res.push_back(read_data);
    }
    return res;
}
