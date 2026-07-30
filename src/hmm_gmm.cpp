#include "hmm_gmm.h"
#include "fileio.h"
#include "gaussianmixmodel.h"
#include "hmm.h"
#include "model.h"
#include "mymath.hpp"
#include "signalmath.hpp"
#include "sparsetransition.h"
#include <cstddef>
#include <fstream>
#include <string>
#include <iostream>

void StepFitHMMGMM::clear()
{
    if (_transition != nullptr)
    {
        delete _transition;
        _transition = nullptr;
    }
    for (int i = 0; i < _emission.size(); ++i)
    {
        delete _emission[i];
        _emission[i] = nullptr;
    }
    _emission.clear();
    _init_prob.clear();
    if (_hmm_model != nullptr)
    {
        delete _hmm_model;
        _hmm_model = nullptr;
    }
}

void StepFitHMMGMM::allocateModel(const std::string& sequence, const int extend_number, const int kmer) {
    clear();
    _state_number = sequence.size() * extend_number;
    _extend_number = extend_number;
    _ref_sequence = sequence;
    _transition = new SparseTransition();
    _emission.resize(_state_number);
    _kmer_name.resize(_state_number);
    const int kmer_pad = kmer / 2;
    std::unordered_map<std::string, GaussianMixModel*> kmer_model_cache;
    for (int i = 0; i < _state_number; ++i) {
        const int ref_pos = i / _extend_number;
        int start = ref_pos - kmer_pad;
        std::string sub_kmer;
        if (start < 0) {
            sub_kmer.append(-start, 'N');
            start = 0;
        }
        const int len = std::min(kmer - static_cast<int>(sub_kmer.size()),
                           static_cast<int>(sequence.size()) - start);
        if (len > 0) {
            sub_kmer += sequence.substr(start, len);
        }
        if (static_cast<int>(sub_kmer.size()) < kmer) {
            sub_kmer.append(kmer - sub_kmer.size(), 'N');
        }
        _kmer_name[i] = sub_kmer;
        if (i % _extend_number != 0) {
            _emission[i] = _emission[i - 1];
            continue;
        }
        if (auto it = kmer_model_cache.find(sub_kmer); it != kmer_model_cache.end()) {
            _emission[i] = it->second;
        } else {
            GaussianMixModel* new_model = new GaussianMixModel(1);
            _emission[i] = new_model;
            kmer_model_cache[sub_kmer] = new_model;
        }
    }
    _init_prob.resize(_state_number);
    std::fill(_init_prob.begin(), _init_prob.end(), static_cast<Real>(1.0) / static_cast<Real>(_state_number));
    _transition = new SparseTransition();
    _hmm_model = new HMM(_emission, _transition, _init_prob, _method);
    std::vector<Real> coo_val;
    std::vector<int> coo_x;
    std::vector<int> coo_y;
    for (int i = 0; i < _state_number; ++i) {
        coo_x.push_back(i);
        coo_y.push_back(i);
        coo_val.push_back(std::exp(-static_cast<Real>(extend_number) / 31));
        int j = i + 1;
        if (j >= _state_number) {
            continue;
        }
        coo_x.push_back(i);
        coo_y.push_back(j);
        coo_val.push_back(1.0 - std::exp(-static_cast<Real>(extend_number) / 31));
    }
    _transition->fill(coo_x.data(), coo_y.data(), coo_val.data(), coo_x.size(), _state_number);
    return;
}

void StepFitHMMGMM::loadModel(const std::string& fn_model, const std::string& sequence, const int extend_number, const int kmer) {
    clear();
    _state_number = sequence.size() * extend_number;
    _extend_number = extend_number;
    _ref_sequence = sequence;
    _transition = new SparseTransition();
    _emission.resize(_state_number);
    _kmer_name.resize(_state_number);
    const int kmer_pad = kmer / 2;
    std::unordered_map<std::string, GaussianMixModel*> kmer_model_cache;
    for (int i = 0; i < _state_number; ++i) {
        const int ref_pos = i / _extend_number;
        int start = ref_pos - kmer_pad;
        std::string sub_kmer;
        if (start < 0) {
            sub_kmer.append(-start, 'N');
            start = 0;
        }
        const int len = std::min(kmer - static_cast<int>(sub_kmer.size()),
                           static_cast<int>(sequence.size()) - start);
        if (len > 0) {
            sub_kmer += sequence.substr(start, len);
        }
        if (static_cast<int>(sub_kmer.size()) < kmer) {
            sub_kmer.append(kmer - sub_kmer.size(), 'N');
        }
        _kmer_name[i] = sub_kmer;
        if (i % _extend_number != 0) {
            _emission[i] = _emission[i - 1];
            continue;
        }
        if (auto it = kmer_model_cache.find(sub_kmer); it != kmer_model_cache.end()) {
            _emission[i] = it->second;
        } else {
            GaussianMixModel* new_model = new GaussianMixModel(1);
            _emission[i] = new_model;
            kmer_model_cache[sub_kmer] = new_model;
        }
    }
    _init_prob.clear();
    _transition = new SparseTransition();
    _hmm_model = new HMM(_emission, _transition, _init_prob, _method);
    std::vector<Real> coo_val;
    std::vector<int> coo_x;
    std::vector<int> coo_y;
    std::vector<std::vector<Real>> emits_table;
    std::ifstream fin(fn_model);
    if (!fin) {
        throw std::runtime_error("Cannot open file.");
    }
    std::string line;
    std::string flag;
    while (std::getline(fin, line))
    {
        if (line.empty())
            continue;
        if (line[0] == '#') {
            flag = line;
            continue;
        }
        if (flag == "#transition") {
            std::stringstream ss(line);
            Real val;
            int x, y;
            ss >> x >> y >> val;
            coo_val.push_back(val);
            coo_x.push_back(x);
            coo_y.push_back(y);
        }
        else if (flag == "#initial") {
            std::stringstream ss(line);
            Real val;
            ss >> val;
            _init_prob.push_back(val);
        }
        else if (flag == "#emission"){
            std::stringstream ss(line);
            emits_table.push_back({});
            Real val;
            while (ss >> val) {
                emits_table.back().push_back(val);
            }
        }
    }
    _transition->fill(coo_x.data(), coo_y.data(), coo_val.data(), coo_x.size(), _state_number);
    return;
}

void StepFitHMMGMM::saveModel(std::string& fn) const {
    std::ofstream file(fn);
    file << "#emission\n";
    for (int i = 0; i < _state_number; ++i)
    {
        std::vector<Real> val = _emission[i]->save();
        for (int j = 0; j < val.size(); ++j)
        {
            file << val[j] << " ";
        }
        file << "\n";
    }
    file << "#initial\n";
    for (int i = 0; i < _init_prob.size(); ++i) {
        file << _init_prob[i] << "\n";
    }
    file << "#transition\n";
    std::vector<int> coo_x(_transition->num_values());
    std::vector<int> coo_y(_transition->num_values());
    std::vector<Real> coo_val(_transition->num_values());
    _transition->save(coo_x.data(), coo_y.data(), coo_val.data());
    for (int i = 0; i < coo_x.size(); ++i)
    {
        file << coo_x[i] << " " << coo_y[i] << " " << coo_val[i] << std::endl;
    }
}

void StepFitHMMGMM::train(const int batch, const int max_iter,
                          const Real rate)
{
    int reads_number = _readsfile.readsNumber;
    _readsfile.reset();
    std::vector<Real> x_data;
    std::vector<size_t> x_batch;
    std::vector<ChunkInfo> x_info;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        int count = 0;
        int tag = _readsfile.readChunk(x_data, x_batch, x_info, 102, batch, 75);
        while(tag == 1) {
            _hmm_model->EM_step(x_data.data(), x_batch.data(), x_batch.size() - 1);
            tag = _readsfile.readChunk(x_data, x_batch, x_info, 102, batch, 75);
            count++;
            if (count >= 10) {break;}
        }
        _readsfile.reset();
        const Real resi = _hmm_model->update(rate);
        printf("iter: %d, prob: %f\n", iter, resi);
    }
}
void StepFitHMMGMM::infer(int batch)
{
    int reads_number = _readsfile.readsNumber;
    std::vector<Real> x_data;
    std::vector<int> x_label;
    std::vector<size_t> x_batch;
    std::vector<ChunkInfo> x_info;
    _readsfile.reset();
    int tag = _readsfile.readChunk(x_data, x_batch, x_info, 1024, 1, 76);
    for (int i = 0; i < 1; ++i) {
        std::string a = _readsfile.getSequence(x_info[i][0]);
        printf(a.c_str());
        printf("tag: %d\n", tag);
    }
    while(tag == 1) {
        x_label.resize(x_data.size());
        _hmm_model->infer(x_data.data(), x_label.data(), x_batch.data(), x_batch.size() - 1);
        std::vector<char> x_mv(x_data.size(), 1);
        for (int i = 1; i < x_data.size() - 1; ++i) {
            if ((x_label[i] != x_label[i - 1]) & (x_label[i] != x_label[i + 1])) {
                int state = x_label[i];
                auto b = _emission[state];
                continue;
            }
            if (i == 0) {
                x_mv[i] = 0;
            }
            else if (x_label[i] != x_label[i - 1]) {
                x_mv[i] = (x_mv[i - 1] == 1)? 0 : 1;
            }
            else {
                x_mv[i] = x_mv[i - 1];
            }
        }

        ReadsFile::save("./data.dat","aaa", x_data, x_mv, "1111111111111111");
        return;
        std::vector<int> val;
        for (int j = 0; j < x_label.size(); ++j) {
            if (val.size() == 0) {
                val.push_back(x_label[j]);
            }
            else if (val.back() != x_label[j]) {
                    val.push_back(x_label[j]);
            }
        }
        auto b = index_to_kmer(val.data(), val.size(), _kmer);
        printf(b.c_str());
        printf("\n");
        tag = _readsfile.readChunk(x_data, x_batch, x_info, 1024, 10, 768);
        for (int i = 0; i < 1; ++i) {
            std::string a = _readsfile.getSequence(x_info[i][0]);
            printf(a.c_str());
            printf("\n");
        }
        printf("tag: %d", tag);
    }
}