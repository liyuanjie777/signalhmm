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

void StepFitHMMGMM::allocateModel(const std::string& fn_fa, const std::string& chrom, const int extend_number, const int peak) {
    clear();
    _sequence = readFA(fn_fa, chrom);
    _chrom = chrom;
    std::reverse(_sequence.begin(), _sequence.end());
    _state_number = _sequence.size() * extend_number;
    _extend_number = extend_number;
    _transition = new SparseTransition();
    _emission.resize(_state_number);
    _peak = peak;
    for (int i = 0; i < _state_number; ++i) {
        if (i % _extend_number != 0) {
            _emission[i] = _emission[i - 1];
            continue;
        }
        if (i == 0) {
            GaussianMixModel* new_model = new GaussianMixModel(peak * 3);
            _emission[i] = new_model;
        }
        else {
            GaussianMixModel* new_model = new GaussianMixModel(peak);
            _emission[i] = new_model;
        }
    }
    _init_prob.resize(_state_number);
    std::fill(_init_prob.begin(), _init_prob.end(), -std::numeric_limits<Real>::infinity());
    _init_prob[0] = 0.0;

    _transition = new SparseTransition();
    _hmm_model = new HMM(_emission, _transition, _init_prob);
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

void StepFitHMMGMM::loadModel(const std::string& fn_model, const std::string& fn_fa, const std::string& chrom, const int extend_number, const int peak) {
    clear();
    _sequence = readFA(fn_fa, chrom);
    _chrom = chrom;
    std::reverse(_sequence.begin(), _sequence.end());
    _state_number = _sequence.size() * extend_number;
    _extend_number = extend_number;
    _transition = new SparseTransition();
    _emission.resize(_state_number);
    _peak = peak;
    for (int i = 0; i < _state_number; ++i) {
        if (i % _extend_number != 0) {
            _emission[i] = _emission[i - 1];
            continue;
        }
        if (i == 0) {
            GaussianMixModel* new_model = new GaussianMixModel(peak * 3);
            _emission[i] = new_model;
        }
        else {
            GaussianMixModel* new_model = new GaussianMixModel(peak);
            _emission[i] = new_model;
        }
    }
    _init_prob.clear();
    _transition = new SparseTransition();
    _hmm_model = new HMM(_emission, _transition, _init_prob);
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
    for (int i = 0; i < _emission.size(); ++i) {
        _emission[i]->setValues(emits_table[i].data(), emits_table[i].size());
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
        file << _sequence[i / _extend_number] << " ";
        for (int j = 0; j < val.size(); ++j)
        {
            file << val[j] << " ";
        }
        file << "\n";
    }
    file << "#initial\n";
    for (int i = 0; i < _init_prob.size(); ++i) {
        file << std::exp(_init_prob[i]) << "\n";
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

void StepFitHMMGMM::train(const int batch, const int data_dim, const int max_iter, const int sampling, const Real rate, const char* method, const int max_band) {
    _readsfile.reset();
    std::vector<Real> x_data;
    std::vector<size_t> x_batch;
    std::vector<ChunkInfo> x_info;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        int count = 0;
        std::vector<Read> tags = _readsfile.readChunk(batch, _chrom);
        while(tags.size() > 0) {
            std::vector<std::vector<Real>> datas;
            std::vector<std::vector<int>> mvs;
            for (auto tag : tags) {
                if (tag.data.size() <= _state_number) {
                    continue;
                }
                datas.push_back(tag.data);
                mvs.push_back({});
                count++;
            }
            _hmm_model->EM_step(datas, data_dim, method, max_band, mvs);
            tags = _readsfile.readChunk(batch, _chrom);
            if (sampling > 0 && count >= sampling) {break;}
        }
        _readsfile.reset();
        const Real resi = _hmm_model->update(rate, method);
        printf("iter: %d, prob: %f\n", iter, resi / static_cast<Real>(count * batch));
    }
}
void StepFitHMMGMM::infer(const int batch, const int data_dim, const std::string& fn_out)
{
    std::fstream ofile(fn_out, std::ios::binary | std::ios::trunc | std::ios::out);
    ofile.close();
    _readsfile.reset();
    std::vector<Read> tags = _readsfile.readChunk(batch, _chrom);
    while(tags.size() > 0) {
        std::vector<std::vector<Real>> datas;
        for (auto tag : tags) {
            datas.push_back(tag.data);
        }
        std::vector<std::vector<int>> labels(datas.size());
        _hmm_model->infer(datas, labels, data_dim);
        for (int i = 0; i < labels.size(); ++i) {
            tags[i].mv = labels[i];
        }
        _readsfile.save(fn_out, tags);
        return;
    }
}