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



StepFitHMMGMM::StepFitHMMGMM(int kmer, int target, const char* method)
{
    _kmer = kmer;
    _state_number = (1ULL << (2 * kmer));
    _transition = new SparseTransition();
    for(size_t i = 0; i < _state_number; ++i) {
        bool has_target = false;
        for (int j = 0; j < kmer; ++j) {
            size_t base = (i >> (j * 2)) & 0b11;
            if (base == target) {
                has_target = true;
            }
        }
        if (has_target) {
            _emission.push_back(new GaussianMixModel(1));
        }
        else {
            _emission.push_back(new GaussianMixModel(1));
        }
    }
    _init_prob.resize(_state_number);
    std::fill(_init_prob.begin(), _init_prob.end(), Real(1.0) / Real(_state_number));
    _transition = new SparseTransition();
    _hmm_model = new HMM(_emission, _transition, _init_prob, 1, _state_number, _state_number * 5 - 4, method);
}

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
    }
    _emission.clear();
    _init_prob.clear();
    if (_hmm_model == nullptr)
    {
        delete _hmm_model;
        _hmm_model = nullptr;
    }
}

void StepFitHMMGMM::preTrain(std::string& fn)
{
    _filename = fn;
    _readsfile.load(fn, 500, true);
    std::string seq;
    std::vector<Real> data;
    std::vector<char> mv;
    constexpr int loop = 1;
    for (int k = 0; k < loop; ++k) {
        for (int i = 0; i < 10000; ++i) {
            _readsfile.read(seq, data, mv);
            std::vector<int> idx = kmer_to_index(seq.data(), seq.size(), _kmer);
            std::vector<Real> current = kmer_current(data, mv);
            for(int j = 0; j < idx.size(); ++j) {
                Real val = current[j + _kmer / 2];
                Real gamma = 0.0;
                _emission[idx[j]]->log_accumulate(&val, &gamma, 1);
            }
        }
        for (int i = 0; i < _emission.size(); ++i) {
            _emission[i]->update(1.0);
        }
    }
    std::vector<Real> coo_val;
    std::vector<int> coo_x;
    std::vector<int> coo_y;
    kmer_matrix(coo_x, coo_y, coo_val, _kmer, Real(0.9));
    _transition->setValues(coo_x.data(), coo_y.data(), coo_val.data(), coo_x.size(), _state_number);
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