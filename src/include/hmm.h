#pragma once
#include "model.h"
#include "variable.h"
#include <cmath>
#include <cstring>

class HMM
{
public:
    HMM(const std::vector<EmissionModel*>& emit, TransitionModel* transit, std::vector<Real>& prob_pi, const char* method);
    void EM_step(const std::vector<std::vector<Real>>& datas, const std::vector<std::string>& sequences, const int data_dim, const int kmer);
    void infer(const std::vector<std::vector<Real>>& datas, std::vector<std::vector<int>>& labels, const int data_dim) const;
    Real update(Real rate);

private:
    void viterbi(const Real* data, int* label, const int n, const int data_dim) const;
    const char* method_;
    int iter_ = 0;
    float residual_ = 0;
    // external variables
    std::vector<EmissionModel*> emit_;
    TransitionModel* transit_;
    std::vector<Real>& log_pi_;
    // cache, temporary, or internal variables
    std::vector<Real> log_pi_cache_;
};
