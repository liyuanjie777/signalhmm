#pragma once
#include "model.h"
#include "variable.h"
#include <cmath>
#include <cstring>

class HMM
{
public:
    HMM(const std::vector<EmissionModel*>& emit, TransitionModel* transit, std::vector<Real>& prob_pi);
    void EM_step(const std::vector<std::vector<Real>>& datas, const int data_dim, const char* method, const int max_band, const std::vector<std::vector<int>>& mvs);
    void infer(const std::vector<std::vector<Real>>& datas, std::vector<std::vector<int>>& labels, const int data_dim) const;
    Real update(Real rate, const char* method);

private:
    void viterbi(const Real* data, int* label, const int n, const int data_dim) const;
    int iter_ = 0;
    float residual_ = 0;
    // external variables
    std::vector<EmissionModel*> emit_;
    TransitionModel* transit_;
    std::vector<Real>& log_pi_;
    // cache, temporary, or internal variables
    std::vector<Real> log_pi_cache_;
    std::vector<std::vector<int>> shared_emit_;
};
