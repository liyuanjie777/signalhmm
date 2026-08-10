#pragma once
#include "model.h"
#include "variable.h"
#include <cmath>
#include <cstring>

class HMM
{
public:
    HMM(const std::vector<EmissionModel*>& emit, TransitionModel* transit, std::vector<Real>& prob_pi);
    void EM_step(const std::vector<std::vector<Real>>& datas, const int data_dim, const char* method, const std::vector<std::vector<std::vector<int>>>& adj_lists);
    void infer(const std::vector<std::vector<Real>>& datas, std::vector<std::vector<int>>& labels, const int data_dim, std::vector<Real>& scores) const;
    Real update(const char* method);

private:
    Real viterbi(const Real* data, int* label, const int n, const int m) const;
    int iter_ = 0;
    float residual_ = 0;
    // external variables
    std::vector<EmissionModel*> emit_;
    TransitionModel* transit_;
    std::vector<Real>& pi_;
    // cache, temporary, or internal variables
    std::vector<Real> pi_cache_;
    std::vector<std::vector<int>> shared_emit_;
};
