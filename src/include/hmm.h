#pragma once
#include "model.h"
#include "variable.h"
#include <cmath>
#include <cstring>

class HMM
{
public:
    HMM(const std::vector<EmissionModel*>& emit, TransitionModel* transit, std::vector<Real>& prob_pi,
        int dim, int state, int coo_num, const char* method);
    void EM_step(const Real* data, const size_t* batch, const int nbatch);
    void infer(const Real* data, int* label, const size_t* batch, const int nbatch) const;
    Real update(Real rate);

private:

    void viterbi(const Real* data, int* label, const int n) const;
    size_t dim_;
    size_t state_;
    size_t coo_num_;
    const char* method_;
    int iter_ = 0;
    float residual_ = 0;
    // external variables
    std::vector<EmissionModel*> emit_;
    TransitionModel* transit_;
    std::vector<Real>& log_pi_;
    // cache, temporary, or internal variables
    std::vector<Real> log_pi_cache_;
    std::vector<std::vector<int>> shared_emit_;
    std::vector<Real> gamma_;
    std::vector<Real> epsilon_;
};
