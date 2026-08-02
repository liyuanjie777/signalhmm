#pragma once
#include "variable.h"
#include <vector>

class EmissionModel
{
public:
    virtual ~EmissionModel(){};
    virtual void log_emission(const Real* obs, Real* log_probs, int num) const = 0;
    virtual void score(const Real* obs, Real* distance, int num) const = 0;
    virtual void update(Real alpha) = 0;
    virtual void log_accumulate(const Real* x, const Real* gamma, int n) = 0;
    virtual void setValues(const Real* values, int n) = 0;
    virtual std::vector<Real> save() = 0;
};

class TransitionModel
{
public:
    virtual ~TransitionModel() {}
    virtual Real log_transition(int i, int j) const = 0;
    virtual SparseVectorView get_row(int i) = 0;
    virtual SparseVectorView get_col(int j) = 0;
    virtual void mulMV(const SparseVectorView& idata, Real* odata) const = 0;
    virtual void mulVM(const SparseVectorView& idata, Real* odata) const = 0;
    virtual int num_states() const = 0;
    virtual int num_values() const = 0;
    virtual void save(int* x, int* y, Real* val) = 0;
    virtual void fill(const int* x, const int* y, const Real* val, int num, int dim) = 0;
    virtual void epsilon_E_step(const SparseVectorView& alpha, const SparseVectorView& beta, Real* odata) const {}
    virtual void epsilon_M_step(const Real* idata, int n) {}
    virtual void update(Real alpha) {};
};


