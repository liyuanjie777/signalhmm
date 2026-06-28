#pragma once
#include "model.h"

class SparseTransition : public TransitionModel
{
public:
    SparseTransition(const int* x, const int* y, const Real* val, int num, int dim);
    SparseTransition(){};
    ~SparseTransition() override{};

    void save(int* x, int* y, Real* val) override;
    void setValues(const int* x, const int* y, const Real* val, int num, int dim) override;

    Real log_transition(int i, int j) const override;
    void getrow(int i, std::vector<int>& id, std::vector<Real>& val) const override;
    void getcol(int j, std::vector<int>& id, std::vector<Real>& val) const override;
    void mulMV(const Real* idata, Real* odata) const override;
    void mulVM(const Real* idata, Real* odata) const override;
    int num_states() const override { return dim_; };
    int num_values() const override { return num_; };
    void epsilon_mstep(const Real* alpha, const Real* beta, Real* odata) const override;
    void epsilon_accumulate(const Real* idata, int n) override;
    void update(Real alpha) override;

private:
    void normalize();
    std::vector<Real> val_csr;
    std::vector<Real> val_csc;
    std::vector<int> indices_csr;
    std::vector<int> indices_csc;
    std::vector<int> ptr_csr;
    std::vector<int> ptr_csc;
    std::vector<int> csr_csc_map;
    int dim_;
    int num_;

    std::vector<Real> cache_;
};