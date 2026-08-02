#pragma once
#include "model.h"

class SparseTransition : public TransitionModel
{
public:
    SparseTransition(const int* x, const int* y, const Real* val, const int num, const int dim);
    SparseTransition(){};
    ~SparseTransition() override{};

    void save(int* x, int* y, Real* val) override;
    void fill(const int* x, const int* y, const Real* val, const int num, const int dim) override;

    Real log_transition(int i, int j) const override;
    SparseVectorView get_row(int i) override;
    SparseVectorView get_col(int j) override;
    void mulMV(const SparseVectorView& idata, Real* odata) const override;
    void mulVM(const SparseVectorView& idata, Real* odata) const override;
    int num_states() const override { return dim_; };
    int num_values() const override { return num_; };
    void epsilon_E_step(const SparseVectorView& alpha, const SparseVectorView& beta, Real* odata) const override;
    void epsilon_M_step(const Real* idata, int n) override;
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