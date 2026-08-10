#pragma once
#include "model.h"

class GaussianMixModel : public EmissionModel
{
public:
    GaussianMixModel(const int n_components, const int dim);
    ~GaussianMixModel() override{};

    void emission(const Real* obs, Real* probs, const int num) const override;
    void score(const Real* obs, Real* distance, const int num) const override;
    void update() override;
    static void update(GaussianMixModel** em, const int size);
    void accumulate(const Real* x, const Real* gamma, const int n) override;
    std::vector<Real> save() override;
    void setValues(const Real* values, const int n) override;
    void reset();

private:
    void loglikelihood(const Real* obs, Real* log_probs) const;

    std::vector<Real> means_;
    std::vector<Real> vars_;
    std::vector<Real> chol_cov_;
    std::vector<Real> log_weights_;
    int n_components_;
    int dim_;

    std::vector<Real> sum_k_gamma_;
    std::vector<Real> sum_k_x_;
    std::vector<Real> sum_k_xx_;
};
