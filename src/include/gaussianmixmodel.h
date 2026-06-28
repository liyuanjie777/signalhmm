#pragma once
#include "model.h"

class GaussianMixModel : public EmissionModel
{
public:
    GaussianMixModel(int npeak);
    ~GaussianMixModel() override{};

    void log_emission(const Real* obs, Real* log_probs, int num) const override;
    void score(const Real* obs, Real* distance, int num) const override;
    void update(Real alpha) override;
    void log_accumulate(const Real* x, const Real* gamma, int n) override;
    std::vector<Real> save() override;
    void setValues(const Real* values, int n) override;
    void reset();

private:
    std::vector<Real> means_;
    std::vector<Real> vars_;
    std::vector<Real> log_weights_;
    int npeak_;

    std::vector<double> sum_k_gamma_;
    std::vector<double> sum_k_x_;
    std::vector<double> sum_k_xx_;
};
