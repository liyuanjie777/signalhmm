#pragma once
#include "model.h"

class GaussianModel : public EmissionModel
{
public:
    GaussianModel(int dim);
    ~GaussianModel() override{};

    void log_emission(const Real* obs, Real* log_probs, int num) const override;
    void score(const Real* obs, Real* distance, int num) const override;
    void update(Real alpha) override;
    void log_accumulate(const Real* x, const Real* gamma, int n) override;
    std::vector<Real> save() override;
    void setValues(const Real* values, int n) override;
    void reset();

private:
    std::vector<Real> mean_;
    std::vector<Real> chol_cov_;
    Real log_norm_;
    Real weight_;
    int dim_;
    // temporary values
    std::vector<Real> M2_;
    std::vector<Real> mean_new_;
    double sum_w_;
    double sum_w2_;
};
