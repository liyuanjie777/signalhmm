#pragma once
#include "model.h"

class HistogramModel : public EmissionModel
{
public:
    HistogramModel(const Real* low_val, const Real* high_val, int* bins_num, int dim);
    ~HistogramModel() override{};

    void log_emission(const Real* obs, Real* log_probs, int num) const override;
    void score(const Real* obs, Real* log_probs, int num) const override;
    void update(Real alpha) override;
    void accumulate(const Real* x, const Real* gamma, int n) override;
    std::vector<Real> save();

private:
    std::vector<Real> count_;
    std::vector<Real> low_val_;
    std::vector<Real> high_val_;
    std::vector<Real> width_;
    std::vector<int> bins_;
    std::vector<Real> strides_;
    int dim_;
    size_t total_num_;

    std::vector<Real> count_accumulate_;
};
