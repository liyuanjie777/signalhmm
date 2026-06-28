#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

using Real = float;
using Mat = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vec = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using ChunkInfo = std::array<int, 3>;
constexpr double PI = 3.14159265358979323846;
constexpr double MINVAL = 1e-12;

class DiagMatrix
{
  public:
    DiagMatrix(const int k, const int m, const std::vector<int> &segment, const std::vector<Real> &koff);
    int k = 0;
    int m = 0;
    void MatVecHad(const Real *beta, const Real *b, Real *y);
    void VecMatHad(const Real *alpha, const Real *b, Real *y);
    void DiagMatVecHad(const Real *alpha, const Real *b, const Real *beta, Real *y);
    Real operator()(int i, int j) const
    {
        if (j < i || (j - i) >= k || j >= m || i >= m || i < 0 || j < 0)
            return 0;
        else
        {
            return mat[i][j - i];
        }
    }
    void getcol(const int j, std::vector<int> &idx, std::vector<Real> &val);
    void getid(const int j, std::vector<int> &idx);
    void update(const Real *x);
    std::vector<std::vector<Real>> mat;
    std::vector<std::vector<Real>> mat_update;
    std::vector<Real> ytmp;
};