#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>


class SparseVector;
class SparseVectorView;
using Real = float;
using Mat = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vec = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using ChunkInfo = std::array<int, 3>;
constexpr double PI = 3.14159265358979323846;
constexpr double MINVAL = 1e-12;

class SparseVectorView {
public:
    SparseVectorView(Real* val, const int* id, const int num) : data(val), indices(id), size(num) {};
    SparseVectorView(): data(nullptr), indices(nullptr), size(0) {};
    explicit SparseVectorView(const SparseVector& other);
    SparseVectorView& operator=(const SparseVector& other);

    void logsumexp(const Real* val);

    Real* data;
    const int* indices;
    int size;
};

class SparseVector {
public:
    SparseVector(): data(nullptr), indices(nullptr), size(0), _capacity(0) {};
    SparseVector(const SparseVector& other) : indices(other.indices), size(other.size), _capacity(other._capacity) {
        data = (other.data) ? new Real[_capacity] : nullptr;
        if (data) std::copy_n(other.data, size, data);
    }
    explicit SparseVector(const SparseVectorView& other) : indices(other.indices), size(other.size), _capacity(other.size) {
        data = (other.data) ? new Real[size] : nullptr;
        if (data) std::copy_n(other.data, size, data);
    }
    SparseVector& operator=(const SparseVectorView& other) {
        if (other.data == nullptr) {
            size = 0;
            indices = nullptr;
            return *this;
        }
        if (_capacity < other.size) {
            delete[] data;
            _capacity = other.size;
            size = other.size;
            indices = other.indices;
            data = new Real[_capacity];
            std::copy_n(other.data, size, data);
        }
        else {
            size = other.size;
            indices = other.indices;
            std::copy_n(other.data, size, data);
        }
        return *this;
    }
    ~SparseVector() {
        if (data != nullptr) {
            delete[] data;
            data = nullptr;
        }
    }
    Real* data;
    const int* indices;
    int size;
private:
    int _capacity;
};
