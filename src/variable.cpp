/*
 *  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-08-07
 *  License: MIT
 */
#include "variable.h" 
#include "mymath.hpp"

SparseVectorView::SparseVectorView(const SparseVector& other): data(other.data), indices(other.indices), size(other.size) {
    if (size == 0) {
        data = nullptr;
        indices = nullptr;
    }
}

SparseVectorView& SparseVectorView::operator=(const SparseVector& other) {
    if (size == 0) {
        data = nullptr;
        indices = nullptr;
    }
    else {
        data = other.data;
        indices = other.indices;
    }
    size = other.size;
    return *this;
}

void SparseVectorView::logsumexp(const Real* val) {
    for (int i = 0; i < size; i++) {
        const int j = indices[i];
        data[i] = logsumexp2x(data[i], val[j]);
    }
}
