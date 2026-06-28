/*
 *  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-08-07
 *  License: MIT
 */
#include "variable.h" 
#include "mymath.hpp"

DiagMatrix::DiagMatrix(const int k, const int m, const std::vector<int>& segment, const std::vector<Real>& koff) {
	this->m = m;
	this->k = k;
	if (k >= m) {
		this->k = m - 1;
	}
	this->mat.resize(m, std::vector<Real>(k + 1, 0.0));
	this->mat_update.resize(m, std::vector<Real>(k + 1, -std::numeric_limits<Real>::infinity()));
	std::vector<Real> kall(m + k, 1.0);
	for (int seg = 0; seg < segment.size() - 1; ++seg) {
		for (int j = segment[seg]; j < segment[seg + 1]; ++j) {
			kall[j] = std::exp(-koff[seg]);
		}
	}
	kall[m - 1] = 1.0;
	for (int j = 0; j < m; ++j) {
		Real prod = 1.0;
		for (int i = 0; i < k; ++i) {
			Real p = 1.0 - kall[j + i];
			mat[j][i] = std::log(prod * (1.0 - p));
			prod *= p;
		}
		mat[j][k] = std::log(prod);
	}
}

void DiagMatrix::MatVecHad(const Real* beta, const Real* b, Real* y) {
	std::vector<Real> res(k + 1);
	for (int j = 0; j < m; ++j) {
		for (int i = 0; i <= k; ++i) {
			int id = j + i;
			if (id < m)
				res[i] = mat[j][i] + b[id] + beta[id];
			else
				res[i] = -std::numeric_limits<Real>::infinity();
		}
		y[j] = logsumexp(res.data(), k + 1);
	}
}

void DiagMatrix::VecMatHad(const Real* alpha, const Real* b, Real* y) {
	std::vector<Real> res(k + 1);
	for (int j = 0; j < m; ++j) {
		for (int i = 0; i <= k; ++i) {
			int id = j - i;
			if (id >= 0)
				res[i] = mat[id][i] + alpha[id];
			else 
				res[i] = -std::numeric_limits<Real>::infinity();
		}
		y[j] = logsumexp(res.data(), k + 1) + b[j];
	}
}

void DiagMatrix::getcol(const int j, std::vector<int>& idx, std::vector<Real>& val) {
	idx.clear();
	val.clear();
	if (idx.capacity() < k + 1)
		idx.reserve(k + 1);
	if (val.capacity() < k + 1)
		val.reserve(k + 1);
	for (int i = 0; i <= k; ++i) {
		int dj = j - i;
		if (dj >= 0) {
			idx.push_back(dj);
			val.push_back(mat[dj][i]);
		}
	}
	return;
}

void DiagMatrix::getid(const int j, std::vector<int>& idx) {
	idx.clear();
	if (idx.capacity() < k + 1)
		idx.reserve(k + 1);
	for (int i = 0; i <= k; ++i) {
		int dj = j - i;
		if (dj >= 0) {
			idx.push_back(dj);
		}
	}
	return;
}

void DiagMatrix::DiagMatVecHad(const Real* alpha, const Real* b, const Real* beta, Real* y) {
	for (int j = 0; j < m; ++j) {
		for (int i = 0; i <= k; ++i) {
			int id = j + i;
			if (id < m)
				y[j * (k + 1) + i] = mat[j][i] + beta[id] + b[id] + alpha[j];
			else
				y[j * (k + 1) + i] = -std::numeric_limits<Real>::infinity();
		}
		log_normalize(y + j * (k + 1), k + 1);
	}
	return;
}

void DiagMatrix::update(const Real* x) {
	for (int j = 0; j < m - 1; ++j)
		for (int i = 0; i <= k; ++i)
			mat[j][i] = x[j * (k + 1) + i];
	return;
}
