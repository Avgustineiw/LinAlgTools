#pragma once
#include "../types/matrix.h"
#include "../types/submatrix.h"

#include <cmath>
#include <cstdint>
#include <utility>

namespace LinAlgTools::Algorithm {
using Index = int64_t;

std::pair<Matrix<double>, double> HouseholderReduction(SubMatrix<double>& x) {
        int m = x.Rows();
        Matrix<double> v(m, 1);

        double sigma = 0.0;
        for (int i = 1; i < m; ++i) {
                sigma += x(i, 0) * x(i, 0);
        }

        if (sigma == 0.0) {
                Matrix<double> v_zero(m, 1, 0.0);
                return {v_zero, (x(0, 0) <= 0) ? 2.0 : 0.0};
        }

        double mu = std::sqrt(x(0, 0) * x(0, 0) + sigma);

        if (x(0, 0) <= 0) {
                v(0, 0) = x(0, 0) - mu;
        }
        else {
                v(0, 0) = -sigma / (x(0, 0) + mu);
        }

        for (int i = 1; i < m; ++i) {
                v(i, 0) = x(i, 0);
        }

        double v0 = v(0, 0);
        double vTv = 1.0;
        for (int i = 1; i < m; ++i) {
                v(i, 0) /= v0;
                vTv += v(i, 0) * v(i, 0);
        }
        v(0, 0) = 1.0;
        double beta = 2.0 / vTv;

        return {v, beta};
}

// void QR_Householder(Matrix<double>& matrix) {
        // for (Index col = 0; col < std::min(matrix.Rows(), matrix.Columns()); col++) {
        //         LinAlgTools::SubMatrix<double> vec = {matrix, {col, matrix.Rows() - 1}, {col, col}};
        //         LinAlgTools::SubMatrix<double> subMatrix = {matrix, {col, matrix.Rows() - 1}, {col, matrix.Columns() - 1}};
        //
        //         auto res = LinAlgTools::Algorithm::HouseholderReduction(vec);
        //         auto v = res.first;
        //         auto beta = res.second;
        //
        //         auto v_transposed = v.Transposed();
        //         auto outer_product = v * v_transposed;
        //
                // auto matrix_res = subMatrix - (beta * v) * (v_transposed * subMatrix);
        // }
// }
}// namespace LinAlgTools::Algorithm
